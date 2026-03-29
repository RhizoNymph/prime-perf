"""Phase 0 task 0.6: Verify optimization signal is above noise floor.

Compiles both naive and tiled matmul, runs each 10 times, and confirms
the improvement ratio (ref - opt) / ref is well above measurement noise.

Run with:
    uv run pytest -m variance tests/validation/test_signal_detection.py -v
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from perf_optimize.config import SandboxConfig
from perf_optimize.sandbox import PerfSandbox
from perf_optimize.types import PerfCounters
from tests.conftest import requires_all_tools

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "c_programs"
N_SIGNAL_SAMPLES = 10
MATMUL_N = 512  # Large enough for clear perf differences


def _make_matmul_input(n: int, seed: int = 42) -> bytes:
    rng = np.random.default_rng(seed)
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    return struct.pack("i", n) + a.tobytes() + b.tobytes()


@requires_all_tools
@pytest.mark.variance
@pytest.mark.asyncio
class TestSignalDetection:
    """Verify that naive vs tiled matmul produces detectable counter differences."""

    async def _collect_samples(
        self,
        sandbox: PerfSandbox,
        source_file: str,
        n_samples: int,
    ) -> list[PerfCounters]:
        """Compile a program and collect perf samples."""
        source = FIXTURES.joinpath(source_file).read_text()
        _compilation, work_dir = await sandbox.compile_only(source)

        binary_path = Path(work_dir) / "solution"
        input_data = _make_matmul_input(MATMUL_N)
        input_path = Path(work_dir) / "perf_input.bin"
        input_path.write_bytes(input_data)

        try:
            samples: list[PerfCounters] = []
            for _ in range(n_samples):
                counters = await sandbox.measure_only(binary_path, input_path)
                samples.append(counters)
            return samples
        finally:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)

    async def test_cycle_improvement_above_noise(self) -> None:
        """Tiled matmul should show cycle improvement well above measurement CV."""
        config = SandboxConfig()
        sandbox = PerfSandbox(config)

        naive_samples = await self._collect_samples(sandbox, "matmul_naive.c", N_SIGNAL_SAMPLES)
        tiled_samples = await self._collect_samples(sandbox, "matmul_tiled.c", N_SIGNAL_SAMPLES)

        naive_cycles = np.array([s.cycles for s in naive_samples])
        tiled_cycles = np.array([s.cycles for s in tiled_samples])

        naive_mean = float(np.mean(naive_cycles))
        tiled_mean = float(np.mean(tiled_cycles))
        improvement = (naive_mean - tiled_mean) / naive_mean

        # CV of the naive measurements (noise floor)
        naive_cv = float(np.std(naive_cycles, ddof=1) / np.mean(naive_cycles))

        print(f"\nnaive cycles: mean={naive_mean:.0f} CV={naive_cv*100:.2f}%")
        print(f"tiled cycles: mean={tiled_mean:.0f}")
        print(f"improvement: {improvement*100:.1f}%")
        print(f"signal/noise ratio: {improvement/naive_cv:.1f}x")

        # The improvement should be at least 3x the measurement noise
        assert improvement > 3 * naive_cv, (
            f"Cycle improvement ({improvement*100:.1f}%) is not clearly above "
            f"noise floor (CV={naive_cv*100:.2f}%, 3x={3*naive_cv*100:.2f}%)"
        )

        # The tiled version should be meaningfully faster
        assert improvement > 0.10, (
            f"Expected >10% cycle improvement from tiling, got {improvement*100:.1f}%"
        )

    async def test_l1_cache_miss_improvement(self) -> None:
        """Tiled matmul should dramatically reduce L1 cache misses."""
        config = SandboxConfig()
        sandbox = PerfSandbox(config)

        naive_samples = await self._collect_samples(sandbox, "matmul_naive.c", N_SIGNAL_SAMPLES)
        tiled_samples = await self._collect_samples(sandbox, "matmul_tiled.c", N_SIGNAL_SAMPLES)

        naive_misses = np.array([s.l1_dcache_load_misses for s in naive_samples])
        tiled_misses = np.array([s.l1_dcache_load_misses for s in tiled_samples])

        naive_mean = float(np.mean(naive_misses))
        tiled_mean = float(np.mean(tiled_misses))
        improvement = (naive_mean - tiled_mean) / naive_mean

        print(f"\nnaive L1 misses: mean={naive_mean:.0f}")
        print(f"tiled L1 misses: mean={tiled_mean:.0f}")
        print(f"improvement: {improvement*100:.1f}%")

        # Tiling should dramatically reduce L1 cache misses
        assert improvement > 0.20, (
            f"Expected >20% L1 cache miss improvement from tiling, got {improvement*100:.1f}%"
        )

    async def test_reward_signal_is_meaningful(self) -> None:
        """The weighted reward signal should be clearly positive for the optimized version."""
        config = SandboxConfig()
        sandbox = PerfSandbox(config)

        naive_samples = await self._collect_samples(sandbox, "matmul_naive.c", N_SIGNAL_SAMPLES)
        tiled_samples = await self._collect_samples(sandbox, "matmul_tiled.c", N_SIGNAL_SAMPLES)

        # Compute reward using the design doc's formula, skipping None counters
        # and redistributing weight among available counters.
        def _improvement(ref_val: float | None, agent_val: float | None) -> float | None:
            if ref_val is None or agent_val is None or ref_val <= 0:
                return None
            return (ref_val - agent_val) / ref_val

        def _median_field(samples: list[PerfCounters], field: str) -> float | None:
            vals = [getattr(s, field) for s in samples]
            if vals[0] is None:
                return None
            return float(np.median(vals))

        naive_med = PerfCounters(
            cycles=float(np.median([s.cycles for s in naive_samples])),
            instructions=float(np.median([s.instructions for s in naive_samples])),
            cache_misses=_median_field(naive_samples, "cache_misses"),
            l1_dcache_load_misses=_median_field(naive_samples, "l1_dcache_load_misses"),
            llc_load_misses=_median_field(naive_samples, "llc_load_misses"),
            branch_misses=_median_field(naive_samples, "branch_misses"),
        )
        tiled_med = PerfCounters(
            cycles=float(np.median([s.cycles for s in tiled_samples])),
            instructions=float(np.median([s.instructions for s in tiled_samples])),
            cache_misses=_median_field(tiled_samples, "cache_misses"),
            l1_dcache_load_misses=_median_field(tiled_samples, "l1_dcache_load_misses"),
            llc_load_misses=_median_field(tiled_samples, "llc_load_misses"),
            branch_misses=_median_field(tiled_samples, "branch_misses"),
        )

        # Weighted improvement: skip None counters, normalize by available weight
        weight_map = {
            "cycles": 0.5,
            "l1_dcache_load_misses": 0.2,
            "llc_load_misses": 0.2,
            "cache_misses": 0.2,  # AMD substitute for LLC
            "branch_misses": 0.1,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for field, weight in weight_map.items():
            imp = _improvement(getattr(naive_med, field), getattr(tiled_med, field))
            if imp is not None:
                weighted_sum += weight * imp
                total_weight += weight

        reward = max(weighted_sum / total_weight, 0.0) if total_weight > 0 else 0.0

        print(f"\nReward for tiled vs naive: {reward:.4f}")
        print(f"Naive IPC: {naive_med.ipc:.2f}")
        print(f"Tiled IPC: {tiled_med.ipc:.2f}")

        assert reward > 0.05, (
            f"Expected reward > 0.05 for tiled vs naive matmul, got {reward:.4f}"
        )
