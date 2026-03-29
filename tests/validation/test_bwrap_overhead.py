"""Phase 0 task 0.5: Compare bwrap vs bare execution variance.

Confirms bwrap adds no measurable overhead to perf counter readings.
Uses Mann-Whitney U test to check if distributions differ significantly.

Run with:
    uv run pytest -m variance tests/validation/test_bwrap_overhead.py -v
"""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import stats

from perf_optimize.config import SandboxConfig
from perf_optimize.perf_parser import parse_perf_output
from perf_optimize.sandbox import PerfSandbox
from tests.conftest import requires_all_tools

if TYPE_CHECKING:
    from perf_optimize.types import PerfCounters

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "c_programs"
N_SAMPLES = 30
MATMUL_N = 256  # Smaller N since we run many times
SIGNIFICANCE_LEVEL = 0.01  # Conservative: only flag if p < 0.01


def _make_matmul_input(n: int, seed: int = 42) -> bytes:
    rng = np.random.default_rng(seed)
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    return struct.pack("i", n) + a.tobytes() + b.tobytes()


async def _measure_bare(
    config: SandboxConfig,
    binary_path: Path,
    input_path: Path,
) -> PerfCounters:
    """Run perf stat directly (no bwrap) for comparison."""
    counter_list = ",".join(c.value for c in config.perf_counters)
    cmd = [
        config.taskset_path,
        "-c",
        str(config.pin_cpu),
        config.perf_path,
        "stat",
        "-r",
        str(config.perf_repeat),
        "-x",
        ",",
        "-e",
        counter_list,
        "--",
        str(binary_path),
    ]

    stdin_data = input_path.read_bytes()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr_raw = await asyncio.wait_for(
        proc.communicate(input=stdin_data),
        timeout=config.perf_timeout_s,
    )

    stderr_str = stderr_raw.decode("utf-8", errors="replace")
    return parse_perf_output(stderr_str)


@requires_all_tools
@pytest.mark.variance
@pytest.mark.asyncio
class TestBwrapOverhead:
    """Compare perf counter distributions with and without bwrap."""

    async def test_bwrap_does_not_alter_cycle_distribution(self) -> None:
        """Mann-Whitney U test: bwrap cycles vs bare cycles should not differ significantly."""
        config = SandboxConfig()
        sandbox = PerfSandbox(config)

        # Compile outside bwrap for the bare test
        source = FIXTURES.joinpath("matmul_naive.c").read_text()
        _compilation, work_dir = await sandbox.compile_only(source)

        binary_path = Path(work_dir) / "solution"
        input_data = _make_matmul_input(MATMUL_N)
        input_path = Path(work_dir) / "perf_input.bin"
        input_path.write_bytes(input_data)

        try:
            # Collect bwrap samples
            bwrap_cycles: list[float] = []
            for _ in range(N_SAMPLES):
                counters = await sandbox.measure_only(binary_path, input_path)
                bwrap_cycles.append(counters.cycles)

            # Collect bare samples
            bare_cycles: list[float] = []
            for _ in range(N_SAMPLES):
                counters = await _measure_bare(config, binary_path, input_path)
                bare_cycles.append(counters.cycles)

            # Mann-Whitney U test (non-parametric, doesn't assume normality)
            u_stat, p_value = stats.mannwhitneyu(
                bwrap_cycles, bare_cycles, alternative="two-sided"
            )

            bwrap_arr = np.array(bwrap_cycles)
            bare_arr = np.array(bare_cycles)

            print(f"\nbwrap cycles: mean={np.mean(bwrap_arr):.0f} std={np.std(bwrap_arr):.0f}")
            print(f"bare  cycles: mean={np.mean(bare_arr):.0f} std={np.std(bare_arr):.0f}")
            print(f"Mann-Whitney U={u_stat:.1f}, p={p_value:.6f}")
            print(
                f"Relative difference: "
                f"{abs(np.mean(bwrap_arr) - np.mean(bare_arr)) / np.mean(bare_arr) * 100:.2f}%"
            )

            assert p_value >= SIGNIFICANCE_LEVEL, (
                f"bwrap significantly alters cycle counts "
                f"(p={p_value:.6f} < {SIGNIFICANCE_LEVEL}). "
                f"bwrap mean={np.mean(bwrap_arr):.0f}, "
                f"bare mean={np.mean(bare_arr):.0f}"
            )
        finally:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)

    async def test_bwrap_does_not_alter_cache_miss_distribution(self) -> None:
        """Mann-Whitney U test: bwrap L1 cache misses vs bare should not differ."""
        config = SandboxConfig()
        sandbox = PerfSandbox(config)

        source = FIXTURES.joinpath("matmul_naive.c").read_text()
        _compilation, work_dir = await sandbox.compile_only(source)

        binary_path = Path(work_dir) / "solution"
        input_data = _make_matmul_input(MATMUL_N)
        input_path = Path(work_dir) / "perf_input.bin"
        input_path.write_bytes(input_data)

        try:
            bwrap_misses: list[float] = []
            for _ in range(N_SAMPLES):
                counters = await sandbox.measure_only(binary_path, input_path)
                bwrap_misses.append(counters.l1_dcache_load_misses)

            bare_misses: list[float] = []
            for _ in range(N_SAMPLES):
                counters = await _measure_bare(config, binary_path, input_path)
                bare_misses.append(counters.l1_dcache_load_misses)

            u_stat, p_value = stats.mannwhitneyu(
                bwrap_misses, bare_misses, alternative="two-sided"
            )

            bwrap_arr = np.array(bwrap_misses)
            bare_arr = np.array(bare_misses)

            print(f"\nbwrap L1 misses: mean={np.mean(bwrap_arr):.0f} std={np.std(bwrap_arr):.0f}")
            print(f"bare  L1 misses: mean={np.mean(bare_arr):.0f} std={np.std(bare_arr):.0f}")
            print(f"Mann-Whitney U={u_stat:.1f}, p={p_value:.6f}")

            assert p_value >= SIGNIFICANCE_LEVEL, (
                f"bwrap significantly alters L1 cache miss counts (p={p_value:.6f}). "
                f"bwrap mean={np.mean(bwrap_arr):.0f}, bare mean={np.mean(bare_arr):.0f}"
            )
        finally:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)
