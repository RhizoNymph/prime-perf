"""Variance validation: run naive matmul 50 times, verify CV per counter.

Phase 0 task 0.4: Target CV < 5% for cycles, < 10% for cache counters.

This test is slow (~5 min). Run explicitly with:
    uv run pytest -m variance tests/validation/test_variance.py -v
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from perf_optimize.config import SandboxConfig
from perf_optimize.sandbox import PerfSandbox
from perf_optimize.types import (
    CompilationSuccess,
    CounterVarianceStats,
    PerfCounters,
    VarianceReport,
)
from tests.conftest import requires_all_tools

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "c_programs"
N_SAMPLES = 50
MATMUL_N = 512


def _make_matmul_input(n: int, seed: int = 42) -> bytes:
    rng = np.random.default_rng(seed)
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    return struct.pack("i", n) + a.tobytes() + b.tobytes()


def _compute_variance_report(
    samples: list[PerfCounters],
    config: SandboxConfig,
) -> VarianceReport:
    """Compute CV for each counter that is not None across samples."""
    stats: dict[str, CounterVarianceStats] = {}

    # Check which fields are populated (not None) in the first sample
    for field_name in config.hardware_profile.mapped_fields():
        first_val = getattr(samples[0], field_name)
        if first_val is None:
            continue

        values = np.array([getattr(s, field_name) for s in samples])

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        cv = std / mean if mean > 0 else 0.0

        threshold = (
            config.cv_threshold_cycles if field_name == "cycles" else config.cv_threshold_cache
        )

        stats[field_name] = CounterVarianceStats(
            counter=field_name,
            mean=mean,
            std=std,
            cv=cv,
            min_val=float(np.min(values)),
            max_val=float(np.max(values)),
            n_samples=len(values),
            threshold=threshold,
        )

    return VarianceReport(stats=stats)


@requires_all_tools
@pytest.mark.variance
@pytest.mark.asyncio
class TestVariance:
    """Run the same binary N_SAMPLES times and check measurement variance."""

    async def _compile_matmul(self, sandbox: PerfSandbox) -> tuple[str, bytes]:
        """Compile matmul_naive.c and return (work_dir, input_data)."""
        source = FIXTURES.joinpath("matmul_naive.c").read_text()
        compilation, work_dir = await sandbox.compile_only(source)
        assert isinstance(compilation, CompilationSuccess)

        input_data = _make_matmul_input(MATMUL_N)
        input_path = Path(work_dir) / "perf_input.bin"
        input_path.write_bytes(input_data)

        return work_dir, input_data

    async def test_cycles_cv_below_threshold(self) -> None:
        """Cycles CV should be < 5% across 50 measurements."""
        config = SandboxConfig()
        sandbox = PerfSandbox(config)
        work_dir, _input_data = await self._compile_matmul(sandbox)

        try:
            binary_path = Path(work_dir) / "solution"
            input_path = Path(work_dir) / "perf_input.bin"

            samples: list[PerfCounters] = []
            for _i in range(N_SAMPLES):
                counters = await sandbox.measure_only(binary_path, input_path)
                samples.append(counters)

            report = _compute_variance_report(samples, config)

            for field_name, stat in report.stats.items():
                print(
                    f"{field_name}: mean={stat.mean:.0f} std={stat.std:.0f} "
                    f"CV={stat.cv:.4f} ({stat.cv*100:.2f}%) "
                    f"threshold={stat.threshold*100:.0f}% "
                    f"{'PASS' if stat.passed else 'FAIL'}"
                )

            cycles_stat = report.stats["cycles"]
            assert cycles_stat.passed, (
                f"Cycles CV={cycles_stat.cv:.4f} ({cycles_stat.cv*100:.2f}%) "
                f"exceeds threshold {config.cv_threshold_cycles*100:.0f}%"
            )
        finally:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)

    async def test_cache_counters_cv_below_threshold(self) -> None:
        """Cache counter CVs should be < 10% across 50 measurements."""
        config = SandboxConfig()
        sandbox = PerfSandbox(config)
        work_dir, _input_data = await self._compile_matmul(sandbox)

        try:
            binary_path = Path(work_dir) / "solution"
            input_path = Path(work_dir) / "perf_input.bin"

            samples: list[PerfCounters] = []
            for _i in range(N_SAMPLES):
                counters = await sandbox.measure_only(binary_path, input_path)
                samples.append(counters)

            report = _compute_variance_report(samples, config)

            cache_fields = ["cache_misses", "l1_dcache_load_misses", "branch_misses"]
            failures = [
                report.stats[f] for f in cache_fields
                if f in report.stats and not report.stats[f].passed
            ]

            for f in failures:
                print(
                    f"FAIL: {f.counter} CV={f.cv:.4f} ({f.cv*100:.2f}%) "
                    f"threshold={f.threshold*100:.0f}%"
                )

            assert not failures, (
                f"{len(failures)} cache counter(s) exceeded variance threshold: "
                + ", ".join(f"{f.counter}={f.cv*100:.2f}%" for f in failures)
            )
        finally:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)

    async def test_all_counters_report(self) -> None:
        """Generate full variance report for all measured counters."""
        config = SandboxConfig()
        sandbox = PerfSandbox(config)
        work_dir, _input_data = await self._compile_matmul(sandbox)

        try:
            binary_path = Path(work_dir) / "solution"
            input_path = Path(work_dir) / "perf_input.bin"

            samples: list[PerfCounters] = []
            for _i in range(N_SAMPLES):
                counters = await sandbox.measure_only(binary_path, input_path)
                samples.append(counters)

            report = _compute_variance_report(samples, config)

            assert report.all_passed, (
                "Variance check failed for: "
                + ", ".join(f"{f.counter}={f.cv*100:.2f}%" for f in report.failures)
            )
        finally:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)
