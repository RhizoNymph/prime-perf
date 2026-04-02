"""Extensive tests for perf-optimize core types."""

from __future__ import annotations

import dataclasses

import pytest

from perf_optimize.types import (
    PERF_COUNTER_FIELDS,
    CompilationFailure,
    CompilationOutcome,
    CompilationSuccess,
    CounterVarianceStats,
    ExecutionResult,
    PerfCounters,
    PerfCSVLine,
    TestReport,
    TestResult,
    VarianceReport,
)

# ---------------------------------------------------------------------------
# PerfCounters dataclass
# ---------------------------------------------------------------------------


def _sample_counters(**overrides: float) -> PerfCounters:
    defaults = {
        "cycles": 1_000_000.0,
        "instructions": 2_000_000.0,
        "cache_references": 50_000.0,
        "cache_misses": 1_000.0,
        "l1_dcache_load_misses": 500.0,
        "llc_load_misses": 100.0,
        "branch_misses": 200.0,
    }
    defaults.update(overrides)
    return PerfCounters(**defaults)


class TestPerfCounters:
    def test_ipc_normal(self) -> None:
        pc = _sample_counters(cycles=1_000_000.0, instructions=2_000_000.0)
        assert pc.ipc == pytest.approx(2.0)

    def test_ipc_zero_cycles(self) -> None:
        pc = _sample_counters(cycles=0.0)
        assert pc.ipc == 0.0

    def test_ipc_low_ratio(self) -> None:
        pc = _sample_counters(cycles=4_000_000.0, instructions=1_000_000.0)
        assert pc.ipc == pytest.approx(0.25)

    def test_to_dict_keys_match_field_names(self) -> None:
        pc = _sample_counters()
        d = pc.to_dict()
        assert set(d.keys()) == set(PERF_COUNTER_FIELDS)

    def test_to_dict_values(self) -> None:
        pc = _sample_counters()
        d = pc.to_dict()
        assert d["cycles"] == 1_000_000.0
        assert d["instructions"] == 2_000_000.0
        assert d["cache_references"] == 50_000.0
        assert d["cache_misses"] == 1_000.0
        assert d["l1_dcache_load_misses"] == 500.0
        assert d["llc_load_misses"] == 100.0
        assert d["branch_misses"] == 200.0

    def test_to_dict_omits_none(self) -> None:
        pc = PerfCounters(cycles=100.0, instructions=200.0)
        d = pc.to_dict()
        assert set(d.keys()) == {"cycles", "instructions"}
        assert "llc_load_misses" not in d

    def test_frozen(self) -> None:
        pc = _sample_counters()
        with pytest.raises(dataclasses.FrozenInstanceError):
            pc.cycles = 999.0  # type: ignore[misc]

    def test_slots(self) -> None:
        pc = _sample_counters()
        with pytest.raises((AttributeError, TypeError)):
            pc.extra_field = 42  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# CompilationSuccess / CompilationFailure
# ---------------------------------------------------------------------------


class TestCompilationResult:
    def test_success_is_empty_marker(self) -> None:
        s = CompilationSuccess()
        assert dataclasses.fields(s) == ()

    def test_failure_has_outcome_and_stderr(self) -> None:
        f = CompilationFailure(outcome=CompilationOutcome.ERROR, stderr="undefined reference")
        assert f.outcome == CompilationOutcome.ERROR
        assert f.stderr == "undefined reference"

    def test_failure_timeout(self) -> None:
        f = CompilationFailure(outcome=CompilationOutcome.TIMEOUT, stderr="")
        assert f.outcome == CompilationOutcome.TIMEOUT

    def test_success_frozen(self) -> None:
        s = CompilationSuccess()
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
            s.x = 1  # type: ignore[attr-defined]

    def test_failure_frozen(self) -> None:
        f = CompilationFailure(outcome=CompilationOutcome.ERROR, stderr="err")
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.stderr = "new"  # type: ignore[misc]

    def test_isinstance_discrimination(self) -> None:
        s: CompilationSuccess | CompilationFailure = CompilationSuccess()
        assert isinstance(s, CompilationSuccess)
        assert not isinstance(s, CompilationFailure)

        f: CompilationSuccess | CompilationFailure = CompilationFailure(
            outcome=CompilationOutcome.ERROR, stderr=""
        )
        assert isinstance(f, CompilationFailure)
        assert not isinstance(f, CompilationSuccess)


# ---------------------------------------------------------------------------
# TestResult / TestReport
# ---------------------------------------------------------------------------


class TestTestReport:
    def test_all_passed(self) -> None:
        report = TestReport(
            results=(
                TestResult(name="test_a", passed=True),
                TestResult(name="test_b", passed=True),
            )
        )
        assert report.passed == 2
        assert report.total == 2
        assert report.all_passed is True
        assert report.errors == []

    def test_some_failed(self) -> None:
        report = TestReport(
            results=(
                TestResult(name="test_a", passed=True),
                TestResult(name="test_b", passed=False, error="assertion failed"),
                TestResult(name="test_c", passed=False, error="segfault"),
            )
        )
        assert report.passed == 1
        assert report.total == 3
        assert report.all_passed is False
        assert report.errors == ["assertion failed", "segfault"]

    def test_empty_report(self) -> None:
        report = TestReport(results=())
        assert report.passed == 0
        assert report.total == 0
        assert report.all_passed is True
        assert report.errors == []

    def test_failed_without_error_message(self) -> None:
        """A failed test with error=None should not appear in errors list."""
        report = TestReport(
            results=(TestResult(name="test_a", passed=False, error=None),)
        )
        assert report.passed == 0
        assert report.total == 1
        assert report.all_passed is False
        assert report.errors == []

    def test_frozen(self) -> None:
        report = TestReport(results=())
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.results = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


def _make_success_execution() -> ExecutionResult:
    """Fully successful execution with all fields populated."""
    return ExecutionResult(
        compilation=CompilationSuccess(),
        test_report=TestReport(
            results=(
                TestResult(name="test_a", passed=True),
                TestResult(name="test_b", passed=True),
            )
        ),
        perf_counters=_sample_counters(),
        wall_clock_ms=42.5,
    )


def _make_compile_failure_execution() -> ExecutionResult:
    """Compilation failed, so no tests or perf."""
    return ExecutionResult(
        compilation=CompilationFailure(
            outcome=CompilationOutcome.ERROR, stderr="error: missing semicolon"
        ),
        test_report=None,
        perf_counters=None,
        wall_clock_ms=None,
    )


def _make_test_failure_execution() -> ExecutionResult:
    """Compiled but tests failed, so no perf counters."""
    return ExecutionResult(
        compilation=CompilationSuccess(),
        test_report=TestReport(
            results=(
                TestResult(name="test_a", passed=True),
                TestResult(name="test_b", passed=False, error="wrong output"),
            )
        ),
        perf_counters=None,
        wall_clock_ms=None,
    )


class TestExecutionResult:
    def test_successful_compiled(self) -> None:
        er = _make_success_execution()
        assert er.compiled is True

    def test_successful_compiler_errors_none(self) -> None:
        er = _make_success_execution()
        assert er.compiler_errors is None

    def test_successful_tests_passed(self) -> None:
        er = _make_success_execution()
        assert er.tests_passed == 2

    def test_successful_tests_total(self) -> None:
        er = _make_success_execution()
        assert er.tests_total == 2

    def test_successful_has_perf_counters(self) -> None:
        er = _make_success_execution()
        assert er.perf_counters is not None
        assert er.wall_clock_ms == pytest.approx(42.5)

    def test_compile_failure_compiled(self) -> None:
        er = _make_compile_failure_execution()
        assert er.compiled is False

    def test_compile_failure_compiler_errors(self) -> None:
        er = _make_compile_failure_execution()
        assert er.compiler_errors == "error: missing semicolon"

    def test_compile_failure_test_report_none(self) -> None:
        er = _make_compile_failure_execution()
        assert er.test_report is None

    def test_compile_failure_perf_counters_none(self) -> None:
        er = _make_compile_failure_execution()
        assert er.perf_counters is None

    def test_compile_failure_tests_passed_zero(self) -> None:
        er = _make_compile_failure_execution()
        assert er.tests_passed == 0

    def test_compile_failure_tests_total_zero(self) -> None:
        er = _make_compile_failure_execution()
        assert er.tests_total == 0

    def test_test_failure_compiled(self) -> None:
        er = _make_test_failure_execution()
        assert er.compiled is True

    def test_test_failure_tests_passed(self) -> None:
        er = _make_test_failure_execution()
        assert er.tests_passed == 1

    def test_test_failure_tests_total(self) -> None:
        er = _make_test_failure_execution()
        assert er.tests_total == 2

    def test_test_failure_perf_counters_none(self) -> None:
        er = _make_test_failure_execution()
        assert er.perf_counters is None

    def test_frozen(self) -> None:
        er = _make_success_execution()
        with pytest.raises(dataclasses.FrozenInstanceError):
            er.wall_clock_ms = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PerfCSVLine
# ---------------------------------------------------------------------------


class TestPerfCSVLine:
    def test_basic_construction(self) -> None:
        line = PerfCSVLine(
            counter_value=1234.0,
            unit="",
            event_name="cycles",
            run_time=100.0,
            percentage=100.0,
        )
        assert line.counter_value == 1234.0
        assert line.variance is None

    def test_with_variance(self) -> None:
        line = PerfCSVLine(
            counter_value=1234.0,
            unit="",
            event_name="cycles",
            run_time=100.0,
            percentage=100.0,
            variance=0.5,
        )
        assert line.variance == 0.5

    def test_frozen(self) -> None:
        line = PerfCSVLine(
            counter_value=1234.0,
            unit="",
            event_name="cycles",
            run_time=100.0,
            percentage=100.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            line.counter_value = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CounterVarianceStats / VarianceReport
# ---------------------------------------------------------------------------


def _make_variance_stats(
    counter: str = "cycles",
    cv: float = 0.03,
    threshold: float = 0.05,
) -> CounterVarianceStats:
    return CounterVarianceStats(
        counter=counter,
        mean=1_000_000.0,
        std=cv * 1_000_000.0,
        cv=cv,
        min_val=900_000.0,
        max_val=1_100_000.0,
        n_samples=5,
        threshold=threshold,
    )


class TestCounterVarianceStats:
    def test_passed_below_threshold(self) -> None:
        s = _make_variance_stats(cv=0.03, threshold=0.05)
        assert s.passed is True

    def test_failed_above_threshold(self) -> None:
        s = _make_variance_stats(cv=0.06, threshold=0.05)
        assert s.passed is False

    def test_failed_at_threshold(self) -> None:
        """cv == threshold should NOT pass (strict less-than)."""
        s = _make_variance_stats(cv=0.05, threshold=0.05)
        assert s.passed is False


class TestVarianceReport:
    def test_all_passed(self) -> None:
        report = VarianceReport(
            stats={
                "cycles": _make_variance_stats(
                    counter="cycles", cv=0.02, threshold=0.05
                ),
                "cache_misses": _make_variance_stats(
                    counter="cache_misses", cv=0.05, threshold=0.10
                ),
            }
        )
        assert report.all_passed is True
        assert report.failures == []

    def test_some_failed(self) -> None:
        cycles_stats = _make_variance_stats(
            counter="cycles", cv=0.08, threshold=0.05
        )
        cache_stats = _make_variance_stats(
            counter="cache_misses", cv=0.03, threshold=0.10
        )
        report = VarianceReport(
            stats={
                "cycles": cycles_stats,
                "cache_misses": cache_stats,
            }
        )
        assert report.all_passed is False
        assert len(report.failures) == 1
        assert report.failures[0].counter == "cycles"

    def test_empty_report(self) -> None:
        report = VarianceReport(stats={})
        assert report.all_passed is True
        assert report.failures == []
