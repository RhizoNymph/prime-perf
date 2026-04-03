"""Core type definitions for perf-optimize.

This module defines the immutable data types that flow through the measurement
pipeline: perf counters, compilation/test results, execution outcomes, and
variance statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# All PerfCounters field names that correspond to hardware counters.
# Used by HardwareProfile and the parser to validate field mappings.
# Ordered tuple for deterministic iteration (e.g. in to_dict(), prompt display).
PERF_COUNTER_FIELDS: tuple[str, ...] = (
    "cycles",
    "instructions",
    "cache_references",
    "cache_misses",
    "l1_dcache_load_misses",
    "llc_load_misses",
    "branch_misses",
)


@dataclass(frozen=True, slots=True)
class PerfCounters:
    """Hardware performance counter readings.

    ``None`` means the counter is not available on the current hardware.
    ``0`` means it was measured and the count was zero.

    ``cycles`` and ``instructions`` are always present on x86. Other fields
    are populated based on the active ``HardwareProfile``.
    """

    cycles: float
    instructions: float
    cache_references: float | None = None
    cache_misses: float | None = None
    l1_dcache_load_misses: float | None = None
    llc_load_misses: float | None = None
    branch_misses: float | None = None

    def __post_init__(self) -> None:
        """Validate that counter values are non-negative."""
        if self.cycles < 0:
            raise ValueError(f"cycles must be non-negative, got {self.cycles}")
        if self.instructions < 0:
            raise ValueError(f"instructions must be non-negative, got {self.instructions}")
        for field_name in ("cache_references", "cache_misses", "l1_dcache_load_misses", "llc_load_misses", "branch_misses"):
            val = getattr(self, field_name)
            if val is not None and val < 0:
                raise ValueError(f"{field_name} must be non-negative, got {val}")

    @property
    def ipc(self) -> float:
        """Instructions per cycle. Returns 0.0 when cycles is zero."""
        if self.cycles == 0:
            return 0.0
        return self.instructions / self.cycles

    def to_dict(self) -> dict[str, float]:
        """Serialize to a dict keyed by field name, omitting None entries."""
        result: dict[str, float] = {}
        for field_name in PERF_COUNTER_FIELDS:
            val = getattr(self, field_name)
            if val is not None:
                result[field_name] = val
        return result


class CompilationOutcome(StrEnum):
    """Possible non-success compilation outcomes."""

    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True, slots=True)
class CompilationSuccess:
    """Marker type indicating compilation succeeded."""


@dataclass(frozen=True, slots=True)
class CompilationFailure:
    """Compilation failed with a specific outcome and stderr output."""

    outcome: CompilationOutcome
    stderr: str


CompilationResult = CompilationSuccess | CompilationFailure


@dataclass(frozen=True, slots=True)
class TestResult:
    """Result of a single test case execution."""

    __test__ = False  # prevent pytest collection

    name: str
    passed: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TestReport:
    """Aggregated results from running all test cases."""

    __test__ = False  # prevent pytest collection

    results: tuple[TestResult, ...]

    @property
    def passed(self) -> int:
        """Count of tests that passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        """Total number of tests."""
        return len(self.results)

    @property
    def all_passed(self) -> bool:
        """Whether every test passed."""
        return all(r.passed for r in self.results)

    @property
    def errors(self) -> list[str]:
        """Error messages from failed tests."""
        return [r.error for r in self.results if not r.passed and r.error is not None]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Full result of compiling, testing, and measuring a C program.

    Composes compilation, test, and perf counter results into a single
    immutable value. Fields are None when earlier stages failed:
    - test_report is None if compilation failed
    - perf_counters and wall_clock_ms are None if tests didn't all pass
    """

    compilation: CompilationResult
    test_report: TestReport | None
    perf_counters: PerfCounters | None
    wall_clock_ms: float | None

    @property
    def compiled(self) -> bool:
        """Whether compilation succeeded."""
        return isinstance(self.compilation, CompilationSuccess)

    @property
    def compiler_errors(self) -> str | None:
        """Compiler stderr if compilation failed, None otherwise."""
        match self.compilation:
            case CompilationFailure(stderr=stderr):
                return stderr
            case _:
                return None

    @property
    def tests_passed(self) -> int:
        """Number of tests that passed, 0 if no test report."""
        if self.test_report is None:
            return 0
        return self.test_report.passed

    @property
    def tests_total(self) -> int:
        """Total number of tests, 0 if no test report."""
        if self.test_report is None:
            return 0
        return self.test_report.total


@dataclass(frozen=True, slots=True)
class PerfCSVLine:
    """Intermediate parse type for a single line of perf stat CSV output."""

    counter_value: float
    unit: str
    event_name: str
    run_time: float
    percentage: float
    variance: float | None = None


@dataclass(frozen=True)
class CounterVarianceStats:
    """Variance statistics for a single performance counter across repeated measurements."""

    counter: str
    mean: float
    std: float
    cv: float
    min_val: float
    max_val: float
    n_samples: int
    threshold: float

    @property
    def passed(self) -> bool:
        """Whether the coefficient of variation is below the threshold."""
        return self.cv < self.threshold


@dataclass(frozen=True)
class VarianceReport:
    """Aggregated variance check across all performance counters."""

    stats: dict[str, CounterVarianceStats]

    @property
    def all_passed(self) -> bool:
        """Whether all counters passed their variance threshold."""
        return all(s.passed for s in self.stats.values())

    @property
    def failures(self) -> list[CounterVarianceStats]:
        """List of counter stats that exceeded their variance threshold."""
        return [s for s in self.stats.values() if not s.passed]
