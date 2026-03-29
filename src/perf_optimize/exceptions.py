"""Custom exception hierarchy for perf-optimize.

Structured exceptions for each failure domain: prerequisites, sandbox
execution, and perf output parsing.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class PerfOptimizeError(Exception):
    """Base exception for all perf-optimize errors."""


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


class PrerequisiteError(PerfOptimizeError):
    """A required system tool or capability is missing."""


class BwrapNotFoundError(PrerequisiteError):
    """bubblewrap (bwrap) is not installed or not on PATH."""

    def __init__(self) -> None:
        super().__init__(
            "bwrap not found on PATH. Install bubblewrap: "
            "apt install bubblewrap (Debian/Ubuntu) or dnf install bubblewrap (Fedora)"
        )


class PerfNotFoundError(PrerequisiteError):
    """Linux perf is not installed or not on PATH."""

    def __init__(self) -> None:
        super().__init__(
            "perf not found on PATH. Install linux-tools: "
            "apt install linux-tools-common linux-tools-$(uname -r)"
        )


class GccNotFoundError(PrerequisiteError):
    """GCC is not installed or not on PATH."""

    def __init__(self) -> None:
        super().__init__("gcc not found on PATH. Install GCC: apt install gcc")


class TasksetNotFoundError(PrerequisiteError):
    """taskset (from util-linux) is not installed or not on PATH."""

    def __init__(self) -> None:
        super().__init__("taskset not found on PATH. Install util-linux: apt install util-linux")


class PerfParanoidError(PrerequisiteError):
    """perf_event_paranoid is set too high for unprivileged perf access."""

    def __init__(self, current_value: int) -> None:
        self.current_value = current_value
        super().__init__(
            f"perf_event_paranoid={current_value} (needs <=1). "
            "Run: sudo sysctl kernel.perf_event_paranoid=1"
        )


# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------


class SandboxError(PerfOptimizeError):
    """An error during sandboxed execution."""


class BwrapInvocationError(SandboxError):
    """bwrap itself failed to execute (bad args, permission denied, etc.)."""

    def __init__(self, message: str = "bwrap invocation failed") -> None:
        super().__init__(message)


class CompilationTimeoutError(SandboxError):
    """Compilation exceeded the configured timeout."""

    def __init__(self, timeout_s: float | None = None) -> None:
        msg = "Compilation timed out"
        if timeout_s is not None:
            msg += f" after {timeout_s}s"
        super().__init__(msg)


class TestTimeoutError(SandboxError):
    """Test execution exceeded the configured timeout."""

    def __init__(self, timeout_s: float | None = None) -> None:
        msg = "Test execution timed out"
        if timeout_s is not None:
            msg += f" after {timeout_s}s"
        super().__init__(msg)


class PerfMeasurementError(SandboxError):
    """perf stat measurement failed during execution."""

    def __init__(self, message: str = "perf measurement failed") -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# Perf output parsing
# ---------------------------------------------------------------------------


class PerfParseError(PerfOptimizeError):
    """Failed to parse perf stat output."""

    def __init__(self, message: str, raw_output: str) -> None:
        self.raw_output = raw_output
        super().__init__(f"{message}\n--- raw output ---\n{raw_output}")


class CounterNotFoundError(PerfParseError):
    """An expected counter was not present in the perf output."""

    def __init__(self, counter: str, raw_output: str = "") -> None:
        self.counter = counter
        super().__init__(f"Counter '{counter}' not found in perf output", raw_output)


class CounterNotCountedError(PerfParseError):
    """A counter was present but perf reported it as <not counted>."""

    def __init__(self, counter: str, raw_output: str = "") -> None:
        self.counter = counter
        super().__init__(f"Counter '{counter}' was <not counted> by perf", raw_output)
