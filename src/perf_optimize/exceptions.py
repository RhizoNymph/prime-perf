"""Structured exception hierarchy for perf-optimize."""

from __future__ import annotations


class PerfOptimizeError(Exception):
    """Base exception for all perf-optimize errors."""


# ── Perf parsing errors ──────────────────────────────────────────────────────


class PerfParseError(PerfOptimizeError):
    """Base class for errors encountered while parsing perf output."""


class CounterNotFoundError(PerfParseError):
    """A required hardware counter was not present in the perf output."""

    def __init__(self, counter_name: str) -> None:
        self.counter_name = counter_name
        super().__init__(f"Required counter not found in perf output: {counter_name}")


class CounterNotCountedError(PerfParseError):
    """A hardware counter reported <not counted> or <not supported>."""

    def __init__(self, counter_name: str, raw_value: str) -> None:
        self.counter_name = counter_name
        self.raw_value = raw_value
        super().__init__(f"Counter '{counter_name}' was not counted (raw value: {raw_value})")


# ── Sandbox / execution errors ───────────────────────────────────────────────


class SandboxError(PerfOptimizeError):
    """Base class for sandbox execution errors."""


class CompilationError(SandboxError):
    """GCC compilation failed."""

    def __init__(self, stderr: str, returncode: int) -> None:
        self.stderr = stderr
        self.returncode = returncode
        super().__init__(f"Compilation failed (exit {returncode}): {stderr[:200]}")


class ExecutionTimeoutError(SandboxError):
    """A sandboxed process exceeded its wall-clock timeout."""

    def __init__(self, phase: str, timeout_s: float) -> None:
        self.phase = phase
        self.timeout_s = timeout_s
        super().__init__(f"{phase} timed out after {timeout_s}s")
