"""Configuration dataclasses for perf-optimize."""

from __future__ import annotations

from dataclasses import dataclass, field

from perf_optimize.types import PerfCounter


def _default_perf_counters() -> tuple[PerfCounter, ...]:
    return tuple(PerfCounter)


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for the bubblewrap sandbox and measurement tools."""

    gcc_path: str = "gcc"
    gcc_flags: tuple[str, ...] = ("-O2", "-lm")
    bwrap_path: str = "bwrap"
    perf_path: str = "perf"
    taskset_path: str = "taskset"
    compile_timeout_s: float = 30.0
    test_timeout_s: float = 10.0
    perf_timeout_s: float = 60.0
    perf_repeat: int = 5
    pin_cpu: int = 0
    ulimit_mem_kb: int = 512_000
    ulimit_procs: int = 32
    ulimit_fsize_kb: int = 10_240
    ro_bind_paths: tuple[str, ...] = (
        "/usr",
        "/lib",
        "/lib64",
        "/etc/alternatives",
        "/etc/ld.so.cache",
    )
    perf_counters: tuple[PerfCounter, ...] = field(default_factory=_default_perf_counters)
    cv_threshold_cycles: float = 0.05
    cv_threshold_cache: float = 0.10
