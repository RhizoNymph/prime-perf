"""Core data types for perf-optimize."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class PerfCounter(enum.Enum):
    """Hardware performance counters collected by perf stat."""

    CYCLES = "cycles"
    INSTRUCTIONS = "instructions"
    CACHE_REFERENCES = "cache-references"
    CACHE_MISSES = "cache-misses"
    L1_DCACHE_LOAD_MISSES = "L1-dcache-load-misses"
    LLC_LOAD_MISSES = "LLC-load-misses"
    BRANCH_MISSES = "branch-misses"


@dataclass(frozen=True)
class PerfCSVLine:
    """A single parsed line from perf stat CSV output."""

    counter_value: int
    unit: str
    event_name: str
    run_time: int
    percentage: float
    variance: float | None = None


@dataclass(frozen=True)
class PerfCounters:
    """Aggregated hardware counter values from a single perf stat run."""

    cycles: int
    instructions: int
    cache_references: int
    cache_misses: int
    l1_dcache_load_misses: int
    llc_load_misses: int
    branch_misses: int

    @property
    def ipc(self) -> float:
        """Instructions per cycle (derived metric)."""
        if self.cycles == 0:
            return 0.0
        return self.instructions / self.cycles
