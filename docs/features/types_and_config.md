# Feature: Types, Config, and Exceptions

## Scope

**In scope:** Core data types, configuration dataclass, and structured exception hierarchy used across all perf-optimize modules.

**Not in scope:** Business logic, I/O, or any behavior beyond data definition.

## Types

### `PerfCounter` (enum)

Maps logical counter names to perf event name strings:
- `CYCLES` -> `"cycles"`
- `INSTRUCTIONS` -> `"instructions"`
- `CACHE_REFERENCES` -> `"cache-references"`
- `CACHE_MISSES` -> `"cache-misses"`
- `L1_DCACHE_LOAD_MISSES` -> `"L1-dcache-load-misses"`
- `LLC_LOAD_MISSES` -> `"LLC-load-misses"`
- `BRANCH_MISSES` -> `"branch-misses"`

### `PerfCSVLine` (frozen dataclass)

Single parsed line from perf CSV: `counter_value`, `unit`, `event_name`, `run_time`, `percentage`, optional `variance`.

### `PerfCounters` (frozen dataclass)

Aggregated counter values with fields for all 7 counters plus a derived `ipc` property.

### `SandboxConfig` (frozen dataclass)

All sandbox and toolchain configuration with sensible defaults: gcc path/flags, bwrap/perf/taskset paths, timeouts, ulimit values, ro_bind_paths, perf_counters, and CV thresholds.

## Exception Hierarchy

```
PerfOptimizeError
├── PerfParseError
│   ├── CounterNotFoundError(counter_name)
│   └── CounterNotCountedError(counter_name, raw_value)
└── SandboxError
    ├── CompilationError(stderr, returncode)
    └── ExecutionTimeoutError(phase, timeout_s)
```

## Files

| File | Role | Key exports |
|------|------|-------------|
| `src/perf_optimize/types.py` | Data types | `PerfCounter`, `PerfCSVLine`, `PerfCounters` |
| `src/perf_optimize/config.py` | Configuration | `SandboxConfig` |
| `src/perf_optimize/exceptions.py` | Exceptions | Full hierarchy |

## Invariants

- All dataclasses are frozen (immutable after construction).
- `PerfCounter` enum values match exact perf event name strings.
- `SandboxConfig.perf_counters` defaults to all 7 `PerfCounter` members.
- All exceptions carry structured data (not just message strings).
- `PerfCounters.ipc` returns 0.0 when cycles is 0 (avoids division by zero).
