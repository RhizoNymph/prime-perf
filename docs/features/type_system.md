# Type System

## Scope

Defines all immutable data types that flow through the perf-optimize measurement pipeline.

### In Scope
- PerfCounter enum (perf CLI event strings)
- PerfCounters container with IPC computation and serialization
- Compilation result types (success/failure discrimination)
- Test result/report aggregation
- ExecutionResult composing all pipeline stages
- PerfCSVLine intermediate parse type
- Variance statistics types

### Not in Scope
- Parsing logic (see perf parser feature)
- Sandbox execution (see sandbox feature)
- Serialization beyond to_dict()

## Data/Control Flow

```
PerfCounter (enum) -- used as keys/identifiers throughout
  |
  v
PerfCSVLine -- raw parsed line from perf stat CSV output
  |
  v
PerfCounters -- typed container for a full set of counter readings
  |
  v
CounterVarianceStats -- per-counter variance statistics
  |
  v
VarianceReport -- aggregated variance check across all counters

CompilationSuccess | CompilationFailure -> CompilationResult
TestResult -> TestReport

ExecutionResult = CompilationResult + TestReport? + PerfCounters? + wall_clock_ms?
```

## Files

| File | Contents | Key Exports |
|------|----------|-------------|
| `src/perf_optimize/types.py` | All type definitions | PerfCounter, PerfCounters, CompilationOutcome, CompilationSuccess, CompilationFailure, CompilationResult, TestResult, TestReport, ExecutionResult, PerfCSVLine, CounterVarianceStats, VarianceReport |
| `tests/unit/test_types.py` | 53 tests covering all types | - |

## Invariants and Constraints

- All dataclasses are frozen (immutable after construction)
- PerfCounters and pipeline dataclasses use `slots=True` for memory efficiency
- PerfCounter enum values are exact `perf stat` CLI event strings (e.g. "L1-dcache-load-misses")
- PerfCounters.ipc returns 0.0 when cycles is 0 (no division by zero)
- PerfCounters.to_dict keys are PerfCounter string values (not Python field names)
- ExecutionResult.test_report is None when compilation failed
- ExecutionResult.perf_counters is None when tests didn't all pass
- CompilationResult is a type alias union, not an abstract base class
- CounterVarianceStats.passed uses strict less-than (cv < threshold)
- TestReport.errors only includes non-None error strings from failed tests
