# Exception Hierarchy

## Scope

Structured domain exceptions for all failure modes in perf-optimize.

### In Scope
- Base PerfOptimizeError
- Prerequisite check failures (missing tools, wrong perf_event_paranoid)
- Sandbox execution failures (bwrap, perf measurement)
- Perf output parsing failures (missing counters, not-counted counters)

### Not in Scope
- Exception handling logic (consumers of these exceptions)
- Recovery strategies

## Data/Control Flow

```
PerfOptimizeError (base)
  |
  +-- PrerequisiteError
  |     +-- BwrapNotFoundError
  |     +-- PerfNotFoundError
  |     +-- GccNotFoundError
  |     +-- TasksetNotFoundError
  |     +-- PerfParanoidError(current_value: int)
  |
  +-- SandboxError
  |     +-- BwrapInvocationError
  |     +-- PerfMeasurementError
  |
  +-- PerfParseError(message, raw_output)
        +-- CounterNotFoundError(counter)
        +-- CounterNotCountedError(counter)
```

## Files

| File | Contents | Key Exports |
|------|----------|-------------|
| `src/perf_optimize/exceptions.py` | All exception classes | PerfOptimizeError, PrerequisiteError, BwrapNotFoundError, PerfNotFoundError, GccNotFoundError, TasksetNotFoundError, PerfParanoidError, SandboxError, BwrapInvocationError, PerfMeasurementError, PerfParseError, CounterNotFoundError, CounterNotCountedError |

## Invariants and Constraints

- All exceptions inherit from PerfOptimizeError
- PerfParanoidError stores current_value and includes fix command in message: "Run: sudo sysctl kernel.perf_event_paranoid=1"
- PerfParseError always stores raw_output for debugging
- CounterNotFoundError and CounterNotCountedError store the counter name
- Prerequisite errors have zero-arg constructors with descriptive default messages (except PerfParanoidError)
