# Perf Parser Feature

## Scope

**In scope:**
- Parsing `perf stat -x ',' -r N` CSV output into typed PerfCounters
- Handling all edge cases: comments, empty lines, derived metrics, `<not counted>`, `<not supported>`
- Line-by-line parsing via `parse_csv_line` and full-output assembly via `parse_perf_output`

**Not in scope:**
- Running perf (that's the sandbox's job)
- Interpreting counter values (that's the reward function's job)

## Data/Control Flow

```
perf stat stderr (str)
    │
    ▼
parse_perf_output(csv_text)
    │
    ├─ Split into lines
    ├─ For each line: parse_csv_line(line)
    │     ├─ Skip: comments (#), empty, derived metrics, non-integer values
    │     ├─ Raise: <not counted>, <not supported>
    │     └─ Return: PerfCSVLine(counter_value, unit, event_name, ...)
    │
    ├─ Filter: only keep lines where event_name matches PerfCounter enum
    ├─ Verify: all 7 required counters present
    └─ Assemble: PerfCounters dataclass
```

## CSV Format

`perf stat -x ','` produces lines with these fields:
```
counter_value,unit,event_name,run_time,percentage[,variance]
```

- `counter_value`: integer count, or `<not counted>`/`<not supported>`
- `unit`: usually empty for hardware counters
- `event_name`: matches PerfCounter enum values (e.g., "cycles", "L1-dcache-load-misses")
- `run_time`: nanoseconds the counter was active
- `percentage`: percentage of time the counter was active (usually 100.00)
- `variance`: only present with `-r` flag (percentage CV from perf's internal repetitions)

## Files

| File | Role | Key exports |
|------|------|-------------|
| `src/perf_optimize/perf_parser.py` | Parser implementation | `parse_perf_output`, `parse_csv_line` |
| `src/perf_optimize/types.py` | `PerfCSVLine` intermediate type, `PerfCounters` output type | |
| `src/perf_optimize/exceptions.py` | `PerfParseError`, `CounterNotFoundError`, `CounterNotCountedError` | |

## Invariants and Constraints

1. **Pure functions**: No I/O, no side effects. Takes strings, returns dataclasses or raises exceptions.
2. **All 7 counters required**: `parse_perf_output` raises `CounterNotFoundError` if any counter is missing.
3. **Integer counter values only**: Floating-point values in the counter_value field indicate derived metrics and are skipped.
4. **Event name matching**: Only lines whose `event_name` exactly matches a `PerfCounter` enum value are collected.
