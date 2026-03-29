# Feature: Perf Parser

## Scope

**In scope:** Parsing the CSV text output of `perf stat -x ','` (with optional `-r N` repeat) into structured Python dataclasses. Pure functions only -- no I/O, no subprocess calls.

**Not in scope:** Running perf, managing subprocesses, variance analysis, or reward computation.

## Data/Control Flow

1. `parse_perf_output(csv_text)` receives the full stderr string from a `perf stat` invocation.
2. Splits text by newlines and calls `parse_csv_line(line)` for each.
3. `parse_csv_line` handles:
   - Comment lines (`#`) and empty lines -> returns `None`
   - `<not counted>` / `<not supported>` -> raises `CounterNotCountedError`
   - Derived metric lines (empty event_name, spaces in event_name, float counter_value) -> returns `None`
   - Valid data lines -> returns `PerfCSVLine` dataclass
4. `parse_perf_output` collects non-None results, matches `event_name` against `PerfCounter` enum values.
5. Verifies all 7 required counters are present (raises `CounterNotFoundError` if any missing).
6. Returns assembled `PerfCounters` dataclass.

## Files

| File | Role | Key exports |
|------|------|-------------|
| `src/perf_optimize/perf_parser.py` | Parser implementation | `parse_csv_line`, `parse_perf_output` |
| `src/perf_optimize/types.py` | Data types | `PerfCSVLine`, `PerfCounters`, `PerfCounter` |
| `src/perf_optimize/exceptions.py` | Error types | `PerfParseError`, `CounterNotFoundError`, `CounterNotCountedError` |
| `tests/unit/test_perf_parser.py` | Unit tests | 37 test cases |

## Invariants

- `parse_csv_line` and `parse_perf_output` are pure functions with no side effects.
- All 7 `PerfCounter` enum members must be present in output for `parse_perf_output` to succeed.
- `PerfCSVLine` and `PerfCounters` are frozen dataclasses (immutable).
- `<not counted>` / `<not supported>` always raises `CounterNotCountedError` with the counter name and raw value.
- Lines with spaces in the event_name field are treated as derived metrics and skipped.
- Float values in the counter_value field are treated as derived metrics and skipped.

## Perf CSV Format

```
counter_value,unit,event_name,run_time,percentage[,variance]
```

- `counter_value`: integer (or `<not counted>` / `<not supported>`)
- `unit`: string, often empty
- `event_name`: string like `cycles`, `cache-misses`, etc.
- `run_time`: integer nanoseconds
- `percentage`: float (0-100)
- `variance`: float, only present with `-r` flag
