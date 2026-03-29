# Configuration

## Scope

Immutable configuration for the sandbox and measurement pipeline, loaded from
environment variables with sensible defaults.

### In Scope
- SandboxConfig frozen dataclass with all sandbox/measurement parameters
- Environment variable loading via PERF_OPT_ prefix
- Default values for all fields

### Not in Scope
- Config file loading (JSON/YAML)
- Config validation beyond type conversion
- Runtime config mutation

## Data/Control Flow

```
Environment variables (PERF_OPT_*)
  |
  v
SandboxConfig.from_env() -- reads os.environ, applies overrides
  |
  v
SandboxConfig (frozen dataclass) -- consumed by sandbox, perf measurement, variance checks
```

## Fields

| Field | Type | Default | Env Var |
|-------|------|---------|---------|
| gcc_path | str | "gcc" | PERF_OPT_GCC_PATH |
| gcc_flags | tuple[str, ...] | ("-O2", "-lm") | PERF_OPT_GCC_FLAGS (comma-separated) |
| bwrap_path | str | "bwrap" | PERF_OPT_BWRAP_PATH |
| perf_path | str | "perf" | PERF_OPT_PERF_PATH |
| taskset_path | str | "taskset" | PERF_OPT_TASKSET_PATH |
| compile_timeout_s | float | 30.0 | PERF_OPT_COMPILE_TIMEOUT_S |
| test_timeout_s | float | 10.0 | PERF_OPT_TEST_TIMEOUT_S |
| perf_timeout_s | float | 60.0 | PERF_OPT_PERF_TIMEOUT_S |
| perf_repeat | int | 5 | PERF_OPT_PERF_REPEAT |
| pin_cpu | int | 0 | PERF_OPT_PIN_CPU |
| ulimit_mem_kb | int | 512000 | PERF_OPT_ULIMIT_MEM_KB |
| ulimit_procs | int | 32 | PERF_OPT_ULIMIT_PROCS |
| ulimit_fsize_kb | int | 10240 | PERF_OPT_ULIMIT_FSIZE_KB |
| ro_bind_paths | tuple[str, ...] | ("/usr", "/lib", "/lib64", "/etc/alternatives", "/etc/ld.so.cache") | - |
| perf_counters | tuple[PerfCounter, ...] | all PerfCounter members | - |
| cv_threshold_cycles | float | 0.05 | PERF_OPT_CV_THRESHOLD_CYCLES |
| cv_threshold_cache | float | 0.10 | PERF_OPT_CV_THRESHOLD_CACHE |

## Files

| File | Contents | Key Exports |
|------|----------|-------------|
| `src/perf_optimize/config.py` | SandboxConfig dataclass | SandboxConfig |
| `tests/unit/test_config.py` | 35 tests for defaults and env loading | - |

## Invariants and Constraints

- SandboxConfig is frozen (immutable after construction)
- from_env() only overrides fields that have corresponding PERF_OPT_ environment variables set
- Type conversion is applied: str for paths, float for timeouts/thresholds, int for counts
- gcc_flags are parsed from comma-separated env var, whitespace is stripped
- perf_counters defaults to all PerfCounter enum members
- ro_bind_paths and perf_counters are not overridable via environment variables
