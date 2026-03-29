# perf-optimize Overview

## Description

An RL environment that teaches LLMs to optimize C code for hardware performance.
The agent receives a naive reference solution and iteratively rewrites it, getting
structured feedback from Linux `perf` hardware performance counters. Reward is derived
from counter improvements (cycles, cache misses, branch mispredictions), gated behind
correctness verification.

## Subsystems

### Sandbox (`src/perf_optimize/sandbox.py`)
Orchestrates code execution inside bubblewrap (bwrap) containers. Handles compilation
(gcc), correctness testing, and performance measurement (perf stat). All operations
are async via `asyncio.create_subprocess_exec`.

### Hardware Profiles (`src/perf_optimize/counters.py`)
Maps logical counter fields (e.g. `llc_load_misses`) to architecture-specific perf
events (e.g. `LLC-load-misses` on Intel, not available on AMD). Auto-detects CPU
vendor. AMD uses 5 events to avoid PMU multiplexing; Intel uses 7.

### Perf Parser (`src/perf_optimize/perf_parser.py`)
Pure-function parser for `perf stat -x ','` CSV output. Takes a `HardwareProfile`
to map hardware event names back to `PerfCounters` fields. Raises on `<not counted>`
(PMU failure) and `<not supported>` (wrong profile).

### Command Builder (`src/perf_optimize/bwrap.py`)
Pure functions that construct command-line argument lists for bwrap, gcc, and perf stat.
Uses the hardware profile to select which events to request.

### Type System (`src/perf_optimize/types.py`)
Immutable dataclasses for the measurement pipeline: `PerfCounters`, `ExecutionResult`,
`CompilationResult` (discriminated union), `TestReport`, `VarianceReport`. Designed so
invalid states are unrepresentable.

### Exception Hierarchy (`src/perf_optimize/exceptions.py`)
Structured domain exceptions for prerequisites, sandbox execution, and perf output parsing.

### Configuration (`src/perf_optimize/config.py`)
`SandboxConfig` frozen dataclass with `PERF_OPT_*` env var overrides. Controls tool
paths, timeouts, resource limits, CPU pinning, perf counter selection, and variance
thresholds.

## Data Flow

```
Source Code (str)
       │
       ▼
   PerfSandbox.compile_and_run()
       │
       ├─ _compile() ──► bwrap + gcc ──► CompilationResult
       │                                    │
       │   ┌────────────────────────────────┘
       │   │ (if success)
       │   ▼
       ├─ _run_tests() ──► bwrap + ./solution ──► TestReport
       │                                            │
       │   ┌────────────────────────────────────────┘
       │   │ (if all pass)
       │   ▼
       └─ _run_perf() ──► bwrap + perf stat ──► stderr CSV
                                                    │
                                              parse_perf_output()
                                                    │
                                                    ▼
                                              PerfCounters
                                                    │
                                                    ▼
                                            ExecutionResult
```

## Features Index

### sandbox
- description: Bubblewrap-sandboxed code compilation, testing, and perf measurement
- entry_points: [PerfSandbox.compile_and_run, PerfSandbox.measure_only, PerfSandbox.check_prerequisites]
- depends_on: [perf_parser, bwrap_builder, config]
- doc: docs/features/sandbox.md

### perf_parser
- description: Parse perf stat CSV output into typed PerfCounters
- entry_points: [parse_perf_output, parse_csv_line]
- depends_on: [type_system]
- doc: docs/features/perf_parser.md

### bwrap_builder
- description: Construct bwrap, gcc, and perf stat command lines
- entry_points: [build_bwrap_command, build_compile_command, build_perf_command]
- depends_on: [config]
- doc: docs/features/bwrap_builder.md

### hardware_profiles
- description: Architecture-aware counter profiles for AMD and Intel
- entry_points: [HardwareProfile, AMD_ZEN, INTEL_CORE, detect_profile]
- depends_on: []
- doc: docs/features/hardware_profiles.md

### type_system
- description: Immutable dataclasses for the measurement pipeline
- entry_points: [PerfCounters, ExecutionResult, CompilationResult, TestReport]
- depends_on: []
- doc: docs/features/type_system.md

### exception_hierarchy
- description: Structured domain exception classes
- entry_points: [src/perf_optimize/exceptions.py]
- depends_on: []
- doc: docs/features/exception_hierarchy.md

### config
- description: Sandbox configuration with env var overrides
- entry_points: [SandboxConfig, SandboxConfig.from_env]
- depends_on: [type_system, hardware_profiles]
- doc: docs/features/config.md

### variance_validation
- description: Statistical tests proving measurement pipeline stability
- entry_points: [tests/validation/test_variance.py, tests/validation/test_bwrap_overhead.py, tests/validation/test_signal_detection.py]
- depends_on: [sandbox]
- doc: docs/features/variance_validation.md
