# perf-optimize Overview

## Description

An RL environment for teaching LLMs to optimize C code using hardware performance counters.
The system compiles C programs in a sandboxed environment, runs tests for correctness, and
measures performance via Linux `perf stat` with CPU pinning and variance validation.

## Subsystems

- **Type System** - Immutable data types that flow through the measurement pipeline: perf
  counters, compilation/test results, execution outcomes, and variance statistics.
- **Exception Hierarchy** - Structured domain exceptions for prerequisites, sandbox execution,
  and perf output parsing.
- **Configuration** - Frozen dataclass config loaded from environment variables with sensible
  defaults.
- **Sandbox** (planned) - bwrap-based sandboxed compilation and execution of C programs.
- **Perf Parser** (planned) - Parsing of `perf stat` CSV output into typed counter values.
- **Variance Validation** (planned) - Statistical checks on counter stability across repeated
  measurements.

## Data Flow

```
C source code
  -> SandboxConfig (compilation flags, timeouts, CPU pin)
  -> Compile (gcc in bwrap) -> CompilationResult
  -> Test execution -> TestReport
  -> perf stat measurement -> PerfCSVLine[] -> PerfCounters
  -> Variance check -> VarianceReport
  -> ExecutionResult (full pipeline outcome)
```

## Features Index

### type_system
- description: Core immutable types for the measurement pipeline
- entry_points: [src/perf_optimize/types.py]
- depends_on: []
- doc: docs/features/type_system.md

### exception_hierarchy
- description: Structured domain exception classes
- entry_points: [src/perf_optimize/exceptions.py]
- depends_on: []
- doc: docs/features/exception_hierarchy.md

### configuration
- description: Sandbox and measurement pipeline configuration
- entry_points: [src/perf_optimize/config.py]
- depends_on: [type_system]
- doc: docs/features/configuration.md
