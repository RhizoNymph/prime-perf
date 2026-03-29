# perf-optimize Overview

## Description

An RL environment that teaches LLMs to optimize C code for hardware performance. Code runs inside a bubblewrap sandbox with measurements from Linux `perf stat` hardware performance counters. Reward is derived from improvements in cycles, cache misses, and branch mispredictions relative to a naive reference solution.

## Subsystems

- **Types & Config** (`src/perf_optimize/types.py`, `config.py`, `exceptions.py`): Core data types, enums, configuration dataclasses, and structured exception hierarchy.
- **Perf Parser** (`src/perf_optimize/perf_parser.py`): Pure-function parser for `perf stat -x ','` CSV output. Converts raw text into structured `PerfCounters`.
- **Bwrap Builder** (`src/perf_optimize/bwrap.py`): Pure-function command-line builders for taskset+bwrap sandbox invocation, GCC compilation, and perf measurement.
- **PerfSandbox** (planned): Async orchestrator that ties compilation, testing, and measurement together inside bubblewrap.
- **Environment** (planned): `MultiTurnEnv` subclass for the verifiers SDK, managing episodes and reward computation.

## Data Flow

1. Agent submits C source code.
2. **Bwrap Builder** constructs compile command -> **PerfSandbox** runs it inside bwrap.
3. If compilation succeeds, bwrap builder constructs test commands -> sandbox runs correctness tests.
4. If tests pass, bwrap builder constructs perf command -> sandbox runs `perf stat`.
5. **Perf Parser** converts perf's CSV stderr output into `PerfCounters`.
6. Environment computes reward from counter improvements vs reference.

## Features Index

### perf_parser
- description: Parses `perf stat -x ','` CSV output into structured counter data.
- entry_points: [`parse_csv_line`, `parse_perf_output`]
- depends_on: [types, exceptions]
- doc: docs/features/perf_parser.md

### bwrap_builder
- description: Builds command-line argument lists for bwrap sandbox, GCC, and perf stat.
- entry_points: [`build_bwrap_command`, `build_compile_command`, `build_perf_command`]
- depends_on: [config, types]
- doc: docs/features/bwrap_builder.md

### types_and_config
- description: Core data types (PerfCounter enum, PerfCSVLine, PerfCounters), configuration (SandboxConfig), and exception hierarchy.
- entry_points: [PerfCounter, PerfCSVLine, PerfCounters, SandboxConfig]
- depends_on: []
- doc: docs/features/types_and_config.md
