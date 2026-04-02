# perf-optimize Overview

## Description

An RL environment that teaches LLMs to optimize code (C, Rust, Python, TypeScript)
for hardware performance. The agent receives a naive reference solution and iteratively
rewrites it in a multi-turn loop, getting structured feedback from Linux `perf` hardware
performance counters. Reward is derived from counter improvements (cycles, cache misses,
branch mispredictions), gated behind correctness verification. Built on the verifiers
SDK (`MultiTurnEnv`) for integration with prime-rl training.

## Subsystems

### Languages (`src/perf_optimize/languages.py`)
Defines `LanguageConfig` for C, Rust, Python, and TypeScript. Each config specifies
how to compile (or syntax-check), execute, and sandbox the language. Runtime paths
(rustup sysroot, nvm node dir) are resolved dynamically.

### Sandbox (`src/perf_optimize/sandbox.py`)
Orchestrates code execution inside bubblewrap (bwrap) containers. Language-aware:
handles compilation or syntax checking, correctness testing, and performance
measurement (perf stat) for all 4 supported languages. All operations are async
via `asyncio.create_subprocess_exec`.

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

### Verifiers Environment (`src/perf_optimize/env.py`)
`PerfOptimizeEnv(MultiTurnEnv)` — the main environment class. Implements the verifiers
`env_response()` hook to process each model turn: extract code from `<code>` tags,
compile/test/measure via the sandbox, and return structured perf feedback. Terminates
via `final_env_response` on `<submit/>` or max turns. Entry point: `load_environment()`
in `__init__.py`.

### Reward (`src/perf_optimize/reward.py`)
Pure functions for reward computation. `correctness_gate` penalizes agents that never
produce correct code (-1.0 compile fail, -0.5 test fail, 0.0 correct). `perf_reward`
computes weighted improvement across available perf counters, with automatic weight
renormalization for missing counters (e.g. AMD lacks LLC-load-misses).

### Prompts (`src/perf_optimize/prompts.py`)
System prompt template and feedback message formatters. Feedback is structured markdown
showing compilation errors, test failures, or perf counter values with improvement
percentages. Only shows counters that are actually available on the hardware.

### Problem Bank (`src/perf_optimize/problems.py`)
Loader for the `problems/` directory structure. Produces HuggingFace Dataset rows with
`question` (problem + reference code), `answer`, and `info` (base64-encoded test data,
reference perf counters). Supports language and hardware profile selection.

## Data Flow

### Measurement Pipeline

```
Source Code (str)
       │
       ▼
   PerfSandbox.compile_and_run()
       │
       ├─ _compile() ──► bwrap + compiler ──► CompilationResult
       │                                         │
       │   ┌─────────────────────────────────────┘
       │   │ (if success)
       │   ▼
       ├─ _run_tests() ──► bwrap + ./solution ──► TestReport
       │                                            │
       │   ┌────────────────────────────────────────┘
       │   │ (if all pass)
       │   ▼
       └─ _run_perf() ──► bwrap + perf stat ──► PerfCounters
```

### Environment Rollout

```
load_environment(language, max_turns)
       │
       ├─ build_dataset_rows() ──► HF Dataset with question/answer/info
       ├─ format_system_prompt()
       ├─ Rubric(correctness_gate, perf_reward)
       │
       ▼
   MultiTurnEnv.rollout(input, client, model, sampling_args)
       │
       ├─ init_state() ──► framework state with client, model, trajectory
       ├─ setup_state() ──► decode base64 test data, init tracking
       │
       └─ LOOP (framework-managed):
            ├─ get_prompt_messages() ──► calls env_response() for turn > 0
            │    env_response():
            │    ├─ extract code from <code>...</code>
            │    ├─ await sandbox.compile_and_run()
            │    ├─ format feedback (compile error / test fail / perf results)
            │    └─ set final_env_response on <submit/> or max turns
            ├─ get_model_response() ──► model generates next response
            ├─ add_model_response() ──► append to trajectory
            └─ is_completed() ──► check has_final_env_response / errors
```

## Features Index

### languages
- description: Multi-language support (C, Rust, Python, TypeScript) with dynamic path resolution
- entry_points: [Language, LanguageConfig, resolve_language_config]
- depends_on: []
- doc: docs/features/languages.md

### sandbox
- description: Bubblewrap-sandboxed code compilation, testing, and perf measurement
- entry_points: [PerfSandbox.compile_and_run, PerfSandbox.measure_only, PerfSandbox.check_prerequisites]
- depends_on: [perf_parser, bwrap_builder, config, languages]
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

### verifiers_environment
- description: Multi-turn verifiers environment for LLM code optimization with perf feedback
- entry_points: [load_environment, PerfOptimizeEnv]
- depends_on: [sandbox, problem_bank, reward, prompts]
- doc: docs/features/verifiers_environment.md

### reward
- description: Profile-aware weighted counter improvement scoring with correctness gating
- entry_points: [compute_weighted_improvement, correctness_gate, perf_reward]
- depends_on: []
- doc: docs/features/verifiers_environment.md

### prompts
- description: System prompt template and structured feedback formatters
- entry_points: [format_system_prompt, format_compile_error, format_test_failure, format_perf_feedback]
- depends_on: []
- doc: docs/features/verifiers_environment.md

### problem_bank
- description: Problem loader producing HuggingFace Dataset rows from problem directories
- entry_points: [build_dataset_rows, load_problem, load_problem_with_reference]
- depends_on: [languages, type_system]
- doc: docs/features/problem_bank.md
