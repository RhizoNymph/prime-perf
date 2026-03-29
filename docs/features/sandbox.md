# Sandbox Feature

## Scope

**In scope:**
- Compiling C source code inside a bwrap sandbox
- Running correctness tests (binary stdin/stdout comparison)
- Measuring hardware performance counters via perf stat
- Resource limits (memory, processes, file size, timeouts)
- CPU pinning for measurement stability
- Prerequisite checking (tool availability, kernel config)

**Not in scope:**
- Multi-language support (Rust, Python) -- Phase 6
- Concurrent measurement on multiple cores -- Phase 2
- Problem bank loading -- Phase 1 (problems.py)
- Reward computation -- Phase 2 (environment module)

## Data/Control Flow

```
compile_and_run(source_code, test_inputs, expected_outputs, perf_input)
    │
    ├─ Write source_code to {tmpdir}/solution.c
    ├─ Write perf_input to {tmpdir}/perf_input.bin
    │
    ├─ _compile(work_dir)
    │     └─ build_bwrap_command(config, work_dir, [gcc, -O2, -lm, -o, /work/solution, /work/solution.c])
    │     └─ _run_subprocess(bwrap_cmd, timeout=30s)
    │     └─ Returns CompilationSuccess | CompilationFailure
    │
    ├─ [if compiled] _run_tests(work_dir, inputs, expected, names)
    │     └─ For each test case:
    │           └─ build_bwrap_command(config, work_dir, [/work/solution])
    │           └─ _run_subprocess(bwrap_cmd, timeout=10s, stdin_data=input)
    │           └─ Compare stdout bytes to expected
    │     └─ Returns TestReport
    │
    └─ [if all tests pass] _run_perf(work_dir)
          └─ build_perf_command(config, /work/solution)
          └─ build_bwrap_command(config, work_dir, perf_cmd)
          └─ _run_subprocess(bwrap_cmd, timeout=60s, stdin_data=perf_input)
          └─ parse_perf_output(stderr)
          └─ Returns PerfCounters
```

## Files

| File | Role | Key exports |
|------|------|-------------|
| `src/perf_optimize/sandbox.py` | Main orchestrator | `PerfSandbox` |
| `src/perf_optimize/bwrap.py` | Command builders | `build_bwrap_command`, `build_compile_command`, `build_perf_command` |
| `src/perf_optimize/perf_parser.py` | CSV parser | `parse_perf_output`, `parse_csv_line` |
| `src/perf_optimize/config.py` | Configuration | `SandboxConfig` |
| `src/perf_optimize/types.py` | Data types | `ExecutionResult`, `PerfCounters`, `CompilationResult`, `TestReport` |
| `src/perf_optimize/exceptions.py` | Error types | `SandboxError`, `PrerequisiteError`, `PerfParseError` |

## Invariants and Constraints

1. **Tmpdir isolation**: Each `compile_and_run` call creates and cleans up its own tmpdir. No shared state between invocations.
2. **Subprocess timeout**: Every subprocess call has a timeout. On timeout, the process is killed before returning.
3. **Perf requires tests to pass**: `_run_perf` is only called after all tests pass. If tests fail, perf_counters is None.
4. **bwrap uses bash**: The ulimit wrapper uses `/usr/bin/bash` (not sh/dash) because dash lacks `ulimit -u`.
5. **taskset wraps bwrap**: CPU affinity is set at the outermost level so all children inherit it.
6. **perf writes to stderr**: The parser reads stderr, not stdout. stdout contains the program's actual output.
7. **Binary I/O for tests**: Test comparison is byte-exact (not text). Programs use fread/fwrite for binary data.
