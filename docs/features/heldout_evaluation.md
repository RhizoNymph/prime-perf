# Held-out Evaluation

## In scope

- A diagnostic pass that runs **after** the normal rollout turn loop. It
  takes the best correct candidate the agent produced, recompiles it inside
  the sandbox, and runs it against:
  - a held-out test set (`tests_heldout/input_*.bin` + `expected_*.bin`)
  - a single held-out perf input (`perf_inputs_heldout/large.bin`)
- Surfacing the divergence between in-dist and held-out cycles speedup as a
  `weight=0` metric so we can detect lookup-table cheating and overfitting
  to the in-dist test structure during evaluation.
- Per-problem held-out fixtures that come from a **different distribution**
  than the in-dist tests (different RNG seed at minimum, different
  problem-instance shape ideally).

## Not in scope

- Held-out data does **not** contribute to the training reward signal: every
  held-out reward function is registered with `weight=0`.
- Held-out is run once per rollout (not per turn). The cleanup is
  idempotent and must not crash the rollout if the sandbox fails.
- Multi-size held-out evaluation is not in scope here (a sibling branch
  rewrites in-dist measurement to be multi-size; held-out remains
  single-size for now).

## Per-problem distributional choices

| Problem    | In-dist                                | Held-out                                            |
|------------|----------------------------------------|-----------------------------------------------------|
| sort       | uniform random floats                  | bimodal / clustered / sorted-with-noise / NaN-mixed |
| matmul     | dense uniform [0, 1) matrices          | low-rank + sparse, ill-conditioned                  |
| nbody      | uniform random positions in [-1, 1]^3  | two-cluster Gaussian initial conditions             |
| stencil    | i.i.d. uniform random initial state    | checkerboard / smooth gradient initial state        |
| hash_table | uniform random ASCII keys              | Zipf-distributed lookups + sequential numeric keys  |

All held-out generators use a separate RNG seed (1337) distinct from the
in-dist seed (42). The held-out perf input is sized comparably to the
in-dist large perf input so cycle counts can be compared 1:1.

## Data / control flow

```
build_dataset_rows()  ──► info dict carries:
    test_inputs, expected_outputs, perf_input          (in-dist)
    heldout_test_inputs, heldout_expected_outputs,
    heldout_perf_input, reference_heldout_perf,
    reference_heldout_wall_clock_ms                    (held-out)

setup_state(state)    ──► base64-decodes both groups, zeros heldout_*
                          tracking fields (heldout_test_passed = None,
                          heldout_test_total = N, ...)

TurnProcessor         ──► whenever a new best correct candidate is
                          selected, records best_candidate_source = code
                          alongside best_perf_dict.

@vf.cleanup
PerfOptimizeEnv._run_heldout_pass(state):
    if state.heldout_test_passed is not None:    return  # idempotent
    if not state.correct_submissions:            return
    if not state.best_candidate_source:          return
    if no held-out fixtures in state:            return
    try:
        result = await self._sandbox.compile_and_run(
            best_candidate_source, heldout_test_inputs,
            heldout_expected_outputs, heldout_perf_input,
            comparison=...,
        )
    except SandboxError as e:
        log warning, return            # diagnostic only, never crash
    state.heldout_test_passed       = result.test_report.all_passed
    state.heldout_test_passed_count = result.test_report.passed
    state.heldout_test_total        = result.test_report.total
    state.heldout_best_perf         = result.perf_counters.to_dict()
    state.heldout_best_wall_clock_ms = result.wall_clock_ms

Rubric metrics (all weight=0):
    heldout_correctness_passed
    heldout_correctness_pass_rate
    heldout_cycles_speedup
    heldout_wall_clock_ms
    cycles_speedup_indist_minus_heldout
```

## Files and key entry points

- `src/perf_optimize/heldout.py` — pure-function `weight=0` reward functions:
  `heldout_correctness_passed`, `heldout_correctness_pass_rate`,
  `heldout_cycles_speedup`, `heldout_wall_clock_ms`,
  `cycles_speedup_indist_minus_heldout`.
- `src/perf_optimize/problems.py` — `_load_heldout_test_files`,
  `_load_heldout_perf_input`, `_load_heldout_reference_perf`. `ProblemSpec`
  gains `heldout_test_inputs`, `heldout_expected_outputs`,
  `heldout_perf_input`. `build_dataset_rows` writes the held-out keys into
  the info dict.
- `src/perf_optimize/env.py` — `PerfOptimizeState` gains the namespaced
  `heldout_*` fields; `setup_state` decodes the base64 held-out data and
  initializes the result fields; `_attach_heldout_metrics(rubric)` wires
  the metrics; `@vf.cleanup` `_run_heldout_pass` runs the cleanup pass.
- `src/perf_optimize/processor.py` — records `best_candidate_source = code`
  whenever the in-dist best is updated.
- `scripts/generate_<problem>_tests.py` — each problem has a held-out
  generation block driven by its own RNG seed and distributional choice.
- `scripts/generate_reference_perf.py` — `_generate_heldout` writes
  `problems/<name>/reference_perf/c_<profile>_heldout.json` with
  `cycles`, `instructions`, ..., and `wall_clock_ms`.

## Invariants

- All held-out reward functions are attached with `weight=0`. They are
  metrics, not training signal.
- The held-out cleanup is idempotent — re-invocation must not change the
  populated state.
- The held-out cleanup never propagates exceptions. On any sandbox failure
  it logs and returns; the diagnostic fields stay None / 0.
- Held-out test inputs and held-out perf inputs come from a **different
  distribution** than in-dist. At minimum a different RNG seed; ideally a
  different problem-instance shape.
- The held-out perf input is sized comparably to the in-dist large perf
  input so cycle counts are directly comparable.
- The `_attach_heldout_metrics(rubric)` call site stays on its own line so
  sibling branches can interleave their own `_attach_*_metrics` calls
  without textual conflicts.

## Merge notes

Several sibling branches edit overlapping regions and must merge
additively:

- `feat/scaling-test` rewrites in-dist state (singular `reference_perf`,
  `best_perf_dict` -> `_by_size` dicts) and rewrites the
  `processor.py` best-tracking. After it merges, the
  `cycles_speedup_indist_minus_heldout` divergence metric must read
  `state["reference_perf_by_size"][largest_label]["cycles"]` and
  `state["best_perf_by_size"][largest_label]["cycles"]` instead of the
  current singular fields.
- `feat/ipc-diagnostics` adds its own `_attach_*_metrics(rubric)` call
  near the rubric construction; both calls coexist trivially.
