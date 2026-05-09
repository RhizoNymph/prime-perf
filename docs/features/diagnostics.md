# Feature: IPC / Instructions Diagnostics

## Scope

In scope:
- Attach `weight=0` rubric metrics that surface raw `instructions` retired and
  `IPC` (instructions / cycles) for both the best correct candidate and the
  naive reference solution.
- Provide a derived `ipc_delta` and `instructions_ratio` so reviewers can read
  off the type of win at a glance.
- Pure-function module with no sandbox or verifiers SDK dependency.

Not in scope:
- Changing the reward signal (these metrics never enter the weighted sum).
- Adding new perf counters or measurement passes — `instructions` is already
  collected on both AMD and Intel profiles (see `src/perf_optimize/types.py`,
  `PERF_COUNTER_FIELDS`).
- Per-size diagnostics — sibling branch `feat/scaling-test` reshapes state to
  `state["best_perf_by_size"][label]` / `state["reference_perf_by_size"][label]`.
  After that branch lands, each function in `diagnostics.py` should be rewired
  to read the per-size dicts at the largest label.

## Why

Reward only tells us the candidate is faster, not why. With these metrics:

- IPC up ⇒ microarchitectural win (better vectorization, branch prediction,
  fewer pipeline stalls).
- Instructions down ⇒ algorithmic win (smarter logic, fewer operations).

These two signals are roughly orthogonal and together explain the bulk of
real-world C/Rust optimization patterns.

## Data Flow

```
state["best_perf_dict"]   ─┐
                           ├─► candidate_instructions ─┐
state["reference_perf"]   ─┤                            │
                           ├─► reference_instructions ─┤
                           │                            ├─► attached as
state["best_perf_dict"]   ─┤                            │   weight=0 metrics
state["reference_perf"]   ─┼─► candidate_ipc ──────────┤   on the Rubric
                           │                            │
                           ├─► reference_ipc ──────────┤
                           │                            │
                           ├─► ipc_delta ──────────────┤
                           │                            │
                           └─► instructions_ratio ─────┘
```

All six functions are synchronous and pure: `(state, **_) -> float`. They never
raise — missing keys, `None`, zero, and non-finite values all map to `0.0`.

## Files

- `src/perf_optimize/diagnostics.py` — the six diagnostic functions plus a
  `_safe_float` helper. No external imports beyond `math`.
- `src/perf_optimize/env.py` — imports the five user-facing diagnostic
  functions and defines `_attach_ipc_metrics(rubric)`. The constructor for
  `PerfOptimizeEnv` calls `_attach_ipc_metrics(rubric)` after the existing
  `Rubric(...)` construction.
- `tests/unit/test_diagnostics.py` — unit tests covering every function: normal
  values, missing keys, `None` perf dicts, zero cycles / instructions, and
  non-finite (`NaN`, `inf`) guards. Plus an end-to-end "story" test that
  asserts a microarchitectural-win scenario produces `ipc_delta > 0` and
  `instructions_ratio == 1.0`, while an algorithmic-win scenario produces
  `ipc_delta == 0` and `instructions_ratio < 1.0`.

## Invariants

- All six functions accept arbitrary `**_kwargs` (the verifiers Rubric may pass
  `completion`, `info`, etc.) and only depend on `state`.
- All return values are `float` and finite. The implementation guards against
  `None`, missing keys, zero / negative cycles, zero instructions, and
  non-finite inputs by returning `0.0`.
- The rubric helper `_attach_ipc_metrics` only calls `rubric.add_metric(...)`
  (which is `add_reward_func(..., weight=0)`); it never adjusts weights of
  existing reward functions.
- Reading state field names (`best_perf_dict`, `reference_perf`) must match
  `PerfOptimizeState` in `env.py`. After the `feat/scaling-test` merge, this
  invariant changes to require reading from `_by_size[largest_label]`.
