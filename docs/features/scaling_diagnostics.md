# Scaling Diagnostics

## Scope

In scope:
- Per-size diagnostic metrics attached to the rubric as `weight=0` so they
  appear in eval logs without contributing to reward.
- Closed-form OLS fit of `log(cycles) ~ beta * log(n)` for power-law exponents
  on both candidate and reference runs.
- Geometric-mean cycles speedup across all measured sizes.

Not in scope:
- Selecting the headline reward metric — that lives in `reward.py`
  (`direct_speedup_reward`, `perf_reward`).
- Per-size measurement itself — driven by `PerfSandbox.compile_and_run_sized`
  and tracked in `state["best_perf_by_size"]`.

## Data / Control Flow

```
state["perf_inputs"]: list[(label, n, data)]
state["best_perf_by_size"]: dict[label, dict[counter, float]]
state["reference_perf_by_size"]: dict[label, dict[counter, float] | None]
state["best_wall_clock_ms_by_size"]: dict[label, float]
        │
        ▼
Rubric metric callables (added via add_metric, weight=0):
  largest_size_cycles_speedup(state)
  largest_size_wall_clock_ms(state)
  cycles_speedup_geomean(state)
  scaling_exponent_candidate(state)
  scaling_exponent_reference(state)
  scaling_exponent_delta(state)   # candidate - reference
        │
        ▼
floats logged in rollout metrics for offline analysis
```

`fit_log_log_exponent(ns, cycles)` is the underlying helper: closed-form OLS,
returns `None` when fewer than 2 valid (positive, finite) points remain.

## Files

- `src/perf_optimize/scaling.py`
  Public surface: all metric functions and `fit_log_log_exponent`. Pure;
  reads only from `state` dict.
- `src/perf_optimize/env.py`
  `_attach_scaling_metrics(rubric)` registers all six metrics. Sibling
  branches add their own `_attach_*_metrics` calls after this hook.
- `tests/unit/test_scaling.py`
  Tests for fit correctness (perfect O(N), O(N^2), O(N^3) data), filtering
  of non-finite / non-positive points, and rubric metric behaviour with
  missing largest-size measurements.

## Invariants

- All metric functions return `float` (never raise) so the rubric can
  always score the rollout.
- `largest_size_*` always picks the entry with the maximum `n` from
  `state["perf_inputs"]`; "largest" is well-defined because
  `_load_sizes_toml` requires strictly increasing `n`.
- `fit_log_log_exponent` returns `None` instead of NaN/0 when the input is
  underdetermined; the rubric metric wrappers translate `None` to `0.0`.
- `cycles_speedup_geomean` returns `0.0` when no valid (ref, cand) pair
  exists — never raises on missing data.
