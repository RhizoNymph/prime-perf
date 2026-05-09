# Benchmark Environment Mode

## Scope

Benchmark mode reuses `PerfOptimizeEnv` with correctness-only rollout feedback and
direct terminal speedup scoring. It is intended for post-training evaluation, not
for dense RL feedback.

## Entrypoints

- `load_environment(..., feedback_mode="correctness", reward_mode="benchmark")`
- `load_benchmark_environment(...)`

Both entrypoints still execute generated code through `PerfSandbox.compile_and_run()`.
That means candidate code is compiled, tested, and measured with the same bubblewrap,
taskset, timeout, and resource-limit path used by training.

## Metrics

`benchmark_metric` selects the direct metric used for terminal reward:

- `cycles` (default): uses `reference_perf["cycles"]` and the best candidate
  `perf stat` cycle count.
- `wall_clock_ms`: uses `reference_wall_clock_ms` and `best_wall_clock_ms` when a
  problem bank includes a reference wall-clock baseline.
- Any other perf counter key can be used if it exists in both `reference_perf` and
  the candidate perf result.

The reward is correctness-gated by the existing `correctness_gate` rubric component.
For correct submissions, `direct_speedup_reward` returns:

```text
(reference_metric - candidate_metric) / reference_metric
```

Regressions are floored at `0.0`.

## Feedback

`feedback_mode="correctness"` hides performance counters from the model. Passing
submissions receive only correctness confirmation, while failed submissions still
receive compiler or test failure feedback. This avoids leaking the training-time
counter signal into held-out benchmark rollouts.

## Eval Config

Use the Prime eval path:

```bash
prime eval run configs/eval/perf-optimize-benchmark.toml
```

The initial config evaluates the current problem bank in benchmark mode. A mature
benchmark should point `problems_dir` or `problems` at a held-out problem set before
being used for post-training claims.
