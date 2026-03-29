# Variance Validation Feature

## Scope

**In scope:**
- Statistical validation that perf counters are stable enough for RL reward signals
- Comparing bwrap vs bare execution overhead
- Confirming optimization signal (tiled vs naive matmul) is above measurement noise

**Not in scope:**
- Actual RL training or reward computation
- Problem-specific calibration

## Tests

### test_variance.py (Phase 0 Task 0.4)
Compiles `matmul_naive.c` once, runs it 50 times through `measure_only()`.
Computes coefficient of variation (CV = std/mean) for each counter.

**Thresholds:**
- cycles: CV < 5%
- cache/branch counters: CV < 10%

### test_bwrap_overhead.py (Phase 0 Task 0.5)
Runs matmul 30 times with bwrap and 30 times without. Uses Mann-Whitney U test
(non-parametric) to check if distributions differ significantly at p < 0.01.

### test_signal_detection.py (Phase 0 Task 0.6)
Compiles both naive and tiled matmul, runs each 10 times. Verifies:
- Cycle improvement ratio is > 3x the measurement CV (signal >> noise)
- Cycle improvement is > 10%
- L1 cache miss improvement is > 20%
- Weighted reward signal (per design doc formula) is > 0.05

## Running

```bash
# These are excluded from default pytest runs
uv run pytest -m variance tests/validation/ -v
```

## Prerequisites
- `perf_event_paranoid <= 1`
- Ideally: CPU governor set to `performance`, turbo boost disabled, `isolcpus` for the pinned core

## Files

| File | Role |
|------|------|
| `tests/validation/test_variance.py` | 50-run CV validation |
| `tests/validation/test_bwrap_overhead.py` | bwrap vs bare Mann-Whitney U |
| `tests/validation/test_signal_detection.py` | naive vs tiled signal detection |
| `fixtures/c_programs/matmul_naive.c` | Reference program (cache-hostile) |
| `fixtures/c_programs/matmul_tiled.c` | Optimized program (cache-friendly) |

## Invariants and Constraints

1. **Deterministic input**: All tests use seed=42 for reproducible matrix generation.
2. **Single-core measurement**: All measurements use `taskset -c 0` for CPU pinning.
3. **perf -r 5**: Each measurement internally repeats 5 times (perf's own repetitions).
4. **N=512 for variance/signal tests**: Large enough for measurable differences, small enough for reasonable runtime.
