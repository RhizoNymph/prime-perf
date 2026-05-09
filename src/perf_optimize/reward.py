"""Reward computation for the perf-optimize environment.

Pure functions — no sandbox or verifiers dependency. Reward functions accept
keyword arguments matching the verifiers Rubric signature introspection
(state, info, completion, etc.).

In sized-perf mode, the headline metric is *cycles at the largest input
size*. ``benchmark_metric="cycles"`` is redefined to mean "cycles at
largest"; ``benchmark_metric="wall_clock_ms"`` likewise reads
``best_wall_clock_ms_by_size[<largest>]``.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

# Default weights for performance counter improvement scoring.
# Keys match PerfCounters.to_dict() field names.
PERF_WEIGHT_MAP: Mapping[str, float] = MappingProxyType({
    "cycles": 0.5,
    "l1_dcache_load_misses": 0.2,
    "cache_misses": 0.2,
    "llc_load_misses": 0.2,
    "branch_misses": 0.1,
})


def compute_weighted_improvement(
    ref: dict[str, float],
    agent: dict[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Compute weighted improvement ratio across available counters.

    For each counter present in both ``ref`` and ``agent``, computes the
    fractional improvement ``(ref - agent) / ref``. Counters missing from
    either dict or with a reference value of zero are skipped. The remaining
    weights are renormalized so they sum to 1.0.

    Args:
        ref: Reference counter values (from the naive solution).
        agent: Agent's counter values (from the optimized solution).
        weights: Per-counter weights. Defaults to ``PERF_WEIGHT_MAP``.

    Returns:
        Weighted improvement score, floored at 0.0.
    """
    if weights is None:
        weights = PERF_WEIGHT_MAP

    total_weight = 0.0
    weighted_sum = 0.0

    for counter, w in weights.items():
        ref_val = ref.get(counter)
        agent_val = agent.get(counter)
        if ref_val is None or agent_val is None:
            continue
        if ref_val == 0:
            continue
        if not math.isfinite(ref_val) or not math.isfinite(agent_val):
            continue
        improvement = (ref_val - agent_val) / ref_val
        weighted_sum += w * improvement
        total_weight += w

    if total_weight == 0:
        return 0.0

    score = weighted_sum / total_weight
    return max(0.0, score)


def correctness_gate(state: dict[str, Any], **_kwargs: Any) -> float:
    """Reward component: penalize if the agent never produced correct code.

    Returns:
        -1.0 if no submission ever compiled successfully.
        -0.5 if compiled but never passed all tests.
         0.0 if at least one submission was correct.
    """
    correct = state.get("correct_submissions", 0)
    if correct > 0:
        return 0.0

    compile_failures = state.get("compile_failures", 0)
    test_failures = state.get("test_failures", 0)

    # If there were test failures, the code compiled at least once
    if test_failures > 0:
        return -0.5

    # Only compile failures (or no submissions at all)
    if compile_failures > 0:
        return -1.0

    # No submissions at all — treat as total failure
    return -1.0


def _largest_label(state: dict[str, Any]) -> str | None:
    """Pick the size label with the largest ``n`` from ``state["perf_inputs"]``."""
    pis = state.get("perf_inputs") or []
    if not pis:
        return None
    largest = max(pis, key=lambda p: p[1])
    return largest[0]


def _safe_speedup(reference: float | None, candidate: float | None) -> float:
    """Fractional speedup ``(reference - candidate) / reference``, floored at 0."""
    if reference is None or candidate is None:
        return 0.0
    if not math.isfinite(reference) or not math.isfinite(candidate):
        return 0.0
    if reference <= 0:
        return 0.0
    return max(0.0, (reference - candidate) / reference)


def perf_reward(state: dict[str, Any], **_kwargs: Any) -> float:
    """Reward component: weighted improvement at the largest size.

    Reads per-size counters from
    ``state["best_perf_by_size"]`` and ``state["reference_perf_by_size"]``,
    keyed on the largest-size label. Returns 0.0 if either is missing.
    """
    label = _largest_label(state)
    if label is None:
        return 0.0
    best = (state.get("best_perf_by_size") or {}).get(label)
    ref = (state.get("reference_perf_by_size") or {}).get(label)
    if best is None or ref is None:
        return 0.0
    return compute_weighted_improvement(ref, best)


def direct_speedup_reward(state: dict[str, Any], **_kwargs: Any) -> float:
    """Reward component for benchmark-style direct performance evaluation.

    Headline metric (``benchmark_metric``):
    - ``"cycles"`` → cycles at the largest input size.
    - ``"wall_clock_ms"`` → wall-clock at the largest input size.

    Returns fractional speedup, floored at 0.0 so regressions don't earn
    positive credit. Correctness is handled separately by ``correctness_gate``.
    """
    metric = state.get("benchmark_metric", "cycles")
    label = _largest_label(state)
    if label is None:
        return 0.0

    if metric == "wall_clock_ms":
        ref = (state.get("reference_wall_clock_ms_by_size") or {}).get(label)
        cand = (state.get("best_wall_clock_ms_by_size") or {}).get(label)
    else:
        ref = ((state.get("reference_perf_by_size") or {}).get(label) or {}).get(metric)
        cand = ((state.get("best_perf_by_size") or {}).get(label) or {}).get(metric)

    return _safe_speedup(ref, cand)
