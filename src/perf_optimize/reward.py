"""Reward computation for the perf-optimize environment.

Pure functions — no sandbox or verifiers dependency. Reward functions accept
keyword arguments matching the verifiers Rubric signature introspection
(state, info, completion, etc.).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

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


def perf_reward(state: dict[str, Any], **_kwargs: Any) -> float:
    """Reward component: weighted improvement from the best correct submission.

    Reads ``state["best_perf_dict"]`` and ``state["reference_perf"]``.
    Returns 0.0 if no correct submission or no reference perf available.
    """
    best = state.get("best_perf_dict")
    ref = state.get("reference_perf")
    if best is None or ref is None:
        return 0.0
    return compute_weighted_improvement(ref, best)
