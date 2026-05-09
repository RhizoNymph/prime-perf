"""Reward computation for the perf-optimize environment.

Pure functions — no sandbox or verifiers dependency. Reward functions accept
keyword arguments matching the verifiers Rubric signature introspection
(state, info, completion, etc.).
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


def direct_speedup_reward(state: dict[str, Any], **_kwargs: Any) -> float:
    """Reward component for benchmark-style direct performance evaluation.

    Uses ``state["benchmark_metric"]`` to select the direct metric:
    - ``"cycles"`` reads candidate/reference CPU cycles from perf counters.
    - ``"wall_clock_ms"`` reads candidate/reference wall-clock timing.

    The return value is fractional speedup ``(reference - candidate) / reference``,
    floored at 0.0 so regressions do not receive positive performance credit.
    Correctness is handled separately by ``correctness_gate``.
    """
    metric = state.get("benchmark_metric", "cycles")

    if metric == "wall_clock_ms":
        reference = state.get("reference_wall_clock_ms")
        candidate = state.get("best_wall_clock_ms")
    else:
        reference_perf = state.get("reference_perf") or {}
        best_perf = state.get("best_perf_dict") or {}
        reference = reference_perf.get(metric)
        candidate = best_perf.get(metric)

    if reference is None or candidate is None:
        return 0.0
    if reference <= 0:
        return 0.0
    if not math.isfinite(reference) or not math.isfinite(candidate):
        return 0.0

    return max(0.0, (reference - candidate) / reference)
