"""Scaling diagnostics — weight=0 metrics attached to the rubric.

All metric functions accept ``state`` plus arbitrary kwargs (verifiers Rubric
introspection passes extras). Each returns ``float`` so they can be used as
``weight=0`` reward functions for observability.

The state shape this module reads:
- ``perf_inputs``: list of ``(label, n, data)`` tuples (sorted ascending by n).
- ``best_perf_by_size``: dict[label -> dict[counter -> float]].
- ``reference_perf_by_size``: dict[label -> dict[counter -> float] | None].
- ``best_wall_clock_ms_by_size``: dict[label -> float].
"""

from __future__ import annotations

import math
from typing import Any


def _largest_label(state: dict[str, Any]) -> str | None:
    """Return the label with the largest ``n``; None if no perf inputs.

    Inputs are normally pre-sorted ascending; we still take the max to be safe.
    """
    pis = state.get("perf_inputs") or []
    if not pis:
        return None
    # Each entry is (label, n, data) — index 1 is n
    largest = max(pis, key=lambda p: p[1])
    return largest[0]


def _safe_speedup(reference: float | None, candidate: float | None) -> float:
    """Fractional speedup ``(reference - candidate) / reference``, floored at 0.

    Returns 0.0 when either value is missing, non-finite, or ``reference <= 0``.
    """
    if reference is None or candidate is None:
        return 0.0
    if not math.isfinite(reference) or not math.isfinite(candidate):
        return 0.0
    if reference <= 0:
        return 0.0
    return max(0.0, (reference - candidate) / reference)


def _safe_ratio(reference: float | None, candidate: float | None) -> float | None:
    """``reference / candidate`` for geomean speedup; None when invalid."""
    if reference is None or candidate is None:
        return None
    if not math.isfinite(reference) or not math.isfinite(candidate):
        return None
    if reference <= 0 or candidate <= 0:
        return None
    return reference / candidate


def fit_log_log_exponent(ns: list[int], cycles: list[float]) -> float | None:
    """Fit a power-law exponent: cycles ~ n^beta.

    Computes the OLS slope of ``log(cycles)`` on ``log(n)`` (closed form).
    Filters out any pair where ``n <= 0``, ``cycles <= 0``, or non-finite.
    Returns ``None`` if fewer than 2 valid points remain or input lengths
    differ.
    """
    if len(ns) != len(cycles):
        return None

    xs: list[float] = []
    ys: list[float] = []
    for n, c in zip(ns, cycles, strict=True):
        if n <= 0 or c <= 0:
            continue
        if not math.isfinite(c):
            continue
        xs.append(math.log(n))
        ys.append(math.log(c))

    if len(xs) < 2:
        return None

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0.0:
        # All x identical (e.g., duplicate n's after filtering) — slope undefined.
        return None
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return cov_xy / var_x


# ── Rubric metric functions ─────────────────────────────────────────────────


def largest_size_cycles_speedup(state: dict[str, Any], **_kwargs: Any) -> float:
    """Cycles speedup at the largest input size — weight=0 diagnostic.

    Distinct from the headline reward in that this *always* uses cycles,
    regardless of ``state["benchmark_metric"]``.
    """
    label = _largest_label(state)
    if label is None:
        return 0.0
    ref = ((state.get("reference_perf_by_size") or {}).get(label) or {}).get("cycles")
    cand = ((state.get("best_perf_by_size") or {}).get(label) or {}).get("cycles")
    return _safe_speedup(ref, cand)


def largest_size_wall_clock_ms(state: dict[str, Any], **_kwargs: Any) -> float:
    """Raw wall-clock ms at the largest size — sanity check, weight=0.

    Returns 0.0 when missing.
    """
    label = _largest_label(state)
    if label is None:
        return 0.0
    val = (state.get("best_wall_clock_ms_by_size") or {}).get(label)
    if val is None or not math.isfinite(val):
        return 0.0
    return float(val)


def cycles_speedup_geomean(state: dict[str, Any], **_kwargs: Any) -> float:
    """Geometric mean of per-size cycles speedup ratios (ref/cand).

    Skips sizes with missing or non-positive values. Returns 0.0 if no valid
    points.
    """
    pis = state.get("perf_inputs") or []
    ref_by_size = state.get("reference_perf_by_size") or {}
    best_by_size = state.get("best_perf_by_size") or {}

    log_sum = 0.0
    n = 0
    for entry in pis:
        label = entry[0]
        ref = (ref_by_size.get(label) or {}).get("cycles")
        cand = (best_by_size.get(label) or {}).get("cycles")
        ratio = _safe_ratio(ref, cand)
        if ratio is None:
            continue
        log_sum += math.log(ratio)
        n += 1

    if n == 0:
        return 0.0
    return math.exp(log_sum / n)


def _exponent_from_state(
    state: dict[str, Any], by_size_key: str
) -> float:
    pis = state.get("perf_inputs") or []
    if not pis:
        return 0.0
    by_size = state.get(by_size_key) or {}
    ns: list[int] = []
    cycles: list[float] = []
    for entry in pis:
        label, n_val = entry[0], entry[1]
        counters = by_size.get(label) or {}
        cyc = counters.get("cycles")
        if cyc is None:
            continue
        ns.append(int(n_val))
        cycles.append(float(cyc))
    beta = fit_log_log_exponent(ns, cycles)
    return 0.0 if beta is None else beta


def scaling_exponent_candidate(state: dict[str, Any], **_kwargs: Any) -> float:
    """Power-law exponent fitted to candidate cycles per size."""
    return _exponent_from_state(state, "best_perf_by_size")


def scaling_exponent_reference(state: dict[str, Any], **_kwargs: Any) -> float:
    """Power-law exponent fitted to reference cycles per size."""
    return _exponent_from_state(state, "reference_perf_by_size")


def scaling_exponent_delta(state: dict[str, Any], **_kwargs: Any) -> float:
    """``beta_candidate - beta_reference``. Positive => candidate scales worse."""
    return scaling_exponent_candidate(state) - scaling_exponent_reference(state)
