"""Held-out evaluation diagnostics.

These reward functions are attached to the rubric with ``weight=0`` so they
appear as metrics in rollout results without contributing to the training
signal. They expose:

* whether the agent's best candidate generalizes to a held-out test set
  (different RNG seed and/or distribution shape than in-dist tests);
* how the held-out cycles speedup compares to the in-dist cycles speedup --
  a positive divergence is a strong signal of overfitting / lookup-table
  cheating to the in-dist test structure.

All functions accept ``state`` as a plain dict and tolerate missing keys so
they can run before the held-out pass populated the fields, or when the
held-out pass was skipped (e.g. no correct submission ever produced).
"""

from __future__ import annotations

import math
from typing import Any


def _safe_speedup(reference: float | None, candidate: float | None) -> float:
    """Floored fractional speedup ``(reference - candidate) / reference``."""
    if reference is None or candidate is None:
        return 0.0
    if not math.isfinite(reference) or not math.isfinite(candidate):
        return 0.0
    if reference <= 0:
        return 0.0
    return max(0.0, (reference - candidate) / reference)


def heldout_correctness_passed(state: dict[str, Any], **_kwargs: Any) -> float:
    """Returns 1.0 iff every held-out test case passed and at least one ran."""
    total = state.get("heldout_test_total", 0) or 0
    passed = state.get("heldout_test_passed_count", 0) or 0
    if total <= 0:
        return 0.0
    return 1.0 if passed == total else 0.0


def heldout_correctness_pass_rate(state: dict[str, Any], **_kwargs: Any) -> float:
    """Fraction of held-out test cases that passed (0.0 if held-out didn't run)."""
    total = state.get("heldout_test_total", 0) or 0
    passed = state.get("heldout_test_passed_count", 0) or 0
    if total <= 0:
        return 0.0
    return passed / total


def heldout_cycles_speedup(state: dict[str, Any], **_kwargs: Any) -> float:
    """Floored cycles speedup against the held-out reference baseline."""
    ref = state.get("reference_heldout_perf") or {}
    best = state.get("heldout_best_perf") or {}
    if not ref or not best:
        return 0.0
    return _safe_speedup(ref.get("cycles"), best.get("cycles"))


def heldout_wall_clock_ms(state: dict[str, Any], **_kwargs: Any) -> float:
    """Raw wall-clock ms of the best candidate on the held-out perf input."""
    val = state.get("heldout_best_wall_clock_ms")
    if val is None:
        return 0.0
    return float(val)


def cycles_speedup_indist_minus_heldout(
    state: dict[str, Any], **_kwargs: Any
) -> float:
    """In-dist cycles speedup minus held-out cycles speedup.

    Positive values indicate the agent improved more on the in-distribution
    perf input than on the held-out perf input -- a signature of overfitting
    or lookup-table cheating to the in-dist test structure.

    In-dist side reads the largest-size entries from
    ``reference_perf_by_size`` and ``best_perf_by_size`` so the comparison is
    apples-to-apples with the headline metric.
    """
    from .reward import _largest_label

    label = _largest_label(state)
    indist_ref = ((state.get("reference_perf_by_size") or {}).get(label) or {}) if label else {}
    indist_best = ((state.get("best_perf_by_size") or {}).get(label) or {}) if label else {}
    heldout_ref = state.get("reference_heldout_perf") or {}
    heldout_best = state.get("heldout_best_perf") or {}

    if not indist_ref or not indist_best:
        return 0.0
    if not heldout_ref or not heldout_best:
        return 0.0

    indist_ref_cycles = indist_ref.get("cycles")
    heldout_ref_cycles = heldout_ref.get("cycles")
    # Both reference cycles must be a positive, finite value before the
    # divergence is meaningful; otherwise the speedup on that side is
    # undefined and we report no signal.
    if indist_ref_cycles is None or heldout_ref_cycles is None:
        return 0.0
    if not math.isfinite(indist_ref_cycles) or not math.isfinite(heldout_ref_cycles):
        return 0.0
    if indist_ref_cycles <= 0 or heldout_ref_cycles <= 0:
        return 0.0

    indist_speedup = _safe_speedup(indist_ref_cycles, indist_best.get("cycles"))
    heldout_speedup = _safe_speedup(heldout_ref_cycles, heldout_best.get("cycles"))
    return indist_speedup - heldout_speedup
