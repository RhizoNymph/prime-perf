"""Diagnostic metrics for the perf-optimize environment.

Pure functions — no sandbox or verifiers dependency. These metrics are intended
to be attached to the rubric with ``weight=0`` (via ``rubric.add_metric``) so
they describe HOW the candidate is winning without affecting the score:

- IPC (instructions / cycles) up   ⇒ microarchitectural win (better
  vectorization, branch prediction, fewer pipeline stalls).
- Instructions retired down        ⇒ algorithmic win (smarter logic,
  fewer operations).

All metrics read from the largest-size entries of the per-size state dicts
(``state["best_perf_by_size"][largest_label]`` and
``state["reference_perf_by_size"][largest_label]``) so they're aligned with
the headline cycles-speedup-at-largest reward.
"""

from __future__ import annotations

import math
from typing import Any


def _safe_float(value: Any) -> float:
    """Coerce ``value`` to a finite float. Returns 0.0 for None / non-numeric / non-finite."""
    if value is None:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def _largest_perf(state: dict[str, Any], by_size_key: str) -> dict[str, Any]:
    """Return the largest-size counter dict from ``state[by_size_key]``.

    Reads the largest label off ``state["perf_inputs"]`` (list of
    ``(label, n, data)`` tuples) and returns the matching dict, or {} if
    nothing usable is present.
    """
    from .reward import _largest_label

    label = _largest_label(state)
    if label is None:
        return {}
    return (state.get(by_size_key) or {}).get(label) or {}


def candidate_instructions(state: dict[str, Any], **_kwargs: Any) -> float:
    """Total instructions retired by the best correct candidate (raw count).

    Returns 0.0 if no correct submission exists or the value is non-finite.
    """
    best = _largest_perf(state, "best_perf_by_size")
    return _safe_float(best.get("instructions"))


def reference_instructions(state: dict[str, Any], **_kwargs: Any) -> float:
    """Total instructions retired by the naive reference solution (raw count).

    Returns 0.0 if the problem lacks a reference or the value is non-finite.
    """
    ref = _largest_perf(state, "reference_perf_by_size")
    return _safe_float(ref.get("instructions"))


def candidate_ipc(state: dict[str, Any], **_kwargs: Any) -> float:
    """Instructions per cycle for the best correct candidate.

    Returns 0.0 if cycles <= 0, instructions <= 0, missing, or non-finite.
    """
    best = _largest_perf(state, "best_perf_by_size")
    cycles = _safe_float(best.get("cycles"))
    instr = _safe_float(best.get("instructions"))
    if cycles <= 0 or instr <= 0:
        return 0.0
    return instr / cycles


def reference_ipc(state: dict[str, Any], **_kwargs: Any) -> float:
    """Instructions per cycle for the naive reference solution.

    Returns 0.0 if cycles <= 0, instructions <= 0, missing, or non-finite.
    """
    ref = _largest_perf(state, "reference_perf_by_size")
    cycles = _safe_float(ref.get("cycles"))
    instr = _safe_float(ref.get("instructions"))
    if cycles <= 0 or instr <= 0:
        return 0.0
    return instr / cycles


def ipc_delta(state: dict[str, Any], **_kwargs: Any) -> float:
    """``candidate_ipc - reference_ipc``.

    Positive ⇒ candidate uses microarchitecture better than the reference.
    Negative ⇒ candidate has worse IPC than the reference (typically when paired
    with an even larger reduction in instructions, i.e. an algorithmic win).
    """
    return candidate_ipc(state) - reference_ipc(state)


def instructions_ratio(state: dict[str, Any], **_kwargs: Any) -> float:
    """``candidate_instructions / reference_instructions``.

    < 1.0 ⇒ algorithmic win (candidate executes fewer instructions).
    Returns 0.0 if reference instructions are 0 or non-finite.
    """
    ref = reference_instructions(state)
    if ref <= 0:
        return 0.0
    return candidate_instructions(state) / ref
