"""Diagnostic metrics for the perf-optimize environment.

Pure functions — no sandbox or verifiers dependency. These metrics are intended
to be attached to the rubric with ``weight=0`` (via ``rubric.add_metric``) so
they describe HOW the candidate is winning without affecting the score:

- IPC (instructions / cycles) up   ⇒ microarchitectural win (better
  vectorization, branch prediction, fewer pipeline stalls).
- Instructions retired down        ⇒ algorithmic win (smarter logic,
  fewer operations).

State shape today (this branch reads from the singular fields):
    state["best_perf_dict"]: dict[str, float] | None  # best correct candidate
    state["reference_perf"]: dict[str, float] | None  # naive baseline

NOTE on merge with ``feat/scaling-test``: that sibling branch reshapes state
to per-size dicts ``state["best_perf_by_size"][label]`` and
``state["reference_perf_by_size"][label]``. At merge time, each function below
should be rewired to read ``state["best_perf_by_size"][largest_label]`` and
``state["reference_perf_by_size"][largest_label]`` (the largest-label helper
is expected to be exposed from ``perf_optimize.scaling`` or
``perf_optimize.reward`` post-merge).
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


def candidate_instructions(state: dict[str, Any], **_kwargs: Any) -> float:
    """Total instructions retired by the best correct candidate (raw count).

    Returns 0.0 if no correct submission exists or the value is non-finite.
    """
    best = state.get("best_perf_dict") or {}
    return _safe_float(best.get("instructions"))


def reference_instructions(state: dict[str, Any], **_kwargs: Any) -> float:
    """Total instructions retired by the naive reference solution (raw count).

    Returns 0.0 if the problem lacks a reference or the value is non-finite.
    """
    ref = state.get("reference_perf") or {}
    return _safe_float(ref.get("instructions"))


def candidate_ipc(state: dict[str, Any], **_kwargs: Any) -> float:
    """Instructions per cycle for the best correct candidate.

    Returns 0.0 if cycles <= 0, instructions <= 0, missing, or non-finite.
    """
    best = state.get("best_perf_dict") or {}
    cycles = _safe_float(best.get("cycles"))
    instr = _safe_float(best.get("instructions"))
    if cycles <= 0 or instr <= 0:
        return 0.0
    return instr / cycles


def reference_ipc(state: dict[str, Any], **_kwargs: Any) -> float:
    """Instructions per cycle for the naive reference solution.

    Returns 0.0 if cycles <= 0, instructions <= 0, missing, or non-finite.
    """
    ref = state.get("reference_perf") or {}
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
