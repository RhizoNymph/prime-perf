"""Tests for perf_optimize.diagnostics — IPC and instructions diagnostic metrics.

These metrics are attached to the rubric with weight=0 so they describe HOW the
candidate is winning (microarchitectural vs algorithmic) without affecting the
score. They read the largest-size entries from the per-size state dicts.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from perf_optimize.diagnostics import (
    candidate_instructions,
    candidate_ipc,
    instructions_ratio,
    ipc_delta,
    reference_instructions,
    reference_ipc,
)


def _state(
    *,
    best: dict[str, float] | None = None,
    ref: dict[str, float] | None = None,
    label: str = "large",
    n: int = 1024,
) -> dict[str, Any]:
    """Build a per-size state with a single largest-size entry."""
    state: dict[str, Any] = {"perf_inputs": [(label, n, b"")]}
    if best is not None:
        state["best_perf_by_size"] = {label: best}
    if ref is not None:
        state["reference_perf_by_size"] = {label: ref}
    return state


# ── candidate_instructions ────────────────────────────────────────────────────


class TestCandidateInstructions:
    def test_returns_value_when_present(self) -> None:
        assert candidate_instructions(
            _state(best={"cycles": 1000.0, "instructions": 2500.0})
        ) == pytest.approx(2500.0)

    def test_returns_zero_when_best_missing(self) -> None:
        assert candidate_instructions(_state(best=None)) == 0.0

    def test_returns_zero_when_state_missing(self) -> None:
        assert candidate_instructions({}) == 0.0

    def test_returns_zero_when_instructions_missing(self) -> None:
        assert candidate_instructions(_state(best={"cycles": 1000.0})) == 0.0

    def test_returns_zero_for_non_finite(self) -> None:
        assert candidate_instructions(
            _state(best={"cycles": 1000.0, "instructions": float("nan")})
        ) == 0.0
        assert candidate_instructions(
            _state(best={"cycles": 1000.0, "instructions": float("inf")})
        ) == 0.0

    def test_ignores_extra_kwargs(self) -> None:
        s = _state(best={"cycles": 1000.0, "instructions": 2500.0})
        assert candidate_instructions(s, completion=[], info={}) == pytest.approx(2500.0)


# ── reference_instructions ────────────────────────────────────────────────────


class TestReferenceInstructions:
    def test_returns_value_when_present(self) -> None:
        assert reference_instructions(
            _state(ref={"cycles": 1000.0, "instructions": 5000.0})
        ) == pytest.approx(5000.0)

    def test_returns_zero_when_ref_missing(self) -> None:
        assert reference_instructions(_state(ref=None)) == 0.0

    def test_returns_zero_when_state_missing(self) -> None:
        assert reference_instructions({}) == 0.0

    def test_returns_zero_when_instructions_missing(self) -> None:
        assert reference_instructions(_state(ref={"cycles": 1000.0})) == 0.0

    def test_returns_zero_for_non_finite(self) -> None:
        assert reference_instructions(
            _state(ref={"cycles": 1000.0, "instructions": float("nan")})
        ) == 0.0
        assert reference_instructions(
            _state(ref={"cycles": 1000.0, "instructions": float("-inf")})
        ) == 0.0


# ── candidate_ipc ─────────────────────────────────────────────────────────────


class TestCandidateIpc:
    def test_normal_ipc(self) -> None:
        assert candidate_ipc(
            _state(best={"cycles": 1000.0, "instructions": 2500.0})
        ) == pytest.approx(2.5)

    def test_returns_zero_when_best_missing(self) -> None:
        assert candidate_ipc(_state(best=None)) == 0.0

    def test_returns_zero_when_state_missing(self) -> None:
        assert candidate_ipc({}) == 0.0

    def test_returns_zero_when_cycles_zero(self) -> None:
        assert candidate_ipc(
            _state(best={"cycles": 0.0, "instructions": 1000.0})
        ) == 0.0

    def test_returns_zero_when_cycles_negative(self) -> None:
        assert candidate_ipc(
            _state(best={"cycles": -1.0, "instructions": 1000.0})
        ) == 0.0

    def test_returns_zero_when_instructions_zero(self) -> None:
        assert candidate_ipc(
            _state(best={"cycles": 1000.0, "instructions": 0.0})
        ) == 0.0

    def test_returns_zero_when_cycles_missing(self) -> None:
        assert candidate_ipc(_state(best={"instructions": 1000.0})) == 0.0

    def test_returns_zero_when_instructions_missing(self) -> None:
        assert candidate_ipc(_state(best={"cycles": 1000.0})) == 0.0

    def test_returns_zero_for_non_finite(self) -> None:
        assert candidate_ipc(
            _state(best={"cycles": float("nan"), "instructions": 2500.0})
        ) == 0.0
        assert candidate_ipc(
            _state(best={"cycles": 1000.0, "instructions": float("inf")})
        ) == 0.0


# ── reference_ipc ─────────────────────────────────────────────────────────────


class TestReferenceIpc:
    def test_normal_ipc(self) -> None:
        assert reference_ipc(
            _state(ref={"cycles": 1000.0, "instructions": 2000.0})
        ) == pytest.approx(2.0)

    def test_returns_zero_when_ref_missing(self) -> None:
        assert reference_ipc(_state(ref=None)) == 0.0

    def test_returns_zero_when_state_missing(self) -> None:
        assert reference_ipc({}) == 0.0

    def test_returns_zero_when_cycles_zero(self) -> None:
        assert reference_ipc(
            _state(ref={"cycles": 0.0, "instructions": 2000.0})
        ) == 0.0

    def test_returns_zero_when_instructions_zero(self) -> None:
        assert reference_ipc(
            _state(ref={"cycles": 1000.0, "instructions": 0.0})
        ) == 0.0

    def test_returns_zero_for_non_finite(self) -> None:
        assert reference_ipc(
            _state(ref={"cycles": float("inf"), "instructions": 2000.0})
        ) == 0.0


# ── ipc_delta ─────────────────────────────────────────────────────────────────


class TestIpcDelta:
    def test_candidate_higher_returns_positive(self) -> None:
        # candidate IPC = 3.0, reference IPC = 2.0 ⇒ delta = +1.0
        s = _state(
            best={"cycles": 1000.0, "instructions": 3000.0},
            ref={"cycles": 1000.0, "instructions": 2000.0},
        )
        assert ipc_delta(s) == pytest.approx(1.0)

    def test_candidate_lower_returns_negative(self) -> None:
        # candidate IPC = 1.0, reference IPC = 2.0 ⇒ delta = -1.0
        s = _state(
            best={"cycles": 1000.0, "instructions": 1000.0},
            ref={"cycles": 1000.0, "instructions": 2000.0},
        )
        assert ipc_delta(s) == pytest.approx(-1.0)

    def test_equal_ipc_returns_zero(self) -> None:
        s = _state(
            best={"cycles": 1000.0, "instructions": 2000.0},
            ref={"cycles": 500.0, "instructions": 1000.0},
        )
        assert ipc_delta(s) == pytest.approx(0.0)

    def test_no_best_returns_negative_reference_ipc(self) -> None:
        s = _state(best=None, ref={"cycles": 1000.0, "instructions": 2000.0})
        assert ipc_delta(s) == pytest.approx(-2.0)

    def test_no_reference_returns_candidate_ipc(self) -> None:
        s = _state(best={"cycles": 1000.0, "instructions": 3000.0}, ref=None)
        assert ipc_delta(s) == pytest.approx(3.0)

    def test_both_missing_returns_zero(self) -> None:
        assert ipc_delta({}) == 0.0


# ── instructions_ratio ────────────────────────────────────────────────────────


class TestInstructionsRatio:
    def test_candidate_uses_fewer_instructions(self) -> None:
        s = _state(
            best={"cycles": 500.0, "instructions": 1000.0},
            ref={"cycles": 1000.0, "instructions": 2000.0},
        )
        assert instructions_ratio(s) == pytest.approx(0.5)
        assert instructions_ratio(s) < 1.0

    def test_candidate_uses_more_instructions(self) -> None:
        s = _state(
            best={"cycles": 1000.0, "instructions": 4000.0},
            ref={"cycles": 1000.0, "instructions": 2000.0},
        )
        assert instructions_ratio(s) == pytest.approx(2.0)

    def test_equal_instructions_returns_one(self) -> None:
        s = _state(
            best={"cycles": 500.0, "instructions": 2000.0},
            ref={"cycles": 1000.0, "instructions": 2000.0},
        )
        assert instructions_ratio(s) == pytest.approx(1.0)

    def test_zero_reference_returns_zero(self) -> None:
        s = _state(
            best={"cycles": 1000.0, "instructions": 2000.0},
            ref={"cycles": 1000.0, "instructions": 0.0},
        )
        assert instructions_ratio(s) == 0.0

    def test_missing_reference_returns_zero(self) -> None:
        s = _state(best={"cycles": 1000.0, "instructions": 2000.0}, ref=None)
        assert instructions_ratio(s) == 0.0

    def test_missing_candidate_returns_zero(self) -> None:
        s = _state(best=None, ref={"cycles": 1000.0, "instructions": 2000.0})
        assert instructions_ratio(s) == 0.0

    def test_non_finite_reference_returns_zero(self) -> None:
        s = _state(
            best={"cycles": 1000.0, "instructions": 2000.0},
            ref={"cycles": 1000.0, "instructions": float("nan")},
        )
        assert instructions_ratio(s) == 0.0


# ── combined diagnostic story ────────────────────────────────────────────────


class TestDiagnosticStory:
    """End-to-end checks that the metrics tell a coherent story."""

    def test_microarchitectural_win(self) -> None:
        """Same instructions, fewer cycles ⇒ higher candidate IPC, ratio = 1.0."""
        s = _state(
            best={"cycles": 500.0, "instructions": 2000.0},
            ref={"cycles": 1000.0, "instructions": 2000.0},
        )
        assert candidate_ipc(s) == pytest.approx(4.0)
        assert reference_ipc(s) == pytest.approx(2.0)
        assert ipc_delta(s) > 0
        assert instructions_ratio(s) == pytest.approx(1.0)

    def test_algorithmic_win(self) -> None:
        """Fewer instructions, same IPC ⇒ ratio < 1.0, delta ~ 0."""
        s = _state(
            best={"cycles": 500.0, "instructions": 1000.0},
            ref={"cycles": 1000.0, "instructions": 2000.0},
        )
        assert candidate_ipc(s) == pytest.approx(2.0)
        assert reference_ipc(s) == pytest.approx(2.0)
        assert ipc_delta(s) == pytest.approx(0.0)
        assert instructions_ratio(s) < 1.0
        assert instructions_ratio(s) == pytest.approx(0.5)

    def test_all_metrics_finite_with_normal_input(self) -> None:
        s = _state(
            best={"cycles": 500.0, "instructions": 1500.0},
            ref={"cycles": 1000.0, "instructions": 2000.0},
        )
        for fn in (
            candidate_instructions,
            reference_instructions,
            candidate_ipc,
            reference_ipc,
            ipc_delta,
            instructions_ratio,
        ):
            value = fn(s)
            assert math.isfinite(value), f"{fn.__name__} returned non-finite {value}"
