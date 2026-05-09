"""Tests for perf_optimize.diagnostics — IPC and instructions diagnostic metrics.

These metrics are attached to the rubric with weight=0 so they describe HOW the
candidate is winning (microarchitectural vs algorithmic) without affecting the
score.
"""

from __future__ import annotations

import math

import pytest

from perf_optimize.diagnostics import (
    candidate_instructions,
    candidate_ipc,
    instructions_ratio,
    ipc_delta,
    reference_instructions,
    reference_ipc,
)

# ── candidate_instructions ────────────────────────────────────────────────────


class TestCandidateInstructions:
    def test_returns_value_when_present(self) -> None:
        state = {"best_perf_dict": {"cycles": 1000.0, "instructions": 2500.0}}
        assert candidate_instructions(state) == pytest.approx(2500.0)

    def test_returns_zero_when_best_perf_dict_none(self) -> None:
        state = {"best_perf_dict": None}
        assert candidate_instructions(state) == 0.0

    def test_returns_zero_when_best_perf_dict_missing(self) -> None:
        state: dict = {}
        assert candidate_instructions(state) == 0.0

    def test_returns_zero_when_instructions_missing(self) -> None:
        state = {"best_perf_dict": {"cycles": 1000.0}}
        assert candidate_instructions(state) == 0.0

    def test_returns_zero_for_non_finite(self) -> None:
        state = {"best_perf_dict": {"cycles": 1000.0, "instructions": float("nan")}}
        assert candidate_instructions(state) == 0.0

        state2 = {"best_perf_dict": {"cycles": 1000.0, "instructions": float("inf")}}
        assert candidate_instructions(state2) == 0.0

    def test_ignores_extra_kwargs(self) -> None:
        state = {"best_perf_dict": {"cycles": 1000.0, "instructions": 2500.0}}
        assert candidate_instructions(state, completion=[], info={}) == pytest.approx(2500.0)


# ── reference_instructions ────────────────────────────────────────────────────


class TestReferenceInstructions:
    def test_returns_value_when_present(self) -> None:
        state = {"reference_perf": {"cycles": 1000.0, "instructions": 5000.0}}
        assert reference_instructions(state) == pytest.approx(5000.0)

    def test_returns_zero_when_reference_perf_none(self) -> None:
        state = {"reference_perf": None}
        assert reference_instructions(state) == 0.0

    def test_returns_zero_when_reference_perf_missing(self) -> None:
        state: dict = {}
        assert reference_instructions(state) == 0.0

    def test_returns_zero_when_instructions_missing(self) -> None:
        state = {"reference_perf": {"cycles": 1000.0}}
        assert reference_instructions(state) == 0.0

    def test_returns_zero_for_non_finite(self) -> None:
        state = {"reference_perf": {"cycles": 1000.0, "instructions": float("nan")}}
        assert reference_instructions(state) == 0.0

        state2 = {"reference_perf": {"cycles": 1000.0, "instructions": float("-inf")}}
        assert reference_instructions(state2) == 0.0


# ── candidate_ipc ─────────────────────────────────────────────────────────────


class TestCandidateIpc:
    def test_normal_ipc(self) -> None:
        state = {"best_perf_dict": {"cycles": 1000.0, "instructions": 2500.0}}
        assert candidate_ipc(state) == pytest.approx(2.5)

    def test_returns_zero_when_best_perf_dict_none(self) -> None:
        state = {"best_perf_dict": None}
        assert candidate_ipc(state) == 0.0

    def test_returns_zero_when_best_perf_dict_missing(self) -> None:
        state: dict = {}
        assert candidate_ipc(state) == 0.0

    def test_returns_zero_when_cycles_zero(self) -> None:
        state = {"best_perf_dict": {"cycles": 0.0, "instructions": 1000.0}}
        assert candidate_ipc(state) == 0.0

    def test_returns_zero_when_cycles_negative(self) -> None:
        state = {"best_perf_dict": {"cycles": -1.0, "instructions": 1000.0}}
        assert candidate_ipc(state) == 0.0

    def test_returns_zero_when_instructions_zero(self) -> None:
        state = {"best_perf_dict": {"cycles": 1000.0, "instructions": 0.0}}
        assert candidate_ipc(state) == 0.0

    def test_returns_zero_when_cycles_missing(self) -> None:
        state = {"best_perf_dict": {"instructions": 1000.0}}
        assert candidate_ipc(state) == 0.0

    def test_returns_zero_when_instructions_missing(self) -> None:
        state = {"best_perf_dict": {"cycles": 1000.0}}
        assert candidate_ipc(state) == 0.0

    def test_returns_zero_for_non_finite(self) -> None:
        state = {"best_perf_dict": {"cycles": float("nan"), "instructions": 2500.0}}
        assert candidate_ipc(state) == 0.0

        state2 = {"best_perf_dict": {"cycles": 1000.0, "instructions": float("inf")}}
        assert candidate_ipc(state2) == 0.0


# ── reference_ipc ─────────────────────────────────────────────────────────────


class TestReferenceIpc:
    def test_normal_ipc(self) -> None:
        state = {"reference_perf": {"cycles": 1000.0, "instructions": 2000.0}}
        assert reference_ipc(state) == pytest.approx(2.0)

    def test_returns_zero_when_reference_perf_none(self) -> None:
        state = {"reference_perf": None}
        assert reference_ipc(state) == 0.0

    def test_returns_zero_when_reference_perf_missing(self) -> None:
        state: dict = {}
        assert reference_ipc(state) == 0.0

    def test_returns_zero_when_cycles_zero(self) -> None:
        state = {"reference_perf": {"cycles": 0.0, "instructions": 2000.0}}
        assert reference_ipc(state) == 0.0

    def test_returns_zero_when_instructions_zero(self) -> None:
        state = {"reference_perf": {"cycles": 1000.0, "instructions": 0.0}}
        assert reference_ipc(state) == 0.0

    def test_returns_zero_for_non_finite(self) -> None:
        state = {"reference_perf": {"cycles": float("inf"), "instructions": 2000.0}}
        assert reference_ipc(state) == 0.0


# ── ipc_delta ─────────────────────────────────────────────────────────────────


class TestIpcDelta:
    def test_candidate_higher_returns_positive(self) -> None:
        # candidate IPC = 3.0, reference IPC = 2.0 ⇒ delta = +1.0
        state = {
            "best_perf_dict": {"cycles": 1000.0, "instructions": 3000.0},
            "reference_perf": {"cycles": 1000.0, "instructions": 2000.0},
        }
        assert ipc_delta(state) == pytest.approx(1.0)

    def test_candidate_lower_returns_negative(self) -> None:
        # candidate IPC = 1.0, reference IPC = 2.0 ⇒ delta = -1.0
        state = {
            "best_perf_dict": {"cycles": 1000.0, "instructions": 1000.0},
            "reference_perf": {"cycles": 1000.0, "instructions": 2000.0},
        }
        assert ipc_delta(state) == pytest.approx(-1.0)

    def test_equal_ipc_returns_zero(self) -> None:
        state = {
            "best_perf_dict": {"cycles": 1000.0, "instructions": 2000.0},
            "reference_perf": {"cycles": 500.0, "instructions": 1000.0},
        }
        assert ipc_delta(state) == pytest.approx(0.0)

    def test_no_best_returns_negative_reference_ipc(self) -> None:
        # candidate IPC = 0.0, reference IPC = 2.0 ⇒ delta = -2.0
        state = {
            "best_perf_dict": None,
            "reference_perf": {"cycles": 1000.0, "instructions": 2000.0},
        }
        assert ipc_delta(state) == pytest.approx(-2.0)

    def test_no_reference_returns_candidate_ipc(self) -> None:
        # candidate IPC = 3.0, reference IPC = 0.0 ⇒ delta = 3.0
        state = {
            "best_perf_dict": {"cycles": 1000.0, "instructions": 3000.0},
            "reference_perf": None,
        }
        assert ipc_delta(state) == pytest.approx(3.0)

    def test_both_missing_returns_zero(self) -> None:
        state: dict = {}
        assert ipc_delta(state) == 0.0


# ── instructions_ratio ────────────────────────────────────────────────────────


class TestInstructionsRatio:
    def test_candidate_uses_fewer_instructions(self) -> None:
        # candidate uses half as many instructions ⇒ ratio = 0.5 (algorithmic win)
        state = {
            "best_perf_dict": {"cycles": 500.0, "instructions": 1000.0},
            "reference_perf": {"cycles": 1000.0, "instructions": 2000.0},
        }
        assert instructions_ratio(state) == pytest.approx(0.5)
        assert instructions_ratio(state) < 1.0

    def test_candidate_uses_more_instructions(self) -> None:
        state = {
            "best_perf_dict": {"cycles": 1000.0, "instructions": 4000.0},
            "reference_perf": {"cycles": 1000.0, "instructions": 2000.0},
        }
        assert instructions_ratio(state) == pytest.approx(2.0)

    def test_equal_instructions_returns_one(self) -> None:
        state = {
            "best_perf_dict": {"cycles": 500.0, "instructions": 2000.0},
            "reference_perf": {"cycles": 1000.0, "instructions": 2000.0},
        }
        assert instructions_ratio(state) == pytest.approx(1.0)

    def test_zero_reference_returns_zero(self) -> None:
        state = {
            "best_perf_dict": {"cycles": 1000.0, "instructions": 2000.0},
            "reference_perf": {"cycles": 1000.0, "instructions": 0.0},
        }
        assert instructions_ratio(state) == 0.0

    def test_missing_reference_returns_zero(self) -> None:
        state = {
            "best_perf_dict": {"cycles": 1000.0, "instructions": 2000.0},
            "reference_perf": None,
        }
        assert instructions_ratio(state) == 0.0

    def test_missing_candidate_returns_zero(self) -> None:
        # candidate_instructions returns 0.0 when best_perf_dict is None ⇒ ratio = 0.0
        state = {
            "best_perf_dict": None,
            "reference_perf": {"cycles": 1000.0, "instructions": 2000.0},
        }
        assert instructions_ratio(state) == 0.0

    def test_non_finite_reference_returns_zero(self) -> None:
        state = {
            "best_perf_dict": {"cycles": 1000.0, "instructions": 2000.0},
            "reference_perf": {"cycles": 1000.0, "instructions": float("nan")},
        }
        assert instructions_ratio(state) == 0.0


# ── combined diagnostic story ────────────────────────────────────────────────


class TestDiagnosticStory:
    """End-to-end checks that the metrics tell a coherent story."""

    def test_microarchitectural_win(self) -> None:
        """Same instructions, fewer cycles ⇒ higher candidate IPC, ratio = 1.0."""
        state = {
            "best_perf_dict": {"cycles": 500.0, "instructions": 2000.0},
            "reference_perf": {"cycles": 1000.0, "instructions": 2000.0},
        }
        # candidate IPC = 4.0, reference IPC = 2.0
        assert candidate_ipc(state) == pytest.approx(4.0)
        assert reference_ipc(state) == pytest.approx(2.0)
        assert ipc_delta(state) > 0
        assert instructions_ratio(state) == pytest.approx(1.0)

    def test_algorithmic_win(self) -> None:
        """Fewer instructions, same IPC ⇒ ratio < 1.0, delta ~ 0."""
        state = {
            "best_perf_dict": {"cycles": 500.0, "instructions": 1000.0},
            "reference_perf": {"cycles": 1000.0, "instructions": 2000.0},
        }
        # Both have IPC = 2.0
        assert candidate_ipc(state) == pytest.approx(2.0)
        assert reference_ipc(state) == pytest.approx(2.0)
        assert ipc_delta(state) == pytest.approx(0.0)
        assert instructions_ratio(state) < 1.0
        assert instructions_ratio(state) == pytest.approx(0.5)

    def test_all_metrics_finite_with_normal_input(self) -> None:
        state = {
            "best_perf_dict": {"cycles": 500.0, "instructions": 1500.0},
            "reference_perf": {"cycles": 1000.0, "instructions": 2000.0},
        }
        for fn in (
            candidate_instructions,
            reference_instructions,
            candidate_ipc,
            reference_ipc,
            ipc_delta,
            instructions_ratio,
        ):
            value = fn(state)
            assert math.isfinite(value), f"{fn.__name__} returned non-finite {value}"
