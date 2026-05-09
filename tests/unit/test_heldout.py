"""Tests for held-out diagnostic metrics in `perf_optimize.heldout`.

These tests cover the `weight=0` diagnostic functions that surface in-dist
vs. held-out divergence so we can detect lookup-table cheating and overfitting
to test-input structure.
"""

from __future__ import annotations

import pytest

from perf_optimize.heldout import (
    cycles_speedup_indist_minus_heldout,
    heldout_correctness_pass_rate,
    heldout_correctness_passed,
    heldout_cycles_speedup,
    heldout_wall_clock_ms,
)


# ── heldout_correctness_passed ───────────────────────────────────────────────


class TestHeldoutCorrectnessPassed:
    def test_full_pass_returns_one(self) -> None:
        state = {"heldout_test_passed_count": 5, "heldout_test_total": 5}
        assert heldout_correctness_passed(state) == 1.0

    def test_partial_pass_returns_zero(self) -> None:
        state = {"heldout_test_passed_count": 4, "heldout_test_total": 5}
        assert heldout_correctness_passed(state) == 0.0

    def test_zero_passed_returns_zero(self) -> None:
        state = {"heldout_test_passed_count": 0, "heldout_test_total": 5}
        assert heldout_correctness_passed(state) == 0.0

    def test_zero_total_returns_zero(self) -> None:
        """Held-out pass didn't run or no held-out cases — must not credit."""
        state = {"heldout_test_passed_count": 0, "heldout_test_total": 0}
        assert heldout_correctness_passed(state) == 0.0

    def test_missing_keys_returns_zero(self) -> None:
        assert heldout_correctness_passed({}) == 0.0

    def test_ignores_extra_kwargs(self) -> None:
        state = {"heldout_test_passed_count": 3, "heldout_test_total": 3}
        assert heldout_correctness_passed(state, completion=[], info={}) == 1.0


# ── heldout_correctness_pass_rate ────────────────────────────────────────────


class TestHeldoutCorrectnessPassRate:
    def test_full_pass(self) -> None:
        state = {"heldout_test_passed_count": 5, "heldout_test_total": 5}
        assert heldout_correctness_pass_rate(state) == pytest.approx(1.0)

    def test_partial_pass(self) -> None:
        state = {"heldout_test_passed_count": 3, "heldout_test_total": 5}
        assert heldout_correctness_pass_rate(state) == pytest.approx(0.6)

    def test_zero_total_returns_zero(self) -> None:
        state = {"heldout_test_passed_count": 0, "heldout_test_total": 0}
        assert heldout_correctness_pass_rate(state) == 0.0

    def test_missing_keys_returns_zero(self) -> None:
        assert heldout_correctness_pass_rate({}) == 0.0


# ── heldout_cycles_speedup ───────────────────────────────────────────────────


class TestHeldoutCyclesSpeedup:
    def test_50_percent_speedup(self) -> None:
        state = {
            "reference_heldout_perf": {"cycles": 10_000.0},
            "heldout_best_perf": {"cycles": 5_000.0},
        }
        assert heldout_cycles_speedup(state) == pytest.approx(0.5)

    def test_no_change_returns_zero(self) -> None:
        state = {
            "reference_heldout_perf": {"cycles": 1000.0},
            "heldout_best_perf": {"cycles": 1000.0},
        }
        assert heldout_cycles_speedup(state) == 0.0

    def test_regression_floors_at_zero(self) -> None:
        state = {
            "reference_heldout_perf": {"cycles": 1000.0},
            "heldout_best_perf": {"cycles": 2000.0},
        }
        assert heldout_cycles_speedup(state) == 0.0

    def test_no_heldout_perf_returns_zero(self) -> None:
        state = {
            "reference_heldout_perf": {"cycles": 1000.0},
            "heldout_best_perf": None,
        }
        assert heldout_cycles_speedup(state) == 0.0

    def test_no_reference_returns_zero(self) -> None:
        state = {
            "reference_heldout_perf": None,
            "heldout_best_perf": {"cycles": 500.0},
        }
        assert heldout_cycles_speedup(state) == 0.0

    def test_zero_reference_returns_zero(self) -> None:
        state = {
            "reference_heldout_perf": {"cycles": 0.0},
            "heldout_best_perf": {"cycles": 0.0},
        }
        assert heldout_cycles_speedup(state) == 0.0

    def test_missing_state_keys_returns_zero(self) -> None:
        assert heldout_cycles_speedup({}) == 0.0


# ── heldout_wall_clock_ms ────────────────────────────────────────────────────


class TestHeldoutWallClockMs:
    def test_returns_raw_value(self) -> None:
        state = {"heldout_best_wall_clock_ms": 12.5}
        assert heldout_wall_clock_ms(state) == pytest.approx(12.5)

    def test_none_returns_zero(self) -> None:
        state = {"heldout_best_wall_clock_ms": None}
        assert heldout_wall_clock_ms(state) == 0.0

    def test_missing_returns_zero(self) -> None:
        assert heldout_wall_clock_ms({}) == 0.0


# ── cycles_speedup_indist_minus_heldout ──────────────────────────────────────


class TestCyclesSpeedupIndistMinusHeldout:
    @staticmethod
    def _state(
        *,
        ref_cycles: float | None,
        best_cycles: float | None,
        held_ref_cycles: float | None,
        held_best_cycles: float | None,
    ) -> dict[str, Any]:
        """Build a state with sized in-dist fields keyed on a single 'large' entry."""
        ref_by_size: dict[str, dict[str, float] | None] = {}
        best_by_size: dict[str, dict[str, float]] = {}
        if ref_cycles is not None:
            ref_by_size = {"large": {"cycles": ref_cycles}}
        if best_cycles is not None:
            best_by_size = {"large": {"cycles": best_cycles}}
        return {
            "perf_inputs": [("large", 1024, b"")],
            "reference_perf_by_size": ref_by_size,
            "best_perf_by_size": best_by_size,
            "reference_heldout_perf": (
                {"cycles": held_ref_cycles} if held_ref_cycles is not None else None
            ),
            "heldout_best_perf": (
                {"cycles": held_best_cycles} if held_best_cycles is not None else None
            ),
        }

    def test_perfect_match_returns_zero(self) -> None:
        """In-dist and held-out improvements equal -> divergence is zero."""
        state = self._state(
            ref_cycles=10_000.0, best_cycles=5_000.0,
            held_ref_cycles=10_000.0, held_best_cycles=5_000.0,
        )
        assert cycles_speedup_indist_minus_heldout(state) == pytest.approx(0.0)

    def test_overfit_gives_positive_divergence(self) -> None:
        """Large in-dist gain, no held-out gain -> positive divergence."""
        state = self._state(
            ref_cycles=10_000.0, best_cycles=1_000.0,  # 90% speedup
            held_ref_cycles=10_000.0, held_best_cycles=10_000.0,  # 0% speedup
        )
        assert cycles_speedup_indist_minus_heldout(state) == pytest.approx(0.9)

    def test_held_out_better_gives_negative_divergence(self) -> None:
        """Held-out outperforms in-dist -> negative divergence (allowed)."""
        state = self._state(
            ref_cycles=10_000.0, best_cycles=8_000.0,  # 20% speedup
            held_ref_cycles=10_000.0, held_best_cycles=5_000.0,  # 50% speedup
        )
        assert cycles_speedup_indist_minus_heldout(state) == pytest.approx(-0.3)

    def test_held_out_didnt_run_returns_zero(self) -> None:
        """If held-out values are missing/None we cannot compute divergence."""
        state = self._state(
            ref_cycles=10_000.0, best_cycles=5_000.0,
            held_ref_cycles=None, held_best_cycles=None,
        )
        assert cycles_speedup_indist_minus_heldout(state) == 0.0

    def test_in_dist_didnt_run_returns_zero(self) -> None:
        """If in-dist values are missing we cannot compute divergence."""
        state = self._state(
            ref_cycles=None, best_cycles=None,
            held_ref_cycles=10_000.0, held_best_cycles=5_000.0,
        )
        assert cycles_speedup_indist_minus_heldout(state) == 0.0

    def test_zero_reference_cycles_returns_zero(self) -> None:
        """Division-by-zero guard on either reference."""
        state = self._state(
            ref_cycles=0.0, best_cycles=0.0,
            held_ref_cycles=10_000.0, held_best_cycles=5_000.0,
        )
        assert cycles_speedup_indist_minus_heldout(state) == 0.0

    def test_regressions_still_yield_floor_difference(self) -> None:
        """Both speedups are floored at 0 before subtraction."""
        state = self._state(
            ref_cycles=10_000.0, best_cycles=20_000.0,  # -1.0 floored to 0
            held_ref_cycles=10_000.0, held_best_cycles=5_000.0,  # +0.5
        )
        # 0.0 - 0.5 = -0.5
        assert cycles_speedup_indist_minus_heldout(state) == pytest.approx(-0.5)
