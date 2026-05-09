"""Tests for scaling diagnostics: log-log fit, geomean speedup, exponent delta."""

from __future__ import annotations

import math

import pytest

from perf_optimize.scaling import (
    cycles_speedup_geomean,
    fit_log_log_exponent,
    largest_size_cycles_speedup,
    largest_size_wall_clock_ms,
    scaling_exponent_candidate,
    scaling_exponent_delta,
    scaling_exponent_reference,
)

# ── fit_log_log_exponent ─────────────────────────────────────────────────────


class TestFitLogLogExponent:
    def test_perfect_o_n_squared(self) -> None:
        ns = [1, 2, 4, 8]
        cycles = [float(n * n) for n in ns]
        beta = fit_log_log_exponent(ns, cycles)
        assert beta is not None
        assert beta == pytest.approx(2.0)

    def test_perfect_linear(self) -> None:
        ns = [1, 2, 4, 8, 16]
        cycles = [float(n) for n in ns]
        beta = fit_log_log_exponent(ns, cycles)
        assert beta is not None
        assert beta == pytest.approx(1.0)

    def test_perfect_o_n_cubed(self) -> None:
        ns = [2, 4, 8, 16]
        cycles = [float(n) ** 3 for n in ns]
        beta = fit_log_log_exponent(ns, cycles)
        assert beta is not None
        assert beta == pytest.approx(3.0)

    def test_log_n(self) -> None:
        # cycles = n * log(n) is harder; just verify between 1 and 2
        ns = [16, 32, 64, 128, 256]
        cycles = [float(n) * math.log(n) for n in ns]
        beta = fit_log_log_exponent(ns, cycles)
        assert beta is not None
        assert 1.0 < beta < 1.5

    def test_single_point_returns_none(self) -> None:
        beta = fit_log_log_exponent([256], [1000.0])
        assert beta is None

    def test_empty_returns_none(self) -> None:
        beta = fit_log_log_exponent([], [])
        assert beta is None

    def test_filters_non_finite(self) -> None:
        ns = [1, 2, 4, 8]
        cycles = [1.0, float("nan"), 16.0, 64.0]
        beta = fit_log_log_exponent(ns, cycles)
        assert beta is not None
        # Without nan, points are (1,1), (4,16), (8,64) — slope is still 2
        assert beta == pytest.approx(2.0)

    def test_filters_zero_n(self) -> None:
        # log(0) is undefined; n=0 should be filtered, leaving fewer points
        ns = [0, 2, 4]
        cycles = [0.0, 4.0, 16.0]
        beta = fit_log_log_exponent(ns, cycles)
        assert beta is not None
        assert beta == pytest.approx(2.0)

    def test_filters_zero_cycles(self) -> None:
        ns = [1, 2, 4]
        cycles = [0.0, 4.0, 16.0]
        beta = fit_log_log_exponent(ns, cycles)
        assert beta is not None
        assert beta == pytest.approx(2.0)

    def test_filters_negative_cycles(self) -> None:
        ns = [1, 2, 4, 8]
        cycles = [1.0, -1.0, 16.0, 64.0]
        beta = fit_log_log_exponent(ns, cycles)
        assert beta is not None
        assert beta == pytest.approx(2.0)

    def test_after_filter_too_few_returns_none(self) -> None:
        ns = [1, 2]
        cycles = [float("inf"), 4.0]
        beta = fit_log_log_exponent(ns, cycles)
        assert beta is None

    def test_mismatched_lengths_returns_none(self) -> None:
        beta = fit_log_log_exponent([1, 2], [1.0])
        assert beta is None


# ── largest_size_cycles_speedup ──────────────────────────────────────────────


class TestLargestSizeCyclesSpeedup:
    def test_speedup_at_largest(self) -> None:
        state = {
            "perf_inputs": [("small", 256, b"x"), ("large", 1024, b"x")],
            "best_perf_by_size": {
                "small": {"cycles": 500.0},
                "large": {"cycles": 2_000.0},
            },
            "reference_perf_by_size": {
                "small": {"cycles": 1000.0},
                "large": {"cycles": 4_000.0},
            },
        }
        # large is largest by n; speedup = (4000 - 2000)/4000 = 0.5
        assert largest_size_cycles_speedup(state) == pytest.approx(0.5)

    def test_missing_largest_returns_zero(self) -> None:
        state = {
            "perf_inputs": [("small", 256, b"x"), ("large", 1024, b"x")],
            "best_perf_by_size": {"small": {"cycles": 500.0}},
            "reference_perf_by_size": {
                "small": {"cycles": 1000.0},
                "large": {"cycles": 4_000.0},
            },
        }
        assert largest_size_cycles_speedup(state) == 0.0

    def test_no_inputs_returns_zero(self) -> None:
        state: dict = {}
        assert largest_size_cycles_speedup(state) == 0.0

    def test_regression_floors_at_zero(self) -> None:
        state = {
            "perf_inputs": [("large", 1024, b"x")],
            "best_perf_by_size": {"large": {"cycles": 6_000.0}},
            "reference_perf_by_size": {"large": {"cycles": 4_000.0}},
        }
        assert largest_size_cycles_speedup(state) == 0.0


# ── largest_size_wall_clock_ms ────────────────────────────────────────────────


class TestLargestSizeWallClockMs:
    def test_returns_largest(self) -> None:
        state = {
            "perf_inputs": [("small", 256, b"x"), ("large", 1024, b"x")],
            "best_wall_clock_ms_by_size": {"small": 1.0, "large": 4.0},
        }
        assert largest_size_wall_clock_ms(state) == pytest.approx(4.0)

    def test_missing_returns_zero(self) -> None:
        state = {
            "perf_inputs": [("small", 256, b"x"), ("large", 1024, b"x")],
            "best_wall_clock_ms_by_size": {"small": 1.0},
        }
        assert largest_size_wall_clock_ms(state) == 0.0

    def test_empty_state_returns_zero(self) -> None:
        assert largest_size_wall_clock_ms({}) == 0.0


# ── cycles_speedup_geomean ──────────────────────────────────────────────────


class TestCyclesSpeedupGeomean:
    def test_geomean_of_ratios(self) -> None:
        state = {
            "perf_inputs": [
                ("small", 100, b"x"),
                ("medium", 200, b"x"),
                ("large", 400, b"x"),
            ],
            "best_perf_by_size": {
                "small": {"cycles": 250.0},
                "medium": {"cycles": 500.0},
                "large": {"cycles": 1000.0},
            },
            "reference_perf_by_size": {
                "small": {"cycles": 1000.0},
                "medium": {"cycles": 2000.0},
                "large": {"cycles": 4000.0},
            },
        }
        # ratios = ref/cand = 4.0, 4.0, 4.0; geomean = 4.0
        result = cycles_speedup_geomean(state)
        assert result == pytest.approx(4.0)

    def test_skips_missing(self) -> None:
        state = {
            "perf_inputs": [
                ("small", 100, b"x"),
                ("large", 400, b"x"),
            ],
            "best_perf_by_size": {
                "small": {"cycles": 250.0},
                # large missing
            },
            "reference_perf_by_size": {
                "small": {"cycles": 1000.0},
                "large": {"cycles": 4000.0},
            },
        }
        # only small contributes; ratio = 4.0
        assert cycles_speedup_geomean(state) == pytest.approx(4.0)

    def test_empty_returns_zero(self) -> None:
        state: dict = {}
        assert cycles_speedup_geomean(state) == 0.0

    def test_no_valid_points_returns_zero(self) -> None:
        state = {
            "perf_inputs": [("small", 100, b"x")],
            "best_perf_by_size": {},
            "reference_perf_by_size": {},
        }
        assert cycles_speedup_geomean(state) == 0.0

    def test_skips_zero_candidate(self) -> None:
        state = {
            "perf_inputs": [
                ("small", 100, b"x"),
                ("large", 400, b"x"),
            ],
            "best_perf_by_size": {
                "small": {"cycles": 0.0},
                "large": {"cycles": 1000.0},
            },
            "reference_perf_by_size": {
                "small": {"cycles": 1000.0},
                "large": {"cycles": 4000.0},
            },
        }
        # Only large counts; geomean = 4.0
        assert cycles_speedup_geomean(state) == pytest.approx(4.0)


# ── scaling_exponent_candidate / reference ──────────────────────────────────


class TestScalingExponents:
    def test_candidate_exponent(self) -> None:
        state = {
            "perf_inputs": [("small", 2, b"x"), ("medium", 4, b"x"), ("large", 8, b"x")],
            "best_perf_by_size": {
                "small": {"cycles": 4.0},
                "medium": {"cycles": 16.0},
                "large": {"cycles": 64.0},
            },
        }
        assert scaling_exponent_candidate(state) == pytest.approx(2.0)

    def test_reference_exponent(self) -> None:
        state = {
            "perf_inputs": [("small", 2, b"x"), ("medium", 4, b"x"), ("large", 8, b"x")],
            "reference_perf_by_size": {
                "small": {"cycles": 8.0},
                "medium": {"cycles": 64.0},
                "large": {"cycles": 512.0},
            },
        }
        # cycles = n^3 -> exponent 3
        assert scaling_exponent_reference(state) == pytest.approx(3.0)

    def test_delta_positive_when_candidate_worse(self) -> None:
        state = {
            "perf_inputs": [("small", 2, b"x"), ("medium", 4, b"x"), ("large", 8, b"x")],
            "best_perf_by_size": {
                "small": {"cycles": 8.0},
                "medium": {"cycles": 64.0},
                "large": {"cycles": 512.0},
            },
            "reference_perf_by_size": {
                "small": {"cycles": 4.0},
                "medium": {"cycles": 16.0},
                "large": {"cycles": 64.0},
            },
        }
        # candidate ~ n^3, reference ~ n^2; delta ~ +1
        delta = scaling_exponent_delta(state)
        assert delta == pytest.approx(1.0)

    def test_delta_zero_when_same(self) -> None:
        state = {
            "perf_inputs": [("small", 2, b"x"), ("medium", 4, b"x"), ("large", 8, b"x")],
            "best_perf_by_size": {
                "small": {"cycles": 4.0},
                "medium": {"cycles": 16.0},
                "large": {"cycles": 64.0},
            },
            "reference_perf_by_size": {
                "small": {"cycles": 4.0},
                "medium": {"cycles": 16.0},
                "large": {"cycles": 64.0},
            },
        }
        assert scaling_exponent_delta(state) == pytest.approx(0.0)

    def test_returns_zero_when_no_data(self) -> None:
        state: dict = {}
        assert scaling_exponent_candidate(state) == 0.0
        assert scaling_exponent_reference(state) == 0.0
        assert scaling_exponent_delta(state) == 0.0

    def test_too_few_points_returns_zero(self) -> None:
        state = {
            "perf_inputs": [("small", 2, b"x")],
            "best_perf_by_size": {"small": {"cycles": 100.0}},
            "reference_perf_by_size": {"small": {"cycles": 100.0}},
        }
        assert scaling_exponent_candidate(state) == 0.0
        assert scaling_exponent_reference(state) == 0.0
        assert scaling_exponent_delta(state) == 0.0
