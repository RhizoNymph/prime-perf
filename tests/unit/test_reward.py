"""Tests for perf_optimize.reward — pure reward computation functions."""

from __future__ import annotations

import pytest

from perf_optimize.reward import (
    PERF_WEIGHT_MAP,
    compute_weighted_improvement,
    correctness_gate,
    direct_speedup_reward,
    perf_reward,
)

# ── compute_weighted_improvement ─────────────────────────────────────────────


class TestComputeWeightedImprovement:
    """Tests for compute_weighted_improvement."""

    def test_identical_counters_returns_zero(self) -> None:
        ref = {"cycles": 1000, "branch_misses": 200}
        assert compute_weighted_improvement(ref, ref) == 0.0

    def test_perfect_improvement(self) -> None:
        ref = {"cycles": 1000, "branch_misses": 200}
        agent = {"cycles": 0, "branch_misses": 0}
        assert compute_weighted_improvement(ref, agent) == 1.0

    def test_50_percent_improvement(self) -> None:
        ref = {"cycles": 1000}
        agent = {"cycles": 500}
        result = compute_weighted_improvement(ref, agent)
        assert result == pytest.approx(0.5)

    def test_regression_floors_at_zero(self) -> None:
        """If the agent is worse than reference, result floors at 0.0."""
        ref = {"cycles": 1000}
        agent = {"cycles": 2000}
        assert compute_weighted_improvement(ref, agent) == 0.0

    def test_missing_counters_skipped(self) -> None:
        """Counters missing from agent or ref are skipped, weights renormalized."""
        ref = {"cycles": 1000, "l1_dcache_load_misses": 500}
        agent = {"cycles": 500}  # l1_dcache_load_misses missing
        result = compute_weighted_improvement(ref, agent)
        # Only cycles contributes, 50% improvement
        assert result == pytest.approx(0.5)

    def test_zero_ref_counter_skipped(self) -> None:
        """A reference counter of 0 is skipped (can't compute improvement)."""
        ref = {"cycles": 0, "branch_misses": 200}
        agent = {"cycles": 0, "branch_misses": 100}
        result = compute_weighted_improvement(ref, agent)
        # Only branch_misses contributes, 50% improvement
        assert result == pytest.approx(0.5)

    def test_custom_weights(self) -> None:
        ref = {"cycles": 1000, "branch_misses": 200}
        agent = {"cycles": 500, "branch_misses": 100}
        # Equal weights: both 50% improvement → 0.5
        weights = {"cycles": 1.0, "branch_misses": 1.0}
        result = compute_weighted_improvement(ref, agent, weights=weights)
        assert result == pytest.approx(0.5)

    def test_empty_dicts_returns_zero(self) -> None:
        assert compute_weighted_improvement({}, {}) == 0.0

    def test_amd_scenario_no_llc(self) -> None:
        """AMD doesn't have llc_load_misses — weights redistribute."""
        ref = {
            "cycles": 10_000_000,
            "l1_dcache_load_misses": 500_000,
            "cache_misses": 100_000,
            "branch_misses": 50_000,
        }
        agent = {
            "cycles": 5_000_000,
            "l1_dcache_load_misses": 250_000,
            "cache_misses": 50_000,
            "branch_misses": 25_000,
        }
        result = compute_weighted_improvement(ref, agent)
        # All counters 50% improvement, should be 0.5
        assert result == pytest.approx(0.5)

    def test_intel_scenario_all_counters(self) -> None:
        """Intel has all counters including llc_load_misses."""
        ref = {
            "cycles": 10_000_000,
            "l1_dcache_load_misses": 500_000,
            "cache_misses": 100_000,
            "llc_load_misses": 200_000,
            "branch_misses": 50_000,
        }
        agent = {
            "cycles": 5_000_000,
            "l1_dcache_load_misses": 250_000,
            "cache_misses": 50_000,
            "llc_load_misses": 100_000,
            "branch_misses": 25_000,
        }
        result = compute_weighted_improvement(ref, agent)
        assert result == pytest.approx(0.5)

    def test_mixed_improvement_and_regression(self) -> None:
        """Some counters improve, some regress."""
        ref = {"cycles": 1000, "branch_misses": 100}
        agent = {"cycles": 500, "branch_misses": 200}
        weights = {"cycles": 1.0, "branch_misses": 1.0}
        # cycles: 50% improvement, branch_misses: -100% regression
        # weighted average = (0.5 - 1.0) / 2 = -0.25, floored to 0.0
        result = compute_weighted_improvement(ref, agent, weights=weights)
        assert result == 0.0

    def test_default_weights_match_constant(self) -> None:
        """Verify default weights are PERF_WEIGHT_MAP."""
        ref = {"cycles": 1000}
        agent = {"cycles": 500}
        result_default = compute_weighted_improvement(ref, agent)
        result_explicit = compute_weighted_improvement(ref, agent, weights=PERF_WEIGHT_MAP)
        assert result_default == result_explicit

    def test_nan_ref_value_skipped(self) -> None:
        """A NaN reference value should be skipped."""
        ref = {"cycles": float("nan"), "branch_misses": 200}
        agent = {"cycles": 500, "branch_misses": 100}
        result = compute_weighted_improvement(ref, agent)
        # Only branch_misses contributes, 50% improvement
        assert result == pytest.approx(0.5)

    def test_nan_agent_value_skipped(self) -> None:
        """A NaN agent value should be skipped."""
        ref = {"cycles": 1000, "branch_misses": 200}
        agent = {"cycles": float("nan"), "branch_misses": 100}
        result = compute_weighted_improvement(ref, agent)
        # Only branch_misses contributes, 50% improvement
        assert result == pytest.approx(0.5)

    def test_inf_values_skipped(self) -> None:
        """Infinite values (positive or negative) should be skipped."""
        ref = {"cycles": float("inf"), "branch_misses": 200}
        agent = {"cycles": 500, "branch_misses": 100}
        result = compute_weighted_improvement(ref, agent)
        # Only branch_misses contributes, 50% improvement
        assert result == pytest.approx(0.5)

        # Negative infinity in agent
        ref2 = {"cycles": 1000, "branch_misses": 200}
        agent2 = {"cycles": float("-inf"), "branch_misses": 100}
        result2 = compute_weighted_improvement(ref2, agent2)
        assert result2 == pytest.approx(0.5)


# ── correctness_gate ─────────────────────────────────────────────────────────


class TestCorrectnessGate:
    """Tests for correctness_gate reward function."""

    def test_correct_submissions_returns_zero(self) -> None:
        state = {"correct_submissions": 3, "compile_failures": 1, "test_failures": 1}
        assert correctness_gate(state) == 0.0

    def test_compile_failures_only_returns_minus_one(self) -> None:
        state = {"correct_submissions": 0, "compile_failures": 5, "test_failures": 0}
        assert correctness_gate(state) == -1.0

    def test_test_failures_returns_minus_half(self) -> None:
        state = {"correct_submissions": 0, "compile_failures": 2, "test_failures": 3}
        assert correctness_gate(state) == -0.5

    def test_no_submissions_returns_minus_one(self) -> None:
        state = {"correct_submissions": 0, "compile_failures": 0, "test_failures": 0}
        assert correctness_gate(state) == -1.0

    def test_missing_keys_treated_as_zero(self) -> None:
        state: dict = {}
        assert correctness_gate(state) == -1.0

    def test_ignores_extra_kwargs(self) -> None:
        """Verifiers rubric may pass extra kwargs — they must be ignored."""
        state = {"correct_submissions": 1}
        assert correctness_gate(state, completion=[], info={}) == 0.0


# ── perf_reward ──────────────────────────────────────────────────────────────


class TestPerfReward:
    """Tests for perf_reward function (now keyed on largest-size)."""

    def _state(self, *, best: float | None, ref: float | None) -> dict:
        s: dict = {
            "perf_inputs": [("large", 1024, b"x")],
            "best_perf_by_size": {},
            "reference_perf_by_size": {},
        }
        if best is not None:
            s["best_perf_by_size"]["large"] = {"cycles": best}
        if ref is not None:
            s["reference_perf_by_size"]["large"] = {"cycles": ref}
        return s

    def test_no_best_perf_returns_zero(self) -> None:
        assert perf_reward(self._state(best=None, ref=1000)) == 0.0

    def test_no_reference_perf_returns_zero(self) -> None:
        assert perf_reward(self._state(best=500, ref=None)) == 0.0

    def test_both_missing_returns_zero(self) -> None:
        state: dict = {}
        assert perf_reward(state) == 0.0

    def test_improvement_returns_positive(self) -> None:
        result = perf_reward(self._state(best=5000, ref=10000))
        assert result == pytest.approx(0.5)

    def test_no_improvement_returns_zero(self) -> None:
        assert perf_reward(self._state(best=10000, ref=10000)) == 0.0

    def test_regression_floors_at_zero(self) -> None:
        assert perf_reward(self._state(best=20000, ref=10000)) == 0.0


class TestDirectSpeedupReward:
    """Tests for benchmark-style direct speedup reward.

    The headline metric is now cycles@largest-size; reads ``_by_size`` state shape.
    """

    def test_cycles_speedup_at_largest(self) -> None:
        state = {
            "benchmark_metric": "cycles",
            "perf_inputs": [("small", 256, b"x"), ("large", 1024, b"x")],
            "best_perf_by_size": {
                "small": {"cycles": 1_000.0},
                "large": {"cycles": 5_000.0},
            },
            "reference_perf_by_size": {
                "small": {"cycles": 2_000.0},
                "large": {"cycles": 10_000.0},
            },
        }
        # large is largest by n; (10000 - 5000)/10000 = 0.5
        assert direct_speedup_reward(state) == pytest.approx(0.5)

    def test_wall_clock_speedup_at_largest(self) -> None:
        state = {
            "benchmark_metric": "wall_clock_ms",
            "perf_inputs": [("small", 256, b"x"), ("large", 1024, b"x")],
            "best_wall_clock_ms_by_size": {"small": 5.0, "large": 25.0},
            "reference_wall_clock_ms_by_size": {"small": 20.0, "large": 100.0},
        }
        assert direct_speedup_reward(state) == pytest.approx(0.75)

    def test_regression_floors_at_zero(self) -> None:
        state = {
            "benchmark_metric": "cycles",
            "perf_inputs": [("large", 1024, b"x")],
            "best_perf_by_size": {"large": {"cycles": 12_000.0}},
            "reference_perf_by_size": {"large": {"cycles": 10_000.0}},
        }
        assert direct_speedup_reward(state) == 0.0

    def test_missing_largest_returns_zero(self) -> None:
        state = {
            "benchmark_metric": "cycles",
            "perf_inputs": [("small", 256, b"x"), ("large", 1024, b"x")],
            "best_perf_by_size": {"small": {"cycles": 5_000.0}},
            "reference_perf_by_size": {
                "small": {"cycles": 10_000.0},
                "large": {"cycles": 40_000.0},
            },
        }
        # No candidate at "large" → 0.0
        assert direct_speedup_reward(state) == 0.0

    def test_missing_reference_returns_zero(self) -> None:
        state = {
            "benchmark_metric": "cycles",
            "perf_inputs": [("large", 1024, b"x")],
            "best_perf_by_size": {"large": {"cycles": 5_000.0}},
        }
        assert direct_speedup_reward(state) == 0.0

    def test_no_perf_inputs_returns_zero(self) -> None:
        state = {
            "benchmark_metric": "cycles",
            "best_perf_by_size": {"large": {"cycles": 5_000.0}},
            "reference_perf_by_size": {"large": {"cycles": 10_000.0}},
        }
        assert direct_speedup_reward(state) == 0.0


class TestPerfRewardSized:
    """perf_reward should also key on largest-size by default."""

    def test_uses_largest_size_counters(self) -> None:
        state = {
            "perf_inputs": [("small", 256, b"x"), ("large", 1024, b"x")],
            "best_perf_by_size": {
                "small": {"cycles": 1_000.0},
                "large": {"cycles": 5_000.0},
            },
            "reference_perf_by_size": {
                "small": {"cycles": 2_000.0},
                "large": {"cycles": 10_000.0},
            },
        }
        # weighted_improvement at large: cycles 50% improvement; weight=0.5,
        # only counter present, renormalizes -> 0.5
        assert perf_reward(state) == pytest.approx(0.5)

    def test_missing_largest_returns_zero(self) -> None:
        state = {
            "perf_inputs": [("large", 1024, b"x")],
            "best_perf_by_size": {},
            "reference_perf_by_size": {"large": {"cycles": 10_000.0}},
        }
        assert perf_reward(state) == 0.0
