"""Tests for perf_optimize.prompts — system prompt and feedback formatters."""

from __future__ import annotations

from perf_optimize.prompts import (
    format_compile_error,
    format_no_code_found,
    format_perf_feedback,
    format_system_prompt,
    format_test_failure,
)


class TestFormatSystemPrompt:
    def test_contains_language(self) -> None:
        result = format_system_prompt("c", 5)
        assert "**Language:** c" in result

    def test_contains_max_turns(self) -> None:
        result = format_system_prompt("rust", 10)
        assert "**Turns remaining:** 10" in result

    def test_contains_code_tag_example(self) -> None:
        result = format_system_prompt("c", 5)
        assert "<code" in result
        assert "</code>" in result

    def test_contains_submit_tag(self) -> None:
        result = format_system_prompt("c", 5)
        assert "<submit/>" in result


class TestFormatCompileError:
    def test_contains_stderr(self) -> None:
        result = format_compile_error("error: undeclared identifier", 1, 5)
        assert "undeclared identifier" in result

    def test_contains_turn_info(self) -> None:
        result = format_compile_error("error", 2, 5)
        assert "turn 2/5" in result

    def test_contains_failed_label(self) -> None:
        result = format_compile_error("error", 1, 5)
        assert "Compilation failed" in result

    def test_truncates_long_stderr(self) -> None:
        long_err = "x" * 5000
        result = format_compile_error(long_err, 1, 5)
        # Should be truncated to 2000 chars
        assert len(result) < 5000


class TestFormatTestFailure:
    def test_shows_pass_count(self) -> None:
        result = format_test_failure(3, 5, ["err1", "err2"], 1, 5)
        assert "3/5 passed" in result

    def test_shows_errors(self) -> None:
        result = format_test_failure(3, 5, ["wrong output at byte 0"], 1, 5)
        assert "wrong output at byte 0" in result

    def test_turn_info(self) -> None:
        result = format_test_failure(0, 5, [], 2, 7)
        assert "turn 2/7" in result

    def test_limits_errors_shown(self) -> None:
        errors = [f"error_{i}" for i in range(20)]
        result = format_test_failure(0, 5, errors, 1, 5)
        # Should show at most 5 errors
        assert "error_4" in result
        assert "error_5" not in result

    def test_empty_errors(self) -> None:
        result = format_test_failure(0, 5, [], 1, 5)
        assert "0/5 passed" in result


class TestFormatPerfFeedback:
    def test_shows_counter_values(self) -> None:
        agent = {"cycles": 5_000_000}
        ref = {"cycles": 10_000_000}
        result = format_perf_feedback(agent, ref, 1, 5)
        assert "5,000,000" in result

    def test_shows_improvement_percentage(self) -> None:
        agent = {"cycles": 5_000_000}
        ref = {"cycles": 10_000_000}
        result = format_perf_feedback(agent, ref, 1, 5)
        assert "+50.0%" in result

    def test_shows_regression(self) -> None:
        agent = {"cycles": 15_000_000}
        ref = {"cycles": 10_000_000}
        result = format_perf_feedback(agent, ref, 1, 5)
        assert "-50.0%" in result

    def test_missing_ref_counter(self) -> None:
        """Counter not in ref should be shown without comparison."""
        agent = {"cycles": 5_000_000, "branch_misses": 100}
        ref = {"cycles": 10_000_000}
        result = format_perf_feedback(agent, ref, 1, 5)
        assert "100" in result

    def test_turn_info(self) -> None:
        result = format_perf_feedback({"cycles": 100}, {"cycles": 200}, 3, 5)
        assert "turn 3/5" in result

    def test_all_tests_passed_label(self) -> None:
        result = format_perf_feedback({"cycles": 100}, {"cycles": 200}, 1, 5)
        assert "All tests passed" in result

    def test_submit_hint(self) -> None:
        result = format_perf_feedback({"cycles": 100}, {"cycles": 200}, 1, 5)
        assert "<submit/>" in result

    def test_filters_to_rewarded_counters(self) -> None:
        """Only rewarded counters should appear when filter is provided."""
        agent = {"cycles": 5_000, "instructions": 10_000, "cache_misses": 200}
        ref = {"cycles": 10_000, "instructions": 8_000, "cache_misses": 400}
        result = format_perf_feedback(
            agent, ref, 1, 5, rewarded_counters={"cycles", "cache_misses"}
        )
        assert "cycles" in result
        assert "cache_misses" in result
        assert "instructions" not in result

    def test_no_filter_shows_all_counters(self) -> None:
        """Without filter, all counters should appear (backward compat)."""
        agent = {"cycles": 5_000, "instructions": 10_000}
        ref = {"cycles": 10_000, "instructions": 8_000}
        result = format_perf_feedback(agent, ref, 1, 5)
        assert "cycles" in result
        assert "instructions" in result


class TestFormatNoCodeFound:
    def test_contains_turn_info(self) -> None:
        result = format_no_code_found(2, 5)
        assert "turn 2/5" in result

    def test_contains_code_tag_hint(self) -> None:
        result = format_no_code_found(1, 5)
        assert "<code" in result
