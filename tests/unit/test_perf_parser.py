"""Tests for perf stat CSV output parser.

All tests use hardcoded perf CSV strings. No system dependencies required.
"""

from __future__ import annotations

import pytest

from perf_optimize.exceptions import CounterNotCountedError, CounterNotFoundError
from perf_optimize.perf_parser import parse_csv_line, parse_perf_output
from perf_optimize.types import PerfCounters

# ── Realistic sample output ──────────────────────────────────────────────────

REALISTIC_PERF_OUTPUT = """\
# started on Thu Mar 28 10:00:00 2026

1523456789,,cycles,2000000000,100.00,0.45
3045678901,,instructions,2000000000,100.00,0.23
45678901,,cache-references,2000000000,100.00,0.89
1234567,,cache-misses,2000000000,100.00,1.23
5678901,,L1-dcache-load-misses,2000000000,100.00,0.67
234567,,LLC-load-misses,2000000000,100.00,1.45
890123,,branch-misses,2000000000,100.00,0.78
"""

REALISTIC_PERF_OUTPUT_NO_VARIANCE = """\
# started on Thu Mar 28 10:00:00 2026

1523456789,,cycles,2000000000,100.00
3045678901,,instructions,2000000000,100.00
45678901,,cache-references,2000000000,100.00
1234567,,cache-misses,2000000000,100.00
5678901,,L1-dcache-load-misses,2000000000,100.00
234567,,LLC-load-misses,2000000000,100.00
890123,,branch-misses,2000000000,100.00
"""


# ═══════════════════════════════════════════════════════════════════════════════
# parse_csv_line
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseCsvLineNormal:
    """Tests for successfully parsed lines."""

    def test_normal_line_with_all_fields(self) -> None:
        line = "1234567,,cycles,1000000,100.00,0.12"
        result = parse_csv_line(line)
        assert result is not None
        assert result.counter_value == 1234567
        assert result.unit == ""
        assert result.event_name == "cycles"
        assert result.run_time == 1000000
        assert result.percentage == 100.00
        assert result.variance == 0.12

    def test_line_without_variance(self) -> None:
        line = "1234567,,cycles,1000000,100.00"
        result = parse_csv_line(line)
        assert result is not None
        assert result.counter_value == 1234567
        assert result.event_name == "cycles"
        assert result.variance is None

    def test_line_with_unit(self) -> None:
        line = "1234567,msec,task-clock,1000000,100.00"
        result = parse_csv_line(line)
        assert result is not None
        assert result.counter_value == 1234567
        assert result.unit == "msec"
        assert result.event_name == "task-clock"

    def test_large_counter_value(self) -> None:
        line = "9999999999,,cycles,2000000000,100.00,0.45"
        result = parse_csv_line(line)
        assert result is not None
        assert result.counter_value == 9999999999

    def test_zero_counter_value(self) -> None:
        line = "0,,cache-misses,1000000,100.00"
        result = parse_csv_line(line)
        assert result is not None
        assert result.counter_value == 0
        assert result.event_name == "cache-misses"

    def test_all_seven_counter_types(self) -> None:
        """Verify parse_csv_line handles all expected counter event names."""
        events = [
            "cycles",
            "instructions",
            "cache-references",
            "cache-misses",
            "L1-dcache-load-misses",
            "LLC-load-misses",
            "branch-misses",
        ]
        for event in events:
            line = f"42,,{event},1000,100.00,0.01"
            result = parse_csv_line(line)
            assert result is not None, f"Failed to parse event: {event}"
            assert result.event_name == event

    def test_empty_variance_field(self) -> None:
        """Variance field present but empty string."""
        line = "1234567,,cycles,1000000,100.00,"
        result = parse_csv_line(line)
        assert result is not None
        assert result.variance is None


class TestParseCsvLineSkipped:
    """Tests for lines that should return None."""

    def test_comment_line(self) -> None:
        line = "# started on Thu Mar 28 2026"
        assert parse_csv_line(line) is None

    def test_comment_line_with_leading_space(self) -> None:
        line = "  # started on Thu Mar 28 2026"
        assert parse_csv_line(line) is None

    def test_empty_line(self) -> None:
        assert parse_csv_line("") is None

    def test_whitespace_only_line(self) -> None:
        assert parse_csv_line("   ") is None

    def test_derived_metric_insn_per_cycle(self) -> None:
        """Derived metrics like 'insn per cycle' should be skipped."""
        line = "1.23,,insn per cycle,,,"
        assert parse_csv_line(line) is None

    def test_empty_event_name(self) -> None:
        """Lines with empty event name are derived metrics."""
        line = "1234567,,,1000000,100.00"
        assert parse_csv_line(line) is None

    def test_empty_counter_value_and_event_name(self) -> None:
        line = ",,,,"
        assert parse_csv_line(line) is None

    def test_empty_counter_value_with_event_name(self) -> None:
        """Empty counter_value with a valid event name is still skipped."""
        line = ",,cycles,1000000,100.00"
        assert parse_csv_line(line) is None

    def test_floating_point_counter_value(self) -> None:
        """Floating-point values in counter_value indicate derived metrics."""
        line = "3.14,,GHz,1000000,100.00"
        assert parse_csv_line(line) is None

    def test_too_few_fields(self) -> None:
        """Lines with fewer than 5 comma-separated fields are skipped."""
        line = "1234,,cycles"
        assert parse_csv_line(line) is None


class TestParseCsvLineErrors:
    """Tests for lines that should raise exceptions."""

    def test_not_counted_raises(self) -> None:
        line = "<not counted>,,cycles,0,0.00"
        with pytest.raises(CounterNotCountedError) as exc_info:
            parse_csv_line(line)
        assert exc_info.value.counter_name == "cycles"
        assert "<not counted>" in exc_info.value.raw_value

    def test_not_supported_raises(self) -> None:
        line = "<not supported>,,cycles,0,0.00"
        with pytest.raises(CounterNotCountedError) as exc_info:
            parse_csv_line(line)
        assert exc_info.value.counter_name == "cycles"
        assert "<not supported>" in exc_info.value.raw_value

    def test_not_counted_with_unknown_event(self) -> None:
        """<not counted> with empty event name still raises."""
        line = "<not counted>,,,0,0.00"
        with pytest.raises(CounterNotCountedError) as exc_info:
            parse_csv_line(line)
        assert exc_info.value.counter_name == "<unknown>"

    def test_not_supported_with_unknown_event(self) -> None:
        line = "<not supported>,,,0,0.00"
        with pytest.raises(CounterNotCountedError) as exc_info:
            parse_csv_line(line)
        assert exc_info.value.counter_name == "<unknown>"


# ═══════════════════════════════════════════════════════════════════════════════
# parse_perf_output
# ═══════════════════════════════════════════════════════════════════════════════


class TestParsePerfOutputValid:
    """Tests for successful full-output parsing."""

    def test_realistic_output_parses_all_counters(self) -> None:
        result = parse_perf_output(REALISTIC_PERF_OUTPUT)
        assert isinstance(result, PerfCounters)
        assert result.cycles == 1523456789
        assert result.instructions == 3045678901
        assert result.cache_references == 45678901
        assert result.cache_misses == 1234567
        assert result.l1_dcache_load_misses == 5678901
        assert result.llc_load_misses == 234567
        assert result.branch_misses == 890123

    def test_realistic_output_without_variance(self) -> None:
        result = parse_perf_output(REALISTIC_PERF_OUTPUT_NO_VARIANCE)
        assert result.cycles == 1523456789
        assert result.instructions == 3045678901

    def test_ipc_derived_correctly(self) -> None:
        result = parse_perf_output(REALISTIC_PERF_OUTPUT)
        expected_ipc = 3045678901 / 1523456789
        assert abs(result.ipc - expected_ipc) < 1e-9

    def test_mixed_valid_comment_and_empty_lines(self) -> None:
        """Parser should skip comments and blanks, collecting only data lines."""
        text = (
            "# this is a comment\n"
            "\n"
            "1000,,cycles,500,100.00\n"
            "# another comment\n"
            "2000,,instructions,500,100.00\n"
            "\n"
            "300,,cache-references,500,100.00\n"
            "40,,cache-misses,500,100.00\n"
            "50,,L1-dcache-load-misses,500,100.00\n"
            "60,,LLC-load-misses,500,100.00\n"
            "70,,branch-misses,500,100.00\n"
        )
        result = parse_perf_output(text)
        assert result.cycles == 1000
        assert result.instructions == 2000
        assert result.cache_references == 300
        assert result.cache_misses == 40
        assert result.l1_dcache_load_misses == 50
        assert result.llc_load_misses == 60
        assert result.branch_misses == 70

    def test_extra_non_counter_lines_are_ignored(self) -> None:
        """Lines for events not in PerfCounter enum are silently ignored."""
        text = (
            "1000,,cycles,500,100.00\n"
            "2000,,instructions,500,100.00\n"
            "300,,cache-references,500,100.00\n"
            "40,,cache-misses,500,100.00\n"
            "50,,L1-dcache-load-misses,500,100.00\n"
            "60,,LLC-load-misses,500,100.00\n"
            "70,,branch-misses,500,100.00\n"
            "12345,msec,task-clock,500,100.00\n"
            "99,,context-switches,500,100.00\n"
        )
        result = parse_perf_output(text)
        assert result.cycles == 1000
        assert result.branch_misses == 70

    def test_output_with_derived_metric_lines(self) -> None:
        """Derived metric lines like 'insn per cycle' should be skipped."""
        text = (
            "1000,,cycles,500,100.00\n"
            "2000,,instructions,500,100.00\n"
            "2.00,,insn per cycle,,,\n"
            "300,,cache-references,500,100.00\n"
            "40,,cache-misses,500,100.00\n"
            "50,,L1-dcache-load-misses,500,100.00\n"
            "60,,LLC-load-misses,500,100.00\n"
            "70,,branch-misses,500,100.00\n"
        )
        result = parse_perf_output(text)
        assert result.cycles == 1000
        assert result.instructions == 2000


class TestParsePerfOutputErrors:
    """Tests for error conditions in full-output parsing."""

    def test_missing_counter_raises(self) -> None:
        """If a required counter is absent, CounterNotFoundError is raised."""
        # Missing branch-misses
        text = (
            "1000,,cycles,500,100.00\n"
            "2000,,instructions,500,100.00\n"
            "300,,cache-references,500,100.00\n"
            "40,,cache-misses,500,100.00\n"
            "50,,L1-dcache-load-misses,500,100.00\n"
            "60,,LLC-load-misses,500,100.00\n"
        )
        with pytest.raises(CounterNotFoundError) as exc_info:
            parse_perf_output(text)
        assert exc_info.value.counter_name == "branch-misses"

    def test_empty_output_raises(self) -> None:
        with pytest.raises(CounterNotFoundError):
            parse_perf_output("")

    def test_only_comments_raises(self) -> None:
        text = "# comment 1\n# comment 2\n"
        with pytest.raises(CounterNotFoundError):
            parse_perf_output(text)

    def test_not_counted_propagates(self) -> None:
        """<not counted> in any line propagates as CounterNotCountedError."""
        text = (
            "1000,,cycles,500,100.00\n"
            "<not counted>,,instructions,0,0.00\n"
            "300,,cache-references,500,100.00\n"
        )
        with pytest.raises(CounterNotCountedError) as exc_info:
            parse_perf_output(text)
        assert exc_info.value.counter_name == "instructions"

    def test_not_supported_propagates(self) -> None:
        text = "<not supported>,,cycles,0,0.00\n2000,,instructions,500,100.00\n"
        with pytest.raises(CounterNotCountedError) as exc_info:
            parse_perf_output(text)
        assert exc_info.value.counter_name == "cycles"

    def test_missing_multiple_counters_reports_first(self) -> None:
        """When multiple counters are missing, the first (by enum order) is reported."""
        # Only cycles present
        text = "1000,,cycles,500,100.00\n"
        with pytest.raises(CounterNotFoundError) as exc_info:
            parse_perf_output(text)
        # instructions is the second counter in the enum, first missing
        assert exc_info.value.counter_name == "instructions"


class TestPerfCountersDataclass:
    """Tests for derived properties on PerfCounters."""

    def test_ipc_zero_cycles(self) -> None:
        counters = PerfCounters(
            cycles=0,
            instructions=1000,
            cache_references=0,
            cache_misses=0,
            l1_dcache_load_misses=0,
            llc_load_misses=0,
            branch_misses=0,
        )
        assert counters.ipc == 0.0

    def test_ipc_normal(self) -> None:
        counters = PerfCounters(
            cycles=1000,
            instructions=3000,
            cache_references=0,
            cache_misses=0,
            l1_dcache_load_misses=0,
            llc_load_misses=0,
            branch_misses=0,
        )
        assert counters.ipc == 3.0

    def test_frozen_dataclass(self) -> None:
        counters = PerfCounters(
            cycles=1000,
            instructions=2000,
            cache_references=300,
            cache_misses=40,
            l1_dcache_load_misses=50,
            llc_load_misses=60,
            branch_misses=70,
        )
        with pytest.raises(AttributeError):
            counters.cycles = 999  # type: ignore[misc]
