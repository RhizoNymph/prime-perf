"""Tests for perf stat CSV output parser.

All tests use hardcoded perf CSV strings. No system dependencies required.
"""

from __future__ import annotations

import pytest

from perf_optimize.exceptions import CounterNotFoundError
from perf_optimize.perf_parser import parse_csv_line, parse_perf_output
from perf_optimize.types import PerfCounters

# ── Realistic sample outputs matching actual perf stat format ────────────────

# With -r 5: value,unit,name,variance%,run_time,percentage[,metric_value,metric_unit]
REALISTIC_PERF_OUTPUT_WITH_R = """\
# started on Thu Mar 28 10:00:00 2026

1523456789,,cycles,0.45%,2000000000,100.00,,
3045678901,,instructions,0.23%,2000000000,100.00,2.00,insn per cycle
45678901,,cache-references,0.89%,2000000000,100.00,,
1234567,,cache-misses,1.23%,2000000000,100.00,2.71,of all cache refs
5678901,,L1-dcache-load-misses,0.67%,2000000000,100.00,,
234567,,LLC-load-misses,1.45%,2000000000,100.00,,
890123,,branch-misses,0.78%,2000000000,100.00,,
"""

# Without -r: value,unit,name,run_time,percentage[,metric_value,metric_unit]
REALISTIC_PERF_OUTPUT_NO_R = """\
# started on Thu Mar 28 10:00:00 2026

1523456789,,cycles,2000000000,100.00,,
3045678901,,instructions,2000000000,100.00,2.00,insn per cycle
45678901,,cache-references,2000000000,100.00,,
1234567,,cache-misses,2000000000,100.00,2.71,of all cache refs
5678901,,L1-dcache-load-misses,2000000000,100.00,,
234567,,LLC-load-misses,2000000000,100.00,,
890123,,branch-misses,2000000000,100.00,,
"""

# Real-world output from AMD Ryzen 7 with -r 5 (LLC-load-misses not supported)
REAL_AMD_OUTPUT = """\
6482066,,cycles,6.49%,2921452,74.00,,
5241051,,instructions,1.58%,2917043,74.00,0.81,insn per cycle
428288,,cache-references,3.55%,2921211,74.00,,
66993,,cache-misses,4.21%,3029615,77.00,15.64,of all cache refs
133393,,L1-dcache-load-misses,1.93%,3916188,100.00,,
<not supported>,,LLC-load-misses,0.00%,0,100.00,,
99442,,branch-misses,1.82%,3875431,98.00,,
"""


# ═══════════════════════════════════════════════════════════════════════════════
# parse_csv_line
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseCsvLineNormal:
    """Tests for successfully parsed lines."""

    def test_line_with_r_flag_variance(self) -> None:
        """With -r: value,unit,name,variance%,run_time,percentage."""
        line = "1234567,,cycles,0.45%,1000000,100.00,,"
        result = parse_csv_line(line)
        assert result is not None
        assert result.counter_value == 1234567
        assert result.unit == ""
        assert result.event_name == "cycles"
        assert result.variance == pytest.approx(0.45)
        assert result.run_time == 1000000
        assert result.percentage == 100.00

    def test_line_without_r_flag(self) -> None:
        """Without -r: value,unit,name,run_time,percentage."""
        line = "1234567,,cycles,1000000,100.00,,"
        result = parse_csv_line(line)
        assert result is not None
        assert result.counter_value == 1234567
        assert result.event_name == "cycles"
        assert result.run_time == 1000000
        assert result.variance is None

    def test_line_without_trailing_fields(self) -> None:
        line = "1234567,,cycles,1000000,100.00"
        result = parse_csv_line(line)
        assert result is not None
        assert result.counter_value == 1234567
        assert result.variance is None

    def test_line_with_unit(self) -> None:
        line = "1234567,msec,task-clock,1000000,100.00"
        result = parse_csv_line(line)
        assert result is not None
        assert result.counter_value == 1234567
        assert result.unit == "msec"
        assert result.event_name == "task-clock"

    def test_large_counter_value(self) -> None:
        line = "9999999999,,cycles,0.45%,2000000000,100.00,,"
        result = parse_csv_line(line)
        assert result is not None
        assert result.counter_value == 9999999999

    def test_zero_counter_value(self) -> None:
        line = "0,,cache-misses,1000000,100.00"
        result = parse_csv_line(line)
        assert result is not None
        assert result.counter_value == 0
        assert result.event_name == "cache-misses"

    def test_all_seven_counter_types_with_r(self) -> None:
        """Verify parse_csv_line handles all event names in -r format."""
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
            line = f"42,,{event},0.01%,1000,100.00,,"
            result = parse_csv_line(line)
            assert result is not None, f"Failed to parse event: {event}"
            assert result.event_name == event


class TestParseCsvLineSkipped:
    """Tests for lines that should return None."""

    def test_comment_line(self) -> None:
        assert parse_csv_line("# started on Thu Mar 28 2026") is None

    def test_comment_line_with_leading_space(self) -> None:
        assert parse_csv_line("  # started on Thu Mar 28 2026") is None

    def test_empty_line(self) -> None:
        assert parse_csv_line("") is None

    def test_whitespace_only_line(self) -> None:
        assert parse_csv_line("   ") is None

    def test_derived_metric_insn_per_cycle(self) -> None:
        line = "1.23,,insn per cycle,,,"
        assert parse_csv_line(line) is None

    def test_empty_event_name(self) -> None:
        line = "1234567,,,1000000,100.00"
        assert parse_csv_line(line) is None

    def test_empty_counter_value_and_event_name(self) -> None:
        assert parse_csv_line(",,,,") is None

    def test_empty_counter_value_with_event_name(self) -> None:
        assert parse_csv_line(",,cycles,1000000,100.00") is None

    def test_floating_point_counter_value(self) -> None:
        assert parse_csv_line("3.14,,GHz,1000000,100.00") is None

    def test_too_few_fields(self) -> None:
        assert parse_csv_line("1234,,cycles") is None

    def test_not_supported_returns_none(self) -> None:
        """<not supported> means hardware doesn't have this counter -- skip gracefully."""
        line = "<not supported>,,LLC-load-misses,0.00%,0,100.00,,"
        assert parse_csv_line(line) is None

    def test_not_supported_without_r(self) -> None:
        line = "<not supported>,,LLC-load-misses,0,100.00,,"
        assert parse_csv_line(line) is None


class TestParseCsvLineNotCounted:
    """Tests for <not counted> lines (PMU scheduling failure)."""

    def test_not_counted_returns_none(self) -> None:
        """<not counted> is a PMU scheduling failure -- skip gracefully."""
        line = "<not counted>,,cycles,0,0.00"
        assert parse_csv_line(line) is None

    def test_not_counted_with_unknown_event(self) -> None:
        line = "<not counted>,,,0,0.00"
        assert parse_csv_line(line) is None


# ═══════════════════════════════════════════════════════════════════════════════
# parse_perf_output
# ═══════════════════════════════════════════════════════════════════════════════


class TestParsePerfOutputValid:
    """Tests for successful full-output parsing."""

    def test_output_with_r_flag(self) -> None:
        result = parse_perf_output(REALISTIC_PERF_OUTPUT_WITH_R)
        assert isinstance(result, PerfCounters)
        assert result.cycles == 1523456789
        assert result.instructions == 3045678901
        assert result.cache_references == 45678901
        assert result.cache_misses == 1234567
        assert result.l1_dcache_load_misses == 5678901
        assert result.llc_load_misses == 234567
        assert result.branch_misses == 890123

    def test_output_without_r_flag(self) -> None:
        result = parse_perf_output(REALISTIC_PERF_OUTPUT_NO_R)
        assert result.cycles == 1523456789
        assert result.instructions == 3045678901

    def test_real_amd_output_with_unsupported_counter(self) -> None:
        """Real AMD output where LLC-load-misses is <not supported>."""
        result = parse_perf_output(REAL_AMD_OUTPUT)
        assert result.cycles == 6482066
        assert result.instructions == 5241051
        assert result.cache_references == 428288
        assert result.cache_misses == 66993
        assert result.l1_dcache_load_misses == 133393
        assert result.llc_load_misses == 0  # <not supported> -> 0
        assert result.branch_misses == 99442

    def test_ipc_derived_correctly(self) -> None:
        result = parse_perf_output(REALISTIC_PERF_OUTPUT_WITH_R)
        expected_ipc = 3045678901 / 1523456789
        assert abs(result.ipc - expected_ipc) < 1e-9

    def test_mixed_valid_comment_and_empty_lines(self) -> None:
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

    def test_unsupported_counter_defaults_to_zero(self) -> None:
        """If a non-essential counter is <not supported>, it defaults to 0."""
        text = (
            "1000,,cycles,500,100.00\n"
            "2000,,instructions,500,100.00\n"
            "300,,cache-references,500,100.00\n"
            "40,,cache-misses,500,100.00\n"
            "50,,L1-dcache-load-misses,500,100.00\n"
            "<not supported>,,LLC-load-misses,0,100.00\n"
            "70,,branch-misses,500,100.00\n"
        )
        result = parse_perf_output(text)
        assert result.llc_load_misses == 0
        assert result.cycles == 1000


class TestParsePerfOutputErrors:
    """Tests for error conditions in full-output parsing."""

    def test_missing_cycles_raises(self) -> None:
        """cycles is a mandatory counter."""
        text = (
            "2000,,instructions,500,100.00\n"
            "300,,cache-references,500,100.00\n"
        )
        with pytest.raises(CounterNotFoundError) as exc_info:
            parse_perf_output(text)
        assert exc_info.value.counter == "cycles"

    def test_missing_instructions_raises(self) -> None:
        """instructions is a mandatory counter."""
        text = "1000,,cycles,500,100.00\n"
        with pytest.raises(CounterNotFoundError) as exc_info:
            parse_perf_output(text)
        assert exc_info.value.counter == "instructions"

    def test_empty_output_raises(self) -> None:
        with pytest.raises(CounterNotFoundError):
            parse_perf_output("")

    def test_only_comments_raises(self) -> None:
        text = "# comment 1\n# comment 2\n"
        with pytest.raises(CounterNotFoundError):
            parse_perf_output(text)

    def test_not_counted_defaults_to_zero(self) -> None:
        """<not counted> counter defaults to 0 if non-mandatory."""
        text = (
            "1000,,cycles,500,100.00\n"
            "2000,,instructions,500,100.00\n"
            "300,,cache-references,500,100.00\n"
            "<not counted>,,cache-misses,0,0.00\n"
            "50,,L1-dcache-load-misses,500,100.00\n"
            "60,,LLC-load-misses,500,100.00\n"
            "70,,branch-misses,500,100.00\n"
        )
        result = parse_perf_output(text)
        assert result.cache_misses == 0
        assert result.cycles == 1000

    def test_not_counted_mandatory_counter_raises(self) -> None:
        """<not counted> on cycles/instructions still fails (they're mandatory)."""
        text = (
            "<not counted>,,cycles,0,0.00\n"
            "2000,,instructions,500,100.00\n"
        )
        with pytest.raises(CounterNotFoundError):
            parse_perf_output(text)


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
