"""Pure-function parser for perf stat CSV output.

This module has no side effects and performs no I/O. It takes string data
from perf stat's CSV output (produced via ``perf stat -x ','``) and returns
structured dataclasses.

Perf CSV field layout:
  Without -r: counter_value,unit,event_name,run_time,percentage[,metric_value,metric_unit]
  With -r:    counter_value,unit,event_name,variance%,run_time,percentage[,metric_value,metric_unit]

The variance field (when present) contains a percentage with a ``%`` suffix.
"""

from __future__ import annotations

from enum import StrEnum
import re
import warnings
from typing import TYPE_CHECKING

from perf_optimize.exceptions import (
    CounterNotCountedError,
    CounterNotFoundError,
    CounterNotSupportedError,
)
from perf_optimize.types import PerfCounters, PerfCSVLine

if TYPE_CHECKING:
    from perf_optimize.counters import HardwareProfile

_LOCALE_NUMBER_RE = re.compile(r"^\d{1,3}(,\d{3})+$")


class PerfLineStatus(StrEnum):
    """Sentinel values for non-data lines in perf CSV output."""

    NOT_SUPPORTED = "not_supported"
    NOT_COUNTED = "not_counted"


def parse_csv_line(line: str) -> PerfCSVLine | PerfLineStatus | None:
    """Parse a single line of ``perf stat -x ','`` CSV output.

    Returns:
        PerfCSVLine for successfully parsed counter lines.
        A string tag (``"not_supported"`` or ``"not_counted"``) for error lines,
        with the event name available from the line itself.
        ``None`` for comment lines, empty lines, and derived metrics.
    """
    stripped = line.strip()

    if not stripped or stripped.startswith("#"):
        return None

    fields = stripped.split(",")

    if len(fields) < 5:
        return None

    raw_value = fields[0]
    unit = fields[1]
    event_name = fields[2]

    if "<not supported>" in raw_value:
        return PerfLineStatus.NOT_SUPPORTED

    if "<not counted>" in raw_value:
        return PerfLineStatus.NOT_COUNTED

    # Skip derived metric lines (empty counter_value or empty event_name)
    if not raw_value or not event_name:
        return None

    # Skip textual derived metrics like "insn per cycle"
    if " " in event_name:
        return None

    # Detect locale-formatted numbers (e.g. "1,234,567") and warn
    if _LOCALE_NUMBER_RE.match(raw_value):
        warnings.warn(
            f"Counter value {raw_value!r} appears locale-formatted; "
            "ensure LC_ALL=C when running perf stat",
            stacklevel=2,
        )
        return None

    # Parse counter value (must be integer for hardware counters)
    try:
        counter_value = int(raw_value)
    except ValueError:
        return None

    # Detect -r mode: field[3] ends with '%' (variance field)
    variance: float | None = None
    field3 = fields[3] if len(fields) > 3 else ""

    if field3.endswith("%"):
        try:
            variance = float(field3.rstrip("%"))
        except ValueError:
            variance = None
        try:
            run_time = int(fields[4]) if len(fields) > 4 and fields[4] else 0
        except ValueError:
            run_time = 0
        try:
            percentage = float(fields[5]) if len(fields) > 5 and fields[5] else 0.0
        except ValueError:
            percentage = 0.0
    else:
        try:
            run_time = int(field3) if field3 else 0
        except ValueError:
            run_time = 0
        try:
            percentage = float(fields[4]) if len(fields) > 4 and fields[4] else 0.0
        except ValueError:
            percentage = 0.0

    return PerfCSVLine(
        counter_value=counter_value,
        unit=unit,
        event_name=event_name,
        run_time=run_time,
        percentage=percentage,
        variance=variance,
    )


def _extract_event_name(line: str) -> str:
    """Extract the event name (field[2]) from a CSV line."""
    fields = line.strip().split(",")
    if len(fields) >= 3 and fields[2]:
        return fields[2]
    return "<unknown>"


def parse_perf_output(csv_text: str, profile: HardwareProfile) -> PerfCounters:
    """Parse the full stderr output from ``perf stat -x ','``.

    Uses the ``HardwareProfile`` to map hardware event names to PerfCounters
    field names. Only fields mapped by the profile are populated; unmapped
    fields remain ``None``.

    Raises:
        CounterNotSupportedError: If a profiled event reports ``<not supported>``.
            This means the profile is wrong for this hardware.
        CounterNotCountedError: If a profiled event reports ``<not counted>``.
            This is a PMU scheduling failure.
        CounterNotFoundError: If ``cycles`` or ``instructions`` is missing
            from the output.
    """
    event_to_fields = profile.event_to_fields()
    expected_events = set(profile.perf_events())
    collected: dict[str, float] = {}

    for line in csv_text.splitlines():
        result = parse_csv_line(line)

        if result is None:
            continue

        if result is PerfLineStatus.NOT_SUPPORTED:
            event = _extract_event_name(line)
            if event in expected_events:
                raise CounterNotSupportedError(event, raw_output=csv_text)
            continue

        if result is PerfLineStatus.NOT_COUNTED:
            event = _extract_event_name(line)
            if event in expected_events:
                raise CounterNotCountedError(counter=event, raw_output=csv_text)
            continue

        # It's a PerfCSVLine
        if result.event_name in event_to_fields:
            for field_name in event_to_fields[result.event_name]:
                collected[field_name] = result.counter_value

    # Mandatory counters
    for required in ("cycles", "instructions"):
        if required not in collected:
            raise CounterNotFoundError(required, raw_output=csv_text)

    return PerfCounters(
        cycles=collected["cycles"],
        instructions=collected["instructions"],
        cache_references=collected.get("cache_references"),
        cache_misses=collected.get("cache_misses"),
        l1_dcache_load_misses=collected.get("l1_dcache_load_misses"),
        llc_load_misses=collected.get("llc_load_misses"),
        branch_misses=collected.get("branch_misses"),
    )
