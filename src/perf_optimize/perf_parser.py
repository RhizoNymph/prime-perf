"""Pure-function parser for perf stat CSV output.

This module has no side effects and performs no I/O. It takes string data
from perf stat's CSV output (produced via ``perf stat -x ','``) and returns
structured dataclasses.
"""

from __future__ import annotations

from perf_optimize.exceptions import CounterNotCountedError, CounterNotFoundError
from perf_optimize.types import PerfCounter, PerfCounters, PerfCSVLine


def parse_csv_line(line: str) -> PerfCSVLine | None:
    """Parse a single line of ``perf stat -x ','`` CSV output.

    Returns ``None`` for comment lines (starting with ``#``), empty lines,
    and derived-metric lines (where event_name is empty or a textual description).

    Raises:
        CounterNotCountedError: If the counter value is ``<not counted>`` or
            ``<not supported>``.
    """
    stripped = line.strip()

    # Skip empty lines and comments
    if not stripped or stripped.startswith("#"):
        return None

    fields = stripped.split(",")

    # Need at least 5 fields: counter_value, unit, event_name, run_time, percentage
    if len(fields) < 5:
        return None

    raw_value = fields[0]
    unit = fields[1]
    event_name = fields[2]

    # Check for <not counted> / <not supported> before anything else
    if "<not counted>" in raw_value:
        raise CounterNotCountedError(
            counter_name=event_name or "<unknown>",
            raw_value=raw_value,
        )
    if "<not supported>" in raw_value:
        raise CounterNotCountedError(
            counter_name=event_name or "<unknown>",
            raw_value=raw_value,
        )

    # Skip derived metric lines (empty counter_value or empty event_name)
    if not raw_value or not event_name:
        return None

    # Skip textual derived metrics like "insn per cycle"
    # Real event names contain only alphanumeric chars, hyphens, and underscores
    if " " in event_name:
        return None

    # Parse numeric fields
    try:
        counter_value = int(raw_value)
    except ValueError:
        # Could be a floating-point derived metric; skip it
        return None

    try:
        run_time = int(fields[3]) if fields[3] else 0
    except ValueError:
        run_time = 0

    try:
        percentage = float(fields[4]) if fields[4] else 0.0
    except ValueError:
        percentage = 0.0

    # Variance field is optional (present with -r flag)
    variance: float | None = None
    if len(fields) >= 6 and fields[5]:
        try:
            variance = float(fields[5])
        except ValueError:
            variance = None

    return PerfCSVLine(
        counter_value=counter_value,
        unit=unit,
        event_name=event_name,
        run_time=run_time,
        percentage=percentage,
        variance=variance,
    )


# Mapping from PerfCounter enum value (the perf event name string)
# to the attribute name on PerfCounters.
_COUNTER_TO_ATTR: dict[str, str] = {
    PerfCounter.CYCLES.value: "cycles",
    PerfCounter.INSTRUCTIONS.value: "instructions",
    PerfCounter.CACHE_REFERENCES.value: "cache_references",
    PerfCounter.CACHE_MISSES.value: "cache_misses",
    PerfCounter.L1_DCACHE_LOAD_MISSES.value: "l1_dcache_load_misses",
    PerfCounter.LLC_LOAD_MISSES.value: "llc_load_misses",
    PerfCounter.BRANCH_MISSES.value: "branch_misses",
}

# Set of all valid event name strings for quick lookup
_VALID_EVENT_NAMES: set[str] = {c.value for c in PerfCounter}


def parse_perf_output(csv_text: str) -> PerfCounters:
    """Parse the full stderr output from ``perf stat -x ','``.

    Splits by newlines, parses each line via :func:`parse_csv_line`,
    collects results, and assembles a :class:`PerfCounters` dataclass.

    Raises:
        CounterNotFoundError: If any of the seven required counters is
            missing from the output.
        CounterNotCountedError: If any counter reports ``<not counted>``
            or ``<not supported>`` (propagated from :func:`parse_csv_line`).
    """
    collected: dict[str, int] = {}

    for line in csv_text.splitlines():
        parsed = parse_csv_line(line)
        if parsed is None:
            continue

        # Only collect lines whose event_name matches a known PerfCounter
        if parsed.event_name in _VALID_EVENT_NAMES:
            attr_name = _COUNTER_TO_ATTR[parsed.event_name]
            collected[attr_name] = parsed.counter_value

    # Verify all required counters are present
    for counter in PerfCounter:
        attr_name = _COUNTER_TO_ATTR[counter.value]
        if attr_name not in collected:
            raise CounterNotFoundError(counter.value)

    return PerfCounters(
        cycles=collected["cycles"],
        instructions=collected["instructions"],
        cache_references=collected["cache_references"],
        cache_misses=collected["cache_misses"],
        l1_dcache_load_misses=collected["l1_dcache_load_misses"],
        llc_load_misses=collected["llc_load_misses"],
        branch_misses=collected["branch_misses"],
    )
