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

import structlog

from perf_optimize.exceptions import CounterNotFoundError
from perf_optimize.types import PerfCounter, PerfCounters, PerfCSVLine

logger = structlog.get_logger(__name__)


def parse_csv_line(line: str) -> PerfCSVLine | None:
    """Parse a single line of ``perf stat -x ','`` CSV output.

    Returns ``None`` for comment lines (starting with ``#``), empty lines,
    derived-metric lines, ``<not supported>`` counters, and ``<not counted>``
    counters (PMU scheduling failure).
    """
    stripped = line.strip()

    # Skip empty lines and comments
    if not stripped or stripped.startswith("#"):
        return None

    fields = stripped.split(",")

    # Need at least 5 fields
    if len(fields) < 5:
        return None

    raw_value = fields[0]
    unit = fields[1]
    event_name = fields[2]

    # <not supported> means hardware doesn't have this counter -- skip gracefully
    # <not counted> means the PMU couldn't schedule it (multiplexing) -- also skip
    # Both are recoverable: the counter will be defaulted to 0 in parse_perf_output
    if "<not supported>" in raw_value or "<not counted>" in raw_value:
        return None

    # Skip derived metric lines (empty counter_value or empty event_name)
    if not raw_value or not event_name:
        return None

    # Skip textual derived metrics like "insn per cycle"
    # Real event names contain only alphanumeric chars, hyphens, and underscores
    if " " in event_name:
        return None

    # Parse counter value (must be integer for hardware counters)
    try:
        counter_value = int(raw_value)
    except ValueError:
        # Floating-point values indicate derived metrics; skip
        return None

    # Detect whether the -r (repeat) flag was used by checking if field[3]
    # ends with '%' (variance field). With -r the layout is:
    #   value,unit,name,variance%,run_time,percentage,...
    # Without -r:
    #   value,unit,name,run_time,percentage,...
    variance: float | None = None
    field3 = fields[3] if len(fields) > 3 else ""

    if field3.endswith("%"):
        # -r mode: field[3] is variance, field[4] is run_time, field[5] is percentage
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
        # No -r mode: field[3] is run_time, field[4] is percentage
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

    Counters that are ``<not supported>`` or ``<not counted>`` are set to 0.
    At least ``cycles`` and ``instructions`` must be present.

    Raises:
        CounterNotFoundError: If ``cycles`` or ``instructions`` is missing.
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

    # cycles and instructions are mandatory -- everything else defaults to 0
    # if not supported by the hardware
    for required in (PerfCounter.CYCLES, PerfCounter.INSTRUCTIONS):
        attr_name = _COUNTER_TO_ATTR[required.value]
        if attr_name not in collected:
            raise CounterNotFoundError(required.value)

    # Fill in unsupported counters with 0
    for counter in PerfCounter:
        attr_name = _COUNTER_TO_ATTR[counter.value]
        if attr_name not in collected:
            logger.info("counter_not_supported", counter=counter.value)
            collected[attr_name] = 0

    return PerfCounters(
        cycles=collected["cycles"],
        instructions=collected["instructions"],
        cache_references=collected["cache_references"],
        cache_misses=collected["cache_misses"],
        l1_dcache_load_misses=collected["l1_dcache_load_misses"],
        llc_load_misses=collected["llc_load_misses"],
        branch_misses=collected["branch_misses"],
    )
