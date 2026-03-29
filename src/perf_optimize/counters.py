"""Hardware counter profiles for different CPU architectures.

Maps logical PerfCounters fields to architecture-specific perf event strings.
Auto-detects the CPU vendor to select the correct profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .exceptions import PrerequisiteError


@dataclass(frozen=True)
class CounterMapping:
    """Maps a logical PerfCounters field to a hardware perf event string.

    Args:
        field: PerfCounters dataclass field name (e.g. "cache_misses").
        perf_event: Hardware event string for perf stat (e.g. "cache-misses").
    """

    field: str
    perf_event: str


@dataclass(frozen=True)
class HardwareProfile:
    """Counter configuration for a specific CPU architecture.

    Defines which perf events to request and how to map them back to
    PerfCounters fields. Only events in the profile are requested from
    perf stat -- if perf reports ``<not supported>`` for any of them,
    the profile is wrong for this hardware.

    Args:
        name: Human-readable profile name (e.g. "amd_zen4").
        vendor: CPU vendor string from /proc/cpuinfo (e.g. "AuthenticAMD").
        counters: Mapping from logical fields to hardware events.
    """

    name: str
    vendor: str
    counters: tuple[CounterMapping, ...]

    def perf_events(self) -> list[str]:
        """Deduplicated hardware event strings for ``perf stat -e``.

        Multiple fields may map to the same hardware event (e.g. on AMD,
        cache_misses and llc_load_misses could both map to "cache-misses").
        This returns each unique event only once.
        """
        seen: set[str] = set()
        events: list[str] = []
        for mapping in self.counters:
            if mapping.perf_event not in seen:
                seen.add(mapping.perf_event)
                events.append(mapping.perf_event)
        return events

    def event_to_fields(self) -> dict[str, list[str]]:
        """Map hardware event name to the PerfCounters field(s) it populates.

        A single hardware event may populate multiple fields.
        """
        result: dict[str, list[str]] = {}
        for mapping in self.counters:
            result.setdefault(mapping.perf_event, []).append(mapping.field)
        return result

    def mapped_fields(self) -> frozenset[str]:
        """Set of PerfCounters field names that this profile populates."""
        return frozenset(m.field for m in self.counters)


# ── Built-in profiles ────────────────────────────────────────────────────────

AMD_ZEN = HardwareProfile(
    name="amd_zen",
    vendor="AuthenticAMD",
    counters=(
        CounterMapping("cycles", "cycles"),
        CounterMapping("instructions", "instructions"),
        CounterMapping("cache_misses", "cache-misses"),
        CounterMapping("l1_dcache_load_misses", "L1-dcache-load-misses"),
        CounterMapping("branch_misses", "branch-misses"),
        # 5 events: fits AMD Zen4's 6 general PMU counters without multiplexing.
        # cache-references omitted to avoid PMU contention that causes <not counted>.
        # LLC-load-misses not available on AMD; cache-misses is the L3 miss counter
        # but we leave llc_load_misses unmapped (None) for honesty.
    ),
)

INTEL_CORE = HardwareProfile(
    name="intel_core",
    vendor="GenuineIntel",
    counters=(
        CounterMapping("cycles", "cycles"),
        CounterMapping("instructions", "instructions"),
        CounterMapping("cache_references", "cache-references"),
        CounterMapping("cache_misses", "cache-misses"),
        CounterMapping("l1_dcache_load_misses", "L1-dcache-load-misses"),
        CounterMapping("llc_load_misses", "LLC-load-misses"),
        CounterMapping("branch_misses", "branch-misses"),
    ),
)

_PROFILES_BY_VENDOR: dict[str, HardwareProfile] = {
    AMD_ZEN.vendor: AMD_ZEN,
    INTEL_CORE.vendor: INTEL_CORE,
}


class UnknownCPUVendorError(PrerequisiteError):
    """CPU vendor not recognized -- no hardware profile available."""

    def __init__(self, vendor: str) -> None:
        self.vendor = vendor
        known = ", ".join(_PROFILES_BY_VENDOR.keys())
        super().__init__(
            f"Unknown CPU vendor '{vendor}'. Known vendors: {known}. "
            "You may need to add a HardwareProfile for this CPU."
        )


def _read_cpu_vendor() -> str:
    """Read the CPU vendor string from /proc/cpuinfo."""
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        raise PrerequisiteError("/proc/cpuinfo not found")
    for line in cpuinfo.read_text().splitlines():
        if line.startswith("vendor_id"):
            return line.split(":", 1)[1].strip()
    raise PrerequisiteError("vendor_id not found in /proc/cpuinfo")


def detect_profile() -> HardwareProfile:
    """Auto-detect the hardware profile from the CPU vendor.

    Raises:
        UnknownCPUVendorError: If the CPU vendor is not recognized.
        PrerequisiteError: If /proc/cpuinfo is unreadable.
    """
    vendor = _read_cpu_vendor()
    profile = _PROFILES_BY_VENDOR.get(vendor)
    if profile is None:
        raise UnknownCPUVendorError(vendor)
    return profile
