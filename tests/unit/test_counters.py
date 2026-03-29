"""Tests for the hardware counter profile system.

Covers built-in profiles (AMD_ZEN, INTEL_CORE), event deduplication,
field mapping, and auto-detection.
"""

from __future__ import annotations

from perf_optimize.counters import (
    AMD_ZEN,
    INTEL_CORE,
    CounterMapping,
    HardwareProfile,
    detect_profile,
)


class TestAMDZenProfile:
    def test_perf_events_count(self) -> None:
        """AMD_ZEN has 5 unique perf events (no LLC-load-misses, no cache-references)."""
        events = AMD_ZEN.perf_events()
        assert len(events) == 5
        assert len(set(events)) == 5

    def test_perf_events_no_llc(self) -> None:
        assert "LLC-load-misses" not in AMD_ZEN.perf_events()

    def test_no_cache_references(self) -> None:
        """cache-references omitted to avoid PMU contention on Zen4."""
        assert "cache-references" not in AMD_ZEN.perf_events()

    def test_event_to_fields_mapping(self) -> None:
        mapping = AMD_ZEN.event_to_fields()
        assert "cycles" in mapping
        assert "instructions" in mapping
        assert "cache-misses" in mapping
        assert "L1-dcache-load-misses" in mapping
        assert "branch-misses" in mapping
        assert "LLC-load-misses" not in mapping
        assert "cache-references" not in mapping

    def test_mapped_fields_excludes_llc_and_cache_refs(self) -> None:
        fields = AMD_ZEN.mapped_fields()
        assert "llc_load_misses" not in fields
        assert "cache_references" not in fields

    def test_mapped_fields_includes_basics(self) -> None:
        fields = AMD_ZEN.mapped_fields()
        assert "cycles" in fields
        assert "instructions" in fields
        assert "cache_misses" in fields
        assert "l1_dcache_load_misses" in fields
        assert "branch_misses" in fields


class TestIntelCoreProfile:
    def test_perf_events_count(self) -> None:
        """INTEL_CORE has 7 unique perf events (includes LLC-load-misses)."""
        events = INTEL_CORE.perf_events()
        assert len(events) == 7
        assert len(set(events)) == 7

    def test_perf_events_includes_llc(self) -> None:
        assert "LLC-load-misses" in INTEL_CORE.perf_events()

    def test_mapped_fields_includes_llc(self) -> None:
        fields = INTEL_CORE.mapped_fields()
        assert "llc_load_misses" in fields

    def test_mapped_fields_complete(self) -> None:
        fields = INTEL_CORE.mapped_fields()
        expected = {
            "cycles",
            "instructions",
            "cache_references",
            "cache_misses",
            "l1_dcache_load_misses",
            "llc_load_misses",
            "branch_misses",
        }
        assert fields == expected


class TestDetectProfile:
    def test_returns_valid_profile(self) -> None:
        """detect_profile() returns a valid HardwareProfile on this system."""
        profile = detect_profile()
        assert isinstance(profile, HardwareProfile)
        assert profile.name
        assert profile.vendor
        assert len(profile.counters) > 0

    def test_detected_profile_has_mandatory_events(self) -> None:
        profile = detect_profile()
        events = profile.perf_events()
        assert "cycles" in events
        assert "instructions" in events


class TestDeduplication:
    def test_duplicate_events_deduplicated(self) -> None:
        """If two fields map to the same event, perf_events() lists it once."""
        profile = HardwareProfile(
            name="test_dedup",
            vendor="TestVendor",
            counters=(
                CounterMapping("cycles", "cycles"),
                CounterMapping("instructions", "instructions"),
                CounterMapping("cache_misses", "cache-misses"),
                CounterMapping("llc_load_misses", "cache-misses"),  # same event
            ),
        )
        events = profile.perf_events()
        assert events.count("cache-misses") == 1
        assert len(events) == 3  # cycles, instructions, cache-misses

    def test_duplicate_events_both_fields_mapped(self) -> None:
        """Both fields appear in event_to_fields for the shared event."""
        profile = HardwareProfile(
            name="test_dedup",
            vendor="TestVendor",
            counters=(
                CounterMapping("cycles", "cycles"),
                CounterMapping("instructions", "instructions"),
                CounterMapping("cache_misses", "cache-misses"),
                CounterMapping("llc_load_misses", "cache-misses"),
            ),
        )
        mapping = profile.event_to_fields()
        assert set(mapping["cache-misses"]) == {"cache_misses", "llc_load_misses"}

    def test_duplicate_events_both_fields_in_mapped_fields(self) -> None:
        """Both fields appear in mapped_fields even though the event is shared."""
        profile = HardwareProfile(
            name="test_dedup",
            vendor="TestVendor",
            counters=(
                CounterMapping("cycles", "cycles"),
                CounterMapping("instructions", "instructions"),
                CounterMapping("cache_misses", "cache-misses"),
                CounterMapping("llc_load_misses", "cache-misses"),
            ),
        )
        fields = profile.mapped_fields()
        assert "cache_misses" in fields
        assert "llc_load_misses" in fields
