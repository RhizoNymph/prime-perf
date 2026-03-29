# Hardware Profiles Feature

## Scope

**In scope:**
- Mapping logical PerfCounters fields to architecture-specific perf event strings
- Built-in profiles for AMD Zen and Intel Core
- Auto-detection of CPU vendor from /proc/cpuinfo
- Ensuring event count fits within PMU limits (no multiplexing)

**Not in scope:**
- ARM or non-x86 architectures
- Raw/uncore PMU events (only generic perf events)
- Per-microarchitecture tuning (Zen3 vs Zen4 vs Zen5)

## Design Rationale

Different CPU architectures expose different hardware performance counters:
- **AMD Zen4**: No `LLC-load-misses`; `cache-misses` is the L3 miss counter. Only 6 general PMU counters — requesting 6+ events causes multiplexing and `<not counted>` errors.
- **Intel Core**: Has `LLC-load-misses` as a separate event. More PMU counters available.

Rather than defaulting missing counters to 0 (which corrupts RL reward signals), each
PerfCounters field is `float | None`. `None` means "not measured on this hardware" —
distinct from 0.0 meaning "measured, zero events occurred."

The Phase 2 reward function will use profile-aware weights to handle missing counters.

## Counter Mappings

### AMD Zen (5 events)

| Field | Perf Event | Notes |
|-------|-----------|-------|
| cycles | cycles | |
| instructions | instructions | |
| cache_misses | cache-misses | = L3 misses on AMD |
| l1_dcache_load_misses | L1-dcache-load-misses | |
| branch_misses | branch-misses | |
| cache_references | — | Omitted to avoid PMU contention |
| llc_load_misses | — | Not available on AMD |

### Intel Core (7 events)

| Field | Perf Event |
|-------|-----------|
| cycles | cycles |
| instructions | instructions |
| cache_references | cache-references |
| cache_misses | cache-misses |
| l1_dcache_load_misses | L1-dcache-load-misses |
| llc_load_misses | LLC-load-misses |
| branch_misses | branch-misses |

## Files

| File | Role | Key exports |
|------|------|-------------|
| `src/perf_optimize/counters.py` | Profile definitions | `HardwareProfile`, `CounterMapping`, `AMD_ZEN`, `INTEL_CORE`, `detect_profile()` |
| `src/perf_optimize/types.py` | `PerfCounters` with `float \| None` fields | `PerfCounters`, `PERF_COUNTER_FIELDS` |
| `src/perf_optimize/config.py` | Stores active profile | `SandboxConfig.hardware_profile` |

## Invariants and Constraints

1. **No multiplexing**: Each profile's event count must fit within the CPU's general PMU counter limit. AMD Zen: max 5 events. Intel Core: max 7.
2. **Mandatory counters**: `cycles` and `instructions` must be in every profile and present in perf output.
3. **None means unmeasured**: A `None` field in `PerfCounters` means the hardware doesn't provide this counter. It must NOT be treated as 0 in any computation.
4. **Profile errors are loud**: `<not supported>` from perf for a profiled event raises `CounterNotSupportedError` — the profile is wrong for this CPU. `<not counted>` raises `CounterNotCountedError` — PMU scheduling failure.
