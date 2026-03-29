"""Shared fixtures and markers for the perf-optimize test suite."""

import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "c_programs"

has_bwrap = shutil.which("bwrap") is not None
has_gcc = shutil.which("gcc") is not None
has_perf = shutil.which("perf") is not None


def _perf_event_paranoid() -> int | None:
    try:
        return int(Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


perf_paranoid = _perf_event_paranoid()
has_perf_access = perf_paranoid is not None and perf_paranoid <= 1

requires_bwrap = pytest.mark.skipif(not has_bwrap, reason="bwrap not installed")
requires_gcc = pytest.mark.skipif(not has_gcc, reason="gcc not installed")
requires_perf = pytest.mark.skipif(not has_perf, reason="perf not installed")
requires_perf_access = pytest.mark.skipif(
    not has_perf_access,
    reason=f"perf_event_paranoid={perf_paranoid} (needs <=1)",
)
requires_all_tools = pytest.mark.skipif(
    not (has_bwrap and has_gcc and has_perf and has_perf_access),
    reason="requires bwrap + gcc + perf with perf_event_paranoid<=1",
)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
