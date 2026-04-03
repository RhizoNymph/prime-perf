"""Shared fixtures and markers for the perf-optimize test suite."""

import shutil
from pathlib import Path

import pytest

FIXTURES_ROOT = Path(__file__).parent.parent / "fixtures"
# Keep FIXTURES_DIR for backward compatibility (C-specific)
FIXTURES_DIR = FIXTURES_ROOT / "c_programs"

has_bwrap = shutil.which("bwrap") is not None
has_gcc = shutil.which("gcc") is not None
has_perf = shutil.which("perf") is not None
has_rustc = shutil.which("rustc") is not None
has_python3 = shutil.which("python3") is not None
has_node = shutil.which("node") is not None


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
requires_rustc = pytest.mark.skipif(not has_rustc, reason="rustc not installed")
requires_python3 = pytest.mark.skipif(not has_python3, reason="python3 not installed")
requires_node = pytest.mark.skipif(not has_node, reason="node not installed")
requires_all_tools = pytest.mark.skipif(
    not (has_bwrap and has_gcc and has_perf and has_perf_access),
    reason="requires bwrap + gcc + perf with perf_event_paranoid<=1",
)


@pytest.fixture
def fixtures_root() -> Path:
    return FIXTURES_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
