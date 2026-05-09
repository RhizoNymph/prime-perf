"""Tests for the sized perf-input problem layout."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from perf_optimize.languages import Language
from perf_optimize.problems import (
    _load_reference_perf,
    build_dataset_rows,
    load_problem,
    load_problem_with_reference,
)


@pytest.fixture
def sized_problem_dir(tmp_path: Path) -> Path:
    """Build a problem directory using the sized perf_inputs layout."""
    d = tmp_path / "sized_problem"
    d.mkdir()
    (d / "spec.md").write_text("# Sized Problem\nDo it.\n")
    (d / "comparison.json").write_text(json.dumps({"mode": "exact"}))

    tests = d / "tests"
    tests.mkdir()
    (tests / "input_0.bin").write_bytes(b"\x01")
    (tests / "expected_0.bin").write_bytes(b"\x02")

    # sizes.toml
    (d / "sizes.toml").write_text(
        """
[[sizes]]
label = "small"
n = 256

[[sizes]]
label = "medium"
n = 512

[[sizes]]
label = "large"
n = 1024
"""
    )

    # perf_inputs/<label>.bin
    pi = d / "perf_inputs"
    pi.mkdir()
    (pi / "small.bin").write_bytes(b"S" * 10)
    (pi / "medium.bin").write_bytes(b"M" * 20)
    (pi / "large.bin").write_bytes(b"L" * 40)

    # references
    ref = d / "reference"
    ref.mkdir()
    (ref / "solution.c").write_text("int main(){}\n")

    # reference perf per size
    rp = d / "reference_perf"
    rp.mkdir()
    (rp / "c_amd_zen_small.json").write_text(
        json.dumps({"cycles": 1000.0, "instructions": 2000.0, "wall_clock_ms": 1.5})
    )
    (rp / "c_amd_zen_medium.json").write_text(
        json.dumps({"cycles": 4000.0, "instructions": 8000.0, "wall_clock_ms": 6.0})
    )
    (rp / "c_amd_zen_large.json").write_text(
        json.dumps({"cycles": 16000.0, "instructions": 32000.0, "wall_clock_ms": 24.0})
    )
    return d


class TestLoadProblemSized:
    def test_loads_perf_inputs_in_order(self, sized_problem_dir: Path) -> None:
        spec = load_problem(sized_problem_dir)
        labels = [p.spec.label for p in spec.perf_inputs]
        assert labels == ["small", "medium", "large"]

    def test_perf_input_data_loaded(self, sized_problem_dir: Path) -> None:
        spec = load_problem(sized_problem_dir)
        small = next(p for p in spec.perf_inputs if p.spec.label == "small")
        assert small.data == b"S" * 10
        large = next(p for p in spec.perf_inputs if p.spec.label == "large")
        assert large.data == b"L" * 40


class TestLoadReferencePerf:
    def test_returns_per_size_measurements(self, sized_problem_dir: Path) -> None:
        from perf_optimize.types import SizeSpec

        sizes = (
            SizeSpec(label="small", n=256),
            SizeSpec(label="medium", n=512),
            SizeSpec(label="large", n=1024),
        )
        ms = _load_reference_perf(sized_problem_dir, Language.C, "amd_zen", sizes)
        assert len(ms) == 3
        labels = [m.spec.label for m in ms]
        assert labels == ["small", "medium", "large"]
        for m in ms:
            assert m.perf_counters is not None
            assert m.wall_clock_ms is not None

    def test_missing_size_yields_empty_measurement(
        self, sized_problem_dir: Path
    ) -> None:
        from perf_optimize.types import SizeSpec

        # Delete medium ref
        (sized_problem_dir / "reference_perf" / "c_amd_zen_medium.json").unlink()
        sizes = (
            SizeSpec(label="small", n=256),
            SizeSpec(label="medium", n=512),
            SizeSpec(label="large", n=1024),
        )
        ms = _load_reference_perf(sized_problem_dir, Language.C, "amd_zen", sizes)
        by_label = {m.spec.label: m for m in ms}
        assert by_label["medium"].perf_counters is None
        assert by_label["medium"].wall_clock_ms is None
        assert by_label["small"].perf_counters is not None
        assert by_label["large"].perf_counters is not None


class TestBuildDatasetRowsSized:
    def test_info_contains_perf_inputs_list(self, sized_problem_dir: Path) -> None:
        rows = build_dataset_rows(sized_problem_dir.parent, Language.C, "amd_zen")
        assert len(rows) == 1
        info = rows[0]["info"]
        assert "perf_inputs" in info
        assert isinstance(info["perf_inputs"], list)
        assert len(info["perf_inputs"]) == 3
        # Each entry has label, n, data_b64
        for entry in info["perf_inputs"]:
            assert set(entry.keys()) >= {"label", "n", "data_b64"}

    def test_info_drops_singular_perf_input_key(self, sized_problem_dir: Path) -> None:
        rows = build_dataset_rows(sized_problem_dir.parent, Language.C, "amd_zen")
        info = rows[0]["info"]
        assert "perf_input" not in info

    def test_info_contains_reference_perf_by_size(
        self, sized_problem_dir: Path
    ) -> None:
        rows = build_dataset_rows(sized_problem_dir.parent, Language.C, "amd_zen")
        info = rows[0]["info"]
        assert "reference_perf_by_size" in info
        rp = info["reference_perf_by_size"]
        assert set(rp.keys()) == {"small", "medium", "large"}
        assert rp["small"]["cycles"] == 1000.0

    def test_info_contains_reference_wall_clock_by_size(
        self, sized_problem_dir: Path
    ) -> None:
        rows = build_dataset_rows(sized_problem_dir.parent, Language.C, "amd_zen")
        info = rows[0]["info"]
        assert "reference_wall_clock_ms_by_size" in info
        rw = info["reference_wall_clock_ms_by_size"]
        assert rw["small"] == 1.5
        assert rw["medium"] == 6.0
        assert rw["large"] == 24.0

    def test_load_problem_with_reference_per_size(self, sized_problem_dir: Path) -> None:
        prob = load_problem_with_reference(sized_problem_dir, Language.C, "amd_zen")
        assert prob.reference_perf is not None
        # tuple of measurements
        assert len(prob.reference_perf) == 3
