"""Tests for the problem bank loader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from perf_optimize.comparison import ComparisonMode
from perf_optimize.languages import Language
from perf_optimize.problems import (
    build_dataset_rows,
    load_problem,
    load_problem_with_reference,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def problem_dir(tmp_path: Path) -> Path:
    """Create a minimal problem directory for testing."""
    d = tmp_path / "test_problem"
    d.mkdir()

    # spec.md
    (d / "spec.md").write_text("# Test Problem\nDo the thing.\n")

    # comparison.json
    (d / "comparison.json").write_text(json.dumps({"mode": "exact"}))

    # tests
    tests = d / "tests"
    tests.mkdir()
    (tests / "input_0.bin").write_bytes(b"\x01\x02\x03")
    (tests / "expected_0.bin").write_bytes(b"\x04\x05\x06")
    (tests / "input_1.bin").write_bytes(b"\x07\x08")
    (tests / "expected_1.bin").write_bytes(b"\x09\x0a")

    # perf_input.bin
    (d / "perf_input.bin").write_bytes(b"\xff" * 100)

    # reference solutions
    ref = d / "reference"
    ref.mkdir()
    (ref / "solution.c").write_text("int main() { return 0; }\n")
    (ref / "solution.rs").write_text("fn main() {}\n")
    (ref / "solution.py").write_text("print('hello')\n")
    (ref / "solution.ts").write_text("console.log('hello')\n")

    # reference perf
    perf_dir = d / "reference_perf"
    perf_dir.mkdir()
    (perf_dir / "c_amd_zen.json").write_text(json.dumps({
        "cycles": 1000000.0,
        "instructions": 2000000.0,
        "cache_misses": 5000.0,
        "l1_dcache_load_misses": 3000.0,
        "branch_misses": 1000.0,
    }))

    return d


@pytest.fixture
def tolerance_problem_dir(tmp_path: Path) -> Path:
    """Create a problem directory with tolerance-based comparison."""
    d = tmp_path / "nbody_test"
    d.mkdir()
    (d / "spec.md").write_text("# N-Body\nSimulate gravity.\n")
    (d / "comparison.json").write_text(json.dumps({"mode": "tolerance", "tolerance": 1e-5}))
    tests = d / "tests"
    tests.mkdir()
    (tests / "input_0.bin").write_bytes(b"\x01")
    (tests / "expected_0.bin").write_bytes(b"\x02")
    (d / "perf_input.bin").write_bytes(b"\x00")
    ref = d / "reference"
    ref.mkdir()
    (ref / "solution.c").write_text("int main() {}\n")
    return d


class TestLoadProblem:
    def test_loads_spec_text(self, problem_dir: Path) -> None:
        spec = load_problem(problem_dir)
        assert "Test Problem" in spec.spec_text

    def test_loads_name_from_dirname(self, problem_dir: Path) -> None:
        spec = load_problem(problem_dir)
        assert spec.name == "test_problem"

    def test_loads_test_inputs(self, problem_dir: Path) -> None:
        spec = load_problem(problem_dir)
        assert len(spec.test_inputs) == 2
        assert spec.test_inputs[0] == b"\x01\x02\x03"

    def test_loads_expected_outputs(self, problem_dir: Path) -> None:
        spec = load_problem(problem_dir)
        assert len(spec.expected_outputs) == 2
        assert spec.expected_outputs[0] == b"\x04\x05\x06"

    def test_loads_perf_input(self, problem_dir: Path) -> None:
        spec = load_problem(problem_dir)
        assert len(spec.perf_input) == 100

    def test_exact_comparison_mode(self, problem_dir: Path) -> None:
        spec = load_problem(problem_dir)
        assert spec.comparison == ComparisonMode.EXACT
        assert spec.tolerance is None

    def test_tolerance_comparison_mode(self, tolerance_problem_dir: Path) -> None:
        spec = load_problem(tolerance_problem_dir)
        assert spec.comparison == ComparisonMode.TOLERANCE
        assert spec.tolerance == pytest.approx(1e-5)

    def test_missing_comparison_json_defaults_to_exact(self, tmp_path: Path) -> None:
        d = tmp_path / "no_comp"
        d.mkdir()
        (d / "spec.md").write_text("# Minimal\n")
        tests = d / "tests"
        tests.mkdir()
        (tests / "input_0.bin").write_bytes(b"\x01")
        (tests / "expected_0.bin").write_bytes(b"\x02")
        spec = load_problem(d)
        assert spec.comparison == ComparisonMode.EXACT

    def test_empty_tests_dir_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "no_tests"
        d.mkdir()
        (d / "spec.md").write_text("# Empty\n")
        (d / "tests").mkdir()
        with pytest.raises(FileNotFoundError, match="No test files found"):
            load_problem(d)

    def test_missing_expected_output_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "mismatched"
        d.mkdir()
        (d / "spec.md").write_text("# Mismatched\n")
        tests = d / "tests"
        tests.mkdir()
        (tests / "input_0.bin").write_bytes(b"\x01")
        # No expected_0.bin
        with pytest.raises(FileNotFoundError, match="Missing expected output file"):
            load_problem(d)

    def test_tolerance_mode_missing_tolerance_raises(self, tmp_path: Path) -> None:
        """comparison.json with mode=tolerance but no tolerance value should raise."""
        d = tmp_path / "bad_tolerance"
        d.mkdir()
        (d / "spec.md").write_text("# Bad\n")
        (d / "comparison.json").write_text(json.dumps({"mode": "tolerance"}))
        tests = d / "tests"
        tests.mkdir()
        (tests / "input_0.bin").write_bytes(b"\x01")
        (tests / "expected_0.bin").write_bytes(b"\x02")
        with pytest.raises(ValueError, match="tolerance mode requires"):
            load_problem(d)

    def test_non_contiguous_test_files_raises(self, tmp_path: Path) -> None:
        """Skipping test file indices (e.g. 0, 2 without 1) should raise."""
        d = tmp_path / "gap_tests"
        d.mkdir()
        (d / "spec.md").write_text("# Gap\n")
        tests = d / "tests"
        tests.mkdir()
        # Create contiguous pair 0
        (tests / "input_0.bin").write_bytes(b"\x01")
        (tests / "expected_0.bin").write_bytes(b"\x02")
        # Skip 1, create pair 2
        (tests / "input_2.bin").write_bytes(b"\x03")
        (tests / "expected_2.bin").write_bytes(b"\x04")
        with pytest.raises(FileNotFoundError, match="Non-contiguous"):
            load_problem(d)


class TestLoadProblemWithReference:
    def test_loads_c_reference(self, problem_dir: Path) -> None:
        p = load_problem_with_reference(problem_dir, Language.C, "amd_zen")
        assert p.language == Language.C
        assert "int main" in p.reference_source

    def test_loads_python_reference(self, problem_dir: Path) -> None:
        p = load_problem_with_reference(problem_dir, Language.PYTHON, "amd_zen")
        assert p.language == Language.PYTHON
        assert "print" in p.reference_source

    def test_loads_reference_perf(self, problem_dir: Path) -> None:
        p = load_problem_with_reference(problem_dir, Language.C, "amd_zen")
        assert p.reference_perf is not None
        assert p.reference_perf.cycles == 1000000.0
        assert p.reference_perf.instructions == 2000000.0
        assert p.reference_perf.llc_load_misses is None  # not in json

    def test_missing_perf_returns_none(self, problem_dir: Path) -> None:
        p = load_problem_with_reference(problem_dir, Language.RUST, "amd_zen")
        assert p.reference_perf is None


class TestBuildDatasetRows:
    def test_produces_rows(self, problem_dir: Path) -> None:
        # problem_dir is inside tmp_path; use its parent as the problems root
        rows = build_dataset_rows(problem_dir.parent, Language.C, "amd_zen")
        assert len(rows) == 1

    def test_row_has_question(self, problem_dir: Path) -> None:
        rows = build_dataset_rows(problem_dir.parent, Language.C, "amd_zen")
        assert "Test Problem" in rows[0]["question"]
        assert "int main" in rows[0]["question"]

    def test_row_has_info(self, problem_dir: Path) -> None:
        rows = build_dataset_rows(problem_dir.parent, Language.C, "amd_zen")
        info = rows[0]["info"]
        assert info["problem_name"] == "test_problem"
        assert info["language"] == "c"
        assert info["comparison"] == "exact"
        assert len(info["test_inputs"]) == 2

    def test_row_answer_is_empty_string(self, problem_dir: Path) -> None:
        rows = build_dataset_rows(problem_dir.parent, Language.C, "amd_zen")
        assert rows[0]["answer"] == ""

    def test_row_has_reference_perf(self, problem_dir: Path) -> None:
        rows = build_dataset_rows(problem_dir.parent, Language.C, "amd_zen")
        perf = rows[0]["info"]["reference_perf"]
        assert perf["cycles"] == 1000000.0
