"""Tests for held-out data loading and info-dict shape in `perf_optimize.problems`.

Held-out test inputs and perf inputs MUST be present for every problem; loaders
raise structured errors if anything is missing so the migration cannot silently
omit held-out coverage for a problem.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from perf_optimize.languages import Language
from perf_optimize.problems import (
    _load_heldout_perf_input,
    _load_heldout_reference_perf,
    _load_heldout_test_files,
    build_dataset_rows,
    load_problem_with_reference,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def problem_dir(tmp_path: Path) -> Path:
    """Create a minimal problem directory with held-out data for testing."""
    d = tmp_path / "test_problem"
    d.mkdir()

    (d / "spec.md").write_text("# Test\nDo it.\n")
    (d / "comparison.json").write_text(json.dumps({"mode": "exact"}))

    tests = d / "tests"
    tests.mkdir()
    (tests / "input_0.bin").write_bytes(b"\x01")
    (tests / "expected_0.bin").write_bytes(b"\x02")

    # Sized perf inputs (required by the multi-size loader).
    (d / "sizes.toml").write_text(
        '[[sizes]]\nlabel = "large"\nn = 1024\n',
    )
    perf_inputs = d / "perf_inputs"
    perf_inputs.mkdir()
    (perf_inputs / "large.bin").write_bytes(b"\xff" * 50)

    ref = d / "reference"
    ref.mkdir()
    (ref / "solution.c").write_text("int main(){return 0;}\n")

    perf_dir = d / "reference_perf"
    perf_dir.mkdir()
    (perf_dir / "c_amd_zen_large.json").write_text(json.dumps({
        "cycles": 1_000_000.0,
        "instructions": 2_000_000.0,
        "wall_clock_ms": 5.0,
    }))

    # Held-out fixtures
    held_tests = d / "tests_heldout"
    held_tests.mkdir()
    (held_tests / "input_0.bin").write_bytes(b"\xaa\xbb")
    (held_tests / "expected_0.bin").write_bytes(b"\xcc\xdd")
    (held_tests / "input_1.bin").write_bytes(b"\xee")
    (held_tests / "expected_1.bin").write_bytes(b"\xff")

    held_perf = d / "perf_inputs_heldout"
    held_perf.mkdir()
    (held_perf / "large.bin").write_bytes(b"\x77" * 200)

    (perf_dir / "c_amd_zen_heldout.json").write_text(json.dumps({
        "cycles": 2_000_000.0,
        "instructions": 4_000_000.0,
        "wall_clock_ms": 12.5,
    }))

    return d


# ── _load_heldout_test_files ─────────────────────────────────────────────────


class TestLoadHeldoutTestFiles:
    def test_loads_inputs_and_expected(self, problem_dir: Path) -> None:
        inputs, expected = _load_heldout_test_files(problem_dir / "tests_heldout")
        assert inputs == (b"\xaa\xbb", b"\xee")
        assert expected == (b"\xcc\xdd", b"\xff")

    def test_returns_empty_tuples_when_dir_missing(self, tmp_path: Path) -> None:
        """An absent tests_heldout dir yields empty tuples (caller decides)."""
        d = tmp_path / "no_held"
        d.mkdir()
        inputs, expected = _load_heldout_test_files(d / "tests_heldout")
        assert inputs == ()
        assert expected == ()

    def test_missing_expected_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "broken"
        d.mkdir()
        td = d / "tests_heldout"
        td.mkdir()
        (td / "input_0.bin").write_bytes(b"\x01")
        # No expected_0.bin
        with pytest.raises(FileNotFoundError, match="Missing expected"):
            _load_heldout_test_files(td)


# ── _load_heldout_perf_input ─────────────────────────────────────────────────


class TestLoadHeldoutPerfInput:
    def test_loads_large_bin(self, problem_dir: Path) -> None:
        data = _load_heldout_perf_input(problem_dir)
        assert len(data) == 200

    def test_missing_raises(self, tmp_path: Path) -> None:
        """Mandatory migration: missing held-out perf input must raise."""
        d = tmp_path / "no_heldout_perf"
        d.mkdir()
        with pytest.raises(FileNotFoundError, match="perf_inputs_heldout/large.bin"):
            _load_heldout_perf_input(d)


# ── _load_heldout_reference_perf ─────────────────────────────────────────────


class TestLoadHeldoutReferencePerf:
    def test_loads_counters_and_wall_clock(self, problem_dir: Path) -> None:
        counters, wall = _load_heldout_reference_perf(problem_dir, Language.C, "amd_zen")
        assert counters is not None
        assert counters.cycles == 2_000_000.0
        assert wall == pytest.approx(12.5)

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        d = tmp_path / "no_perf"
        d.mkdir()
        counters, wall = _load_heldout_reference_perf(d, Language.C, "amd_zen")
        assert counters is None
        assert wall is None


# ── ProblemSpec held-out fields and info dict ────────────────────────────────


class TestProblemSpecHeldout:
    def test_problem_spec_has_heldout_fields(self, problem_dir: Path) -> None:
        p = load_problem_with_reference(problem_dir, Language.C, "amd_zen")
        assert p.spec.heldout_test_inputs == (b"\xaa\xbb", b"\xee")
        assert p.spec.heldout_expected_outputs == (b"\xcc\xdd", b"\xff")
        assert len(p.spec.heldout_perf_input) == 200


class TestBuildDatasetRowsHeldout:
    def test_info_dict_has_heldout_keys(self, problem_dir: Path) -> None:
        rows = build_dataset_rows(problem_dir.parent, Language.C, "amd_zen")
        info = rows[0]["info"]
        assert "heldout_test_inputs" in info
        assert "heldout_expected_outputs" in info
        assert "heldout_perf_input" in info
        assert "reference_heldout_perf" in info
        assert "reference_heldout_wall_clock_ms" in info

    def test_heldout_inputs_count_matches(self, problem_dir: Path) -> None:
        rows = build_dataset_rows(problem_dir.parent, Language.C, "amd_zen")
        info = rows[0]["info"]
        assert len(info["heldout_test_inputs"]) == 2
        assert len(info["heldout_expected_outputs"]) == 2

    def test_heldout_perf_is_base64(self, problem_dir: Path) -> None:
        import base64

        rows = build_dataset_rows(problem_dir.parent, Language.C, "amd_zen")
        info = rows[0]["info"]
        decoded = base64.b64decode(info["heldout_perf_input"])
        assert len(decoded) == 200

    def test_reference_heldout_perf_dict_shape(self, problem_dir: Path) -> None:
        rows = build_dataset_rows(problem_dir.parent, Language.C, "amd_zen")
        info = rows[0]["info"]
        ref = info["reference_heldout_perf"]
        assert ref is not None
        assert ref["cycles"] == 2_000_000.0
        assert info["reference_heldout_wall_clock_ms"] == pytest.approx(12.5)

    def test_in_dist_keys_unchanged(self, problem_dir: Path) -> None:
        """Sanity: held-out additions don't disturb the original keys."""
        rows = build_dataset_rows(problem_dir.parent, Language.C, "amd_zen")
        info = rows[0]["info"]
        assert "test_inputs" in info
        assert "expected_outputs" in info
        assert "perf_inputs" in info  # sized layout: list of {label, n, data_b64}
