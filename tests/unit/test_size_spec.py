"""Tests for SizeSpec, SizedPerfInput, SizedMeasurement, SizedExecutionResult.

Also covers _load_sizes_toml validation in problems.py.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from perf_optimize.types import (
    CompilationSuccess,
    PerfCounters,
    SizedExecutionResult,
    SizedMeasurement,
    SizedPerfInput,
    SizeSpec,
    TestReport,
    TestResult,
)

# ── SizeSpec ─────────────────────────────────────────────────────────────────


class TestSizeSpec:
    def test_basic_construction(self) -> None:
        spec = SizeSpec(label="small", n=256)
        assert spec.label == "small"
        assert spec.n == 256

    def test_frozen(self) -> None:
        spec = SizeSpec(label="small", n=256)
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.label = "medium"  # type: ignore[misc]

    def test_empty_label_raises(self) -> None:
        with pytest.raises(ValueError, match="label"):
            SizeSpec(label="", n=256)

    def test_label_with_invalid_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="label"):
            SizeSpec(label="some label", n=256)

    def test_label_with_dash_raises(self) -> None:
        with pytest.raises(ValueError, match="label"):
            SizeSpec(label="some-label", n=256)

    def test_label_with_underscore_ok(self) -> None:
        spec = SizeSpec(label="extra_large", n=2048)
        assert spec.label == "extra_large"

    def test_label_with_digits_ok(self) -> None:
        spec = SizeSpec(label="size4", n=512)
        assert spec.label == "size4"

    def test_zero_n_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be positive"):
            SizeSpec(label="small", n=0)

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be positive"):
            SizeSpec(label="small", n=-1)


# ── SizedPerfInput ───────────────────────────────────────────────────────────


class TestSizedPerfInput:
    def test_basic_construction(self) -> None:
        spec = SizeSpec(label="small", n=256)
        sip = SizedPerfInput(spec=spec, data=b"\x01\x02\x03")
        assert sip.spec.label == "small"
        assert sip.data == b"\x01\x02\x03"

    def test_frozen(self) -> None:
        spec = SizeSpec(label="small", n=256)
        sip = SizedPerfInput(spec=spec, data=b"\x00")
        with pytest.raises(dataclasses.FrozenInstanceError):
            sip.data = b"\x01"  # type: ignore[misc]

    def test_empty_data_raises(self) -> None:
        spec = SizeSpec(label="small", n=256)
        with pytest.raises(ValueError, match="data must be non-empty"):
            SizedPerfInput(spec=spec, data=b"")


# ── SizedMeasurement ────────────────────────────────────────────────────────


class TestSizedMeasurement:
    def test_succeeded_when_both_present(self) -> None:
        spec = SizeSpec(label="small", n=256)
        m = SizedMeasurement(
            spec=spec,
            perf_counters=PerfCounters(cycles=100.0, instructions=200.0),
            wall_clock_ms=1.5,
        )
        assert m.succeeded is True

    def test_succeeded_false_if_counters_missing(self) -> None:
        spec = SizeSpec(label="small", n=256)
        m = SizedMeasurement(spec=spec, perf_counters=None, wall_clock_ms=1.5)
        assert m.succeeded is False

    def test_succeeded_false_if_wall_clock_missing(self) -> None:
        spec = SizeSpec(label="small", n=256)
        pc = PerfCounters(cycles=100.0, instructions=200.0)
        m = SizedMeasurement(spec=spec, perf_counters=pc, wall_clock_ms=None)
        assert m.succeeded is False

    def test_frozen(self) -> None:
        spec = SizeSpec(label="small", n=256)
        m = SizedMeasurement(spec=spec, perf_counters=None, wall_clock_ms=None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.wall_clock_ms = 2.0  # type: ignore[misc]


# ── SizedExecutionResult ────────────────────────────────────────────────────


class TestSizedExecutionResult:
    def test_basic_construction(self) -> None:
        spec_s = SizeSpec(label="small", n=256)
        spec_l = SizeSpec(label="large", n=1024)
        ms = (
            SizedMeasurement(
                spec=spec_s,
                perf_counters=PerfCounters(cycles=100.0, instructions=200.0),
                wall_clock_ms=0.5,
            ),
            SizedMeasurement(
                spec=spec_l,
                perf_counters=PerfCounters(cycles=400.0, instructions=800.0),
                wall_clock_ms=2.0,
            ),
        )
        ser = SizedExecutionResult(
            compilation=CompilationSuccess(),
            test_report=TestReport(results=(TestResult(name="t0", passed=True),)),
            measurements=ms,
        )
        assert len(ser.measurements) == 2
        assert ser.measurements[0].spec.label == "small"

    def test_frozen(self) -> None:
        ser = SizedExecutionResult(
            compilation=CompilationSuccess(),
            test_report=None,
            measurements=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ser.measurements = ()  # type: ignore[misc]


# ── _load_sizes_toml ────────────────────────────────────────────────────────


@pytest.fixture
def problem_dir_with_sizes(tmp_path: Path) -> Path:
    d = tmp_path / "prob"
    d.mkdir()
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
    return d


class TestLoadSizesToml:
    def test_parses_in_ascending_order(self, problem_dir_with_sizes: Path) -> None:
        from perf_optimize.problems import _load_sizes_toml

        sizes = _load_sizes_toml(problem_dir_with_sizes)
        assert [s.label for s in sizes] == ["small", "medium", "large"]
        assert [s.n for s in sizes] == [256, 512, 1024]

    def test_rejects_duplicate_labels(self, tmp_path: Path) -> None:
        from perf_optimize.problems import _load_sizes_toml

        d = tmp_path / "prob"
        d.mkdir()
        (d / "sizes.toml").write_text(
            """
[[sizes]]
label = "small"
n = 256

[[sizes]]
label = "small"
n = 512
"""
        )
        with pytest.raises(ValueError, match="duplicate"):
            _load_sizes_toml(d)

    def test_rejects_non_monotone_n(self, tmp_path: Path) -> None:
        from perf_optimize.problems import _load_sizes_toml

        d = tmp_path / "prob"
        d.mkdir()
        (d / "sizes.toml").write_text(
            """
[[sizes]]
label = "small"
n = 512

[[sizes]]
label = "medium"
n = 256
"""
        )
        with pytest.raises(ValueError, match=r"monotone|ascending|increas"):
            _load_sizes_toml(d)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from perf_optimize.problems import _load_sizes_toml

        d = tmp_path / "no_sizes"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            _load_sizes_toml(d)

    def test_empty_sizes_raises(self, tmp_path: Path) -> None:
        from perf_optimize.problems import _load_sizes_toml

        d = tmp_path / "empty"
        d.mkdir()
        (d / "sizes.toml").write_text("")
        with pytest.raises(ValueError, match="at least one"):
            _load_sizes_toml(d)
