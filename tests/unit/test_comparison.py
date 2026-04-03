"""Tests for output comparison utilities."""

from __future__ import annotations

import struct

from perf_optimize.comparison import ComparisonMode, compare_outputs


class TestExactComparison:
    def test_identical_bytes(self) -> None:
        assert compare_outputs(b"hello", b"hello") is None

    def test_empty_bytes(self) -> None:
        assert compare_outputs(b"", b"") is None

    def test_different_length(self) -> None:
        result = compare_outputs(b"abc", b"ab")
        assert result is not None
        assert "size mismatch" in result.lower()

    def test_different_content(self) -> None:
        result = compare_outputs(b"abc", b"abd")
        assert result is not None
        assert "byte 2" in result

    def test_first_byte_differs(self) -> None:
        result = compare_outputs(b"\x00", b"\x01")
        assert result is not None
        assert "byte 0" in result


class TestToleranceComparison:
    def _pack_floats(self, *values: float) -> bytes:
        return struct.pack(f"<{len(values)}f", *values)

    def test_identical_floats(self) -> None:
        data = self._pack_floats(1.0, 2.0, 3.0)
        assert compare_outputs(data, data, ComparisonMode.TOLERANCE, 1e-5) is None

    def test_within_tolerance(self) -> None:
        actual = self._pack_floats(1.0, 2.0000001, 3.0)
        expected = self._pack_floats(1.0, 2.0, 3.0)
        assert compare_outputs(actual, expected, ComparisonMode.TOLERANCE, 1e-5) is None

    def test_exceeds_tolerance(self) -> None:
        actual = self._pack_floats(1.0, 2.1, 3.0)
        expected = self._pack_floats(1.0, 2.0, 3.0)
        result = compare_outputs(actual, expected, ComparisonMode.TOLERANCE, 1e-5)
        assert result is not None
        assert "Float[1]" in result

    def test_both_nan_is_ok(self) -> None:
        nan = float("nan")
        actual = self._pack_floats(nan)
        expected = self._pack_floats(nan)
        assert compare_outputs(actual, expected, ComparisonMode.TOLERANCE, 1e-5) is None

    def test_one_nan_is_mismatch(self) -> None:
        nan = float("nan")
        actual = self._pack_floats(nan)
        expected = self._pack_floats(1.0)
        result = compare_outputs(actual, expected, ComparisonMode.TOLERANCE, 1e-5)
        assert result is not None
        assert "NaN" in result

    def test_zero_expected(self) -> None:
        actual = self._pack_floats(1e-7)
        expected = self._pack_floats(0.0)
        assert compare_outputs(actual, expected, ComparisonMode.TOLERANCE, 1e-5) is None

    def test_zero_expected_exceeds(self) -> None:
        actual = self._pack_floats(1.0)
        expected = self._pack_floats(0.0)
        result = compare_outputs(actual, expected, ComparisonMode.TOLERANCE, 1e-5)
        assert result is not None

    def test_different_length(self) -> None:
        actual = self._pack_floats(1.0)
        expected = self._pack_floats(1.0, 2.0)
        result = compare_outputs(actual, expected, ComparisonMode.TOLERANCE, 1e-5)
        assert result is not None
        assert "size mismatch" in result.lower()

    def test_not_multiple_of_4(self) -> None:
        result = compare_outputs(b"abc", b"abc", ComparisonMode.TOLERANCE, 1e-5)
        assert result is not None
        assert "multiple of 4" in result

    def test_tolerance_required(self) -> None:
        data = self._pack_floats(1.0)
        result = compare_outputs(data, data, ComparisonMode.TOLERANCE, None)
        assert result is not None
        assert "requires" in result.lower()

    def test_both_pos_inf_is_ok(self) -> None:
        inf = float("inf")
        data = self._pack_floats(inf)
        assert compare_outputs(data, data, ComparisonMode.TOLERANCE, 1e-5) is None

    def test_both_neg_inf_is_ok(self) -> None:
        neg_inf = float("-inf")
        data = self._pack_floats(neg_inf)
        assert compare_outputs(data, data, ComparisonMode.TOLERANCE, 1e-5) is None

    def test_finite_vs_inf_is_mismatch(self) -> None:
        actual = self._pack_floats(1.0)
        expected = self._pack_floats(float("inf"))
        result = compare_outputs(actual, expected, ComparisonMode.TOLERANCE, 1e-5)
        assert result is not None
        assert "infinity mismatch" in result

    def test_pos_inf_vs_neg_inf_is_mismatch(self) -> None:
        actual = self._pack_floats(float("inf"))
        expected = self._pack_floats(float("-inf"))
        result = compare_outputs(actual, expected, ComparisonMode.TOLERANCE, 1e-5)
        assert result is not None
        assert "infinity mismatch" in result
