"""Output comparison utilities for test validation.

Supports exact binary match and tolerance-based float comparison.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import StrEnum


class ComparisonMode(StrEnum):
    """How to compare actual vs expected output."""

    EXACT = "exact"
    TOLERANCE = "tolerance"


@dataclass(frozen=True, slots=True, eq=False)
class ComparisonConfig:
    """Bundles comparison mode with its tolerance parameter.

    Eliminates the repeated (mode, tolerance) parameter pair.
    """

    mode: ComparisonMode = ComparisonMode.EXACT
    tolerance: float | None = None

    def __eq__(self, other: object) -> bool:
        """Support comparison with ComparisonMode and str for backward compatibility."""
        if isinstance(other, ComparisonMode):
            return self.mode == other
        if isinstance(other, str):
            return self.mode.value == other
        if isinstance(other, ComparisonConfig):
            return self.mode == other.mode and self.tolerance == other.tolerance
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.mode, self.tolerance))


def compare_outputs(
    actual: bytes,
    expected: bytes,
    mode: ComparisonMode = ComparisonMode.EXACT,
    tolerance: float | None = None,
) -> str | None:
    """Compare actual output bytes against expected.

    Args:
        actual: Output produced by the solution.
        expected: Expected output from the reference.
        mode: Comparison strategy.
        tolerance: Relative tolerance for float comparison (required when mode=TOLERANCE).

    Returns:
        None if outputs match, or an error message string describing the mismatch.
    """
    if mode == ComparisonMode.EXACT:
        return _compare_exact(actual, expected)
    if mode == ComparisonMode.TOLERANCE:
        if tolerance is None:
            return "tolerance mode requires a tolerance value"
        return _compare_tolerance(actual, expected, tolerance)
    return f"unknown comparison mode: {mode}"


def _compare_exact(actual: bytes, expected: bytes) -> str | None:
    """Byte-exact comparison."""
    if actual == expected:
        return None
    if len(actual) != len(expected):
        return f"Output size mismatch: got {len(actual)} bytes, expected {len(expected)} bytes"
    # Find first differing byte position
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if a != e:
            return (
                f"Output differs at byte {i}: got 0x{a:02x}, expected 0x{e:02x} "
                f"(total {len(actual)} bytes)"
            )
    return None  # unreachable but satisfies type checker


def _compare_tolerance(actual: bytes, expected: bytes, tolerance: float) -> str | None:
    """Float32-level comparison with relative tolerance.

    Interprets both byte arrays as sequences of float32 values and
    compares each pair with relative tolerance.
    """
    if len(actual) != len(expected):
        return f"Output size mismatch: got {len(actual)} bytes, expected {len(expected)} bytes"

    if len(actual) % 4 != 0:
        return f"Output size {len(actual)} is not a multiple of 4 (expected float32 array)"

    n_floats = len(actual) // 4
    actual_floats = struct.unpack(f"<{n_floats}f", actual)
    expected_floats = struct.unpack(f"<{n_floats}f", expected)

    for i, (a, e) in enumerate(zip(actual_floats, expected_floats, strict=True)):
        # Handle NaN: both NaN is OK, only one NaN is a mismatch
        a_nan = a != a
        e_nan = e != e
        if a_nan and e_nan:
            continue
        if a_nan != e_nan:
            return f"Float[{i}]: one is NaN, other is {a if e_nan else e}"

        # Handle exact zero
        if e == 0.0:
            if abs(a) > tolerance:
                return f"Float[{i}]: got {a}, expected 0.0 (abs diff {abs(a)} > {tolerance})"
            continue

        # Relative comparison
        rel_diff = abs(a - e) / abs(e)
        if rel_diff > tolerance:
            return (
                f"Float[{i}]: got {a}, expected {e} "
                f"(relative diff {rel_diff:.2e} > {tolerance:.2e})"
            )

    return None
