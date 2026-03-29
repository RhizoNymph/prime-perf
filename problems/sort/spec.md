# Float32 Array Sort

Sort an array of single-precision floats in ascending order with IEEE-aware handling.

## I/O Format (binary, little-endian)

**Input** (stdin):
- `int32` N -- array length
- `N float32` -- values to sort

**Output** (stdout):
- `N float32` -- sorted values (ascending)

## Sorting Rules

1. Sort ascending by numeric value.
2. NaN values sort to the END of the array (after +inf).
3. -0.0 sorts before +0.0 (IEEE totalOrder-like semantics).
4. Multiple NaN values maintain their relative order (stable for NaN).

## Constraints
- 1 <= N <= 10,000,000
- Values may include NaN, +inf, -inf, -0.0, and +0.0

## Notes
The naive algorithm uses a comparison-based sort with a custom comparator
that handles NaN and signed-zero semantics. The comparison function:
- If both are NaN: maintain order (stable)
- If only a is NaN: a goes after b
- If only b is NaN: b goes after a
- Otherwise: compare as normal floats; for -0.0 vs +0.0, use sign bit

A practical approach: partition NaN values to the end first, then sort the
non-NaN prefix with standard comparison. Cache-unfriendly access patterns
and branch mispredictions are the primary performance bottlenecks.
