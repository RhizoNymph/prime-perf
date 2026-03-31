# Matrix Multiply

Multiply two NxN matrices of single-precision floats.

## I/O Format (binary, little-endian)

**Input** (stdin):
- `int32` N — matrix dimension
- `N*N float32` — matrix A (row-major)
- `N*N float32` — matrix B (row-major)

**Output** (stdout):
- `N*N float32` — matrix C = A × B (row-major)

## Constraints
- 1 ≤ N ≤ 2048
- All values are finite (no NaN, no inf)

## Notes
The naive algorithm is O(N³) with poor cache behavior. The inner loop
accesses B column-by-column, causing a cache miss on every access for
large N. Cache tiling, loop reordering, and SIMD are effective optimizations.
