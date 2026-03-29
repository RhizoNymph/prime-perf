# 2D 5-Point Stencil

Iterated 2D 5-point stencil computation on a grid of single-precision floats.

## I/O Format (binary, little-endian)

**Input** (stdin):
- `int32` W — grid width
- `int32` H — grid height
- `int32` iters — number of iterations
- `W*H float32` — initial grid values (row-major)

**Output** (stdout):
- `W*H float32` — final grid values after all iterations (row-major)

## Computation

For each iteration, compute a new grid where each interior cell is the average
of itself and its four cardinal neighbors:

```
new[i][j] = (old[i-1][j] + old[i+1][j] + old[i][j-1] + old[i][j+1] + old[i][j]) / 5.0
```

Boundary cells (first/last row/column) remain unchanged from the initial grid.
After each iteration, the new grid becomes the old grid for the next iteration.

## Constraints
- 1 ≤ W, H ≤ 2048
- 0 ≤ iters ≤ 1000
- All values are finite (no NaN, no inf)

## Notes
The naive algorithm uses two allocated grids and swaps them after each iteration.
Cache tiling, vectorization, and parallelism are effective optimizations. The
boundary condition means that a 1-cell border is fixed for all iterations.
