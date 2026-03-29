# N-body Gravitational Simulation

Simulate N bodies interacting via Newtonian gravity using leapfrog integration.

## I/O Format (binary, little-endian)

**Input** (stdin):
- `int32` N — number of bodies
- `int32` steps — number of simulation steps
- `float32` dt — timestep
- `N * 7 float32` — body data per body: (x, y, z, vx, vy, vz, mass)

**Output** (stdout):
- `N * 7 float32` — final body state per body: (x, y, z, vx, vy, vz, mass)

## Constants
- Gravitational constant G = 1.0
- Softening epsilon^2 = 1e-6

## Algorithm

For each step:

1. **Compute accelerations** from pairwise gravitational forces:
   ```
   for each pair (i, j) where i != j:
       dx = x[j] - x[i]
       dy = y[j] - y[i]
       dz = z[j] - z[i]
       dist_sq = dx*dx + dy*dy + dz*dz + epsilon_sq
       inv_dist = 1.0 / sqrt(dist_sq)
       inv_dist3 = inv_dist * inv_dist * inv_dist
       ax[i] += mass[j] * dx * inv_dist3
       ay[i] += mass[j] * dy * inv_dist3
       az[i] += mass[j] * dz * inv_dist3
   ```

2. **Update velocities**: `v += a * dt`
3. **Update positions**: `x += v * dt`

Mass does not change.

## Constraints
- 1 <= N <= 4096
- 1 <= steps <= 1000
- All values are finite (no NaN, no inf)

## Notes
The naive algorithm is O(N^2) per step. Force computation dominates runtime.
The inner loop has good opportunities for SIMD, tiling, and parallelization.
Barnes-Hut tree-based approximation is out of scope for exact comparison.
