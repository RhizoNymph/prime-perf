#!/usr/bin/env python3
"""Generate test inputs and expected outputs for the nbody problem.

Compiles the C reference and runs it to produce expected outputs.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROBLEM_DIR = Path(__file__).parent.parent / "problems" / "nbody"
SEED = 42

# Test cases: (N, steps, dt)
TEST_CASES = [
    (3, 1, 0.01),      # minimal
    (3, 10, 0.001),     # more steps, smaller dt
    (10, 5, 0.01),      # medium
    (20, 3, 0.01),      # more bodies
    (5, 50, 0.001),     # many steps
]

# Perf input
PERF_N = 1024
PERF_STEPS = 10
PERF_DT = 0.01


def make_input(
    n: int,
    steps: int,
    dt: float,
    rng: np.random.Generator,
) -> bytes:
    """Generate binary input for nbody problem."""
    header = struct.pack("<iif", n, steps, dt)

    # positions in [-1, 1], velocities in [-0.1, 0.1], masses in [0.1, 10.0]
    positions = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    velocities = rng.uniform(-0.1, 0.1, (n, 3)).astype(np.float32)
    masses = rng.uniform(0.1, 10.0, (n, 1)).astype(np.float32)

    # Pack as 7 floats per body: x, y, z, vx, vy, vz, mass
    body_data = np.hstack([positions, velocities, masses])
    return header + body_data.tobytes()


def main() -> None:
    rng = np.random.default_rng(SEED)
    tests_dir = PROBLEM_DIR / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Compile C reference
    with tempfile.TemporaryDirectory() as tmpdir:
        binary = Path(tmpdir) / "nbody_ref"
        subprocess.run(
            [
                "gcc", "-O2", "-o", str(binary),
                str(PROBLEM_DIR / "reference" / "solution.c"),
                "-lm",
            ],
            check=True,
        )

        # Generate test inputs and expected outputs
        for i, (n, steps, dt) in enumerate(TEST_CASES):
            input_data = make_input(n, steps, dt, rng)
            (tests_dir / f"input_{i}.bin").write_bytes(input_data)

            result = subprocess.run(
                [str(binary)],
                input=input_data,
                capture_output=True,
                timeout=30,
            )
            assert result.returncode == 0, (
                f"C reference failed for N={n}, steps={steps}: "
                f"{result.stderr.decode()}"
            )
            (tests_dir / f"expected_{i}.bin").write_bytes(result.stdout)

            expected_floats = len(result.stdout) // 4
            print(
                f"  test_{i}: N={n}, steps={steps}, dt={dt}, "
                f"input={len(input_data)} bytes, "
                f"output={len(result.stdout)} bytes ({expected_floats} floats)"
            )

        # Generate perf input
        perf_input = make_input(PERF_N, PERF_STEPS, PERF_DT, rng)
        (PROBLEM_DIR / "perf_input.bin").write_bytes(perf_input)
        print(f"  perf_input: N={PERF_N}, steps={PERF_STEPS}, {len(perf_input)} bytes")

    print(f"Generated {len(TEST_CASES)} tests + perf input for nbody")


if __name__ == "__main__":
    main()
