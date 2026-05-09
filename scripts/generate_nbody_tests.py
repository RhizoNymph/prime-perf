#!/usr/bin/env python3
"""Generate test inputs and expected outputs for the nbody problem.

Compiles the C reference and runs it. Writes sizes.toml plus per-size perf
inputs under perf_inputs/.
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
# Keep first 5 identical to prior version so existing tests are preserved;
# add 7 more covering 2-body, larger N, and varied step/dt combos.
TEST_CASES = [
    (3, 1, 0.01),      # minimal
    (3, 10, 0.001),     # more steps, smaller dt
    (10, 5, 0.01),      # medium
    (20, 3, 0.01),      # more bodies
    (5, 50, 0.001),     # many steps
    (2, 1, 0.01),       # 2-body
    (4, 20, 0.005),     # small N, many steps
    (50, 2, 0.01),      # medium-large N, few steps
    (100, 1, 0.01),     # large N, single step
    (8, 100, 0.0001),   # very small dt, many steps
    (30, 5, 0.01),      # moderate everything
    (15, 20, 0.005),    # balanced
]

# Perf sizes: (label, N). Hold steps and dt constant so n parameterizes scaling.
PERF_STEPS = 10
PERF_DT = 0.01
PERF_SIZES = [
    ("small", 512),
    ("medium", 1024),
    ("large", 2048),
]

# Held-out distribution choice for nbody:
#   In-dist tests use positions sampled uniformly in [-1,1]^3. Held-out uses
#   *clustered* initial conditions: a Plummer-like radial distribution plus a
#   second offset cluster. Clustered initial conditions yield strongly
#   non-uniform pairwise force magnitudes and very different memory-access
#   patterns vs. uniform ones, so an agent that overfit to uniform layouts
#   should regress here.
HELDOUT_SEED = 1337
HELDOUT_TEST_CASES = [
    (3, 1, 0.01),
    (10, 5, 0.01),
    (32, 3, 0.005),
    (64, 2, 0.005),
]
HELDOUT_PERF_N = 1024
HELDOUT_PERF_STEPS = 10
HELDOUT_PERF_DT = 0.01


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


def make_heldout_input(
    n: int,
    steps: int,
    dt: float,
    rng: np.random.Generator,
) -> bytes:
    """Held-out nbody input: clustered (two-cluster) initial conditions.

    Half of the bodies are drawn from a tight Gaussian centered at the origin
    (a Plummer-like core), the other half from a Gaussian centered at
    (+0.6, 0, 0). Velocities and masses keep the same support as in-dist so
    only the spatial distribution diverges, isolating the cache/memory
    effects of clustering on pairwise-force computation.
    """
    header = struct.pack("<iif", n, steps, dt)

    half = n // 2
    pos_a = rng.normal(loc=(0.0, 0.0, 0.0), scale=0.15, size=(half, 3))
    pos_b = rng.normal(loc=(0.6, 0.0, 0.0), scale=0.15, size=(n - half, 3))
    positions = np.vstack([pos_a, pos_b]).astype(np.float32)
    rng.shuffle(positions)
    velocities = rng.uniform(-0.1, 0.1, (n, 3)).astype(np.float32)
    masses = rng.uniform(0.1, 10.0, (n, 1)).astype(np.float32)

    body_data = np.hstack([positions, velocities, masses])
    return header + body_data.tobytes()


def write_sizes_toml(path: Path, sizes: list[tuple[str, int]]) -> None:
    lines: list[str] = []
    for label, n in sizes:
        lines.append("[[sizes]]")
        lines.append(f'label = "{label}"')
        lines.append(f"n = {n}")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    rng = np.random.default_rng(SEED)
    tests_dir = PROBLEM_DIR / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    perf_dir = PROBLEM_DIR / "perf_inputs"
    perf_dir.mkdir(parents=True, exist_ok=True)

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

        # Generate per-size perf inputs (steps + dt held constant)
        for label, n in PERF_SIZES:
            perf_input = make_input(n, PERF_STEPS, PERF_DT, rng)
            (perf_dir / f"{label}.bin").write_bytes(perf_input)
            print(
                f"  perf_input[{label}]: N={n}, steps={PERF_STEPS}, dt={PERF_DT}, "
                f"{len(perf_input)} bytes"
            )

        # Held-out tests + perf input.
        # Distribution: clustered (two Gaussian groups) instead of uniform cube.
        held_rng = np.random.default_rng(HELDOUT_SEED)
        held_tests_dir = PROBLEM_DIR / "tests_heldout"
        held_tests_dir.mkdir(parents=True, exist_ok=True)
        held_perf_dir = PROBLEM_DIR / "perf_inputs_heldout"
        held_perf_dir.mkdir(parents=True, exist_ok=True)

        for i, (n, steps, dt) in enumerate(HELDOUT_TEST_CASES):
            input_data = make_heldout_input(n, steps, dt, held_rng)
            (held_tests_dir / f"input_{i}.bin").write_bytes(input_data)
            result = subprocess.run(
                [str(binary)], input=input_data, capture_output=True, timeout=30,
            )
            assert result.returncode == 0, (
                f"C reference failed for held-out N={n}, steps={steps}: "
                f"{result.stderr.decode()}"
            )
            (held_tests_dir / f"expected_{i}.bin").write_bytes(result.stdout)
            print(
                f"  heldout_test_{i}: N={n}, steps={steps}, dt={dt}, "
                f"input={len(input_data)} bytes, output={len(result.stdout)} bytes"
            )

        held_perf_input = make_heldout_input(
            HELDOUT_PERF_N, HELDOUT_PERF_STEPS, HELDOUT_PERF_DT, held_rng,
        )
        (held_perf_dir / "large.bin").write_bytes(held_perf_input)
        print(
            f"  heldout_perf large.bin: N={HELDOUT_PERF_N}, "
            f"steps={HELDOUT_PERF_STEPS}, {len(held_perf_input)} bytes"
        )

    write_sizes_toml(PROBLEM_DIR / "sizes.toml", PERF_SIZES)
    legacy = PROBLEM_DIR / "perf_input.bin"
    if legacy.exists():
        legacy.unlink()
        print(f"  removed legacy {legacy.name}")

    print(
        f"Generated {len(TEST_CASES)} tests + {len(PERF_SIZES)} perf inputs "
        f"+ {len(HELDOUT_TEST_CASES)} held-out tests for nbody"
    )


if __name__ == "__main__":
    main()
