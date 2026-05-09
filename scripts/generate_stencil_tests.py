#!/usr/bin/env python3
"""Generate test inputs and expected outputs for the stencil problem.

Compiles the C reference and runs it to produce expected outputs.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROBLEM_DIR = Path(__file__).parent.parent / "problems" / "stencil"
# Keep first 5 entries identical to prior version so existing tests are preserved;
# add 7 more with non-square shapes, iters=0 identity, and larger grids.
TEST_PARAMS = [
    (8, 8, 1),
    (8, 8, 10),
    (16, 16, 5),
    (32, 32, 3),
    (64, 64, 1),
    (16, 8, 5),
    (8, 16, 5),
    (5, 5, 0),
    (48, 16, 3),
    (16, 48, 3),
    (128, 128, 2),
    (24, 40, 7),
]
PERF_W = 1024
PERF_H = 1024
PERF_ITERS = 100
SEED = 42

# Held-out distribution choice for stencil:
#   In-dist tests use uniform random initial state. Held-out alternates
#   between high-frequency (checkerboard) and smooth (linear gradient)
#   initial states. These have very different smoothing trajectories under
#   a 5-point stencil; an agent that overfit to noise should regress on the
#   structured cases.
HELDOUT_SEED = 1337
HELDOUT_TEST_PARAMS = [
    (16, 16, 5),
    (32, 32, 3),
    (64, 64, 2),
    (24, 40, 5),
]
HELDOUT_PERF_W = 1024
HELDOUT_PERF_H = 1024
HELDOUT_PERF_ITERS = 100


def make_input(w: int, h: int, iters: int, rng: np.random.Generator) -> bytes:
    grid = rng.random((h, w), dtype=np.float32)
    return struct.pack("<iii", w, h, iters) + grid.tobytes()


def _checkerboard(h: int, w: int) -> np.ndarray:
    yy, xx = np.indices((h, w))
    return ((yy + xx) % 2).astype(np.float32)


def _gradient(h: int, w: int) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    return (0.5 * (yy + xx)).astype(np.float32)


def make_heldout_input(
    w: int, h: int, iters: int, idx: int, rng: np.random.Generator,
) -> bytes:
    """Held-out stencil input: alternating checkerboard / smooth gradient.

    Even idx -> high-frequency checkerboard with tiny noise.
    Odd  idx -> smooth gradient with tiny noise.
    Both diverge sharply from i.i.d. uniform initial state.
    """
    if idx % 2 == 0:
        base = _checkerboard(h, w)
    else:
        base = _gradient(h, w)
    noise = rng.uniform(-0.01, 0.01, size=(h, w)).astype(np.float32)
    grid = (base + noise).astype(np.float32)
    return struct.pack("<iii", w, h, iters) + grid.tobytes()


def make_heldout_perf_input(
    w: int, h: int, iters: int, rng: np.random.Generator,
) -> bytes:
    """Held-out perf input: smooth gradient + tiny noise, sized comparably."""
    base = _gradient(h, w)
    noise = rng.uniform(-0.01, 0.01, size=(h, w)).astype(np.float32)
    grid = (base + noise).astype(np.float32)
    return struct.pack("<iii", w, h, iters) + grid.tobytes()


def main() -> None:
    rng = np.random.default_rng(SEED)
    tests_dir = PROBLEM_DIR / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Compile C reference
    with tempfile.TemporaryDirectory() as tmpdir:
        binary = Path(tmpdir) / "stencil_ref"
        subprocess.run(
            ["gcc", "-O2", "-lm", "-o", str(binary),
             str(PROBLEM_DIR / "reference" / "solution.c")],
            check=True,
        )

        # Generate test inputs and expected outputs
        for i, (w, h, iters) in enumerate(TEST_PARAMS):
            input_data = make_input(w, h, iters, rng)
            (tests_dir / f"input_{i}.bin").write_bytes(input_data)

            result = subprocess.run(
                [str(binary)], input=input_data, capture_output=True, timeout=30,
            )
            assert result.returncode == 0, (
                f"C reference failed for W={w}, H={h}, iters={iters}: "
                f"{result.stderr.decode()}"
            )
            (tests_dir / f"expected_{i}.bin").write_bytes(result.stdout)

            expected_floats = len(result.stdout) // 4
            print(f"  test_{i}: W={w}, H={h}, iters={iters}, "
                  f"input={len(input_data)} bytes, "
                  f"output={len(result.stdout)} bytes ({expected_floats} floats)")

        # Generate perf input
        perf_input = make_input(PERF_W, PERF_H, PERF_ITERS, rng)
        (PROBLEM_DIR / "perf_input.bin").write_bytes(perf_input)
        print(f"  perf_input: W={PERF_W}, H={PERF_H}, iters={PERF_ITERS}, "
              f"{len(perf_input)} bytes")

        # Held-out tests + perf input.
        # Distribution: checkerboard / smooth gradient instead of i.i.d.
        # uniform random initial state.
        held_rng = np.random.default_rng(HELDOUT_SEED)
        held_tests_dir = PROBLEM_DIR / "tests_heldout"
        held_tests_dir.mkdir(parents=True, exist_ok=True)
        held_perf_dir = PROBLEM_DIR / "perf_inputs_heldout"
        held_perf_dir.mkdir(parents=True, exist_ok=True)

        for i, (w, h, iters) in enumerate(HELDOUT_TEST_PARAMS):
            input_data = make_heldout_input(w, h, iters, i, held_rng)
            (held_tests_dir / f"input_{i}.bin").write_bytes(input_data)
            result = subprocess.run(
                [str(binary)], input=input_data, capture_output=True, timeout=30,
            )
            assert result.returncode == 0, (
                f"C reference failed for held-out W={w}, H={h}, iters={iters}: "
                f"{result.stderr.decode()}"
            )
            (held_tests_dir / f"expected_{i}.bin").write_bytes(result.stdout)
            print(
                f"  heldout_test_{i}: W={w}, H={h}, iters={iters}, "
                f"input={len(input_data)} bytes, output={len(result.stdout)} bytes"
            )

        held_perf_input = make_heldout_perf_input(
            HELDOUT_PERF_W, HELDOUT_PERF_H, HELDOUT_PERF_ITERS, held_rng,
        )
        (held_perf_dir / "large.bin").write_bytes(held_perf_input)
        print(
            f"  heldout_perf large.bin: W={HELDOUT_PERF_W}, "
            f"H={HELDOUT_PERF_H}, iters={HELDOUT_PERF_ITERS}, "
            f"{len(held_perf_input)} bytes"
        )

    print(f"Generated {len(TEST_PARAMS)} tests + perf input for stencil")


if __name__ == "__main__":
    main()
