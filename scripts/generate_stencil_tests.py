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
TEST_PARAMS = [
    (8, 8, 1),
    (8, 8, 10),
    (16, 16, 5),
    (32, 32, 3),
    (64, 64, 1),
]
PERF_W = 1024
PERF_H = 1024
PERF_ITERS = 100
SEED = 42


def make_input(w: int, h: int, iters: int, rng: np.random.Generator) -> bytes:
    grid = rng.random((h, w), dtype=np.float32)
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

    print(f"Generated {len(TEST_PARAMS)} tests + perf input for stencil")


if __name__ == "__main__":
    main()
