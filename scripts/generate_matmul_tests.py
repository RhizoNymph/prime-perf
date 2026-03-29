#!/usr/bin/env python3
"""Generate test inputs and expected outputs for the matmul problem.

Compiles the C reference and runs it to produce expected outputs.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROBLEM_DIR = Path(__file__).parent.parent / "problems" / "matmul"
TEST_SIZES = [2, 4, 8, 16, 32]
PERF_SIZE = 1024
SEED = 42


def make_input(n: int, rng: np.random.Generator) -> bytes:
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    return struct.pack("<i", n) + a.tobytes() + b.tobytes()


def main() -> None:
    rng = np.random.default_rng(SEED)
    tests_dir = PROBLEM_DIR / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Compile C reference
    with tempfile.TemporaryDirectory() as tmpdir:
        binary = Path(tmpdir) / "matmul_ref"
        subprocess.run(
            ["gcc", "-O2", "-lm", "-o", str(binary),
             str(PROBLEM_DIR / "reference" / "solution.c")],
            check=True,
        )

        # Generate test inputs and expected outputs
        for i, n in enumerate(TEST_SIZES):
            input_data = make_input(n, rng)
            (tests_dir / f"input_{i}.bin").write_bytes(input_data)

            result = subprocess.run(
                [str(binary)], input=input_data, capture_output=True, timeout=10,
            )
            assert result.returncode == 0, f"C reference failed for N={n}: {result.stderr.decode()}"
            (tests_dir / f"expected_{i}.bin").write_bytes(result.stdout)

            expected_floats = len(result.stdout) // 4
            print(f"  test_{i}: N={n}, input={len(input_data)} bytes, "
                  f"output={len(result.stdout)} bytes ({expected_floats} floats)")

        # Generate perf input
        perf_input = make_input(PERF_SIZE, rng)
        (PROBLEM_DIR / "perf_input.bin").write_bytes(perf_input)
        print(f"  perf_input: N={PERF_SIZE}, {len(perf_input)} bytes")

    print(f"Generated {len(TEST_SIZES)} tests + perf input for matmul")


if __name__ == "__main__":
    main()
