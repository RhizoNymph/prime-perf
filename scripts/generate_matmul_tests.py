#!/usr/bin/env python3
"""Generate test inputs and expected outputs for the matmul problem.

Compiles the C reference and runs it to produce expected outputs. Writes
sizes.toml plus per-size perf inputs under perf_inputs/.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROBLEM_DIR = Path(__file__).parent.parent / "problems" / "matmul"
# Keep first 5 sizes identical to prior version so existing tests are preserved;
# add 7 more with varied (prime / non-pow2 / larger) sizes.
TEST_SIZES = [2, 4, 8, 16, 32, 1, 3, 7, 15, 23, 48, 64]
PERF_SIZES = [
    ("small", 512),
    ("medium", 1024),
    ("large", 1536),
]
SEED = 42


def make_input(n: int, rng: np.random.Generator) -> bytes:
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    return struct.pack("<i", n) + a.tobytes() + b.tobytes()


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

        # Generate per-size perf inputs
        for label, n in PERF_SIZES:
            perf_input = make_input(n, rng)
            (perf_dir / f"{label}.bin").write_bytes(perf_input)
            print(f"  perf_input[{label}]: N={n}, {len(perf_input)} bytes")

    write_sizes_toml(PROBLEM_DIR / "sizes.toml", PERF_SIZES)
    # Delete legacy singular perf_input.bin if present.
    legacy = PROBLEM_DIR / "perf_input.bin"
    if legacy.exists():
        legacy.unlink()
        print(f"  removed legacy {legacy.name}")

    print(f"Generated {len(TEST_SIZES)} tests + {len(PERF_SIZES)} perf inputs for matmul")


if __name__ == "__main__":
    main()
