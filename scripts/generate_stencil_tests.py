#!/usr/bin/env python3
"""Generate test inputs and expected outputs for the stencil problem.

Compiles the C reference and runs it. Writes sizes.toml plus per-size perf
inputs under perf_inputs/.
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
PERF_ITERS = 100
# (label, square-grid edge length); n = edge length (W=H=n).
PERF_SIZES = [
    ("small", 512),
    ("medium", 1024),
    ("large", 1536),
]
SEED = 42


def make_input(w: int, h: int, iters: int, rng: np.random.Generator) -> bytes:
    grid = rng.random((h, w), dtype=np.float32)
    return struct.pack("<iii", w, h, iters) + grid.tobytes()


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

        # Generate per-size perf inputs (square WxW with PERF_ITERS held constant)
        for label, n in PERF_SIZES:
            perf_input = make_input(n, n, PERF_ITERS, rng)
            (perf_dir / f"{label}.bin").write_bytes(perf_input)
            print(f"  perf_input[{label}]: W=H={n}, iters={PERF_ITERS}, "
                  f"{len(perf_input)} bytes")

    write_sizes_toml(PROBLEM_DIR / "sizes.toml", PERF_SIZES)
    legacy = PROBLEM_DIR / "perf_input.bin"
    if legacy.exists():
        legacy.unlink()
        print(f"  removed legacy {legacy.name}")

    print(f"Generated {len(TEST_PARAMS)} tests + {len(PERF_SIZES)} perf inputs for stencil")


if __name__ == "__main__":
    main()
