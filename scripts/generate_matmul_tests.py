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

# Held-out distribution choice for matmul:
#   In-dist tests use uniform random matrices in [0,1). Held-out uses
#   ill-conditioned / sparse / structured matrices (low-rank perturbation,
#   diagonal-dominant, sparse). These exercise different cache and FLOP
#   patterns than dense uniform matrices, making it harder for an agent to
#   exploit any incidental property of [0,1) uniform inputs.
HELDOUT_SEED = 1337
HELDOUT_TEST_SIZES = [4, 11, 32, 64]
HELDOUT_PERF_SIZE = 1024


def make_input(n: int, rng: np.random.Generator) -> bytes:
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    return struct.pack("<i", n) + a.tobytes() + b.tobytes()


def make_heldout_input(n: int, rng: np.random.Generator) -> bytes:
    """Held-out matmul input: ill-conditioned / structured matrices.

    Combines a low-rank base (rank ~= n/4) with a small diagonal regularizer
    so the matrices are reproducible and finite, but spectrally very
    different from i.i.d. uniform matrices. For small n we fall back to a
    sparse-ish mask to keep the construction well-defined.
    """
    if n <= 1:
        a = rng.standard_normal((n, n)).astype(np.float32)
        b = rng.standard_normal((n, n)).astype(np.float32)
    else:
        rank = max(1, n // 4)
        ua = rng.standard_normal((n, rank)).astype(np.float32)
        va = rng.standard_normal((rank, n)).astype(np.float32)
        ub = rng.standard_normal((n, rank)).astype(np.float32)
        vb = rng.standard_normal((rank, n)).astype(np.float32)
        a = (ua @ va) + 0.01 * np.eye(n, dtype=np.float32)
        b = (ub @ vb) + 0.01 * np.eye(n, dtype=np.float32)
        # Sparsify B at ~70% density to perturb cache traffic patterns.
        mask = rng.random((n, n)) < 0.7
        b = (b * mask).astype(np.float32)
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

        # Held-out tests + perf input.
        # Distribution: low-rank + sparse instead of dense uniform [0,1).
        held_rng = np.random.default_rng(HELDOUT_SEED)
        held_tests_dir = PROBLEM_DIR / "tests_heldout"
        held_tests_dir.mkdir(parents=True, exist_ok=True)
        held_perf_dir = PROBLEM_DIR / "perf_inputs_heldout"
        held_perf_dir.mkdir(parents=True, exist_ok=True)

        for i, n in enumerate(HELDOUT_TEST_SIZES):
            input_data = make_heldout_input(n, held_rng)
            (held_tests_dir / f"input_{i}.bin").write_bytes(input_data)
            result = subprocess.run(
                [str(binary)], input=input_data, capture_output=True, timeout=30,
            )
            assert result.returncode == 0, (
                f"C reference failed for held-out N={n}: {result.stderr.decode()}"
            )
            (held_tests_dir / f"expected_{i}.bin").write_bytes(result.stdout)
            print(
                f"  heldout_test_{i}: N={n}, input={len(input_data)} bytes, "
                f"output={len(result.stdout)} bytes"
            )

        held_perf_input = make_heldout_input(HELDOUT_PERF_SIZE, held_rng)
        (held_perf_dir / "large.bin").write_bytes(held_perf_input)
        print(
            f"  heldout_perf large.bin: N={HELDOUT_PERF_SIZE}, "
            f"{len(held_perf_input)} bytes"
        )

    write_sizes_toml(PROBLEM_DIR / "sizes.toml", PERF_SIZES)
    # Delete legacy singular perf_input.bin if present.
    legacy = PROBLEM_DIR / "perf_input.bin"
    if legacy.exists():
        legacy.unlink()
        print(f"  removed legacy {legacy.name}")

    print(
        f"Generated {len(TEST_SIZES)} tests + {len(PERF_SIZES)} perf inputs "
        f"+ {len(HELDOUT_TEST_SIZES)} held-out tests for matmul"
    )


if __name__ == "__main__":
    main()
