#!/usr/bin/env python3
"""Generate test inputs and expected outputs for the sort problem.

Compiles the C reference and runs it. Writes sizes.toml plus per-size perf
inputs under perf_inputs/.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROBLEM_DIR = Path(__file__).parent.parent / "problems" / "sort"
SEED = 42

PERF_SIZES = [
    ("small", 1_000_000),
    ("medium", 4_000_000),
    ("large", 10_000_000),
]


def make_basic_small(rng: np.random.Generator) -> bytes:
    """Test 0: N=10, basic small array, all finite."""
    n = 10
    values = rng.uniform(-100.0, 100.0, size=n).astype(np.float32)
    return struct.pack("<i", n) + values.tobytes()


def make_random_medium(rng: np.random.Generator) -> bytes:
    """Test 1: N=100, random values in [-1000, 1000]."""
    n = 100
    values = rng.uniform(-1000.0, 1000.0, size=n).astype(np.float32)
    return struct.pack("<i", n) + values.tobytes()


def make_all_same(rng: np.random.Generator) -> bytes:
    """Test 2: N=50, all same value (stability test)."""
    n = 50
    values = np.full(n, 3.14, dtype=np.float32)
    return struct.pack("<i", n) + values.tobytes()


def make_special_values(rng: np.random.Generator) -> bytes:
    """Test 3: N=20, includes NaN, +inf, -inf, -0.0, +0.0."""
    values = np.array([
        1.0,
        float("nan"),
        float("-inf"),
        -0.0,
        0.0,
        float("inf"),
        float("nan"),
        -1.5,
        2.5,
        float("-inf"),
        float("nan"),
        0.0,
        -0.0,
        100.0,
        -100.0,
        float("inf"),
        3.14,
        -3.14,
        float("nan"),
        42.0,
    ], dtype=np.float32)
    n = len(values)
    assert n == 20
    return struct.pack("<i", n) + values.tobytes()


def make_large_random(rng: np.random.Generator) -> bytes:
    """Test 4: N=1000, larger random array."""
    n = 1000
    values = rng.uniform(-1e6, 1e6, size=n).astype(np.float32)
    return struct.pack("<i", n) + values.tobytes()


def make_already_sorted(rng: np.random.Generator) -> bytes:
    """Test 5: N=200, already sorted ascending."""
    n = 200
    values = np.sort(rng.uniform(-1000.0, 1000.0, size=n).astype(np.float32))
    return struct.pack("<i", n) + values.tobytes()


def make_reverse_sorted(rng: np.random.Generator) -> bytes:
    """Test 6: N=200, reverse sorted."""
    n = 200
    values = np.sort(rng.uniform(-1000.0, 1000.0, size=n).astype(np.float32))[::-1].copy()
    return struct.pack("<i", n) + values.tobytes()


def make_trivial_single(rng: np.random.Generator) -> bytes:
    """Test 7: N=1, trivial single-element."""
    n = 1
    values = rng.uniform(-100.0, 100.0, size=n).astype(np.float32)
    return struct.pack("<i", n) + values.tobytes()


def make_pair(rng: np.random.Generator) -> bytes:
    """Test 8: N=2, two elements."""
    n = 2
    values = rng.uniform(-100.0, 100.0, size=n).astype(np.float32)
    return struct.pack("<i", n) + values.tobytes()


def make_all_nan(rng: np.random.Generator) -> bytes:
    """Test 9: N=30, all NaN (stability of NaN ordering)."""
    n = 30
    values = np.full(n, float("nan"), dtype=np.float32)
    return struct.pack("<i", n) + values.tobytes()


def make_zeros_mix(rng: np.random.Generator) -> bytes:
    """Test 10: N=60, heavy mix of +0.0 and -0.0 with small values."""
    n = 60
    values = rng.uniform(-5.0, 5.0, size=n).astype(np.float32)
    # Sprinkle in many signed zeros
    for i in range(0, n, 2):
        values[i] = np.float32(-0.0 if i % 4 == 0 else 0.0)
    return struct.pack("<i", n) + values.tobytes()


def make_many_duplicates(rng: np.random.Generator) -> bytes:
    """Test 11: N=500, heavy duplicates drawn from a small value pool."""
    n = 500
    pool = rng.uniform(-50.0, 50.0, size=15).astype(np.float32)
    values = pool[rng.integers(0, len(pool), size=n)]
    return struct.pack("<i", n) + values.tobytes()


def make_perf_input(n: int, rng: np.random.Generator) -> bytes:
    """Build a perf input of size N with ~0.1% NaN and a few infs/zeros."""
    values = rng.uniform(-1e6, 1e6, size=n).astype(np.float32)

    # Insert ~0.1% NaN
    nan_count = max(1, n // 1000)
    nan_indices = rng.choice(n, size=nan_count, replace=False)
    values[nan_indices] = np.float32(float("nan"))

    # Insert a few inf/-inf
    inf_count = max(1, n // 5000)
    inf_indices = rng.choice(n, size=inf_count, replace=False)
    for i, idx in enumerate(inf_indices):
        values[idx] = np.float32(float("inf") if i % 2 == 0 else float("-inf"))

    # Insert some -0.0 and +0.0
    zero_count = max(1, n // 2000)
    zero_indices = rng.choice(n, size=zero_count, replace=False)
    for i, idx in enumerate(zero_indices):
        values[idx] = np.float32(-0.0 if i % 2 == 0 else 0.0)

    return struct.pack("<i", n) + values.tobytes()


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

    generators = [
        make_basic_small,
        make_random_medium,
        make_all_same,
        make_special_values,
        make_large_random,
        make_already_sorted,
        make_reverse_sorted,
        make_trivial_single,
        make_pair,
        make_all_nan,
        make_zeros_mix,
        make_many_duplicates,
    ]

    # Compile C reference
    with tempfile.TemporaryDirectory() as tmpdir:
        binary = Path(tmpdir) / "sort_ref"
        subprocess.run(
            [
                "gcc",
                "-O2",
                "-lm",
                "-o",
                str(binary),
                str(PROBLEM_DIR / "reference" / "solution.c"),
            ],
            check=True,
        )

        # Generate test inputs and expected outputs
        for i, gen in enumerate(generators):
            input_data = gen(rng)
            (tests_dir / f"input_{i}.bin").write_bytes(input_data)

            result = subprocess.run(
                [str(binary)],
                input=input_data,
                capture_output=True,
                timeout=30,
            )
            assert (
                result.returncode == 0
            ), f"C reference failed for test {i}: {result.stderr.decode()}"
            (tests_dir / f"expected_{i}.bin").write_bytes(result.stdout)

            n = struct.unpack("<i", input_data[:4])[0]
            expected_floats = len(result.stdout) // 4
            print(
                f"  test_{i}: N={n}, input={len(input_data)} bytes, "
                f"output={len(result.stdout)} bytes ({expected_floats} floats)"
            )

        # Generate per-size perf inputs
        for label, n in PERF_SIZES:
            perf_input = make_perf_input(n, rng)
            (perf_dir / f"{label}.bin").write_bytes(perf_input)
            print(f"  perf_input[{label}]: N={n}, {len(perf_input)} bytes")

    write_sizes_toml(PROBLEM_DIR / "sizes.toml", PERF_SIZES)
    legacy = PROBLEM_DIR / "perf_input.bin"
    if legacy.exists():
        legacy.unlink()
        print(f"  removed legacy {legacy.name}")

    print(f"Generated {len(generators)} tests + {len(PERF_SIZES)} perf inputs for sort")


if __name__ == "__main__":
    main()
