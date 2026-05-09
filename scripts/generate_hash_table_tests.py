#!/usr/bin/env python3
"""Generate test inputs and expected outputs for the hash_table problem.

Compiles the C reference and runs it. Writes sizes.toml plus per-size perf
inputs under perf_inputs/.
"""

from __future__ import annotations

import random
import string
import struct
import subprocess
import tempfile
from pathlib import Path

PROBLEM_DIR = Path(__file__).parent.parent / "problems" / "hash_table"
SEED = 42

CHARSET = string.ascii_letters + string.digits

PERF_SIZES = [
    ("small", 50_000),
    ("medium", 100_000),
    ("large", 200_000),
]


def rand_string(rng: random.Random, min_len: int, max_len: int) -> str:
    length = rng.randint(min_len, max_len)
    return "".join(rng.choices(CHARSET, k=length))


def build_input(
    rng: random.Random,
    n_insert: int,
    n_lookup: int,
    *,
    all_present: bool = True,
    duplicate_keys: bool = False,
    missing_fraction: float = 0.0,
) -> bytes:
    """Build a binary test input."""
    parts: list[bytes] = []

    keys: list[str] = []
    for _ in range(n_insert):
        keys.append(rand_string(rng, 5, 20))

    if duplicate_keys:
        n_dupes = n_insert // 5  # ~20% duplicates
        for _ in range(n_dupes):
            idx = rng.randint(0, len(keys) - 1)
            keys.append(keys[idx])
        n_insert = len(keys)

    insert_entries: list[tuple[str, str]] = []
    for key in keys:
        val = rand_string(rng, 10, 50)
        insert_entries.append((key, val))

    parts.append(struct.pack("<i", n_insert))

    for key, val in insert_entries:
        key_bytes = key.encode("utf-8")
        val_bytes = val.encode("utf-8")
        parts.append(struct.pack("<i", len(key_bytes)))
        parts.append(key_bytes)
        parts.append(struct.pack("<i", len(val_bytes)))
        parts.append(val_bytes)

    unique_keys = list(dict(insert_entries).keys())

    lookup_keys: list[str] = []
    if all_present and not missing_fraction:
        for _ in range(n_lookup):
            lookup_keys.append(rng.choice(unique_keys))
    else:
        n_missing = int(n_lookup * missing_fraction)
        n_present = n_lookup - n_missing
        for _ in range(n_present):
            lookup_keys.append(rng.choice(unique_keys))

        unique_key_set = set(unique_keys)
        for _ in range(n_missing):
            while True:
                missing_key = rand_string(rng, 5, 20)
                if missing_key not in unique_key_set:
                    lookup_keys.append(missing_key)
                    break

        rng.shuffle(lookup_keys)

    parts.append(struct.pack("<i", n_lookup))

    for key in lookup_keys:
        key_bytes = key.encode("utf-8")
        parts.append(struct.pack("<i", len(key_bytes)))
        parts.append(key_bytes)

    return b"".join(parts)


def write_sizes_toml(path: Path, sizes: list[tuple[str, int]]) -> None:
    lines: list[str] = []
    for label, n in sizes:
        lines.append("[[sizes]]")
        lines.append(f'label = "{label}"')
        lines.append(f"n = {n}")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    rng = random.Random(SEED)
    tests_dir = PROBLEM_DIR / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    perf_dir = PROBLEM_DIR / "perf_inputs"
    perf_dir.mkdir(parents=True, exist_ok=True)

    test_configs = [
        (10, 10, {"all_present": True}, "10 inserts, 10 lookups (all present)"),
        (100, 100, {"all_present": True}, "100 inserts, 100 lookups (all present)"),
        (50, 100, {"all_present": False, "missing_fraction": 0.5},
         "50 inserts, 100 lookups (~50% missing)"),
        (100, 100, {"all_present": True, "duplicate_keys": True},
         "100 inserts with duplicates, 100 lookups"),
        (1000, 1000, {"all_present": True}, "1000 inserts, 1000 lookups"),
        (500, 500, {"all_present": True}, "500 inserts, 500 lookups (all present)"),
        (200, 200, {"all_present": False, "missing_fraction": 1.0},
         "200 inserts, 200 lookups (all missing)"),
        (20, 100, {"all_present": True, "duplicate_keys": True},
         "20 inserts with duplicates, 100 lookups (heavy lookups)"),
        (2000, 500, {"all_present": True}, "2000 inserts, 500 lookups (insert-heavy)"),
        (500, 2000, {"all_present": True}, "500 inserts, 2000 lookups (lookup-heavy)"),
        (300, 600, {"all_present": False, "missing_fraction": 0.2},
         "300 inserts, 600 lookups (~20% missing)"),
        (200, 500, {"all_present": False, "missing_fraction": 0.8},
         "200 inserts, 500 lookups (~80% missing)"),
    ]

    test_inputs: list[bytes] = []
    for n_insert, n_lookup, kwargs, desc in test_configs:
        input_data = build_input(rng, n_insert, n_lookup, **kwargs)
        test_inputs.append(input_data)
        print(f"  Generated: {desc} ({len(input_data)} bytes)")

    with tempfile.TemporaryDirectory() as tmpdir:
        binary = Path(tmpdir) / "hash_table_ref"
        subprocess.run(
            ["gcc", "-O2", "-o", str(binary),
             str(PROBLEM_DIR / "reference" / "solution.c")],
            check=True,
        )

        for i, (input_data, (_, _, _, desc)) in enumerate(
            zip(test_inputs, test_configs, strict=False)
        ):
            (tests_dir / f"input_{i}.bin").write_bytes(input_data)

            result = subprocess.run(
                [str(binary)], input=input_data, capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                print(f"  FAILED: {desc}")
                print(f"    stderr: {result.stderr.decode()}")
                raise RuntimeError(f"C reference failed for test {i}")

            (tests_dir / f"expected_{i}.bin").write_bytes(result.stdout)
            print(f"  test_{i}: {desc}, output={len(result.stdout)} bytes")

        # Generate per-size perf inputs (n inserts + n lookups, all present)
        for label, n in PERF_SIZES:
            perf_input = build_input(rng, n, n, all_present=True)
            (perf_dir / f"{label}.bin").write_bytes(perf_input)
            print(f"  perf_input[{label}]: {n} inserts + {n} lookups, "
                  f"{len(perf_input)} bytes")

    write_sizes_toml(PROBLEM_DIR / "sizes.toml", PERF_SIZES)
    legacy = PROBLEM_DIR / "perf_input.bin"
    if legacy.exists():
        legacy.unlink()
        print(f"  removed legacy {legacy.name}")

    print(f"Generated {len(test_configs)} tests + {len(PERF_SIZES)} perf inputs "
          f"for hash_table")


if __name__ == "__main__":
    main()
