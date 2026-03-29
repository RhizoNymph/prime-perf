#!/usr/bin/env python3
"""Generate test inputs and expected outputs for the hash_table problem.

Compiles the C reference and runs it to produce expected outputs.
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
    """Build a binary test input.

    Args:
        rng: seeded random instance
        n_insert: number of inserts
        n_lookup: number of lookups
        all_present: if True, all lookups are for inserted keys
        duplicate_keys: if True, some keys appear 2-3 times with different values
        missing_fraction: fraction of lookups for keys not in the table
    """
    parts: list[bytes] = []

    # Generate insert keys and values
    keys: list[str] = []
    for _ in range(n_insert):
        keys.append(rand_string(rng, 5, 20))

    # If duplicate_keys, reuse some keys with new values
    if duplicate_keys:
        n_dupes = n_insert // 5  # ~20% duplicates
        for _ in range(n_dupes):
            # Pick a random existing key and append it again
            idx = rng.randint(0, len(keys) - 1)
            keys.append(keys[idx])

        # Update n_insert to include duplicates
        n_insert = len(keys)

    # Generate values for each insert entry
    insert_entries: list[tuple[str, str]] = []
    for key in keys:
        val = rand_string(rng, 10, 50)
        insert_entries.append((key, val))

    # Write insert count
    parts.append(struct.pack("<i", n_insert))

    # Write each insert
    for key, val in insert_entries:
        key_bytes = key.encode("utf-8")
        val_bytes = val.encode("utf-8")
        parts.append(struct.pack("<i", len(key_bytes)))
        parts.append(key_bytes)
        parts.append(struct.pack("<i", len(val_bytes)))
        parts.append(val_bytes)

    # Determine unique inserted keys (for lookup generation)
    unique_keys = list(dict(insert_entries).keys())  # preserves last-wins order

    # Generate lookups
    lookup_keys: list[str] = []
    if all_present and not missing_fraction:
        # All lookups hit existing keys
        for _ in range(n_lookup):
            lookup_keys.append(rng.choice(unique_keys))
    else:
        # Some lookups are for missing keys
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

        # Shuffle to interleave hits and misses
        rng.shuffle(lookup_keys)

    # Write lookup count
    parts.append(struct.pack("<i", n_lookup))

    # Write each lookup
    for key in lookup_keys:
        key_bytes = key.encode("utf-8")
        parts.append(struct.pack("<i", len(key_bytes)))
        parts.append(key_bytes)

    return b"".join(parts)


def main() -> None:
    rng = random.Random(SEED)
    tests_dir = PROBLEM_DIR / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Define test cases
    test_configs = [
        # (n_insert, n_lookup, kwargs, description)
        (10, 10, {"all_present": True}, "10 inserts, 10 lookups (all present)"),
        (100, 100, {"all_present": True}, "100 inserts, 100 lookups (all present)"),
        (50, 100, {"all_present": False, "missing_fraction": 0.5},
         "50 inserts, 100 lookups (~50% missing)"),
        (100, 100, {"all_present": True, "duplicate_keys": True},
         "100 inserts with duplicates, 100 lookups"),
        (1000, 1000, {"all_present": True}, "1000 inserts, 1000 lookups"),
    ]

    # Generate test inputs
    test_inputs: list[bytes] = []
    for n_insert, n_lookup, kwargs, desc in test_configs:
        input_data = build_input(rng, n_insert, n_lookup, **kwargs)
        test_inputs.append(input_data)
        print(f"  Generated: {desc} ({len(input_data)} bytes)")

    # Compile C reference
    with tempfile.TemporaryDirectory() as tmpdir:
        binary = Path(tmpdir) / "hash_table_ref"
        subprocess.run(
            ["gcc", "-O2", "-o", str(binary),
             str(PROBLEM_DIR / "reference" / "solution.c")],
            check=True,
        )

        # Run C reference to produce expected outputs
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

        # Generate perf input: 100,000 inserts + 100,000 lookups (all present)
        perf_input = build_input(rng, 100_000, 100_000, all_present=True)
        (PROBLEM_DIR / "perf_input.bin").write_bytes(perf_input)
        print(f"  perf_input: 100k inserts + 100k lookups, {len(perf_input)} bytes")

    print(f"Generated {len(test_configs)} tests + perf input for hash_table")


if __name__ == "__main__":
    main()
