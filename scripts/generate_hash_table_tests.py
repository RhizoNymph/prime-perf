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

# Held-out distribution choice for hash_table:
#   In-dist tests use uniform random key sampling. Held-out switches to
#   *Zipf-distributed* key access (heavy collisions on a small head set) and
#   sequential numeric keys. These exercise probe-chain / cache hit-rate
#   behavior very differently from uniform key access, making it harder to
#   overfit to incidental properties of uniform-random ASCII keys.
HELDOUT_SEED = 1337


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


def _zipf_indices(rng: random.Random, num_unique: int, num_samples: int, s: float = 1.5) -> list[int]:
    """Sample ``num_samples`` indices in [0, num_unique) from a Zipf(s) distribution."""
    weights = [1.0 / ((i + 1) ** s) for i in range(num_unique)]
    total = sum(weights)
    cdf: list[float] = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cdf.append(acc)

    indices: list[int] = []
    for _ in range(num_samples):
        r = rng.random()
        # Linear scan is fine for our small num_unique; bisect would be tidier
        # but adds an import and is unnecessary at these sizes.
        for i, c in enumerate(cdf):
            if r <= c:
                indices.append(i)
                break
        else:
            indices.append(num_unique - 1)
    return indices


def build_heldout_zipf(
    rng: random.Random, n_insert: int, n_lookup: int, *, zipf_s: float = 1.5,
) -> bytes:
    """Held-out: uniform random inserts, but lookups follow a Zipf distribution.

    Heavy-tail lookup pattern stresses cache locality and probe-chain length
    completely differently from uniform-random lookups.
    """
    parts: list[bytes] = []
    inserts: list[tuple[str, str]] = []
    for _ in range(n_insert):
        key = rand_string(rng, 5, 20)
        val = rand_string(rng, 10, 50)
        inserts.append((key, val))

    parts.append(struct.pack("<i", n_insert))
    for key, val in inserts:
        kb, vb = key.encode("utf-8"), val.encode("utf-8")
        parts.append(struct.pack("<i", len(kb)))
        parts.append(kb)
        parts.append(struct.pack("<i", len(vb)))
        parts.append(vb)

    unique_keys = list(dict(inserts).keys())
    sampled = _zipf_indices(rng, len(unique_keys), n_lookup, s=zipf_s)
    lookup_keys = [unique_keys[i] for i in sampled]

    parts.append(struct.pack("<i", n_lookup))
    for key in lookup_keys:
        kb = key.encode("utf-8")
        parts.append(struct.pack("<i", len(kb)))
        parts.append(kb)

    return b"".join(parts)


def build_heldout_sequential(rng: random.Random, n_insert: int, n_lookup: int) -> bytes:
    """Held-out: sequential numeric keys (e.g. ``key_000001``).

    Sequential keys hit the same hash buckets in a deterministic pattern
    very different from uniform random ASCII strings, exposing any
    optimization that assumed uniform hash dispersion.
    """
    parts: list[bytes] = []
    inserts: list[tuple[str, str]] = []
    for i in range(n_insert):
        key = f"key_{i:08d}"
        val = rand_string(rng, 10, 50)
        inserts.append((key, val))

    parts.append(struct.pack("<i", n_insert))
    for key, val in inserts:
        kb, vb = key.encode("utf-8"), val.encode("utf-8")
        parts.append(struct.pack("<i", len(kb)))
        parts.append(kb)
        parts.append(struct.pack("<i", len(vb)))
        parts.append(vb)

    unique_keys = [k for k, _ in inserts]
    lookup_keys = [unique_keys[i % len(unique_keys)] for i in range(n_lookup)]
    rng.shuffle(lookup_keys)

    parts.append(struct.pack("<i", n_lookup))
    for key in lookup_keys:
        kb = key.encode("utf-8")
        parts.append(struct.pack("<i", len(kb)))
        parts.append(kb)
    return b"".join(parts)


def main() -> None:
    rng = random.Random(SEED)
    tests_dir = PROBLEM_DIR / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    # Define test cases
    # Keep first 5 identical to prior version so existing tests are preserved;
    # add 7 more covering all-missing, asymmetric insert/lookup ratios, and larger sizes.
    test_configs = [
        # (n_insert, n_lookup, kwargs, description)
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

        # Held-out tests + perf input.
        # Distribution: Zipf-skewed lookups and sequential numeric keys
        # rather than uniform-random ASCII keys.
        held_rng = random.Random(HELDOUT_SEED)
        held_tests_dir = PROBLEM_DIR / "tests_heldout"
        held_tests_dir.mkdir(parents=True, exist_ok=True)
        held_perf_dir = PROBLEM_DIR / "perf_inputs_heldout"
        held_perf_dir.mkdir(parents=True, exist_ok=True)

        heldout_specs = [
            ("zipf_small", build_heldout_zipf(held_rng, 100, 200)),
            ("zipf_medium", build_heldout_zipf(held_rng, 500, 1000, zipf_s=1.2)),
            ("sequential_small", build_heldout_sequential(held_rng, 200, 400)),
            ("sequential_medium", build_heldout_sequential(held_rng, 1000, 2000)),
        ]
        for i, (label, input_data) in enumerate(heldout_specs):
            (held_tests_dir / f"input_{i}.bin").write_bytes(input_data)
            result = subprocess.run(
                [str(binary)], input=input_data, capture_output=True, timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"C reference failed for held-out {label}: "
                    f"{result.stderr.decode()}"
                )
            (held_tests_dir / f"expected_{i}.bin").write_bytes(result.stdout)
            print(
                f"  heldout_test_{i}: {label}, input={len(input_data)} bytes, "
                f"output={len(result.stdout)} bytes"
            )

        # Held-out perf input: Zipf-distributed lookups, sized comparably to
        # the in-dist perf input.
        held_perf_input = build_heldout_zipf(held_rng, 100_000, 100_000, zipf_s=1.2)
        (held_perf_dir / "large.bin").write_bytes(held_perf_input)
        print(
            f"  heldout_perf large.bin: 100k inserts + 100k Zipf lookups, "
            f"{len(held_perf_input)} bytes"
        )

    print(f"Generated {len(test_configs)} tests + perf input for hash_table")


if __name__ == "__main__":
    main()
