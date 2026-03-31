# Problem Bank

## Scope

**In scope:**
- Binary I/O problems with reference solutions in C, Rust, Python, TypeScript
- Test generation scripts that compile C reference and produce expected outputs
- Performance inputs for benchmarking
- Exact-match comparison of binary outputs

**Not in scope:**
- Floating-point tolerance comparison (handled by comparison.json mode)
- Problem difficulty ratings or curriculum ordering
- Agent training logic

## Data/Control Flow

```
scripts/generate_<problem>_tests.py
    │
    ├─ Generates test input .bin files (seeded RNG)
    ├─ Compiles C reference solution
    ├─ Runs C reference to produce expected output .bin files
    └─ Generates perf_input.bin for benchmarking
```

At runtime, the sandbox loads a problem's test inputs, runs the candidate solution,
and compares output against expected using the mode from `comparison.json`.

## Directory Structure

Each problem follows this layout:
```
problems/<name>/
    spec.md              # Problem description and I/O format
    comparison.json      # {"mode": "exact"} or {"mode": "tolerance", ...}
    perf_input.bin       # Large input for performance measurement
    reference/
        solution.c       # C reference (used to generate expected outputs)
        solution.rs      # Rust reference
        solution.py      # Python reference
        solution.ts      # TypeScript reference
    tests/
        input_0.bin      # Test input
        expected_0.bin   # Expected output (from C reference)
        ...
```

## Problems

### matmul
- **Description:** NxN matrix multiplication (float32)
- **Files:** problems/matmul/
- **Generator:** scripts/generate_matmul_tests.py
- **Test cases:** 5 sizes (2, 4, 8, 16, 32), perf size 1024

### hash_table
- **Description:** String key-value hash table insert + lookup
- **Files:** problems/hash_table/
- **Generator:** scripts/generate_hash_table_tests.py
- **Test cases:**
  - test_0: 10 inserts, 10 lookups (all present)
  - test_1: 100 inserts, 100 lookups (all present)
  - test_2: 50 inserts, 100 lookups (~50% missing)
  - test_3: 100 inserts with duplicate keys (last wins)
  - test_4: 1000 inserts, 1000 lookups
- **Perf input:** 100k inserts + 100k lookups

## Invariants

- All 4 language references must produce byte-identical output for all test cases
- C reference is the ground truth (generates expected outputs)
- Test generation is deterministic (seeded RNG)
- Binary I/O uses little-endian int32 for all lengths/counts
- For hash_table: duplicate key inserts use last-value-wins semantics
