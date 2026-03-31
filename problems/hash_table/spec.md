# Hash Table

String key-value hash table: insert key-value pairs, then look up keys.

## I/O Format (binary, little-endian)

**Input** (stdin):
- `int32 N_insert` — number of key-value pairs to insert
- For each insert:
  - `int32 key_len` — key byte length
  - `key_len bytes` — key data (UTF-8 string)
  - `int32 val_len` — value byte length
  - `val_len bytes` — value data (UTF-8 string)
- `int32 N_lookup` — number of lookups to perform
- For each lookup:
  - `int32 key_len` — key byte length
  - `key_len bytes` — key data

**Output** (stdout):
For each lookup:
- `int32 found` — 1 if key found, 0 if not
- If found:
  - `int32 val_len` — value byte length
  - `val_len bytes` — value data

## Constraints
- 1 ≤ N_insert ≤ 200,000
- 1 ≤ N_lookup ≤ 200,000
- Key lengths: 5–20 bytes (alphanumeric ASCII)
- Value lengths: 10–50 bytes (alphanumeric ASCII)
- If a key is inserted multiple times, the last value wins

## Notes
The naive implementation uses a simple chaining hash table with linked lists.
Optimization opportunities include better hash functions, open addressing,
SIMD-accelerated string comparison, cache-friendly memory layout, and
pre-allocated memory pools.
