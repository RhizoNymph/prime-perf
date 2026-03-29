import struct
import sys


def main() -> None:
    data = sys.stdin.buffer.read()
    offset = 0

    (n_insert,) = struct.unpack_from("<i", data, offset)
    offset += 4

    # Use a plain dict with bytes keys for simplicity (this is the naive reference)
    table: dict[bytes, bytes] = {}

    for _ in range(n_insert):
        (key_len,) = struct.unpack_from("<i", data, offset)
        offset += 4
        key = data[offset : offset + key_len]
        offset += key_len

        (val_len,) = struct.unpack_from("<i", data, offset)
        offset += 4
        val = data[offset : offset + val_len]
        offset += val_len

        table[key] = val  # last write wins for duplicates

    (n_lookup,) = struct.unpack_from("<i", data, offset)
    offset += 4

    out_parts: list[bytes] = []

    for _ in range(n_lookup):
        (key_len,) = struct.unpack_from("<i", data, offset)
        offset += 4
        key = data[offset : offset + key_len]
        offset += key_len

        val = table.get(key)
        if val is not None:
            out_parts.append(struct.pack("<i", 1))
            out_parts.append(struct.pack("<i", len(val)))
            out_parts.append(val)
        else:
            out_parts.append(struct.pack("<i", 0))

    sys.stdout.buffer.write(b"".join(out_parts))


if __name__ == "__main__":
    main()
