import sys

import numpy as np


def main() -> None:
    data = sys.stdin.buffer.read()
    offset = 0

    w = int(np.frombuffer(data[offset : offset + 4], dtype=np.int32)[0])
    offset += 4
    h = int(np.frombuffer(data[offset : offset + 4], dtype=np.int32)[0])
    offset += 4
    iters = int(np.frombuffer(data[offset : offset + 4], dtype=np.int32)[0])
    offset += 4

    n = w * h
    old = np.frombuffer(data[offset : offset + n * 4], dtype=np.float32).copy()
    new = old.copy()

    five = np.float32(5.0)

    for _iter in range(iters):
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                new[i * w + j] = (
                    old[(i - 1) * w + j]
                    + old[(i + 1) * w + j]
                    + old[i * w + (j - 1)]
                    + old[i * w + (j + 1)]
                    + old[i * w + j]
                ) / five
        # Swap old and new
        old, new = new, old
        # Copy boundary into new for next iteration
        if _iter < iters - 1:
            new[:w] = old[:w]
            new[(h - 1) * w : h * w] = old[(h - 1) * w : h * w]
            for i in range(1, h - 1):
                new[i * w] = old[i * w]
                new[i * w + w - 1] = old[i * w + w - 1]

    sys.stdout.buffer.write(old.tobytes())


if __name__ == "__main__":
    main()
