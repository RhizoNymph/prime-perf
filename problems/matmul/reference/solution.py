import sys

import numpy as np


def main() -> None:
    data = sys.stdin.buffer.read()
    offset = 0

    n = int(np.frombuffer(data[offset : offset + 4], dtype=np.int32)[0])
    offset += 4

    floats_per_matrix = n * n
    a = np.frombuffer(data[offset : offset + floats_per_matrix * 4], dtype=np.float32).copy()
    offset += floats_per_matrix * 4
    b = np.frombuffer(data[offset : offset + floats_per_matrix * 4], dtype=np.float32).copy()

    c = np.zeros(n * n, dtype=np.float32)

    # Naive i,j,k loop using float32 arithmetic (matches C's accumulation order)
    for i in range(n):
        for j in range(n):
            s = np.float32(0.0)
            for k in range(n):
                s += a[i * n + k] * b[k * n + j]
            c[i * n + j] = s

    sys.stdout.buffer.write(c.tobytes())


if __name__ == "__main__":
    main()
