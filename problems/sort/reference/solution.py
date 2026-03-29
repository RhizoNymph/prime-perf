import math
import struct
import sys

import numpy as np


def sort_key(val: np.float32) -> tuple[int, float, int]:
    """Key function for sorting float32 values with NaN/signed-zero handling.

    Returns a tuple (nan_flag, value, sign_flag) where:
    - nan_flag: 0 for normal values, 1 for NaN (pushes NaN to end)
    - value: the float value (0.0 for NaN, doesn't matter)
    - sign_flag: 0 for negative sign, 1 for positive (puts -0.0 before +0.0)
    """
    f = float(val)
    if math.isnan(f):
        return (1, 0.0, 0)
    sign_positive = 0 if math.copysign(1.0, f) < 0 else 1
    return (0, f, sign_positive)


def main() -> None:
    data = sys.stdin.buffer.read()
    offset = 0

    n = int(np.frombuffer(data[offset : offset + 4], dtype=np.int32)[0])
    offset += 4

    arr = np.frombuffer(data[offset : offset + n * 4], dtype=np.float32).copy()

    # Convert to list, sort with custom key, convert back
    values = list(arr)
    values.sort(key=sort_key)

    result = np.array(values, dtype=np.float32)
    sys.stdout.buffer.write(result.tobytes())


if __name__ == "__main__":
    main()
