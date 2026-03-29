#!/usr/bin/env python3
"""Generate binary input for matmul test programs.

Usage: python gen_matmul_input.py <N> <output_file> [--seed SEED]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matmul binary input")
    parser.add_argument("n", type=int, help="Matrix dimension")
    parser.add_argument("output", type=Path, help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    a = rng.random((args.n, args.n), dtype=np.float32)
    b = rng.random((args.n, args.n), dtype=np.float32)

    with open(args.output, "wb") as f:
        f.write(struct.pack("i", args.n))
        f.write(a.tobytes())
        f.write(b.tobytes())

    print(f"Generated {args.output}: N={args.n}, {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
