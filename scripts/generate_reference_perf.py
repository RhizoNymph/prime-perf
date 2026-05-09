#!/usr/bin/env python3
"""Measure reference perf counters for each problem's C reference solution.

For every problem under ``problems/``:

1. Compile ``reference/solution.c`` inside the sandbox (gcc -O2 -lm).
2. Run ``perf stat`` on the binary with ``perf_input.bin`` on stdin.
3. Write ``reference_perf/c_<profile>.json`` with the measured counters.

The profile name is auto-detected from the CPU vendor (e.g. ``amd_zen``),
matching how ``load_problem_with_reference`` looks up baselines.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

# Make ``src/`` importable when run as a script without installation.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from perf_optimize.config import SandboxConfig  # noqa: E402
from perf_optimize.languages import Language  # noqa: E402
from perf_optimize.sandbox import PerfSandbox  # noqa: E402
from perf_optimize.types import CompilationFailure, PerfCounters  # noqa: E402

PROBLEMS = ("matmul", "hash_table", "nbody", "sort", "stencil")


def _counters_to_json(counters: PerfCounters) -> dict[str, float | None]:
    """Serialize PerfCounters to JSON, preserving null for unmapped counters."""
    return {f.name: getattr(counters, f.name) for f in dc_fields(counters)}


async def _measure_problem(sandbox: PerfSandbox, problem_dir: Path) -> PerfCounters:
    source = (problem_dir / "reference" / "solution.c").read_text()
    perf_input = problem_dir / "perf_input.bin"
    if not perf_input.exists():
        raise FileNotFoundError(f"{perf_input} missing — run the test generator first")

    compilation, work_dir = await sandbox.compile_only(source)
    try:
        if isinstance(compilation, CompilationFailure):
            raise RuntimeError(
                f"failed to compile reference for {problem_dir.name}: {compilation.stderr[:500]}"
            )
        binary = Path(work_dir) / sandbox._config.language.output_file
        return await sandbox.measure_only(binary, perf_input)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def main() -> None:
    sandbox = PerfSandbox(SandboxConfig.from_env(language=Language.C))
    await sandbox.check_prerequisites()

    profile = sandbox._config.hardware_profile.name
    print(f"Hardware profile: c_{profile}")

    problems_root = REPO_ROOT / "problems"
    for name in PROBLEMS:
        problem_dir = problems_root / name
        counters = await _measure_problem(sandbox, problem_dir)

        out_dir = problem_dir / "reference_perf"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"c_{profile}.json"

        payload = _counters_to_json(counters)
        out_file.write_text(json.dumps(payload, indent=2) + "\n")

        summary = ", ".join(
            f"{k}={int(v):,}" for k, v in payload.items() if v is not None
        )
        print(f"  {name}: {summary}")
        print(f"    -> {out_file.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
