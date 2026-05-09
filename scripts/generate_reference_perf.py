#!/usr/bin/env python3
"""Measure reference perf counters for each problem's C reference solution.

For every problem under ``problems/``:

1. Compile ``reference/solution.c`` inside the sandbox (gcc -O2 -lm).
2. For each size declared in ``sizes.toml``, run ``perf stat`` on the binary
   with ``perf_inputs/<label>.bin`` on stdin, recording cycles/instructions
   and wall-clock.
3. Write ``reference_perf/c_<profile>_<label>.json`` per size.

The profile name is auto-detected from the CPU vendor (e.g. ``amd_zen``),
matching how ``load_problem_with_reference`` looks up baselines.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from dataclasses import fields as dc_fields
from pathlib import Path

# Make ``src/`` importable when run as a script without installation.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from perf_optimize.config import SandboxConfig  # noqa: E402
from perf_optimize.languages import Language  # noqa: E402
from perf_optimize.problems import _load_sizes_toml  # noqa: E402
from perf_optimize.sandbox import PerfSandbox  # noqa: E402
from perf_optimize.types import CompilationFailure, PerfCounters  # noqa: E402

PROBLEMS = ("matmul", "hash_table", "nbody", "sort", "stencil")


def _counters_to_json(counters: PerfCounters) -> dict[str, float | None]:
    """Serialize PerfCounters to JSON, preserving null for unmapped counters."""
    return {f.name: getattr(counters, f.name) for f in dc_fields(counters)}


async def _measure_one_size(
    sandbox: PerfSandbox,
    binary_path: Path,
    perf_input_path: Path,
) -> tuple[PerfCounters, float]:
    """Run perf on a single sized input; return counters + wall-clock ms."""
    t0 = time.perf_counter()
    counters = await sandbox.measure_only(binary_path, perf_input_path)
    wall_clock_ms = (time.perf_counter() - t0) * 1000.0
    return counters, wall_clock_ms


async def _measure_problem(
    sandbox: PerfSandbox, problem_dir: Path
) -> list[tuple[str, int, PerfCounters, float]]:
    """Compile reference once and measure each declared size."""
    source = (problem_dir / "reference" / "solution.c").read_text()
    sizes = _load_sizes_toml(problem_dir)

    perf_dir = problem_dir / "perf_inputs"
    if not perf_dir.is_dir():
        raise FileNotFoundError(
            f"{perf_dir} missing — run the test generator first"
        )

    compilation, work_dir = await sandbox.compile_only(source)
    try:
        if isinstance(compilation, CompilationFailure):
            raise RuntimeError(
                f"failed to compile reference for {problem_dir.name}: "
                f"{compilation.stderr[:500]}"
            )
        binary = Path(work_dir) / sandbox._config.language.output_file
        results: list[tuple[str, int, PerfCounters, float]] = []
        for spec in sizes:
            perf_input_path = perf_dir / f"{spec.label}.bin"
            if not perf_input_path.exists():
                raise FileNotFoundError(
                    f"{perf_input_path} missing — run the test generator first"
                )
            counters, ms = await _measure_one_size(sandbox, binary, perf_input_path)
            results.append((spec.label, spec.n, counters, ms))
        return results
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def _measure_heldout(
    sandbox: PerfSandbox, problem_dir: Path,
) -> tuple[PerfCounters, float]:
    """Measure the C reference on the held-out perf input.

    Held-out perf input lives at ``perf_inputs_heldout/large.bin``. Returns
    ``(counters, wall_clock_ms)``. Sized comparably to the in-dist large perf
    input so cycle counts can be compared 1:1 with the in-dist baseline.
    """
    import asyncio
    import time

    source = (problem_dir / "reference" / "solution.c").read_text()
    perf_input = problem_dir / "perf_inputs_heldout" / "large.bin"
    if not perf_input.exists():
        raise FileNotFoundError(
            f"{perf_input} missing — run the test generator to produce held-out data",
        )

    compilation, work_dir = await sandbox.compile_only(source)
    try:
        if isinstance(compilation, CompilationFailure):
            raise RuntimeError(
                f"failed to compile reference for {problem_dir.name} (heldout): "
                f"{compilation.stderr[:500]}",
            )
        binary = Path(work_dir) / sandbox._config.language.output_file
        t0 = time.monotonic()
        counters = await sandbox.measure_only(binary, perf_input)
        wall_ms = (time.monotonic() - t0) * 1000.0
        # Yield once so the event loop drains -- keeps this script async-friendly.
        await asyncio.sleep(0)
        return counters, wall_ms
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def _generate_heldout(
    sandbox: PerfSandbox, problem_dir: Path, profile: str,
) -> None:
    """Persist held-out reference perf for one problem.

    Kept as a separate function so the sibling per-size rewrite can interleave
    a per-size loop with this held-out pass without textual conflicts.
    """
    counters, wall_ms = await _measure_heldout(sandbox, problem_dir)
    out_dir = problem_dir / "reference_perf"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"c_{profile}_heldout.json"

    payload: dict[str, float | None] = _counters_to_json(counters)
    payload["wall_clock_ms"] = wall_ms
    out_file.write_text(json.dumps(payload, indent=2) + "\n")
    summary = ", ".join(
        f"{k}={int(v):,}"
        for k, v in payload.items()
        if v is not None and k != "wall_clock_ms"
    )
    print(f"    heldout: {summary}, wall={wall_ms:.1f}ms")
    print(f"      -> {out_file.relative_to(REPO_ROOT)}")


async def main() -> None:
    sandbox = PerfSandbox(SandboxConfig.from_env(language=Language.C))
    await sandbox.check_prerequisites()

    profile = sandbox._config.hardware_profile.name
    print(f"Hardware profile: c_{profile}")

    problems_root = REPO_ROOT / "problems"
    for name in PROBLEMS:
        problem_dir = problems_root / name
        results = await _measure_problem(sandbox, problem_dir)

        out_dir = problem_dir / "reference_perf"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Remove any legacy `c_<profile>.json` (single-size reference).
        legacy = out_dir / f"c_{profile}.json"
        if legacy.exists():
            legacy.unlink()
            print(f"  removed legacy {legacy.relative_to(REPO_ROOT)}")

        for label, n, counters, wall_clock_ms in results:
            out_file = out_dir / f"c_{profile}_{label}.json"
            payload: dict[str, float | None] = _counters_to_json(counters)
            payload["wall_clock_ms"] = wall_clock_ms

            out_file.write_text(json.dumps(payload, indent=2) + "\n")

            summary = ", ".join(
                f"{k}={int(v):,}"
                for k, v in payload.items()
                if v is not None and k not in ("wall_clock_ms",)
            )
            print(
                f"  {name}[{label}, n={n}]: {summary}, "
                f"wall_clock_ms={wall_clock_ms:.1f}"
            )
            print(f"    -> {out_file.relative_to(REPO_ROOT)}")

        # Held-out reference perf. Kept as a separate call so the sibling
        # per-size rewrite can interleave its per-size loop above this line.
        await _generate_heldout(sandbox, problem_dir, profile)


if __name__ == "__main__":
    asyncio.run(main())
