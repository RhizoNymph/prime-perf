"""End-to-end integration test for the held-out cleanup pass.

Spins up a tiny `add_one` problem with held-out tests and a held-out perf
input, injects a known-correct candidate into the rollout state, and invokes
the held-out cleanup directly. Asserts the state is populated.

Requires bwrap, gcc, perf, taskset.
"""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

verifiers = pytest.importorskip("verifiers", reason="verifiers SDK not installed")

from perf_optimize.env import PerfOptimizeEnv  # noqa: E402
from perf_optimize.languages import Language  # noqa: E402

CORRECT_C_SOURCE = """\
#include <stdio.h>
#include <stdint.h>
int main() {
    int32_t x;
    while (fread(&x, sizeof(x), 1, stdin) == 1) {
        x += 1;
        fwrite(&x, sizeof(x), 1, stdout);
    }
    return 0;
}
"""


def _make_problem_dir(tmp_path: Path) -> Path:
    """Create a tiny C problem with held-out fixtures for the cleanup pass."""
    problems = tmp_path / "problems"
    p = problems / "add_one"
    p.mkdir(parents=True)

    (p / "spec.md").write_text(
        "# Add One\n\nRead int32s from stdin, add 1, write to stdout.\n"
    )
    (p / "comparison.json").write_text(json.dumps({"mode": "exact"}))

    ref = p / "reference"
    ref.mkdir()
    (ref / "solution.c").write_text(CORRECT_C_SOURCE)

    # In-dist tests
    tests = p / "tests"
    tests.mkdir()
    (tests / "input_0.bin").write_bytes(struct.pack("<i", 5))
    (tests / "expected_0.bin").write_bytes(struct.pack("<i", 6))

    (p / "perf_input.bin").write_bytes(struct.pack("<i", 42))

    # Reference perf baseline (required by env warning logic)
    from perf_optimize.counters import detect_profile

    profile = detect_profile()
    perf_dir = p / "reference_perf"
    perf_dir.mkdir()
    baseline = {
        "cycles": 500_000.0,
        "instructions": 1_000_000.0,
    }
    (perf_dir / f"c_{profile.name}.json").write_text(json.dumps(baseline))

    # Held-out tests (different distribution: negative inputs)
    held = p / "tests_heldout"
    held.mkdir()
    (held / "input_0.bin").write_bytes(struct.pack("<i", -10))
    (held / "expected_0.bin").write_bytes(struct.pack("<i", -9))
    (held / "input_1.bin").write_bytes(struct.pack("<i", 999))
    (held / "expected_1.bin").write_bytes(struct.pack("<i", 1000))

    held_perf = p / "perf_inputs_heldout"
    held_perf.mkdir()
    # Same shape as in-dist perf input, but a different value
    (held_perf / "large.bin").write_bytes(struct.pack("<i", 7))

    # Held-out reference perf
    (perf_dir / f"c_{profile.name}_heldout.json").write_text(json.dumps({
        "cycles": 600_000.0,
        "instructions": 1_100_000.0,
        "wall_clock_ms": 5.0,
    }))

    return problems


@pytest.mark.integration
@pytest.mark.asyncio
async def test_heldout_pass_populates_state(tmp_path: Path) -> None:
    problems_dir = _make_problem_dir(tmp_path)

    env = PerfOptimizeEnv(
        language=Language.C,
        max_turns=3,
        problems_dir=problems_dir,
    )

    # Build a state matching what setup_state would produce, then add the
    # tracking fields the cleanup pass expects.
    row = env.dataset[0]
    info = row["info"]

    state: dict = {"info": info, "trajectory": []}
    state = await env.setup_state(state)

    # Simulate that the rollout produced one correct submission with this
    # source as the best candidate.
    state["correct_submissions"] = 1
    state["best_candidate_source"] = CORRECT_C_SOURCE

    # Run the held-out cleanup directly.
    await env._run_heldout_pass(state)

    assert state["heldout_test_total"] == 2
    assert state["heldout_test_passed_count"] == 2
    assert state["heldout_test_passed"] is True
    assert state["heldout_best_perf"] is not None
    assert "cycles" in state["heldout_best_perf"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_heldout_pass_skipped_without_correct_submission(
    tmp_path: Path,
) -> None:
    """Cleanup must early-exit when no correct submission was produced."""
    problems_dir = _make_problem_dir(tmp_path)

    env = PerfOptimizeEnv(
        language=Language.C,
        max_turns=3,
        problems_dir=problems_dir,
    )

    row = env.dataset[0]
    info = row["info"]

    state: dict = {"info": info, "trajectory": []}
    state = await env.setup_state(state)

    state["correct_submissions"] = 0
    state["best_candidate_source"] = None

    await env._run_heldout_pass(state)

    # Did not run -> result fields stay None / 0
    assert state["heldout_test_passed"] is None
    assert state["heldout_test_passed_count"] == 0
    assert state["heldout_best_perf"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_heldout_pass_idempotent(tmp_path: Path) -> None:
    """Calling the cleanup twice must not re-run the pipeline."""
    problems_dir = _make_problem_dir(tmp_path)

    env = PerfOptimizeEnv(
        language=Language.C,
        max_turns=3,
        problems_dir=problems_dir,
    )

    row = env.dataset[0]
    info = row["info"]
    state: dict = {"info": info, "trajectory": []}
    state = await env.setup_state(state)

    state["correct_submissions"] = 1
    state["best_candidate_source"] = CORRECT_C_SOURCE

    await env._run_heldout_pass(state)
    first_perf = state["heldout_best_perf"]
    assert first_perf is not None

    # Mutate the source so a re-run would change the result -- but it shouldn't.
    state["best_candidate_source"] = "this would not compile"
    await env._run_heldout_pass(state)
    assert state["heldout_best_perf"] is first_perf
