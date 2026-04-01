"""Integration test for PerfOptimizeEnv with a mock OpenAI client.

Requires: bwrap, gcc, perf, taskset. Marked as integration test.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice, ChatCompletionMessage

verifiers = pytest.importorskip("verifiers", reason="verifiers SDK not installed")

from perf_optimize.env import PerfOptimizeEnv  # noqa: E402
from perf_optimize.languages import Language  # noqa: E402


def _make_problem_dir(tmp_path: Path) -> Path:
    """Create a minimal C problem for integration testing."""
    problems = tmp_path / "problems"
    p = problems / "add_one"
    p.mkdir(parents=True)

    # Simple problem: read an int32, add 1, write it back
    (p / "spec.md").write_text(
        "# Add One\n\nRead a 32-bit integer from stdin, add 1, write to stdout.\n"
    )
    (p / "comparison.json").write_text(json.dumps({"mode": "exact"}))

    # Reference solution
    ref = p / "reference"
    ref.mkdir()
    (ref / "solution.c").write_text("""\
#include <stdio.h>
#include <stdint.h>
int main() {
    int32_t x;
    fread(&x, sizeof(x), 1, stdin);
    x += 1;
    fwrite(&x, sizeof(x), 1, stdout);
    return 0;
}
""")

    # Test cases: input 5 → expect 6, input 0 → expect 1
    import struct

    tests = p / "tests"
    tests.mkdir()
    (tests / "input_0.bin").write_bytes(struct.pack("<i", 5))
    (tests / "expected_0.bin").write_bytes(struct.pack("<i", 6))
    (tests / "input_1.bin").write_bytes(struct.pack("<i", 0))
    (tests / "expected_1.bin").write_bytes(struct.pack("<i", 1))

    # Perf input (same format)
    (p / "perf_input.bin").write_bytes(struct.pack("<i", 42))

    # Reference perf baselines (required by PerfOptimizeEnv validation).
    # Use the hardware profile detected at runtime.
    from perf_optimize.counters import detect_profile

    profile = detect_profile()
    perf_dir = p / "reference_perf"
    perf_dir.mkdir()
    baseline = {
        "cycles": 500_000.0,
        "instructions": 1_000_000.0,
        "cache_misses": 1_000.0,
        "l1_dcache_load_misses": 2_000.0,
        "branch_misses": 500.0,
    }
    (perf_dir / f"c_{profile.name}.json").write_text(json.dumps(baseline))

    return problems


def _mock_chat_completion(content: str) -> ChatCompletion:
    """Create a real ChatCompletion for use with the verifiers framework."""
    return ChatCompletion(
        id="test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                ),
            )
        ],
        created=0,
        model="test-model",
        object="chat.completion",
    )


def _rollout_input(env: PerfOptimizeEnv, idx: int = 0) -> dict:
    """Build a RolloutInput dict from the env's dataset row."""
    row = env.dataset[idx]
    return {
        "prompt": row["prompt"],
        "example_id": row["example_id"],
        "task": row["task"],
        "answer": row.get("answer", ""),
        "info": row["info"],
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_env_rollout_correct_submission(tmp_path: Path) -> None:
    """Test a rollout where the agent submits correct (identical) code then submits."""
    problems_dir = _make_problem_dir(tmp_path)

    env = PerfOptimizeEnv(
        language=Language.C,
        max_turns=3,
        problems_dir=problems_dir,
    )

    # The agent's response: submit the same correct code
    correct_code = """\
#include <stdio.h>
#include <stdint.h>
int main() {
    int32_t x;
    fread(&x, sizeof(x), 1, stdin);
    x += 1;
    fwrite(&x, sizeof(x), 1, stdout);
    return 0;
}"""

    # Mock client: first turn returns code, second turn submits
    mock_client = AsyncMock()
    responses = [
        _mock_chat_completion(f'<code lang="c">\n{correct_code}\n</code>'),
        _mock_chat_completion("<submit/>"),
    ]
    mock_client.chat.completions.create = AsyncMock(side_effect=responses)

    rollout_input = _rollout_input(env)
    state = await env.rollout(rollout_input, mock_client, "test-model")

    # Verify state tracking
    assert state["correct_submissions"] >= 1
    assert state["compile_failures"] == 0
    assert state["submitted"] is True
    assert state["best_perf_dict"] is not None
    assert "cycles" in state["best_perf_dict"]

    # Completion should have messages
    assert state["completion"] is not None
    assert len(state["completion"]) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_env_rollout_compile_failure_then_fix(tmp_path: Path) -> None:
    """Test a rollout where first submission fails to compile, second succeeds."""
    problems_dir = _make_problem_dir(tmp_path)

    env = PerfOptimizeEnv(
        language=Language.C,
        max_turns=5,
        problems_dir=problems_dir,
    )

    bad_code = "this is not valid C code!!!"
    good_code = """\
#include <stdio.h>
#include <stdint.h>
int main() {
    int32_t x;
    fread(&x, sizeof(x), 1, stdin);
    x += 1;
    fwrite(&x, sizeof(x), 1, stdout);
    return 0;
}"""

    mock_client = AsyncMock()
    responses = [
        _mock_chat_completion(f'<code lang="c">\n{bad_code}\n</code>'),
        _mock_chat_completion(f'<code lang="c">\n{good_code}\n</code>'),
        _mock_chat_completion("<submit/>"),
    ]
    mock_client.chat.completions.create = AsyncMock(side_effect=responses)

    rollout_input = _rollout_input(env)
    state = await env.rollout(rollout_input, mock_client, "test-model")

    assert state["compile_failures"] == 1
    assert state["correct_submissions"] >= 1
    assert state["submitted"] is True
