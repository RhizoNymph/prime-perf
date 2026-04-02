"""Tests for PerfOptimizeEnv — code extraction, completion detection, state setup."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, patch

import pytest

verifiers = pytest.importorskip("verifiers", reason="verifiers SDK not installed")

from perf_optimize.env import _extract_code, _has_submit  # noqa: E402


class TestExtractCode:
    def test_basic_code_block(self) -> None:
        text = '<code lang="c">int main() { return 0; }</code>'
        assert _extract_code(text) == "int main() { return 0; }"

    def test_code_block_no_lang(self) -> None:
        text = "<code>int main() { return 0; }</code>"
        assert _extract_code(text) == "int main() { return 0; }"

    def test_multiline_code(self) -> None:
        text = '<code lang="c">\nint main() {\n    return 0;\n}\n</code>'
        result = _extract_code(text)
        assert result is not None
        assert "int main()" in result
        assert "return 0;" in result

    def test_no_code_block(self) -> None:
        text = "I think we should optimize the loop."
        assert _extract_code(text) is None

    def test_code_with_surrounding_text(self) -> None:
        text = 'Here is my solution:\n<code lang="rust">fn main() {}</code>\nDone.'
        assert _extract_code(text) == "fn main() {}"

    def test_code_with_python_lang(self) -> None:
        text = '<code lang="python">print("hello")</code>'
        assert _extract_code(text) == 'print("hello")'

    def test_strips_whitespace(self) -> None:
        text = "<code>   int x = 1;   </code>"
        assert _extract_code(text) == "int x = 1;"

    def test_literal_close_tag_in_source(self) -> None:
        """Code containing literal </code> (e.g. printf) must not truncate."""
        text = '<code lang="c">printf("</code>");\nreturn 0;</code>'
        result = _extract_code(text)
        assert result is not None
        assert 'printf("</code>");' in result
        assert "return 0;" in result

    def test_multiple_code_blocks_takes_last(self) -> None:
        """When multiple code blocks exist, the last one is extracted."""
        text = '<code lang="c">int first() {}</code>\nHere is v2:\n<code lang="c">int second() {}</code>'
        result = _extract_code(text)
        assert result == "int second() {}"

    def test_empty_code_block_returns_none(self) -> None:
        """An empty code block returns None."""
        text = "<code></code>"
        assert _extract_code(text) is None

    def test_whitespace_only_code_block_returns_none(self) -> None:
        """A whitespace-only code block returns None."""
        text = "<code>   </code>"
        assert _extract_code(text) is None


class TestHasSubmit:
    def test_submit_on_own_line(self) -> None:
        assert _has_submit("I'm done.\n<submit/>")

    def test_submit_with_surrounding_whitespace(self) -> None:
        assert _has_submit("I'm done.\n  <submit/>  \nExtra text.")

    def test_submit_tag_with_space(self) -> None:
        assert _has_submit("I'm done.\n<submit />")

    def test_no_submit(self) -> None:
        assert not _has_submit("Here is my code.")

    def test_submit_inline_in_prose_does_not_match(self) -> None:
        assert not _has_submit("I'll use <submit/> after one more iteration.")

    def test_submit_in_backticks_does_not_match(self) -> None:
        assert not _has_submit("Use `<submit/>` when ready.")

    def test_submit_inside_code_block_does_not_match(self) -> None:
        text = '<code lang="c">// <submit/>\nint main() {}</code>'
        assert not _has_submit(text)

    def test_submit_on_own_line_inside_code_block_does_not_match(self) -> None:
        text = '<code lang="c">\n<submit/>\nint main() {}</code>'
        assert not _has_submit(text)

    def test_submit_after_code_block(self) -> None:
        text = '<code lang="c">int main() {}</code>\n<submit/>'
        assert _has_submit(text)

    def test_submit_between_code_blocks_detected(self) -> None:
        """A <submit/> between two code blocks is outside both and should match."""
        text = "<code>x</code>\n<submit/>\n<code>y</code>"
        assert _has_submit(text)

    def test_submit_in_markdown_fence_does_not_match(self) -> None:
        """A <submit/> inside a markdown fenced code block should not match."""
        text = "Here is how to submit:\n```\n<submit/>\n```\nThat's it."
        assert not _has_submit(text)


class TestSetupState:
    @pytest.mark.asyncio
    async def test_decodes_test_data(self) -> None:
        """setup_state should decode base64 test data from info."""
        from perf_optimize.env import PerfOptimizeEnv

        test_input = b"\x01\x02\x03"
        expected_output = b"\x04\x05\x06"
        perf_input = b"\xff" * 10

        state = {
            "info": {
                "test_inputs": [base64.b64encode(test_input).decode()],
                "expected_outputs": [base64.b64encode(expected_output).decode()],
                "perf_input": base64.b64encode(perf_input).decode(),
                "comparison": "exact",
                "tolerance": None,
                "reference_perf": {"cycles": 1000.0, "instructions": 2000.0},
            },
            "prompt": [],
            "completion": [],
            "answer": "",
            "task": "default",
        }

        # Call setup_state as an unbound method — it doesn't use self
        result = await PerfOptimizeEnv.setup_state(None, state)  # type: ignore[arg-type]

        assert result["test_inputs"] == [test_input]
        assert result["expected_outputs"] == [expected_output]
        assert result["perf_input"] == perf_input
        assert result["comparison"] == "exact"
        assert result["reference_perf"] == {"cycles": 1000.0, "instructions": 2000.0}
        assert result["best_perf_dict"] is None
        assert result["submitted"] is False
        assert result["compile_failures"] == 0
        assert result["test_failures"] == 0
        assert result["correct_submissions"] == 0


class TestSandboxErrorHandling:
    """_process_turn should catch SandboxError and return infrastructure error feedback."""

    @pytest.mark.asyncio
    async def test_bwrap_invocation_error_returns_feedback(self) -> None:
        from perf_optimize.env import PerfOptimizeEnv
        from perf_optimize.exceptions import BwrapInvocationError

        state = {
            "test_inputs": [b"input"],
            "expected_outputs": [b"output"],
            "perf_input": b"perf",
            "comparison": "exact",
            "tolerance": None,
            "reference_perf": {"cycles": 1000.0},
            "best_perf_dict": None,
            "best_wall_clock_ms": None,
            "submitted": False,
            "compile_failures": 0,
            "test_failures": 0,
            "correct_submissions": 0,
        }

        content = '<code lang="c">int main() { return 0; }</code>'

        env = PerfOptimizeEnv.__new__(PerfOptimizeEnv)
        env._sandbox = AsyncMock()
        env._sandbox.compile_and_run = AsyncMock(
            side_effect=BwrapInvocationError("bwrap failed to start")
        )

        result = await env._process_turn(content, state, turn=1, max_turns=5)

        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"
        assert "Infrastructure error" in msg["content"]
        assert "bwrap failed to start" in msg["content"]
        assert "not a problem with your code" in msg["content"]
