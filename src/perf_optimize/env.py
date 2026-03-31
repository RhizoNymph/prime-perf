"""PerfOptimizeEnv: multi-turn verifiers environment for code optimization.

The LLM agent receives a naive reference solution and iteratively optimizes it,
getting structured feedback from hardware performance counters on each turn.
"""

from __future__ import annotations

import base64
import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.rubrics import Rubric
from verifiers.types import ChatCompletion, ChatMessage, Info, Messages, SamplingArgs, State

from .config import SandboxConfig
from .languages import Language
from .problems import build_dataset_rows
from .prompts import (
    format_compile_error,
    format_no_code_found,
    format_perf_feedback,
    format_system_prompt,
    format_test_failure,
)
from .reward import correctness_gate, perf_reward
from .sandbox import PerfSandbox
from .types import CompilationFailure

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = structlog.get_logger(__name__)

# Regex to extract code from <code lang="...">...</code> blocks.
# The lang attribute is optional.
_CODE_PATTERN = re.compile(r"<code(?:\s+lang=\"[^\"]*\")?>\s*(.*?)\s*</code>", re.DOTALL)

# Regex to detect <submit/> tag.
_SUBMIT_PATTERN = re.compile(r"<submit\s*/?>")

# Default problems directory relative to this package.
_DEFAULT_PROBLEMS_DIR = Path(__file__).parent.parent.parent / "problems"


def _extract_code(text: str) -> str | None:
    """Extract code from <code>...</code> tags in model output."""
    match = _CODE_PATTERN.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def _has_submit(text: str) -> bool:
    """Check if the model output contains a <submit/> tag."""
    return _SUBMIT_PATTERN.search(text) is not None


class PerfOptimizeEnv(MultiTurnEnv):
    """Multi-turn environment for LLM code performance optimization.

    The agent receives a naive reference solution and submits optimized code.
    Each submission is compiled, tested, and measured with hardware perf counters.
    Feedback includes counter values and improvement percentages.

    Args:
        language: Target programming language.
        max_turns: Maximum interaction turns.
        problems_dir: Path to problems directory (default: project's problems/).
        problems: Optional list of problem names to include (None = all).
    """

    def __init__(
        self,
        language: Language = Language.C,
        max_turns: int = 5,
        problems_dir: Path | None = None,
        problems: list[str] | None = None,
    ) -> None:
        if problems_dir is None:
            problems_dir = _DEFAULT_PROBLEMS_DIR

        self._sandbox_config = SandboxConfig.from_env(language)
        self._sandbox = PerfSandbox(self._sandbox_config)
        self._language = language
        self._problem_filter = problems

        profile_name = self._sandbox_config.hardware_profile.name
        rows = build_dataset_rows(problems_dir, language, profile_name)

        if problems is not None:
            rows = [r for r in rows if r["info"]["problem_name"] in problems]

        # verifiers expects a HuggingFace Dataset with "question" column
        from datasets import Dataset as HFDataset

        dataset = HFDataset.from_list(rows)

        system_prompt = format_system_prompt(language.value, max_turns)

        rubric = Rubric(
            funcs=[correctness_gate, perf_reward],
            weights=[1.0, 1.0],
        )

        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt,
            rubric=rubric,
            max_turns=max_turns,
            message_type="chat",
        )

    def setup_state(self, state: State, **_kwargs: Any) -> State:
        """Initialize environment-specific tracking in state.

        Decodes base64 test data from info and sets up perf tracking fields.
        """
        info = state["info"]

        # Decode binary test data from base64
        state["test_inputs"] = [base64.b64decode(t) for t in info["test_inputs"]]
        state["expected_outputs"] = [base64.b64decode(t) for t in info["expected_outputs"]]
        state["perf_input"] = base64.b64decode(info["perf_input"])
        state["comparison"] = info["comparison"]
        state["tolerance"] = info.get("tolerance")
        state["reference_perf"] = info.get("reference_perf")

        # Tracking fields
        state["best_perf_dict"] = None
        state["best_wall_clock_ms"] = None
        state["submitted"] = False
        state["compile_failures"] = 0
        state["test_failures"] = 0
        state["correct_submissions"] = 0

        return state

    def is_completed(self, messages: Messages, state: State, **_kwargs: Any) -> bool:
        """Check if the rollout should end.

        Ends when the agent submits or the last assistant message has <submit/>.
        """
        if state.get("submitted", False):
            return True

        # Check if the last assistant message contains <submit/>
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict) and last.get("role") == "assistant":
                content = last.get("content", "")
                if content and _has_submit(content):
                    state["submitted"] = True
                    return True

        return False

    async def _async_env_response(
        self, messages: Messages, state: State
    ) -> tuple[list[ChatMessage], State]:
        """Process the agent's code submission asynchronously.

        Extracts code from the model's response, compiles, tests, and measures
        performance. Returns formatted feedback as a user message.
        """
        assert isinstance(messages, list)

        # Get the last assistant message
        last_msg = messages[-1]
        assert isinstance(last_msg, dict) and last_msg["role"] == "assistant"
        content = last_msg.get("content", "")

        turn = state["turn"]
        max_turns = self.max_turns

        # Extract code
        code = _extract_code(content)
        if code is None:
            feedback = format_no_code_found(turn, max_turns)
            return [{"role": "user", "content": feedback}], state

        # Run the full pipeline: compile → test → perf
        from .comparison import ComparisonMode

        result = await self._sandbox.compile_and_run(
            source_code=code,
            test_inputs=state["test_inputs"],
            expected_outputs=state["expected_outputs"],
            perf_input=state["perf_input"],
            comparison=ComparisonMode(state["comparison"]),
            tolerance=state["tolerance"],
        )

        # Handle compilation failure
        if isinstance(result.compilation, CompilationFailure):
            state["compile_failures"] += 1
            stderr = result.compilation.stderr
            feedback = format_compile_error(stderr, turn, max_turns)
            return [{"role": "user", "content": feedback}], state

        # Handle test failure
        if result.test_report is not None and not result.test_report.all_passed:
            state["test_failures"] += 1
            feedback = format_test_failure(
                result.test_report.passed,
                result.test_report.total,
                result.test_report.errors,
                turn,
                max_turns,
            )
            return [{"role": "user", "content": feedback}], state

        # Tests passed — record correct submission
        state["correct_submissions"] += 1

        if result.perf_counters is not None:
            agent_perf = result.perf_counters.to_dict()

            # Track best performance (by cycles)
            current_best = state.get("best_perf_dict")
            if current_best is None or agent_perf.get("cycles", float("inf")) < current_best.get(
                "cycles", float("inf")
            ):
                state["best_perf_dict"] = agent_perf
                state["best_wall_clock_ms"] = result.wall_clock_ms

            ref_perf = state.get("reference_perf", {})
            feedback = format_perf_feedback(agent_perf, ref_perf, turn, max_turns)
        else:
            feedback = (
                f"**All tests passed** (turn {turn}/{max_turns}), "
                "but perf measurement unavailable."
            )

        return [{"role": "user", "content": feedback}], state

    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info | None = None,
        sampling_args: SamplingArgs | None = None,
        **kwargs: Any,
    ) -> tuple[Messages, State]:
        """Run a multi-turn rollout with async env_response.

        This is a copy of MultiTurnEnv.rollout() with the single change of
        calling ``await self._async_env_response(...)`` instead of the sync
        ``self.env_response(...)``.
        """
        info = info or {}
        is_completed = False
        state: State = {
            "prompt": prompt,
            "completion": [],
            "answer": answer,
            "task": task,
            "info": info,
            "responses": [],
            "turn": 0,
        }
        state = self.setup_state(state)

        assert isinstance(prompt, list)
        completion: list[ChatMessage] = []
        rollout: list[ChatMessage] = deepcopy(prompt)

        while not is_completed:
            if self.is_completed(rollout, state, **kwargs):
                is_completed = True
                break

            response = await self.get_model_response(
                client=client,
                model=model,
                prompt=rollout,
                oai_tools=info.get("oai_tools", None),
                sampling_args=sampling_args,
                message_type=self.message_type,
            )
            state["responses"].append(response)

            assert isinstance(response, ChatCompletion)
            response_text: str = response.choices[0].message.content or ""
            response_message: ChatMessage = {
                "role": "assistant",
                "content": response_text,
            }
            if response.choices[0].message.tool_calls:
                response_message["tool_calls"] = response.choices[0].message.tool_calls  # type: ignore[assignment]

            rollout.append(response_message)
            completion.append(response_message)

            state["turn"] += 1

            if self.is_completed(rollout, state, **kwargs) or state["turn"] >= self.max_turns:
                is_completed = True
            else:
                # The key difference: await async env_response
                env_msgs, state = await self._async_env_response(rollout, state)
                assert isinstance(env_msgs, list)
                rollout += env_msgs
                completion += env_msgs

        return completion, state

    def env_response(
        self, messages: Messages, state: State, **_kwargs: Any
    ) -> tuple[Messages, State]:
        """Sync stub required by MultiTurnEnv ABC.

        Not called in practice — our rollout() override uses _async_env_response.
        """
        raise NotImplementedError(
            "PerfOptimizeEnv.env_response() is not callable synchronously. "
            "Use rollout() which calls _async_env_response() instead."
        )
