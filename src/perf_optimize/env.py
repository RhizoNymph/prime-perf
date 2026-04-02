"""PerfOptimizeEnv: multi-turn verifiers environment for code optimization.

The LLM agent receives a naive reference solution and iteratively optimizes it,
getting structured feedback from hardware performance counters on each turn.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

import structlog
import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.rubrics.rubric import Rubric
from verifiers.types import ChatMessage, Messages, State

from .config import SandboxConfig
from .exceptions import (
    CounterNotCountedError,
    CounterNotFoundError,
    CounterNotSupportedError,
    PerfMeasurementError,
    PerfParseError,
)
from .languages import Language
from .problems import build_dataset_rows
from .prompts import (
    format_compile_error,
    format_no_code_found,
    format_perf_feedback,
    format_system_prompt,
    format_test_failure,
)
from .reward import PERF_WEIGHT_MAP, compute_weighted_improvement, correctness_gate, perf_reward
from .sandbox import PerfSandbox
from .types import CompilationFailure

logger = structlog.get_logger(__name__)

# For extraction: find opening tags
_CODE_OPEN_PATTERN = re.compile(r"<code(?:\s+lang=\"[^\"]*\")?>")
_CODE_CLOSE = "</code>"

# For stripping in _has_submit: non-greedy to strip each block individually
_CODE_STRIP_PATTERN = re.compile(r"<code(?:\s+lang=\"[^\"]*\")?>.*?</code>", re.DOTALL)

# Regex to detect <submit/> tag as a standalone command (on its own line).
# Prevents false positives from mentions in prose or inside <code> blocks.
_SUBMIT_PATTERN = re.compile(r"^\s*<submit\s*/?>\s*$", re.MULTILINE)

# Regex to strip markdown fenced code blocks before submit detection.
_MARKDOWN_FENCE_PATTERN = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)


def _default_problems_dir() -> Path:
    """Resolve the default problems directory.

    Tries two locations:
    1. Bundled with the package (wheel install): perf_optimize/problems/
    2. Repo checkout: ../../problems/ relative to this file
    """
    # Wheel-installed location: problems/ is force-included next to the package
    pkg_dir = Path(__file__).parent / "problems"
    if pkg_dir.is_dir():
        return pkg_dir

    # Repo checkout: problems/ at repository root
    repo_dir = Path(__file__).parent.parent.parent / "problems"
    if repo_dir.is_dir():
        return repo_dir

    msg = (
        "Cannot find problems directory. Pass problems_dir explicitly, "
        "or ensure the package was installed with problem data."
    )
    raise FileNotFoundError(msg)


def _extract_code(text: str) -> str | None:
    """Extract code from the last <code>...</code> block in model output."""
    close_idx = text.rfind(_CODE_CLOSE)
    if close_idx == -1:
        return None

    # Find the last opening tag before the closing tag
    prefix = text[:close_idx]
    match = None
    for m in _CODE_OPEN_PATTERN.finditer(prefix):
        match = m

    if match is None:
        return None

    result = text[match.end():close_idx].strip()
    return result or None  # empty string -> None (Bug 14)


def _has_submit(text: str) -> bool:
    """Check if the model output contains a <submit/> tag outside code blocks."""
    stripped = _CODE_STRIP_PATTERN.sub("", text)
    stripped = _MARKDOWN_FENCE_PATTERN.sub("", stripped)
    return _SUBMIT_PATTERN.search(stripped) is not None


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
            problems_dir = _default_problems_dir()

        self._sandbox_config = SandboxConfig.from_env(language)
        self._sandbox = PerfSandbox(self._sandbox_config)
        self._language = language
        self._problem_filter = problems

        profile_name = self._sandbox_config.hardware_profile.name
        rows = build_dataset_rows(problems_dir, language, profile_name)

        if problems is not None:
            rows = [r for r in rows if r["info"]["problem_name"] in problems]

        # Warn about problems missing reference perf baselines — perf_reward()
        # returns 0.0 for these, so training degrades to correctness-only.
        missing = [r["info"]["problem_name"] for r in rows if "reference_perf" not in r["info"]]
        if missing:
            logger.warning(
                "problems_missing_baselines",
                problems=missing,
                language=language.value,
                profile=profile_name,
                hint="Run perf measurement to enable perf-based rewards for these problems.",
            )

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

    async def setup_state(self, state: State, **_kwargs: Any) -> State:
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

    def _check_submitted(self, state: State) -> bool:
        """Check if the rollout should end.

        Only checks the ``submitted`` flag. The rollout loop is responsible for
        setting this flag *after* processing any code in the same message.
        """
        return state.get("submitted", False)

    @vf.stop
    async def max_turns_reached(self, state: State) -> bool:
        """Disabled — we handle max turns in env_response via final_env_response.

        This ensures the last model response is always processed (compiled,
        tested, measured) before termination, so the rubric scores correctly.
        """
        return False

    async def env_response(
        self, messages: Messages, state: State, **_kwargs: Any
    ) -> Messages:
        """Process the agent's code submission and return feedback.

        Called by the framework's rollout loop via get_prompt_messages() after
        each model response. Extracts code, compiles, tests, measures perf, and
        returns formatted feedback. Sets ``state["final_env_response"]`` when
        the rollout should terminate (submit tag or max turns).
        """
        assert isinstance(messages, list)

        last_msg = messages[-1]
        assert isinstance(last_msg, dict) and last_msg["role"] == "assistant"
        content = last_msg.get("content", "")

        turn = len(state["trajectory"])
        max_turns = self.max_turns

        has_code = _extract_code(content) is not None
        wants_submit = _has_submit(content)
        at_limit = turn >= max_turns

        # Process code if present, or provide "no code" feedback when not terminating.
        should_process = has_code or (not at_limit and not wants_submit)
        if should_process:
            feedback_msgs = await self._process_turn(content, state, turn, max_turns)
        else:
            feedback_msgs = []

        # Signal termination to the framework via final_env_response.
        if wants_submit or at_limit:
            state["submitted"] = True
            state["final_env_response"] = feedback_msgs

        return feedback_msgs

    async def _process_turn(
        self,
        content: str,
        state: State,
        turn: int,
        max_turns: int,
    ) -> list[ChatMessage]:
        """Compile, test, and measure the agent's code submission.

        Returns feedback messages. Mutates state tracking fields in-place.
        """
        code = _extract_code(content)
        if code is None:
            feedback = format_no_code_found(turn, max_turns)
            return [{"role": "user", "content": feedback}]

        # Run the full pipeline: compile → test → perf
        from .comparison import ComparisonMode

        perf_error: str | None = None
        try:
            result = await self._sandbox.compile_and_run(
                source_code=code,
                test_inputs=state["test_inputs"],
                expected_outputs=state["expected_outputs"],
                perf_input=state["perf_input"],
                comparison=ComparisonMode(state["comparison"]),
                tolerance=state["tolerance"],
            )
        except (
            PerfMeasurementError,
            CounterNotSupportedError,
            CounterNotCountedError,
            CounterNotFoundError,
            PerfParseError,
        ) as exc:
            # Raised only after tests pass (during _run_perf / parse_perf_output),
            # so compilation and correctness are fine — we just lack perf data.
            logger.warning("perf_measurement_failed", error=str(exc))
            perf_error = str(exc)
            from .types import CompilationSuccess, ExecutionResult, TestReport, TestResult

            result = ExecutionResult(
                compilation=CompilationSuccess(),
                test_report=TestReport(results=(TestResult(name="assumed", passed=True),)),
                perf_counters=None,
                wall_clock_ms=None,
            )

        # Handle compilation failure
        if isinstance(result.compilation, CompilationFailure):
            state["compile_failures"] += 1
            stderr = result.compilation.stderr
            feedback = format_compile_error(stderr, turn, max_turns)
            return [{"role": "user", "content": feedback}]

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
            return [{"role": "user", "content": feedback}]

        # Tests passed — record correct submission
        state["correct_submissions"] += 1

        if result.perf_counters is not None:
            agent_perf = result.perf_counters.to_dict()

            # Track best performance by weighted reward score
            ref_perf = state.get("reference_perf") or {}
            current_best = state.get("best_perf_dict")
            if current_best is None:
                state["best_perf_dict"] = agent_perf
                state["best_wall_clock_ms"] = result.wall_clock_ms
            else:
                new_score = compute_weighted_improvement(ref_perf, agent_perf)
                best_score = compute_weighted_improvement(ref_perf, current_best)
                if new_score > best_score:
                    state["best_perf_dict"] = agent_perf
                    state["best_wall_clock_ms"] = result.wall_clock_ms

            feedback = format_perf_feedback(
                agent_perf, ref_perf, turn, max_turns, rewarded_counters=set(PERF_WEIGHT_MAP)
            )
        else:
            detail = f": {perf_error}" if perf_error else ""
            feedback = (
                f"**All tests passed** (turn {turn}/{max_turns}), "
                f"but perf measurement unavailable{detail}. Try again."
            )

        return [{"role": "user", "content": feedback}]
