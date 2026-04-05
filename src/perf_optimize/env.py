"""PerfOptimizeEnv: multi-turn verifiers environment for code optimization.

The LLM agent receives a naive reference solution and iteratively optimizes it,
getting structured feedback from hardware performance counters on each turn.
"""

from __future__ import annotations

import base64
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict

import structlog
import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, State

from .config import SandboxConfig, _detect_unshare_net
from .languages import Language
from .problems import build_dataset_rows
from .processor import TurnProcessor
from .prompts import format_system_prompt
from .reward import correctness_gate, perf_reward
from .sandbox import PerfSandbox

logger = structlog.get_logger(__name__)


class PerfOptimizeState(TypedDict):
    """Environment-specific state fields set by setup_state."""

    test_inputs: list[bytes]
    expected_outputs: list[bytes]
    perf_input: bytes
    comparison: str
    tolerance: float | None
    reference_perf: dict[str, float] | None
    best_perf_dict: dict[str, float] | None
    best_wall_clock_ms: float | None
    submitted: bool
    compile_failures: int
    test_failures: int
    correct_submissions: int


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

        config = SandboxConfig.from_env(language)
        if config.unshare_net and not _detect_unshare_net(config.bwrap_path):
            logger.warning(
                "unshare_net_unavailable",
                hint="bwrap --unshare-net failed on this system; "
                     "sandbox will run without network namespace isolation",
            )
            config = replace(config, unshare_net=False)
        self._sandbox_config = config
        self._sandbox = PerfSandbox(self._sandbox_config)
        self._processor = TurnProcessor(self._sandbox)
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
        from .comparison import ComparisonConfig, ComparisonMode

        state["comparison"] = ComparisonConfig(
            mode=ComparisonMode(info["comparison"]),
            tolerance=info.get("tolerance"),
        )
        state["reference_perf"] = info.get("reference_perf")

        # Tracking fields
        state["best_perf_dict"] = None
        state["best_wall_clock_ms"] = None
        state["submitted"] = False
        state["compile_failures"] = 0
        state["test_failures"] = 0
        state["correct_submissions"] = 0

        return state

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
        assert last_msg["role"] == "assistant"
        content = last_msg["content"] or ""

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
    ) -> Messages:
        """Compile, test, and measure the agent's code submission.

        Delegates to TurnProcessor for domain logic and applies state updates.
        """
        code = _extract_code(content)
        # Lazily create processor if not set (supports __new__-based test setup)
        if not hasattr(self, "_processor"):
            self._processor = TurnProcessor(self._sandbox)
        outcome = await self._processor.process(
            code=code,
            test_inputs=state["test_inputs"],
            expected_outputs=state["expected_outputs"],
            perf_input=state["perf_input"],
            comparison=state["comparison"],
            reference_perf=state.get("reference_perf"),
            best_perf_dict=state.get("best_perf_dict"),
            best_wall_clock_ms=state.get("best_wall_clock_ms"),
            turn=turn,
            max_turns=max_turns,
        )

        # Apply state mutations from the processor
        for key, value in outcome.state_updates.items():
            if key.endswith("_delta"):
                field = key.removesuffix("_delta")
                state[field] = state.get(field, 0) + value
            else:
                state[key] = value

        return [{"role": "user", "content": outcome.feedback}]
