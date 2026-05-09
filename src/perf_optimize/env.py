"""PerfOptimizeEnv: multi-turn verifiers environment for code optimization.

The LLM agent receives a naive reference solution and iteratively optimizes it,
getting structured feedback from hardware performance counters on each turn.
"""

from __future__ import annotations

import base64
import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import structlog
import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.rubrics.rubric import Rubric

from .config import SandboxConfig, _detect_unshare_net
from .languages import Language
from .problems import build_dataset_rows
from .processor import TurnProcessor
from .prompts import format_system_prompt
from .reward import correctness_gate, direct_speedup_reward, perf_reward
from .sandbox import PerfSandbox
from .scaling import (
    cycles_speedup_geomean,
    largest_size_cycles_speedup,
    largest_size_wall_clock_ms,
    scaling_exponent_candidate,
    scaling_exponent_delta,
    scaling_exponent_reference,
)
from .types import SizedPerfInput, SizeSpec

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from verifiers.types import Messages, State


class PerfOptimizeState(TypedDict):
    """Environment-specific state fields set by setup_state.

    Sized perf-input layout: ``perf_inputs`` is a list of (label, n, data)
    tuples; per-size best tracking via ``best_perf_by_size`` and
    ``best_wall_clock_ms_by_size``.
    """

    test_inputs: list[bytes]
    expected_outputs: list[bytes]
    perf_inputs: list[tuple[str, int, bytes]]
    comparison: str
    tolerance: float | None
    reference_perf_by_size: dict[str, dict[str, float] | None]
    reference_wall_clock_ms_by_size: dict[str, float | None]
    best_perf_by_size: dict[str, dict[str, float]]
    best_wall_clock_ms_by_size: dict[str, float]
    benchmark_metric: str
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


def _build_base_rubric(reward_mode: str) -> Rubric:
    """Build the base rubric (correctness + headline reward)."""
    reward_func = direct_speedup_reward if reward_mode == "benchmark" else perf_reward
    return Rubric(funcs=[correctness_gate, reward_func], weights=[1.0, 1.0])


def _attach_scaling_metrics(rubric: Rubric) -> None:
    """Register weight=0 scaling diagnostics on the rubric."""
    rubric.add_metric(largest_size_cycles_speedup)
    rubric.add_metric(largest_size_wall_clock_ms)
    rubric.add_metric(cycles_speedup_geomean)
    rubric.add_metric(scaling_exponent_candidate)
    rubric.add_metric(scaling_exponent_reference)
    rubric.add_metric(scaling_exponent_delta)


class PerfOptimizeEnv(MultiTurnEnv):
    """Multi-turn environment for LLM code performance optimization.

    The agent receives a naive reference solution and submits optimized code.
    Each submission is compiled, tested, and measured with hardware perf counters
    at multiple input sizes; the headline reward is cycles speedup at the
    largest size.

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
        feedback_mode: str = "full",
        reward_mode: str = "training",
        benchmark_metric: str = "cycles",
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
        self._feedback_mode = feedback_mode
        self._reward_mode = reward_mode
        self._benchmark_metric = benchmark_metric

        profile_name = self._sandbox_config.hardware_profile.name
        rows = build_dataset_rows(problems_dir, language, profile_name)

        if problems is not None:
            rows = [r for r in rows if r["info"]["problem_name"] in problems]

        # Warn about problems missing reference perf baselines at any size.
        missing = [
            r["info"]["problem_name"]
            for r in rows
            if not any(
                v is not None
                for v in (r["info"].get("reference_perf_by_size") or {}).values()
            )
        ]
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

        system_prompt = format_system_prompt(
            language.value,
            max_turns,
            feedback_mode=feedback_mode,
        )

        rubric = _build_base_rubric(reward_mode)
        _attach_scaling_metrics(rubric)
        # Sibling branches will add ``_attach_heldout_metrics(rubric)`` and
        # ``_attach_ipc_metrics(rubric)`` after this call. Keep this hook
        # site stable to minimize merge-conflict surface.

        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt,
            rubric=rubric,
            max_turns=max_turns,
            message_type="chat",
        )

    async def setup_state(self, state: State, **_kwargs: Any) -> State:
        """Initialize environment-specific tracking in state.

        Decodes base64 test data and per-size perf inputs from info.
        """
        info = state["info"]

        state["test_inputs"] = [base64.b64decode(t) for t in info["test_inputs"]]
        state["expected_outputs"] = [base64.b64decode(t) for t in info["expected_outputs"]]

        # Sized perf inputs: list of (label, n, data) tuples, sorted by ascending n.
        perf_inputs_info = info.get("perf_inputs") or []
        decoded: list[tuple[str, int, bytes]] = []
        for entry in perf_inputs_info:
            decoded.append((
                entry["label"],
                int(entry["n"]),
                base64.b64decode(entry["data_b64"]),
            ))
        # Defensive sort — generators write in order, but enforce here too.
        decoded.sort(key=lambda t: t[1])
        state["perf_inputs"] = decoded

        from .comparison import ComparisonConfig, ComparisonMode

        state["comparison"] = ComparisonConfig(
            mode=ComparisonMode(info["comparison"]),
            tolerance=info.get("tolerance"),
        )
        state["reference_perf_by_size"] = info.get("reference_perf_by_size") or {}
        state["reference_wall_clock_ms_by_size"] = (
            info.get("reference_wall_clock_ms_by_size") or {}
        )
        state["benchmark_metric"] = getattr(self, "_benchmark_metric", "cycles")

        # Tracking fields
        state["best_perf_by_size"] = {}
        state["best_wall_clock_ms_by_size"] = {}
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

        # Hydrate state's tuple-encoded perf inputs into SizedPerfInput objects.
        perf_inputs = [
            SizedPerfInput(spec=SizeSpec(label=label, n=n), data=data)
            for label, n, data in state["perf_inputs"]
        ]

        outcome = await self._processor.process(
            code=code,
            test_inputs=state["test_inputs"],
            expected_outputs=state["expected_outputs"],
            perf_inputs=perf_inputs,
            comparison=state["comparison"],
            reference_perf_by_size=state.get("reference_perf_by_size"),
            best_perf_by_size=state.get("best_perf_by_size") or {},
            best_wall_clock_ms_by_size=state.get("best_wall_clock_ms_by_size") or {},
            turn=turn,
            max_turns=max_turns,
            feedback_mode=getattr(self, "_feedback_mode", "full"),
            selection_metric=(
                getattr(self, "_benchmark_metric", "cycles")
                if getattr(self, "_reward_mode", "training") == "benchmark"
                else "counter_reward"
            ),
        )

        # Apply state mutations from the processor
        for key, value in outcome.state_updates.items():
            if key.endswith("_delta"):
                field = key.removesuffix("_delta")
                state[field] = state.get(field, 0) + value
            else:
                state[key] = value

        return [{"role": "user", "content": outcome.feedback}]
