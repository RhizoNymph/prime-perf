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
from .exceptions import SandboxError
from .heldout import (
    cycles_speedup_indist_minus_heldout,
    heldout_correctness_pass_rate,
    heldout_correctness_passed,
    heldout_cycles_speedup,
    heldout_wall_clock_ms,
)
from .languages import Language
from .problems import build_dataset_rows
from .processor import TurnProcessor
from .prompts import format_system_prompt
from .reward import correctness_gate, direct_speedup_reward, perf_reward
from .sandbox import PerfSandbox

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from verifiers.types import Messages, State


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
    reference_wall_clock_ms: float | None
    benchmark_metric: str
    submitted: bool
    compile_failures: int
    test_failures: int
    correct_submissions: int
    # Held-out evaluation fields. Namespaced ``heldout_*`` to keep merges with
    # the sibling ``feat/scaling-test`` branch (which rewrites the singular
    # in-dist fields above into ``_by_size`` dicts) additive.
    best_candidate_source: str | None
    heldout_test_inputs: list[bytes]
    heldout_expected_outputs: list[bytes]
    heldout_perf_input: bytes
    reference_heldout_perf: dict[str, float] | None
    reference_heldout_wall_clock_ms: float | None
    heldout_test_passed: bool | None
    heldout_test_total: int
    heldout_test_passed_count: int
    heldout_best_perf: dict[str, float] | None
    heldout_best_wall_clock_ms: float | None


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


def _attach_heldout_metrics(rubric: Rubric) -> None:
    """Attach held-out diagnostic metrics to a rubric (all weight=0).

    Kept as a small standalone helper so it merges additively alongside the
    sibling branches (``feat/scaling-test``, ``feat/ipc-diagnostics``) which
    each add their own ``_attach_*_metrics`` call near the rubric construction.
    """
    rubric.add_metric(heldout_correctness_passed)
    rubric.add_metric(heldout_correctness_pass_rate)
    rubric.add_metric(heldout_cycles_speedup)
    rubric.add_metric(heldout_wall_clock_ms)
    rubric.add_metric(cycles_speedup_indist_minus_heldout)


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

        system_prompt = format_system_prompt(
            language.value,
            max_turns,
            feedback_mode=feedback_mode,
        )

        reward_func = direct_speedup_reward if reward_mode == "benchmark" else perf_reward

        # NOTE (merge): sibling branches ``feat/scaling-test`` and
        # ``feat/ipc-diagnostics`` will add their own ``_attach_*_metrics(rubric)``
        # calls right below this construction. Each call should stay on its
        # own line so the merges remain additive.
        rubric = Rubric(
            funcs=[correctness_gate, reward_func],
            weights=[1.0, 1.0],
        )
        _attach_heldout_metrics(rubric)

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
        state["reference_wall_clock_ms"] = info.get("reference_wall_clock_ms")
        state["benchmark_metric"] = getattr(self, "_benchmark_metric", "cycles")

        # Tracking fields
        state["best_perf_dict"] = None
        state["best_wall_clock_ms"] = None
        state["submitted"] = False
        state["compile_failures"] = 0
        state["test_failures"] = 0
        state["correct_submissions"] = 0

        # Held-out diagnostic state. Decoded eagerly so the cleanup pass does
        # not need to re-touch info; namespaced ``heldout_*`` so merges with
        # the sibling multi-size branch stay additive.
        heldout_inputs = info.get("heldout_test_inputs") or []
        heldout_expected = info.get("heldout_expected_outputs") or []
        heldout_perf_b64 = info.get("heldout_perf_input") or ""
        state["heldout_test_inputs"] = [base64.b64decode(t) for t in heldout_inputs]
        state["heldout_expected_outputs"] = [
            base64.b64decode(t) for t in heldout_expected
        ]
        state["heldout_perf_input"] = (
            base64.b64decode(heldout_perf_b64) if heldout_perf_b64 else b""
        )
        state["reference_heldout_perf"] = info.get("reference_heldout_perf")
        state["reference_heldout_wall_clock_ms"] = info.get(
            "reference_heldout_wall_clock_ms",
        )
        state["best_candidate_source"] = None
        state["heldout_test_passed"] = None
        state["heldout_test_total"] = len(state["heldout_test_inputs"])
        state["heldout_test_passed_count"] = 0
        state["heldout_best_perf"] = None
        state["heldout_best_wall_clock_ms"] = None

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

    @vf.cleanup
    async def _run_heldout_pass(self, state: State) -> None:
        """Recompile the best correct candidate and evaluate on held-out data.

        Diagnostic-only: populates ``heldout_*`` state fields so the rubric's
        ``weight=0`` metrics (see ``_attach_heldout_metrics``) can surface
        in-dist vs. held-out divergence. Must not propagate exceptions -- any
        sandbox failure is logged and the held-out fields stay at their
        initial null values so the metrics return zero.

        Idempotent: returns immediately if it already populated
        ``heldout_test_passed``.
        """
        # Idempotency guard: cleanup may be invoked multiple times (e.g. on
        # cancellation paths). Skip if a previous call already produced a
        # verdict.
        if state.get("heldout_test_passed") is not None:
            return

        # Skip cleanly when the rollout never produced a correct submission.
        if not state.get("correct_submissions", 0):
            return
        if not state.get("best_candidate_source"):
            return

        heldout_inputs = state.get("heldout_test_inputs") or []
        heldout_expected = state.get("heldout_expected_outputs") or []
        heldout_perf_input = state.get("heldout_perf_input") or b""
        if not heldout_inputs or not heldout_perf_input:
            logger.warning(
                "heldout_pass_skipped_missing_data",
                problem=state.get("info", {}).get("problem_name"),
                heldout_test_count=len(heldout_inputs),
                heldout_perf_bytes=len(heldout_perf_input),
            )
            return

        from .comparison import ComparisonConfig

        comparison = state.get("comparison")
        if not isinstance(comparison, ComparisonConfig):
            from .comparison import ComparisonMode

            comparison = ComparisonConfig(
                mode=ComparisonMode(state.get("comparison", "exact")),
                tolerance=state.get("tolerance"),
            )

        try:
            result = await self._sandbox.compile_and_run(
                source_code=state["best_candidate_source"],
                test_inputs=list(heldout_inputs),
                expected_outputs=list(heldout_expected),
                perf_input=heldout_perf_input,
                comparison=comparison,
            )
        except SandboxError as exc:
            logger.warning(
                "heldout_pass_sandbox_error",
                error=str(exc),
                problem=state.get("info", {}).get("problem_name"),
            )
            return
        except Exception as exc:  # noqa: BLE001 -- diagnostic must not crash
            logger.warning(
                "heldout_pass_unexpected_error",
                error=str(exc),
                error_type=type(exc).__name__,
                problem=state.get("info", {}).get("problem_name"),
            )
            return

        report = result.test_report
        passed_count = report.passed if report is not None else 0
        total = report.total if report is not None else len(heldout_inputs)
        all_passed = report is not None and report.all_passed

        state["heldout_test_total"] = total
        state["heldout_test_passed_count"] = passed_count
        state["heldout_test_passed"] = bool(all_passed)
        state["heldout_best_perf"] = (
            result.perf_counters.to_dict() if result.perf_counters is not None else None
        )
        state["heldout_best_wall_clock_ms"] = result.wall_clock_ms
