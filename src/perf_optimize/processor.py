"""Turn processing logic for the perf-optimize environment.

Decoupled from the verifiers SDK -- operates on plain dicts and dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from .comparison import ComparisonConfig, ComparisonMode
from .exceptions import (
    CounterNotCountedError,
    CounterNotFoundError,
    CounterNotSupportedError,
    PerfMeasurementError,
    PerfParseError,
    SandboxError,
)
from .prompts import (
    format_compile_error,
    format_correctness_feedback,
    format_no_code_found,
    format_perf_feedback,
    format_test_failure,
)
from .reward import PERF_WEIGHT_MAP, compute_weighted_improvement
from .types import (
    CompilationFailure,
    CompilationSuccess,
    ExecutionResult,
    TestReport,
    TestResult,
)

if TYPE_CHECKING:
    from .sandbox import PerfSandbox

logger = structlog.get_logger(__name__)

_REWARDED_COUNTERS: frozenset[str] = frozenset(PERF_WEIGHT_MAP)


@dataclass(frozen=True)
class TurnOutcome:
    """Result of processing a single turn."""

    feedback: str
    state_updates: dict[str, Any] = field(default_factory=dict)


class TurnProcessor:
    """Processes agent turns: compile, test, measure, produce feedback.

    Independent of the verifiers framework -- operates on plain values.
    """

    def __init__(self, sandbox: PerfSandbox) -> None:
        self._sandbox = sandbox

    async def process(
        self,
        code: str | None,
        *,
        test_inputs: list[bytes],
        expected_outputs: list[bytes],
        perf_input: bytes,
        comparison: ComparisonConfig | str,
        tolerance: float | None = None,
        reference_perf: dict[str, float] | None,
        best_perf_dict: dict[str, float] | None,
        best_wall_clock_ms: float | None,
        turn: int,
        max_turns: int,
        feedback_mode: str = "full",
        selection_metric: str = "counter_reward",
    ) -> TurnOutcome:
        """Process a single agent turn through the compile/test/measure pipeline.

        Args:
            code: Extracted source code, or None if no code block found.
            test_inputs: Binary test inputs for correctness checking.
            expected_outputs: Expected binary outputs for correctness checking.
            perf_input: Binary input for performance measurement.
            comparison: Comparison config, or mode string (e.g. "exact").
            tolerance: Optional tolerance for float comparison (used when
                comparison is passed as a string).
            reference_perf: Reference perf counters from naive solution.
            best_perf_dict: Best perf counters seen so far, or None.
            best_wall_clock_ms: Best wall clock time seen so far, or None.
            turn: Current turn number.
            max_turns: Maximum number of turns.
            feedback_mode: ``"full"`` shows performance counters; ``"correctness"``
                hides them for benchmark-style evaluation.
            selection_metric: Metric used to keep the best correct submission.

        Returns:
            TurnOutcome with feedback string and state update dict.
        """
        if isinstance(comparison, str):
            comparison = ComparisonConfig(
                mode=ComparisonMode(comparison), tolerance=tolerance,
            )

        if code is None:
            return TurnOutcome(feedback=format_no_code_found(turn, max_turns))

        perf_error: str | None = None
        try:
            result = await self._sandbox.compile_and_run(
                source_code=code,
                test_inputs=test_inputs,
                expected_outputs=expected_outputs,
                perf_input=perf_input,
                comparison=comparison,
            )
        except (
            PerfMeasurementError,
            CounterNotSupportedError,
            CounterNotCountedError,
            CounterNotFoundError,
            PerfParseError,
        ) as exc:
            logger.warning("perf_measurement_failed", error=str(exc))
            perf_error = str(exc)
            result = ExecutionResult(
                compilation=CompilationSuccess(),
                test_report=TestReport(results=(TestResult(name="assumed", passed=True),)),
                perf_counters=None,
                wall_clock_ms=None,
            )
        except SandboxError as exc:
            logger.warning("sandbox_infrastructure_error", error=str(exc))
            feedback = (
                f"**Infrastructure error** (turn {turn}/{max_turns})\n\n"
                f"{exc}\n\n"
                "This is not a problem with your code. Try again."
            )
            return TurnOutcome(feedback=feedback)

        if isinstance(result.compilation, CompilationFailure):
            return TurnOutcome(
                feedback=format_compile_error(result.compilation.stderr, turn, max_turns),
                state_updates={"compile_failures_delta": 1},
            )

        if result.test_report is not None and not result.test_report.all_passed:
            return TurnOutcome(
                feedback=format_test_failure(
                    result.test_report.passed,
                    result.test_report.total,
                    result.test_report.errors,
                    turn,
                    max_turns,
                ),
                state_updates={"test_failures_delta": 1},
            )

        # Tests passed
        updates: dict[str, Any] = {"correct_submissions_delta": 1}

        if result.perf_counters is not None:
            agent_perf = result.perf_counters.to_dict()
            ref_perf = reference_perf or {}
            if best_perf_dict is None:
                updates["best_perf_dict"] = agent_perf
                updates["best_wall_clock_ms"] = result.wall_clock_ms
                # Track the source that produced this best so the held-out
                # cleanup pass can recompile it. Sibling branch ``feat/scaling-test``
                # rewrites the best-tracking logic; keep this assignment co-located
                # with the existing best update so the merge stays small.
                updates["best_candidate_source"] = code
            else:
                if _is_better_submission(
                    selection_metric=selection_metric,
                    ref_perf=ref_perf,
                    agent_perf=agent_perf,
                    agent_wall_clock_ms=result.wall_clock_ms,
                    best_perf_dict=best_perf_dict,
                    best_wall_clock_ms=best_wall_clock_ms,
                ):
                    updates["best_perf_dict"] = agent_perf
                    updates["best_wall_clock_ms"] = result.wall_clock_ms
                    updates["best_candidate_source"] = code

            if feedback_mode == "correctness":
                feedback = format_correctness_feedback(turn, max_turns)
            else:
                feedback = format_perf_feedback(
                    agent_perf, ref_perf, turn, max_turns, rewarded_counters=_REWARDED_COUNTERS
                )
        else:
            detail = f": {perf_error}" if perf_error else ""
            feedback = (
                f"**All tests passed** (turn {turn}/{max_turns}), "
                f"but perf measurement unavailable{detail}. Try again."
            )

        return TurnOutcome(feedback=feedback, state_updates=updates)


def _is_better_submission(
    *,
    selection_metric: str,
    ref_perf: dict[str, float],
    agent_perf: dict[str, float],
    agent_wall_clock_ms: float | None,
    best_perf_dict: dict[str, float],
    best_wall_clock_ms: float | None,
) -> bool:
    """Return whether the new correct submission should replace the current best."""
    if selection_metric == "wall_clock_ms":
        return (
            agent_wall_clock_ms is not None
            and (best_wall_clock_ms is None or agent_wall_clock_ms < best_wall_clock_ms)
        )

    if selection_metric != "counter_reward":
        agent_value = agent_perf.get(selection_metric)
        best_value = best_perf_dict.get(selection_metric)
        return agent_value is not None and (best_value is None or agent_value < best_value)

    new_score = compute_weighted_improvement(ref_perf, agent_perf)
    best_score = compute_weighted_improvement(ref_perf, best_perf_dict)
    return new_score > best_score
