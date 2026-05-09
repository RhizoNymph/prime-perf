"""Turn processing logic for the perf-optimize environment.

Decoupled from the verifiers SDK -- operates on plain dicts and dataclasses.

In the sized perf-input layout, ``best_perf_by_size`` and
``best_wall_clock_ms_by_size`` track per-label bests, and the whole
submission wins or loses on the largest-size selection metric.
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
    SizedExecutionResult,
    SizedMeasurement,
    SizedPerfInput,
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
        perf_inputs: list[SizedPerfInput],
        comparison: ComparisonConfig | str,
        tolerance: float | None = None,
        reference_perf_by_size: dict[str, dict[str, float] | None] | None,
        best_perf_by_size: dict[str, dict[str, float]],
        best_wall_clock_ms_by_size: dict[str, float],
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
            perf_inputs: Per-size perf inputs (sorted ascending by ``n``).
            comparison: Comparison config, or mode string (e.g. "exact").
            tolerance: Optional tolerance for float comparison (used when
                comparison is passed as a string).
            reference_perf_by_size: Per-size reference perf counters from
                naive solution (None values mean "no baseline available").
            best_perf_by_size: Per-size best counters seen so far.
            best_wall_clock_ms_by_size: Per-size best wall-clock seen so far.
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
            result = await self._sandbox.compile_and_run_sized(
                source_code=code,
                test_inputs=test_inputs,
                expected_outputs=expected_outputs,
                perf_inputs=perf_inputs,
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
            result = SizedExecutionResult(
                compilation=CompilationSuccess(),
                test_report=TestReport(results=(TestResult(name="assumed", passed=True),)),
                measurements=tuple(
                    SizedMeasurement(spec=p.spec, perf_counters=None, wall_clock_ms=None)
                    for p in perf_inputs
                ),
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

        if result.measurements:
            largest_label = _largest_label(perf_inputs)
            largest_measurement = next(
                (m for m in result.measurements if m.spec.label == largest_label),
                None,
            )

            should_promote = (
                largest_measurement is not None
                and largest_measurement.succeeded
                and _is_better_submission(
                    selection_metric=selection_metric,
                    largest_label=largest_label,
                    largest_measurement=largest_measurement,
                    reference_perf_by_size=reference_perf_by_size or {},
                    best_perf_by_size=best_perf_by_size,
                    best_wall_clock_ms_by_size=best_wall_clock_ms_by_size,
                )
            )

            if should_promote:
                new_best_perf, new_best_wall = _measurements_to_dicts(result.measurements)
                updates["best_perf_by_size"] = new_best_perf
                updates["best_wall_clock_ms_by_size"] = new_best_wall

            largest_perf = (
                largest_measurement.perf_counters.to_dict()
                if largest_measurement is not None and largest_measurement.perf_counters is not None
                else {}
            )
            ref_perf_largest = (reference_perf_by_size or {}).get(largest_label) or {}

            if feedback_mode == "correctness":
                feedback = format_correctness_feedback(turn, max_turns)
            elif largest_perf:
                feedback = format_perf_feedback(
                    largest_perf,
                    ref_perf_largest,
                    turn,
                    max_turns,
                    rewarded_counters=_REWARDED_COUNTERS,
                )
            else:
                detail = f": {perf_error}" if perf_error else ""
                feedback = (
                    f"**All tests passed** (turn {turn}/{max_turns}), "
                    f"but perf measurement unavailable at largest size{detail}. Try again."
                )
        else:
            detail = f": {perf_error}" if perf_error else ""
            feedback = (
                f"**All tests passed** (turn {turn}/{max_turns}), "
                f"but perf measurement unavailable{detail}. Try again."
            )

        return TurnOutcome(feedback=feedback, state_updates=updates)


def _largest_label(perf_inputs: list[SizedPerfInput]) -> str:
    """Return the label of the largest input. Caller guarantees non-empty."""
    largest = max(perf_inputs, key=lambda p: p.spec.n)
    return largest.spec.label


def _measurements_to_dicts(
    measurements: tuple[SizedMeasurement, ...],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Convert sized measurements into per-label dicts for state storage.

    Skips entries where the measurement failed.
    """
    by_perf: dict[str, dict[str, float]] = {}
    by_wall: dict[str, float] = {}
    for m in measurements:
        if m.perf_counters is not None:
            by_perf[m.spec.label] = m.perf_counters.to_dict()
        if m.wall_clock_ms is not None:
            by_wall[m.spec.label] = m.wall_clock_ms
    return by_perf, by_wall


def _is_better_submission(
    *,
    selection_metric: str,
    largest_label: str,
    largest_measurement: SizedMeasurement,
    reference_perf_by_size: dict[str, dict[str, float] | None],
    best_perf_by_size: dict[str, dict[str, float]],
    best_wall_clock_ms_by_size: dict[str, float],
) -> bool:
    """Decide whether the new submission replaces the current best.

    Selection key is always the largest-size measurement; the whole
    submission wins or loses (no per-size mosaic of winners).
    """
    if selection_metric == "wall_clock_ms":
        agent_ms = largest_measurement.wall_clock_ms
        best_ms = best_wall_clock_ms_by_size.get(largest_label)
        return agent_ms is not None and (best_ms is None or agent_ms < best_ms)

    agent_counters = largest_measurement.perf_counters.to_dict() if largest_measurement.perf_counters else {}
    best_counters = best_perf_by_size.get(largest_label, {})

    if selection_metric != "counter_reward":
        agent_value = agent_counters.get(selection_metric)
        best_value = best_counters.get(selection_metric)
        return agent_value is not None and (best_value is None or agent_value < best_value)

    ref_counters = reference_perf_by_size.get(largest_label) or {}
    new_score = compute_weighted_improvement(ref_counters, agent_counters)
    best_score = compute_weighted_improvement(ref_counters, best_counters) if best_counters else -1.0
    return new_score > best_score
