"""Tests for TurnProcessor — domain logic decoupled from verifiers SDK."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from perf_optimize.processor import TurnOutcome, TurnProcessor, _REWARDED_COUNTERS
from perf_optimize.types import (
    CompilationFailure,
    CompilationOutcome,
    CompilationSuccess,
    ExecutionResult,
    PerfCounters,
    TestReport,
    TestResult,
)


@pytest.fixture
def mock_sandbox() -> AsyncMock:
    """Return a mock PerfSandbox with compile_and_run as an AsyncMock."""
    sandbox = AsyncMock()
    sandbox.compile_and_run = AsyncMock()
    return sandbox


@pytest.fixture
def processor(mock_sandbox: AsyncMock) -> TurnProcessor:
    """Return a TurnProcessor wired to the mock sandbox."""
    return TurnProcessor(mock_sandbox)


# Common kwargs shared across process() calls.
_BASE_KWARGS = {
    "test_inputs": [b"\x01"],
    "expected_outputs": [b"\x02"],
    "perf_input": b"\xff",
    "comparison": "exact",
    "tolerance": None,
    "reference_perf": {"cycles": 10_000.0, "instructions": 20_000.0},
    "best_perf_dict": None,
    "best_wall_clock_ms": None,
    "turn": 1,
    "max_turns": 5,
}


class TestNoCodeFound:
    @pytest.mark.asyncio
    async def test_returns_no_code_feedback(self, processor: TurnProcessor) -> None:
        outcome = await processor.process(code=None, **_BASE_KWARGS)
        assert "No code found" in outcome.feedback
        assert outcome.state_updates == {}

    @pytest.mark.asyncio
    async def test_sandbox_not_called(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        await processor.process(code=None, **_BASE_KWARGS)
        mock_sandbox.compile_and_run.assert_not_called()


class TestCompilationFailure:
    @pytest.mark.asyncio
    async def test_returns_compile_error_feedback(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        mock_sandbox.compile_and_run.return_value = ExecutionResult(
            compilation=CompilationFailure(
                outcome=CompilationOutcome.ERROR, stderr="undefined reference to 'foo'"
            ),
            test_report=None,
            perf_counters=None,
            wall_clock_ms=None,
        )
        outcome = await processor.process(code="int main() {}", **_BASE_KWARGS)
        assert "Compilation failed" in outcome.feedback
        assert "undefined reference" in outcome.feedback
        assert outcome.state_updates == {"compile_failures_delta": 1}


class TestTestFailure:
    @pytest.mark.asyncio
    async def test_returns_test_failure_feedback(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        mock_sandbox.compile_and_run.return_value = ExecutionResult(
            compilation=CompilationSuccess(),
            test_report=TestReport(
                results=(
                    TestResult(name="test_1", passed=True),
                    TestResult(name="test_2", passed=False, error="wrong output"),
                )
            ),
            perf_counters=None,
            wall_clock_ms=None,
        )
        outcome = await processor.process(code="int main() {}", **_BASE_KWARGS)
        assert "Tests failed" in outcome.feedback
        assert "1/2 passed" in outcome.feedback
        assert outcome.state_updates == {"test_failures_delta": 1}


class TestPerfSuccess:
    @pytest.mark.asyncio
    async def test_returns_perf_feedback(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        mock_sandbox.compile_and_run.return_value = ExecutionResult(
            compilation=CompilationSuccess(),
            test_report=TestReport(results=(TestResult(name="test_1", passed=True),)),
            perf_counters=PerfCounters(cycles=5_000.0, instructions=15_000.0),
            wall_clock_ms=1.5,
        )
        outcome = await processor.process(code="int main() {}", **_BASE_KWARGS)
        assert "All tests passed" in outcome.feedback
        assert outcome.state_updates["correct_submissions_delta"] == 1
        assert outcome.state_updates["best_perf_dict"] == {
            "cycles": 5_000.0,
            "instructions": 15_000.0,
        }
        assert outcome.state_updates["best_wall_clock_ms"] == 1.5

    @pytest.mark.asyncio
    async def test_updates_best_when_improved(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        """When new score beats existing best, best_perf_dict should update."""
        mock_sandbox.compile_and_run.return_value = ExecutionResult(
            compilation=CompilationSuccess(),
            test_report=TestReport(results=(TestResult(name="test_1", passed=True),)),
            perf_counters=PerfCounters(cycles=3_000.0, instructions=15_000.0),
            wall_clock_ms=0.8,
        )
        kwargs = {
            **_BASE_KWARGS,
            "best_perf_dict": {"cycles": 7_000.0, "instructions": 15_000.0},
            "best_wall_clock_ms": 2.0,
        }
        outcome = await processor.process(code="int main() {}", **kwargs)
        assert "best_perf_dict" in outcome.state_updates
        assert outcome.state_updates["best_perf_dict"]["cycles"] == 3_000.0

    @pytest.mark.asyncio
    async def test_does_not_update_best_when_worse(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        """When new score is worse, best_perf_dict should NOT be in updates."""
        mock_sandbox.compile_and_run.return_value = ExecutionResult(
            compilation=CompilationSuccess(),
            test_report=TestReport(results=(TestResult(name="test_1", passed=True),)),
            perf_counters=PerfCounters(cycles=9_000.0, instructions=15_000.0),
            wall_clock_ms=3.0,
        )
        kwargs = {
            **_BASE_KWARGS,
            "best_perf_dict": {"cycles": 5_000.0, "instructions": 15_000.0},
            "best_wall_clock_ms": 1.5,
        }
        outcome = await processor.process(code="int main() {}", **kwargs)
        assert "best_perf_dict" not in outcome.state_updates


class TestPerfMeasurementFailure:
    @pytest.mark.asyncio
    async def test_perf_error_produces_unavailable_feedback(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        from perf_optimize.exceptions import PerfMeasurementError

        mock_sandbox.compile_and_run.side_effect = PerfMeasurementError("PMU busy")
        outcome = await processor.process(code="int main() {}", **_BASE_KWARGS)
        assert "perf measurement unavailable" in outcome.feedback
        assert "PMU busy" in outcome.feedback
        assert outcome.state_updates["correct_submissions_delta"] == 1


class TestRewardedCounters:
    def test_rewarded_counters_is_frozenset(self) -> None:
        assert isinstance(_REWARDED_COUNTERS, frozenset)

    def test_rewarded_counters_matches_weight_map(self) -> None:
        from perf_optimize.reward import PERF_WEIGHT_MAP

        assert _REWARDED_COUNTERS == frozenset(PERF_WEIGHT_MAP)


class TestTurnOutcome:
    def test_default_state_updates_empty(self) -> None:
        outcome = TurnOutcome(feedback="hello")
        assert outcome.state_updates == {}

    def test_frozen(self) -> None:
        outcome = TurnOutcome(feedback="hello")
        with pytest.raises(AttributeError):
            outcome.feedback = "changed"  # type: ignore[misc]
