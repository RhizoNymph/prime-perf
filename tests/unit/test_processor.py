"""Tests for TurnProcessor — domain logic decoupled from verifiers SDK.

Now operates on the sized perf_inputs layout: per-size measurement and
per-label best tracking. Whole-submission selection: largest-size cycles win
or lose; no per-size mosaic of winners.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from perf_optimize.processor import _REWARDED_COUNTERS, TurnOutcome, TurnProcessor
from perf_optimize.types import (
    CompilationFailure,
    CompilationOutcome,
    CompilationSuccess,
    PerfCounters,
    SizedExecutionResult,
    SizedMeasurement,
    SizedPerfInput,
    SizeSpec,
    TestReport,
    TestResult,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_sandbox() -> AsyncMock:
    """Return a mock PerfSandbox with compile_and_run_sized as an AsyncMock."""
    sandbox = AsyncMock()
    sandbox.compile_and_run_sized = AsyncMock()
    return sandbox


@pytest.fixture
def processor(mock_sandbox: AsyncMock) -> TurnProcessor:
    """Return a TurnProcessor wired to the mock sandbox."""
    return TurnProcessor(mock_sandbox)


_SMALL = SizeSpec(label="small", n=256)
_LARGE = SizeSpec(label="large", n=1024)
_PERF_INPUTS_TWO = [
    SizedPerfInput(spec=_SMALL, data=b"\xff" * 4),
    SizedPerfInput(spec=_LARGE, data=b"\xff" * 16),
]


def _success_result(
    *,
    small_cycles: float = 5_000.0,
    large_cycles: float = 20_000.0,
    small_ms: float | None = 0.5,
    large_ms: float | None = 2.0,
) -> SizedExecutionResult:
    return SizedExecutionResult(
        compilation=CompilationSuccess(),
        test_report=TestReport(results=(TestResult(name="t0", passed=True),)),
        measurements=(
            SizedMeasurement(
                spec=_SMALL,
                perf_counters=PerfCounters(cycles=small_cycles, instructions=10_000.0),
                wall_clock_ms=small_ms,
            ),
            SizedMeasurement(
                spec=_LARGE,
                perf_counters=PerfCounters(cycles=large_cycles, instructions=40_000.0),
                wall_clock_ms=large_ms,
            ),
        ),
    )


_BASE_KWARGS: dict = {
    "test_inputs": [b"\x01"],
    "expected_outputs": [b"\x02"],
    "perf_inputs": _PERF_INPUTS_TWO,
    "comparison": "exact",
    "tolerance": None,
    "reference_perf_by_size": {
        "small": {"cycles": 10_000.0, "instructions": 20_000.0},
        "large": {"cycles": 40_000.0, "instructions": 80_000.0},
    },
    "best_perf_by_size": {},
    "best_wall_clock_ms_by_size": {},
    "turn": 1,
    "max_turns": 5,
}


# ── Test classes ────────────────────────────────────────────────────────────


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
        mock_sandbox.compile_and_run_sized.assert_not_called()


class TestCompilationFailure:
    @pytest.mark.asyncio
    async def test_returns_compile_error_feedback(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        mock_sandbox.compile_and_run_sized.return_value = SizedExecutionResult(
            compilation=CompilationFailure(
                outcome=CompilationOutcome.ERROR, stderr="undefined reference to 'foo'"
            ),
            test_report=None,
            measurements=(),
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
        mock_sandbox.compile_and_run_sized.return_value = SizedExecutionResult(
            compilation=CompilationSuccess(),
            test_report=TestReport(
                results=(
                    TestResult(name="test_1", passed=True),
                    TestResult(name="test_2", passed=False, error="wrong output"),
                )
            ),
            measurements=(),
        )
        outcome = await processor.process(code="int main() {}", **_BASE_KWARGS)
        assert "Tests failed" in outcome.feedback
        assert "1/2 passed" in outcome.feedback
        assert outcome.state_updates == {"test_failures_delta": 1}


class TestPerfSuccess:
    @pytest.mark.asyncio
    async def test_returns_perf_feedback_and_initial_best(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        mock_sandbox.compile_and_run_sized.return_value = _success_result(
            small_cycles=5_000.0, large_cycles=20_000.0,
        )
        outcome = await processor.process(code="int main() {}", **_BASE_KWARGS)
        assert "All tests passed" in outcome.feedback
        assert outcome.state_updates["correct_submissions_delta"] == 1
        # First submission becomes the best per-label
        bp = outcome.state_updates["best_perf_by_size"]
        assert bp["small"]["cycles"] == 5_000.0
        assert bp["large"]["cycles"] == 20_000.0
        bw = outcome.state_updates["best_wall_clock_ms_by_size"]
        assert bw["small"] == 0.5
        assert bw["large"] == 2.0

    @pytest.mark.asyncio
    async def test_updates_best_when_largest_improves(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        """Whole submission wins when largest-size cycles improves vs. best."""
        mock_sandbox.compile_and_run_sized.return_value = _success_result(
            small_cycles=4_000.0, large_cycles=15_000.0,
        )
        kwargs = {
            **_BASE_KWARGS,
            "best_perf_by_size": {
                "small": {"cycles": 5_000.0},
                "large": {"cycles": 25_000.0},
            },
            "best_wall_clock_ms_by_size": {"small": 0.6, "large": 3.0},
        }
        outcome = await processor.process(code="int main() {}", **kwargs)
        assert "best_perf_by_size" in outcome.state_updates
        assert outcome.state_updates["best_perf_by_size"]["large"]["cycles"] == 15_000.0
        # The whole submission replaced the prior best — small also updated atomically
        assert outcome.state_updates["best_perf_by_size"]["small"]["cycles"] == 4_000.0

    @pytest.mark.asyncio
    async def test_does_not_update_when_largest_worse(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        """Whole submission loses when largest-size cycles regresses."""
        mock_sandbox.compile_and_run_sized.return_value = _success_result(
            small_cycles=1_000.0, large_cycles=30_000.0,  # large regressed
        )
        kwargs = {
            **_BASE_KWARGS,
            "best_perf_by_size": {
                "small": {"cycles": 5_000.0},
                "large": {"cycles": 25_000.0},
            },
            "best_wall_clock_ms_by_size": {"small": 0.6, "large": 3.0},
        }
        outcome = await processor.process(code="int main() {}", **kwargs)
        assert "best_perf_by_size" not in outcome.state_updates
        assert "best_wall_clock_ms_by_size" not in outcome.state_updates

    @pytest.mark.asyncio
    async def test_largest_size_failed_does_not_promote(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        """Failed largest-size measurement keeps previous best, even if smaller succeeded."""
        result = SizedExecutionResult(
            compilation=CompilationSuccess(),
            test_report=TestReport(results=(TestResult(name="t0", passed=True),)),
            measurements=(
                SizedMeasurement(
                    spec=_SMALL,
                    perf_counters=PerfCounters(cycles=2_000.0, instructions=4_000.0),
                    wall_clock_ms=0.4,
                ),
                # large measurement failed (e.g. timeout)
                SizedMeasurement(spec=_LARGE, perf_counters=None, wall_clock_ms=None),
            ),
        )
        mock_sandbox.compile_and_run_sized.return_value = result
        kwargs = {
            **_BASE_KWARGS,
            "best_perf_by_size": {
                "small": {"cycles": 5_000.0},
                "large": {"cycles": 25_000.0},
            },
            "best_wall_clock_ms_by_size": {"small": 0.6, "large": 3.0},
        }
        outcome = await processor.process(code="int main() {}", **kwargs)
        # Even though small improved, we did NOT promote because large failed
        assert "best_perf_by_size" not in outcome.state_updates

    @pytest.mark.asyncio
    async def test_correctness_feedback_hides_counters(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        mock_sandbox.compile_and_run_sized.return_value = _success_result()
        outcome = await processor.process(
            code="int main() {}",
            **_BASE_KWARGS,
            feedback_mode="correctness",
        )
        assert "All tests passed" in outcome.feedback
        assert "cycles" not in outcome.feedback
        mock_sandbox.compile_and_run_sized.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_benchmark_selection_wall_clock(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        """selection_metric=wall_clock_ms uses wall_clock at largest size."""
        mock_sandbox.compile_and_run_sized.return_value = _success_result(
            large_cycles=30_000.0,  # cycles regressed
            large_ms=1.0,  # but wall_clock improved
        )
        kwargs = {
            **_BASE_KWARGS,
            "best_perf_by_size": {
                "small": {"cycles": 5_000.0},
                "large": {"cycles": 25_000.0},
            },
            "best_wall_clock_ms_by_size": {"small": 0.6, "large": 3.0},
        }
        outcome = await processor.process(
            code="int main() {}",
            **kwargs,
            selection_metric="wall_clock_ms",
        )
        assert "best_perf_by_size" in outcome.state_updates
        assert outcome.state_updates["best_wall_clock_ms_by_size"]["large"] == 1.0


class TestPerfMeasurementFailure:
    @pytest.mark.asyncio
    async def test_perf_error_produces_unavailable_feedback(
        self, processor: TurnProcessor, mock_sandbox: AsyncMock
    ) -> None:
        from perf_optimize.exceptions import PerfMeasurementError

        mock_sandbox.compile_and_run_sized.side_effect = PerfMeasurementError("PMU busy")
        outcome = await processor.process(code="int main() {}", **_BASE_KWARGS)
        assert "perf measurement unavailable" in outcome.feedback
        assert "PMU busy" in outcome.feedback
        assert outcome.state_updates["correct_submissions_delta"] == 1


class TestRewardedCounters:
    def test_rewarded_counters_is_frozenset(self) -> None:
        assert isinstance(_REWARDED_COUNTERS, frozenset)

    def test_rewarded_counters_matches_weight_map(self) -> None:
        from perf_optimize.reward import PERF_WEIGHT_MAP

        assert frozenset(PERF_WEIGHT_MAP) == _REWARDED_COUNTERS


class TestTurnOutcome:
    def test_default_state_updates_empty(self) -> None:
        outcome = TurnOutcome(feedback="hello")
        assert outcome.state_updates == {}

    def test_frozen(self) -> None:
        outcome = TurnOutcome(feedback="hello")
        with pytest.raises(AttributeError):
            outcome.feedback = "changed"  # type: ignore[misc]
