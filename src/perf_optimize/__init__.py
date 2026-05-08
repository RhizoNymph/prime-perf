"""perf-optimize: RL environment for teaching LLMs to write performant code."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .env import PerfOptimizeEnv


def load_environment(
    language: str = "c",
    max_turns: int = 5,
    problems_dir: str | None = None,
    problems: list[str] | None = None,
    feedback_mode: str = "full",
    reward_mode: str = "training",
    benchmark_metric: str = "cycles",
) -> PerfOptimizeEnv:
    """Entry point for the verifiers SDK.

    Args:
        language: Target language ("c", "rust", "python", "typescript").
        max_turns: Maximum interaction turns per problem.
        problems_dir: Override path to problems directory.
        problems: Filter to specific problem names (None = all).
        feedback_mode: "full" for training counter feedback, "correctness" to
            hide performance counters during benchmark-style evaluation.
        reward_mode: "training" for weighted counter reward, "benchmark" for
            direct speedup reward.
        benchmark_metric: Direct metric for benchmark mode ("cycles" or
            "wall_clock_ms"; other perf counter names are also accepted).

    Returns:
        Configured PerfOptimizeEnv instance.
    """
    from .env import PerfOptimizeEnv
    from .languages import Language

    return PerfOptimizeEnv(
        language=Language(language),
        max_turns=max_turns,
        problems_dir=Path(problems_dir) if problems_dir else None,
        problems=problems,
        feedback_mode=feedback_mode,
        reward_mode=reward_mode,
        benchmark_metric=benchmark_metric,
    )


def load_benchmark_environment(
    language: str = "c",
    max_turns: int = 5,
    problems_dir: str | None = None,
    problems: list[str] | None = None,
    benchmark_metric: str = "cycles",
) -> PerfOptimizeEnv:
    """Entry point for correctness-only benchmark evaluation.

    Generated code is still compiled, tested, and measured through ``PerfSandbox``.
    Performance counters are hidden from rollout feedback, and terminal reward is
    the direct speedup on ``benchmark_metric``.
    """
    return load_environment(
        language=language,
        max_turns=max_turns,
        problems_dir=problems_dir,
        problems=problems,
        feedback_mode="correctness",
        reward_mode="benchmark",
        benchmark_metric=benchmark_metric,
    )
