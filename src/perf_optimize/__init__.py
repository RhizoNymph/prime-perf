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
) -> PerfOptimizeEnv:
    """Entry point for the verifiers SDK.

    Args:
        language: Target language ("c", "rust", "python", "typescript").
        max_turns: Maximum interaction turns per problem.
        problems_dir: Override path to problems directory.
        problems: Filter to specific problem names (None = all).

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
    )
