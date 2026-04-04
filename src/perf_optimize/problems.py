"""Problem bank loader for the perf-optimize environment.

Loads problem specifications, reference solutions, and test data from the
problem directory structure. Produces HuggingFace Datasets compatible with
the verifiers SDK.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .comparison import ComparisonConfig, ComparisonMode
from .languages import Language, resolve_language_config
from .types import PerfCounters

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ProblemSpec:
    """A problem definition with test data, independent of language.

    Args:
        name: Problem identifier (directory name, e.g. "matmul").
        spec_text: Problem description from spec.md.
        test_inputs: Binary inputs for each test case.
        expected_outputs: Expected binary outputs for each test case.
        perf_input: Larger binary input for performance measurement.
        comparison: Comparison configuration (mode and optional tolerance).
    """

    name: str
    spec_text: str
    test_inputs: tuple[bytes, ...]
    expected_outputs: tuple[bytes, ...]
    perf_input: bytes
    comparison: ComparisonConfig

    @property
    def tolerance(self) -> float | None:
        """Shortcut for comparison.tolerance."""
        return self.comparison.tolerance


@dataclass(frozen=True)
class ProblemWithReference:
    """A problem with a language-specific reference solution and perf baseline."""

    spec: ProblemSpec
    language: Language
    reference_source: str
    reference_perf: PerfCounters | None


def _load_comparison(problem_dir: Path) -> ComparisonConfig:
    """Load comparison.json from a problem directory."""
    comp_file = problem_dir / "comparison.json"
    if not comp_file.exists():
        return ComparisonConfig()
    data = json.loads(comp_file.read_text())
    mode = ComparisonMode(data["mode"])
    tolerance = data.get("tolerance")
    if mode == ComparisonMode.TOLERANCE and tolerance is None:
        raise ValueError(
            "tolerance mode requires a 'tolerance' value in comparison.json"
        )
    return ComparisonConfig(mode=mode, tolerance=tolerance)


def _load_test_files(tests_dir: Path) -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
    """Load test input/expected pairs from the tests directory.

    Files are expected to be named input_0.bin, input_1.bin, ...
    and expected_0.bin, expected_1.bin, ...
    """
    inputs: list[bytes] = []
    outputs: list[bytes] = []

    i = 0
    while True:
        input_file = tests_dir / f"input_{i}.bin"
        expected_file = tests_dir / f"expected_{i}.bin"
        if not input_file.exists():
            break
        inputs.append(input_file.read_bytes())
        if not expected_file.exists():
            raise FileNotFoundError(
                f"Missing expected output file: {expected_file} "
                f"(input_{i}.bin exists but expected_{i}.bin does not)"
            )
        outputs.append(expected_file.read_bytes())
        i += 1

    # Check for non-contiguous files (e.g., input_0, input_1, input_3 — missing input_2)
    all_input_files = list(tests_dir.glob("input_*.bin"))
    if len(all_input_files) > i:
        extra = sorted(
            f.name for f in all_input_files
            if f.name not in {f"input_{j}.bin" for j in range(i)}
        )
        raise FileNotFoundError(
            f"Non-contiguous test files in {tests_dir}: found {extra} "
            f"beyond contiguous range input_0..input_{i - 1}"
        )

    return tuple(inputs), tuple(outputs)


def _load_reference_perf(
    problem_dir: Path, language: Language, profile_name: str
) -> PerfCounters | None:
    """Load reference perf counters for a specific language and hardware profile."""
    perf_file = problem_dir / "reference_perf" / f"{language.value}_{profile_name}.json"
    if not perf_file.exists():
        return None

    data = json.loads(perf_file.read_text())
    return PerfCounters(
        cycles=data["cycles"],
        instructions=data["instructions"],
        cache_references=data.get("cache_references"),
        cache_misses=data.get("cache_misses"),
        l1_dcache_load_misses=data.get("l1_dcache_load_misses"),
        llc_load_misses=data.get("llc_load_misses"),
        branch_misses=data.get("branch_misses"),
    )


def load_problem(problem_dir: Path) -> ProblemSpec:
    """Load a problem specification from its directory.

    Args:
        problem_dir: Path to the problem directory (e.g. problems/matmul/).

    Returns:
        ProblemSpec with test data loaded.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    name = problem_dir.name
    spec_text = (problem_dir / "spec.md").read_text()

    comparison = _load_comparison(problem_dir)

    tests_dir = problem_dir / "tests"
    test_inputs, expected_outputs = _load_test_files(tests_dir)
    if not test_inputs:
        msg = f"No test files found in {tests_dir} — at least one input_0.bin is required"
        raise FileNotFoundError(msg)

    perf_input_file = problem_dir / "perf_input.bin"
    perf_input = perf_input_file.read_bytes() if perf_input_file.exists() else b""

    return ProblemSpec(
        name=name,
        spec_text=spec_text,
        test_inputs=test_inputs,
        expected_outputs=expected_outputs,
        perf_input=perf_input,
        comparison=comparison,
    )


def load_problem_with_reference(
    problem_dir: Path, language: Language, profile_name: str
) -> ProblemWithReference:
    """Load a problem with its language-specific reference solution.

    Args:
        problem_dir: Path to the problem directory.
        language: Which language's reference to load.
        profile_name: Hardware profile name (e.g. "amd_zen").

    Returns:
        ProblemWithReference including source code and perf baseline.
    """
    spec = load_problem(problem_dir)

    ext = resolve_language_config(language).file_extension
    ref_file = problem_dir / "reference" / f"solution{ext}"
    reference_source = ref_file.read_text()

    reference_perf = _load_reference_perf(problem_dir, language, profile_name)

    return ProblemWithReference(
        spec=spec,
        language=language,
        reference_source=reference_source,
        reference_perf=reference_perf,
    )


def _format_prompt(
    problem: ProblemWithReference,
    rewarded_counters: set[str] | None = None,
) -> str:
    """Format a problem into a prompt for the LLM agent.

    Args:
        problem: The problem with reference solution.
        rewarded_counters: If provided, only display these counters in the
            prompt. ``None`` displays all available counters.
    """
    lines = [
        problem.spec.spec_text.strip(),
        "",
        f"## Language: {problem.language.value}",
        "",
        "## Reference Solution (naive — optimize this)",
        "```",
        problem.reference_source.strip(),
        "```",
    ]

    if problem.reference_perf is not None:
        lines.extend([
            "",
            "## Reference Performance",
        ])
        perf_dict = problem.reference_perf.to_dict()
        if rewarded_counters is not None:
            perf_dict = {k: v for k, v in perf_dict.items() if k in rewarded_counters}
        for counter, value in perf_dict.items():
            lines.append(f"  {counter}: {value:,.0f}")
        if problem.reference_perf.ipc > 0:
            lines.append(f"  IPC: {problem.reference_perf.ipc:.2f}")

    lines.extend([
        "",
        "Write an optimized solution.",
    ])

    return "\n".join(lines)


def _encode_bytes(data: bytes) -> str:
    """Encode bytes as base64 for Dataset storage."""
    return base64.b64encode(data).decode("ascii")


def build_dataset_rows(
    problems_dir: Path,
    language: Language,
    profile_name: str,
    rewarded_counters: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build dataset rows from the problem bank.

    Args:
        problems_dir: Root directory containing problem subdirectories.
        language: Which language to use for reference solutions.
        profile_name: Hardware profile name for perf baselines.
        rewarded_counters: If provided, only display these counters in the
            prompt. ``None`` displays all available counters.

    Returns:
        List of dicts with prompt, answer, and info columns.
    """
    rows: list[dict[str, Any]] = []

    for problem_dir in sorted(problems_dir.iterdir()):
        if not problem_dir.is_dir():
            continue
        if not (problem_dir / "spec.md").exists():
            continue

        problem = load_problem_with_reference(problem_dir, language, profile_name)
        prompt = _format_prompt(problem, rewarded_counters=rewarded_counters)

        info: dict[str, Any] = {
            "problem_name": problem.spec.name,
            "language": language.value,
            "test_inputs": [_encode_bytes(t) for t in problem.spec.test_inputs],
            "expected_outputs": [_encode_bytes(t) for t in problem.spec.expected_outputs],
            "perf_input": _encode_bytes(problem.spec.perf_input),
            "comparison": problem.spec.comparison.mode.value,
            "tolerance": problem.spec.comparison.tolerance,
        }

        if problem.reference_perf is not None:
            info["reference_perf"] = problem.reference_perf.to_dict()

        rows.append({
            "question": prompt,
            "answer": "",
            "info": info,
        })

    return rows
