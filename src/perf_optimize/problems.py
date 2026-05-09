"""Problem bank loader for the perf-optimize environment.

Loads problem specifications, reference solutions, and test data from the
problem directory structure. Produces HuggingFace Datasets compatible with
the verifiers SDK.

The on-disk layout is::

    problems/<name>/
      sizes.toml                # ordered list of {label, n}
      perf_inputs/<label>.bin   # raw input per size
      reference_perf/<lang>_<profile>_<label>.json
      tests/{input,expected}_<i>.bin
      reference/solution.<ext>
      comparison.json
      spec.md
"""

from __future__ import annotations

import base64
import json
import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .comparison import ComparisonConfig, ComparisonMode
from .languages import Language, resolve_language_config
from .types import (
    PerfCounters,
    SizedMeasurement,
    SizedPerfInput,
    SizeSpec,
)

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
        perf_inputs: Per-size perf measurement inputs, sorted ascending by n.
        comparison: Comparison configuration (mode and optional tolerance).
        heldout_test_inputs: Binary inputs for the held-out diagnostic test
            set. Empty tuple if the problem has no held-out coverage yet.
        heldout_expected_outputs: Expected outputs paired with
            ``heldout_test_inputs``.
        heldout_perf_input: Single held-out perf input sized comparably to the
            in-dist large perf input. Empty bytes if the problem has no
            held-out coverage yet.
    """

    name: str
    spec_text: str
    test_inputs: tuple[bytes, ...]
    expected_outputs: tuple[bytes, ...]
    perf_inputs: tuple[SizedPerfInput, ...]
    comparison: ComparisonConfig
    heldout_test_inputs: tuple[bytes, ...] = ()
    heldout_expected_outputs: tuple[bytes, ...] = ()
    heldout_perf_input: bytes = b""

    @property
    def tolerance(self) -> float | None:
        """Shortcut for comparison.tolerance."""
        return self.comparison.tolerance


@dataclass(frozen=True)
class ProblemWithReference:
    """A problem with a language-specific reference solution and per-size baseline."""

    spec: ProblemSpec
    language: Language
    reference_source: str
    reference_perf: tuple[SizedMeasurement, ...]


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


def _load_sizes_toml(problem_dir: Path) -> tuple[SizeSpec, ...]:
    """Parse sizes.toml; return SizeSpecs sorted ascending by ``n``.

    Validates: file exists, has at least one entry, no duplicate labels,
    and ``n`` values are strictly increasing in declared order.
    """
    sizes_file = problem_dir / "sizes.toml"
    if not sizes_file.exists():
        raise FileNotFoundError(
            f"sizes.toml not found in {problem_dir}; required for sized perf inputs"
        )
    raw = tomllib.loads(sizes_file.read_text())
    entries = raw.get("sizes", [])
    if not entries:
        raise ValueError(
            f"sizes.toml in {problem_dir} must declare at least one [[sizes]] entry"
        )

    specs: list[SizeSpec] = []
    seen_labels: set[str] = set()
    for entry in entries:
        label = entry["label"]
        n = int(entry["n"])
        if label in seen_labels:
            raise ValueError(
                f"duplicate size label {label!r} in {sizes_file}"
            )
        seen_labels.add(label)
        specs.append(SizeSpec(label=label, n=n))

    # Strict-monotone increasing in declared order ensures "largest" is well-defined.
    for prev, cur in zip(specs, specs[1:], strict=False):
        if cur.n <= prev.n:
            raise ValueError(
                f"sizes in {sizes_file} must be monotone increasing in n; "
                f"got {prev.label}={prev.n} then {cur.label}={cur.n}"
            )
    return tuple(specs)


def _load_perf_inputs(
    problem_dir: Path, sizes: tuple[SizeSpec, ...]
) -> tuple[SizedPerfInput, ...]:
    """Load perf_inputs/<label>.bin for each size; raise on missing."""
    pi_dir = problem_dir / "perf_inputs"
    if not pi_dir.is_dir():
        raise FileNotFoundError(
            f"perf_inputs/ directory not found in {problem_dir}"
        )
    out: list[SizedPerfInput] = []
    for spec in sizes:
        path = pi_dir / f"{spec.label}.bin"
        if not path.exists():
            raise FileNotFoundError(
                f"missing perf input {path} for size {spec.label!r}"
            )
        out.append(SizedPerfInput(spec=spec, data=path.read_bytes()))
    return tuple(out)


def _load_reference_perf(
    problem_dir: Path,
    language: Language,
    profile_name: str,
    sizes: tuple[SizeSpec, ...],
) -> tuple[SizedMeasurement, ...]:
    """Load per-size reference perf JSONs for a language/hardware profile.

    For each size, returns a ``SizedMeasurement``. If the JSON is missing,
    returns a measurement with ``perf_counters=None`` and
    ``wall_clock_ms=None`` for that size (caller treats these as N/A).
    """
    out: list[SizedMeasurement] = []
    rp_dir = problem_dir / "reference_perf"
    for spec in sizes:
        path = rp_dir / f"{language.value}_{profile_name}_{spec.label}.json"
        if not path.exists():
            out.append(
                SizedMeasurement(spec=spec, perf_counters=None, wall_clock_ms=None)
            )
            continue
        data = json.loads(path.read_text())
        wall_clock_ms = data.get("wall_clock_ms")
        counters = PerfCounters(
            cycles=data["cycles"],
            instructions=data["instructions"],
            cache_references=data.get("cache_references"),
            cache_misses=data.get("cache_misses"),
            l1_dcache_load_misses=data.get("l1_dcache_load_misses"),
            llc_load_misses=data.get("llc_load_misses"),
            branch_misses=data.get("branch_misses"),
        )
        out.append(
            SizedMeasurement(
                spec=spec,
                perf_counters=counters,
                wall_clock_ms=wall_clock_ms,
            )
        )
    return tuple(out)


def _load_heldout_test_files(
    heldout_dir: Path,
) -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
    """Load held-out test input/expected pairs from ``tests_heldout/``.

    Returns empty tuples when the directory does not exist; the caller
    decides whether the absence of held-out tests is fatal for its workflow.
    Files are expected to be named ``input_0.bin`` / ``expected_0.bin`` ...
    """
    if not heldout_dir.exists():
        return (), ()

    inputs: list[bytes] = []
    outputs: list[bytes] = []

    i = 0
    while True:
        input_file = heldout_dir / f"input_{i}.bin"
        expected_file = heldout_dir / f"expected_{i}.bin"
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

    all_input_files = list(heldout_dir.glob("input_*.bin"))
    if len(all_input_files) > i:
        extra = sorted(
            f.name
            for f in all_input_files
            if f.name not in {f"input_{j}.bin" for j in range(i)}
        )
        raise FileNotFoundError(
            f"Non-contiguous test files in {heldout_dir}: found {extra} "
            f"beyond contiguous range input_0..input_{i - 1}"
        )

    return tuple(inputs), tuple(outputs)


def _load_heldout_perf_input(problem_dir: Path) -> bytes:
    """Load ``perf_inputs_heldout/large.bin`` for a problem.

    Mandatory for the held-out evaluation pass: raises ``FileNotFoundError``
    with a structured message when the file is missing so the migration
    cannot silently leave a problem without held-out perf coverage.
    """
    perf_file = problem_dir / "perf_inputs_heldout" / "large.bin"
    if not perf_file.exists():
        raise FileNotFoundError(
            f"Missing held-out perf input: {perf_file} "
            "(expected at perf_inputs_heldout/large.bin)"
        )
    return perf_file.read_bytes()


def _load_heldout_reference_perf(
    problem_dir: Path, language: Language, profile_name: str
) -> tuple[PerfCounters | None, float | None]:
    """Load held-out reference perf counters and wall-clock for a profile.

    Returns ``(None, None)`` when the file does not exist. The JSON file is
    expected at ``reference_perf/<lang>_<profile>_heldout.json`` and may
    optionally include a ``wall_clock_ms`` field alongside the perf counters.
    """
    perf_file = (
        problem_dir / "reference_perf" / f"{language.value}_{profile_name}_heldout.json"
    )
    if not perf_file.exists():
        return None, None

    data = json.loads(perf_file.read_text())
    counters = PerfCounters(
        cycles=data["cycles"],
        instructions=data["instructions"],
        cache_references=data.get("cache_references"),
        cache_misses=data.get("cache_misses"),
        l1_dcache_load_misses=data.get("l1_dcache_load_misses"),
        llc_load_misses=data.get("llc_load_misses"),
        branch_misses=data.get("branch_misses"),
    )
    wall_clock_ms = data.get("wall_clock_ms")
    return counters, wall_clock_ms


def load_problem(problem_dir: Path) -> ProblemSpec:
    """Load a problem specification from its directory.

    Args:
        problem_dir: Path to the problem directory (e.g. problems/matmul/).

    Returns:
        ProblemSpec with sized test data loaded.

    Raises:
        FileNotFoundError: If required files (spec.md, sizes.toml, perf inputs) missing.
    """
    name = problem_dir.name
    spec_text = (problem_dir / "spec.md").read_text()

    comparison = _load_comparison(problem_dir)

    tests_dir = problem_dir / "tests"
    test_inputs, expected_outputs = _load_test_files(tests_dir)
    if not test_inputs:
        msg = f"No test files found in {tests_dir} — at least one input_0.bin is required"
        raise FileNotFoundError(msg)

    sizes = _load_sizes_toml(problem_dir)
    perf_inputs = _load_perf_inputs(problem_dir, sizes)

    # Held-out fixtures are optional at the loader level so that problems can
    # be migrated incrementally; the env raises during setup_state if a
    # held-out pass is requested but inputs are missing.
    heldout_inputs, heldout_expected = _load_heldout_test_files(
        problem_dir / "tests_heldout",
    )
    heldout_perf_file = problem_dir / "perf_inputs_heldout" / "large.bin"
    heldout_perf_input = (
        heldout_perf_file.read_bytes() if heldout_perf_file.exists() else b""
    )

    return ProblemSpec(
        name=name,
        spec_text=spec_text,
        test_inputs=test_inputs,
        expected_outputs=expected_outputs,
        perf_inputs=perf_inputs,
        comparison=comparison,
        heldout_test_inputs=heldout_inputs,
        heldout_expected_outputs=heldout_expected,
        heldout_perf_input=heldout_perf_input,
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
        ProblemWithReference with per-size reference baselines.
    """
    spec = load_problem(problem_dir)

    ext = resolve_language_config(language).file_extension
    ref_file = problem_dir / "reference" / f"solution{ext}"
    reference_source = ref_file.read_text()

    sizes = tuple(p.spec for p in spec.perf_inputs)
    reference_perf = _load_reference_perf(problem_dir, language, profile_name, sizes)

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

    Displays the largest-size reference perf (the headline metric).
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

    # Pick the largest-size measurement that has counters (sizes are
    # ascending; iterate from the back).
    headline: SizedMeasurement | None = None
    for m in reversed(problem.reference_perf):
        if m.perf_counters is not None:
            headline = m
            break

    if headline is not None and headline.perf_counters is not None:
        lines.extend([
            "",
            f"## Reference Performance (size={headline.spec.label}, n={headline.spec.n})",
        ])
        perf_dict = headline.perf_counters.to_dict()
        if rewarded_counters is not None:
            perf_dict = {k: v for k, v in perf_dict.items() if k in rewarded_counters}
        for counter, value in perf_dict.items():
            lines.append(f"  {counter}: {value:,.0f}")
        if headline.perf_counters.ipc > 0:
            lines.append(f"  IPC: {headline.perf_counters.ipc:.2f}")

    lines.extend([
        "",
        "Write an optimized solution.",
    ])

    return "\n".join(lines)


def _encode_bytes(data: bytes) -> str:
    """Encode bytes as base64 for Dataset storage."""
    return base64.b64encode(data).decode("ascii")


def _info_perf_inputs(perf_inputs: tuple[SizedPerfInput, ...]) -> list[dict[str, Any]]:
    """Serialize sized perf inputs for HF Dataset info column."""
    return [
        {
            "label": p.spec.label,
            "n": p.spec.n,
            "data_b64": _encode_bytes(p.data),
        }
        for p in perf_inputs
    ]


def _info_reference_perf_by_size(
    measurements: tuple[SizedMeasurement, ...],
) -> dict[str, dict[str, float] | None]:
    out: dict[str, dict[str, float] | None] = {}
    for m in measurements:
        out[m.spec.label] = m.perf_counters.to_dict() if m.perf_counters is not None else None
    return out


def _info_wall_clock_by_size(
    measurements: tuple[SizedMeasurement, ...],
) -> dict[str, float | None]:
    return {m.spec.label: m.wall_clock_ms for m in measurements}


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
            "perf_inputs": _info_perf_inputs(problem.spec.perf_inputs),
            "comparison": problem.spec.comparison.mode.value,
            "tolerance": problem.spec.comparison.tolerance,
            "reference_perf_by_size": _info_reference_perf_by_size(problem.reference_perf),
            "reference_wall_clock_ms_by_size": _info_wall_clock_by_size(
                problem.reference_perf
            ),
        }

        # Held-out diagnostic data — namespaced separately from in-dist sized
        # data so they evolve independently.
        info["heldout_test_inputs"] = [
            _encode_bytes(t) for t in problem.spec.heldout_test_inputs
        ]
        info["heldout_expected_outputs"] = [
            _encode_bytes(t) for t in problem.spec.heldout_expected_outputs
        ]
        info["heldout_perf_input"] = _encode_bytes(problem.spec.heldout_perf_input)

        heldout_counters, heldout_wall_clock = _load_heldout_reference_perf(
            problem_dir, language, profile_name,
        )
        info["reference_heldout_perf"] = (
            heldout_counters.to_dict() if heldout_counters is not None else None
        )
        info["reference_heldout_wall_clock_ms"] = heldout_wall_clock


        rows.append({
            "question": prompt,
            "answer": "",
            "info": info,
        })

    return rows
