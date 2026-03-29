"""PerfSandbox: async orchestrator for compiling, testing, and measuring code.

Supports C, Rust, Python, and TypeScript via LanguageConfig.

All code execution happens inside a bubblewrap sandbox with:
- Filesystem namespace isolation (ro system libs, rw workdir only)
- Network disabled (--unshare-net)
- PID namespace isolation (--unshare-pid)
- Resource limits (memory, processes, file size via ulimit)
- CPU pinning (taskset) for measurement stability
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from .bwrap import build_bwrap_command, build_compile_command, build_perf_command, build_run_command
from .comparison import ComparisonMode, compare_outputs
from .exceptions import (
    BwrapInvocationError,
    BwrapNotFoundError,
    PerfMeasurementError,
    PerfNotFoundError,
    PerfParanoidError,
    PrerequisiteError,
    TasksetNotFoundError,
)
from .perf_parser import parse_perf_output
from .types import (
    CompilationFailure,
    CompilationOutcome,
    CompilationResult,
    CompilationSuccess,
    ExecutionResult,
    PerfCounters,
    TestReport,
    TestResult,
)

if TYPE_CHECKING:
    from .config import SandboxConfig

logger = structlog.get_logger(__name__)

_PARANOID_PATH = Path("/proc/sys/kernel/perf_event_paranoid")


class PerfSandbox:
    """Async orchestrator for compiling, testing, and measuring programs.

    Supports multiple languages via LanguageConfig. All subprocess calls use
    asyncio.create_subprocess_exec. Each invocation creates its own temporary
    directory for isolation.

    Args:
        config: Sandbox configuration (paths, timeouts, limits, language).
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config

    async def check_prerequisites(self) -> None:
        """Verify all required system tools and capabilities are available.

        Checks bwrap, perf, taskset, and the language-specific compiler/runtime.

        Raises:
            BwrapNotFoundError: bwrap not on PATH.
            PrerequisiteError: Language compiler/runtime not on PATH.
            PerfNotFoundError: perf not on PATH.
            TasksetNotFoundError: taskset not on PATH.
            PerfParanoidError: perf_event_paranoid > 1.
        """
        if shutil.which(self._config.bwrap_path) is None:
            raise BwrapNotFoundError()
        if shutil.which(self._config.perf_path) is None:
            raise PerfNotFoundError()
        if shutil.which(self._config.taskset_path) is None:
            raise TasksetNotFoundError()

        lang = self._config.language
        if lang.compiler_path and shutil.which(lang.compiler_path) is None:
            raise PrerequisiteError(
                f"{lang.language.value} compiler '{lang.compiler_path}' not found"
            )
        if lang.runtime_path and shutil.which(lang.runtime_path) is None:
            raise PrerequisiteError(
                f"{lang.language.value} runtime '{lang.runtime_path}' not found"
            )

        if _PARANOID_PATH.exists():
            paranoid = int(_PARANOID_PATH.read_text().strip())
            if paranoid > 1:
                raise PerfParanoidError(paranoid)

    async def compile_and_run(
        self,
        source_code: str,
        test_inputs: list[bytes],
        expected_outputs: list[bytes],
        perf_input: bytes,
        *,
        test_names: list[str] | None = None,
        comparison: ComparisonMode = ComparisonMode.EXACT,
        tolerance: float | None = None,
    ) -> ExecutionResult:
        """Full pipeline: compile, test, measure.

        Args:
            source_code: Source code as a string (language determined by config).
            test_inputs: Binary input for each test case (fed to stdin).
            expected_outputs: Expected binary output for each test case.
            perf_input: Binary input for the performance measurement run.
            test_names: Optional names for each test case.
            comparison: How to compare outputs (exact or tolerance-based).
            tolerance: Relative tolerance for float comparison.

        Returns:
            ExecutionResult with compilation, test, and perf counter data.
        """
        if test_names is None:
            test_names = [f"test_{i}" for i in range(len(test_inputs))]

        work_dir = tempfile.mkdtemp(prefix="perf_opt_")
        try:
            return await self._run_pipeline(
                source_code, test_inputs, expected_outputs, perf_input, test_names,
                work_dir, comparison, tolerance,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def measure_only(self, binary_path: str | Path, input_path: str | Path) -> PerfCounters:
        """Run perf stat on a pre-compiled binary.

        Used for variance validation where recompilation is unnecessary.

        Args:
            binary_path: Path to the compiled binary on the host.
            input_path: Path to the binary input file on the host.

        Returns:
            PerfCounters from the measurement.
        """
        binary_path = Path(binary_path)
        input_path = Path(input_path)

        work_dir = tempfile.mkdtemp(prefix="perf_opt_measure_")
        try:
            work = Path(work_dir)
            shutil.copy2(binary_path, work / "solution")
            (work / "solution").chmod(0o755)
            shutil.copy2(input_path, work / "perf_input.bin")

            return await self._run_perf(work_dir)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def compile_only(self, source_code: str) -> tuple[CompilationResult, str]:
        """Compile source code and return the result and work directory.

        The caller is responsible for cleaning up the work directory.
        Returns (result, work_dir_path).
        """
        work_dir = tempfile.mkdtemp(prefix="perf_opt_compile_")
        work = Path(work_dir)
        lang = self._config.language

        source_file = work / f"solution{lang.file_extension}"
        source_file.write_text(source_code)

        result = await self._compile(work_dir)
        return result, work_dir

    # ── Internal pipeline ─────────────────────────────────────────────────

    async def _run_pipeline(
        self,
        source_code: str,
        test_inputs: list[bytes],
        expected_outputs: list[bytes],
        perf_input: bytes,
        test_names: list[str],
        work_dir: str,
        comparison: ComparisonMode = ComparisonMode.EXACT,
        tolerance: float | None = None,
    ) -> ExecutionResult:
        work = Path(work_dir)
        lang = self._config.language

        # Write source with correct extension
        source_file = work / f"solution{lang.file_extension}"
        source_file.write_text(source_code)

        # Write perf input
        (work / "perf_input.bin").write_bytes(perf_input)

        # Step 1: Compile
        compilation = await self._compile(work_dir)
        if isinstance(compilation, CompilationFailure):
            logger.info("compilation_failed", outcome=compilation.outcome)
            return ExecutionResult(
                compilation=compilation,
                test_report=None,
                perf_counters=None,
                wall_clock_ms=None,
            )

        # Step 2: Run tests
        test_report = await self._run_tests(
            work_dir, test_inputs, expected_outputs, test_names, comparison, tolerance
        )
        if not test_report.all_passed:
            logger.info(
                "tests_failed",
                passed=test_report.passed,
                total=test_report.total,
            )
            return ExecutionResult(
                compilation=compilation,
                test_report=test_report,
                perf_counters=None,
                wall_clock_ms=None,
            )

        # Step 3: Measure perf
        perf_counters = await self._run_perf(work_dir)
        logger.info(
            "perf_measured",
            cycles=perf_counters.cycles,
            ipc=f"{perf_counters.ipc:.2f}",
        )

        return ExecutionResult(
            compilation=compilation,
            test_report=test_report,
            perf_counters=perf_counters,
            wall_clock_ms=None,
        )

    async def _compile(self, work_dir: str) -> CompilationResult:
        """Compile or syntax-check the solution inside the bwrap sandbox."""
        lang = self._config.language
        compile_cmd = build_compile_command(
            self._config,
            source_file=f"/work/solution{lang.file_extension}",
            output_file=f"/work/{lang.output_file}",
        )
        bwrap_cmd = build_bwrap_command(self._config, work_dir, compile_cmd)

        try:
            returncode, _stdout, stderr = await self._run_subprocess(
                bwrap_cmd,
                timeout=self._config.compile_timeout_s,
            )
        except TimeoutError:
            return CompilationFailure(
                outcome=CompilationOutcome.TIMEOUT,
                stderr=f"Compilation timed out after {self._config.compile_timeout_s}s",
            )

        if returncode != 0:
            return CompilationFailure(
                outcome=CompilationOutcome.ERROR,
                stderr=stderr,
            )

        return CompilationSuccess()

    async def _run_tests(
        self,
        work_dir: str,
        test_inputs: list[bytes],
        expected_outputs: list[bytes],
        test_names: list[str],
        comparison: ComparisonMode = ComparisonMode.EXACT,
        tolerance: float | None = None,
    ) -> TestReport:
        """Run correctness tests sequentially inside bwrap."""
        results: list[TestResult] = []

        for name, input_data, expected in zip(
            test_names, test_inputs, expected_outputs, strict=True
        ):
            try:
                result = await self._run_single_test(
                    work_dir, name, input_data, expected, comparison, tolerance
                )
            except TimeoutError:
                result = TestResult(
                    name=name,
                    passed=False,
                    error=f"Test timed out after {self._config.test_timeout_s}s",
                )
            results.append(result)

        return TestReport(results=tuple(results))

    async def _run_single_test(
        self,
        work_dir: str,
        name: str,
        input_data: bytes,
        expected: bytes,
        comparison: ComparisonMode = ComparisonMode.EXACT,
        tolerance: float | None = None,
    ) -> TestResult:
        """Run one test case: feed input, compare output to expected."""
        inner_cmd = build_run_command(self._config)
        bwrap_cmd = build_bwrap_command(self._config, work_dir, inner_cmd)

        returncode, stdout_bytes, stderr = await self._run_subprocess(
            bwrap_cmd,
            timeout=self._config.test_timeout_s,
            stdin_data=input_data,
            capture_stdout_bytes=True,
        )

        if returncode != 0:
            if returncode < 0:
                error = f"Killed by signal {abs(returncode)}"
            elif returncode > 128:
                error = f"Killed by signal {returncode - 128}"
            else:
                error = f"Exit code {returncode}"
            if stderr:
                error += f": {stderr[:500]}"
            return TestResult(name=name, passed=False, error=error)

        if not stdout_bytes:
            return TestResult(name=name, passed=False, error="No output produced")

        mismatch = compare_outputs(stdout_bytes, expected, comparison, tolerance)
        if mismatch is not None:
            return TestResult(name=name, passed=False, error=mismatch)

        return TestResult(name=name, passed=True)

    async def _run_perf(self, work_dir: str) -> PerfCounters:
        """Run perf stat on the solution with perf_input.bin as stdin."""
        perf_cmd = build_perf_command(self._config)
        bwrap_cmd = build_bwrap_command(self._config, work_dir, perf_cmd)

        # Read perf input from the work directory
        perf_input_path = Path(work_dir) / "perf_input.bin"
        stdin_data = perf_input_path.read_bytes() if perf_input_path.exists() else b""

        try:
            returncode, _stdout, stderr = await self._run_subprocess(
                bwrap_cmd,
                timeout=self._config.perf_timeout_s,
                stdin_data=stdin_data,
            )
        except TimeoutError:
            raise PerfMeasurementError(
                f"perf stat timed out after {self._config.perf_timeout_s}s"
            ) from None

        if returncode != 0:
            # perf stat exits non-zero if the measured command fails,
            # but it still writes CSV to stderr. Try to parse anyway.
            logger.warning("perf_nonzero_exit", returncode=returncode)

        if not stderr.strip():
            raise PerfMeasurementError("perf stat produced no output")

        # perf stat writes CSV to stderr
        return parse_perf_output(stderr, self._config.hardware_profile)

    async def _run_subprocess(
        self,
        cmd: list[str],
        timeout: float,
        stdin_data: bytes | None = None,
        capture_stdout_bytes: bool = False,
    ) -> tuple[int, bytes | str, str]:
        """Run a subprocess with timeout.

        Args:
            cmd: Command and arguments.
            timeout: Timeout in seconds.
            stdin_data: Data to feed to stdin.
            capture_stdout_bytes: If True, return stdout as bytes; else as str.

        Returns:
            (returncode, stdout, stderr) where stdout type depends on capture_stdout_bytes.

        Raises:
            asyncio.TimeoutError: If the process doesn't finish in time.
            BwrapInvocationError: If the process can't be started.
        """
        log = logger.bind(cmd_head=cmd[0] if cmd else "<empty>")
        log.debug("subprocess_start", cmd=cmd[:6])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=(
                    asyncio.subprocess.PIPE
                    if stdin_data is not None
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise BwrapInvocationError(f"Command not found: {cmd[0]}") from exc
        except PermissionError as exc:
            raise BwrapInvocationError(f"Permission denied: {cmd[0]}") from exc

        try:
            stdout_raw, stderr_raw = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        returncode = proc.returncode if proc.returncode is not None else -1

        stderr_str = stderr_raw.decode("utf-8", errors="replace")

        if capture_stdout_bytes:
            return returncode, stdout_raw, stderr_str
        return returncode, stdout_raw.decode("utf-8", errors="replace"), stderr_str
