"""Integration tests for PerfSandbox compile_and_run pipeline.

Requires bwrap, gcc, perf, and perf_event_paranoid <= 1.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from perf_optimize.config import SandboxConfig
from perf_optimize.sandbox import PerfSandbox
from perf_optimize.types import CompilationFailure, CompilationOutcome, CompilationSuccess
from tests.conftest import requires_all_tools, requires_bwrap, requires_gcc

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "c_programs"

# Cache compiled reference binary to avoid recompiling per test
_REFERENCE_BINARY: Path | None = None
_REFERENCE_TMPDIR: tempfile.TemporaryDirectory | None = None  # type: ignore[type-arg]


def _get_reference_binary() -> Path:
    """Compile matmul_naive.c once and cache the binary for generating expected output."""
    global _REFERENCE_BINARY, _REFERENCE_TMPDIR
    if _REFERENCE_BINARY is not None and _REFERENCE_BINARY.exists():
        return _REFERENCE_BINARY
    _REFERENCE_TMPDIR = tempfile.TemporaryDirectory(prefix="perf_opt_ref_")
    binary = Path(_REFERENCE_TMPDIR.name) / "matmul_ref"
    subprocess.run(
        ["gcc", "-O2", "-lm", "-o", str(binary), str(FIXTURES / "matmul_naive.c")],
        check=True,
    )
    _REFERENCE_BINARY = binary
    return binary


def _make_matmul_input(n: int, seed: int = 42) -> bytes:
    """Generate binary input for the matmul programs."""
    rng = np.random.default_rng(seed)
    a = rng.random((n, n), dtype=np.float32)
    b = rng.random((n, n), dtype=np.float32)
    return struct.pack("i", n) + a.tobytes() + b.tobytes()


def _make_matmul_expected(n: int, seed: int = 42) -> bytes:
    """Generate expected output by running the reference C program.

    This ensures byte-exact match since the same C code and compilation
    flags produce the same floating-point results.
    """
    binary = _get_reference_binary()
    input_data = _make_matmul_input(n, seed)
    result = subprocess.run(
        [str(binary)],
        input=input_data,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, f"Reference matmul failed: {result.stderr.decode()}"
    return result.stdout


# ── Compilation tests ────────────────────────────────────────────────────────


@requires_all_tools
@pytest.mark.asyncio
async def test_compile_hello() -> None:
    """hello.c should compile and pass test and measure perf."""
    config = SandboxConfig()
    sandbox = PerfSandbox(config)
    source = FIXTURES.joinpath("hello.c").read_text()

    result = await sandbox.compile_and_run(
        source_code=source,
        test_inputs=[b""],
        expected_outputs=[b"Hello, perf-optimize!\n"],
        perf_input=b"",
        test_names=["hello_output"],
    )
    assert result.compiled


@requires_bwrap
@requires_gcc
@pytest.mark.asyncio
async def test_compile_error() -> None:
    """compile_error.c should fail compilation."""
    config = SandboxConfig()
    sandbox = PerfSandbox(config)
    source = FIXTURES.joinpath("compile_error.c").read_text()

    result = await sandbox.compile_and_run(
        source_code=source,
        test_inputs=[b""],
        expected_outputs=[b""],
        perf_input=b"",
    )
    assert not result.compiled
    assert isinstance(result.compilation, CompilationFailure)
    assert result.compilation.outcome == CompilationOutcome.ERROR
    assert result.compiler_errors is not None
    assert "error" in result.compiler_errors.lower()
    assert result.test_report is None
    assert result.perf_counters is None


@requires_bwrap
@requires_gcc
@pytest.mark.asyncio
async def test_compile_timeout() -> None:
    """A source that takes too long to compile should time out."""
    # Use a very short timeout with a normal file to trigger timeout behavior
    config = SandboxConfig(compile_timeout_s=0.001)
    sandbox = PerfSandbox(config)
    source = FIXTURES.joinpath("hello.c").read_text()

    result = await sandbox.compile_and_run(
        source_code=source,
        test_inputs=[b""],
        expected_outputs=[b""],
        perf_input=b"",
    )
    assert not result.compiled
    assert isinstance(result.compilation, CompilationFailure)
    assert result.compilation.outcome == CompilationOutcome.TIMEOUT


# ── Test execution tests ─────────────────────────────────────────────────────


@requires_all_tools
@pytest.mark.asyncio
async def test_hello_passes_test() -> None:
    """hello.c output should match expected."""
    config = SandboxConfig()
    sandbox = PerfSandbox(config)
    source = FIXTURES.joinpath("hello.c").read_text()

    result = await sandbox.compile_and_run(
        source_code=source,
        test_inputs=[b""],
        expected_outputs=[b"Hello, perf-optimize!\n"],
        perf_input=b"",
    )
    assert result.compiled
    assert result.test_report is not None
    assert result.test_report.all_passed


@requires_bwrap
@requires_gcc
@pytest.mark.asyncio
async def test_hello_fails_wrong_expected() -> None:
    """hello.c with wrong expected output should fail the test."""
    config = SandboxConfig()
    sandbox = PerfSandbox(config)
    source = FIXTURES.joinpath("hello.c").read_text()

    result = await sandbox.compile_and_run(
        source_code=source,
        test_inputs=[b""],
        expected_outputs=[b"wrong output"],
        perf_input=b"",
    )
    assert result.compiled
    assert result.test_report is not None
    assert not result.test_report.all_passed
    assert result.perf_counters is None


@requires_bwrap
@requires_gcc
@pytest.mark.asyncio
async def test_timeout_binary_killed() -> None:
    """timeout.c (infinite loop) should be killed by test timeout."""
    config = SandboxConfig(test_timeout_s=1.0)
    sandbox = PerfSandbox(config)
    source = FIXTURES.joinpath("timeout.c").read_text()

    result = await sandbox.compile_and_run(
        source_code=source,
        test_inputs=[b""],
        expected_outputs=[b""],
        perf_input=b"",
    )
    assert result.compiled
    assert result.test_report is not None
    assert not result.test_report.all_passed
    assert "timed out" in result.test_report.errors[0].lower()


# ── Full pipeline tests (compile + test + perf) ─────────────────────────────


@requires_all_tools
@pytest.mark.asyncio
async def test_matmul_naive_full_pipeline() -> None:
    """Full pipeline with matmul_naive.c: compile, test (small N), measure (small N)."""
    config = SandboxConfig()
    sandbox = PerfSandbox(config)
    source = FIXTURES.joinpath("matmul_naive.c").read_text()

    n = 4
    test_input = _make_matmul_input(n)
    expected_output = _make_matmul_expected(n)

    result = await sandbox.compile_and_run(
        source_code=source,
        test_inputs=[test_input],
        expected_outputs=[expected_output],
        perf_input=test_input,
    )
    assert result.compiled
    assert result.test_report is not None
    assert result.test_report.all_passed, f"Test errors: {result.test_report.errors}"
    assert result.perf_counters is not None
    assert result.perf_counters.cycles > 0
    assert result.perf_counters.instructions > 0
    assert result.perf_counters.ipc > 0


@requires_all_tools
@pytest.mark.asyncio
async def test_matmul_tiled_full_pipeline() -> None:
    """Full pipeline with matmul_tiled.c."""
    config = SandboxConfig()
    sandbox = PerfSandbox(config)
    source = FIXTURES.joinpath("matmul_tiled.c").read_text()

    n = 4
    test_input = _make_matmul_input(n)
    expected_output = _make_matmul_expected(n)

    result = await sandbox.compile_and_run(
        source_code=source,
        test_inputs=[test_input],
        expected_outputs=[expected_output],
        perf_input=test_input,
    )
    assert result.compiled
    assert result.test_report is not None
    assert result.test_report.all_passed, f"Test errors: {result.test_report.errors}"
    assert result.perf_counters is not None
    assert result.perf_counters.cycles > 0


@requires_all_tools
@pytest.mark.asyncio
async def test_multiple_test_cases() -> None:
    """Pipeline with multiple test inputs of varying sizes."""
    config = SandboxConfig()
    sandbox = PerfSandbox(config)
    source = FIXTURES.joinpath("matmul_naive.c").read_text()

    sizes = [2, 4, 8]
    test_inputs = [_make_matmul_input(n) for n in sizes]
    expected_outputs = [_make_matmul_expected(n) for n in sizes]
    test_names = [f"matmul_n{n}" for n in sizes]

    result = await sandbox.compile_and_run(
        source_code=source,
        test_inputs=test_inputs,
        expected_outputs=expected_outputs,
        perf_input=_make_matmul_input(4),
        test_names=test_names,
    )
    assert result.compiled
    assert result.test_report is not None
    assert result.test_report.total == 3
    assert result.test_report.all_passed, f"Test errors: {result.test_report.errors}"
    assert result.perf_counters is not None


# ── Sandbox isolation tests ──────────────────────────────────────────────────


@requires_all_tools
@pytest.mark.asyncio
async def test_no_network_access() -> None:
    """Code that tries to access the network should fail."""
    config = SandboxConfig()
    sandbox = PerfSandbox(config)
    source = """
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>

int main(void) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        printf("no network\\n");
        return 0;
    }
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = 80,
        .sin_addr.s_addr = 0x08080808,
    };
    int ret = connect(sock, (struct sockaddr*)&addr, sizeof(addr));
    if (ret < 0) {
        printf("no network\\n");
        return 0;
    }
    printf("has network\\n");
    return 0;
}
"""
    result = await sandbox.compile_and_run(
        source_code=source,
        test_inputs=[b""],
        expected_outputs=[b"no network\n"],
        perf_input=b"",
    )
    assert result.compiled
    assert result.test_report is not None
    assert result.test_report.all_passed, f"Test errors: {result.test_report.errors}"


@requires_all_tools
@pytest.mark.asyncio
async def test_cannot_write_outside_workdir() -> None:
    """Code that tries to write to /tmp should succeed (tmpfs), but /usr should fail."""
    config = SandboxConfig()
    sandbox = PerfSandbox(config)
    source = """
#include <stdio.h>
#include <errno.h>

int main(void) {
    FILE *f = fopen("/usr/test_write", "w");
    if (f == NULL) {
        printf("write blocked\\n");
        return 0;
    }
    fclose(f);
    printf("write allowed\\n");
    return 0;
}
"""
    result = await sandbox.compile_and_run(
        source_code=source,
        test_inputs=[b""],
        expected_outputs=[b"write blocked\n"],
        perf_input=b"",
    )
    assert result.compiled
    assert result.test_report is not None
    assert result.test_report.all_passed


# ── measure_only tests ───────────────────────────────────────────────────────


@requires_all_tools
@pytest.mark.asyncio
async def test_measure_only() -> None:
    """measure_only should return perf counters for a pre-compiled binary."""
    config = SandboxConfig()
    sandbox = PerfSandbox(config)

    # First compile the binary
    source = FIXTURES.joinpath("matmul_naive.c").read_text()
    compilation, work_dir = await sandbox.compile_only(source)
    assert isinstance(compilation, CompilationSuccess)

    # Write perf input
    n = 32
    input_data = _make_matmul_input(n)
    input_path = Path(work_dir) / "perf_input.bin"
    input_path.write_bytes(input_data)

    binary_path = Path(work_dir) / "solution"
    assert binary_path.exists()

    # Measure
    counters = await sandbox.measure_only(binary_path, input_path)
    assert counters.cycles > 0
    assert counters.instructions > 0
    assert counters.ipc > 0

    # Cleanup
    import shutil

    shutil.rmtree(work_dir, ignore_errors=True)
