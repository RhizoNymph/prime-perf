"""Integration tests for PerfSandbox.check_prerequisites()."""

from __future__ import annotations

from dataclasses import replace

import pytest

from perf_optimize.config import SandboxConfig
from perf_optimize.exceptions import (
    BwrapNotFoundError,
    PerfNotFoundError,
    PrerequisiteError,
    TasksetNotFoundError,
)
from perf_optimize.sandbox import PerfSandbox
from tests.conftest import requires_all_tools


@requires_all_tools
@pytest.mark.asyncio
async def test_check_prerequisites_passes_on_valid_system() -> None:
    """On a properly configured system, check_prerequisites should not raise."""
    config = SandboxConfig()
    sandbox = PerfSandbox(config)
    await sandbox.check_prerequisites()


@pytest.mark.asyncio
async def test_bwrap_not_found_raises() -> None:
    config = SandboxConfig(bwrap_path="/nonexistent/bwrap")
    sandbox = PerfSandbox(config)
    with pytest.raises(BwrapNotFoundError):
        await sandbox.check_prerequisites()


@pytest.mark.asyncio
async def test_compiler_not_found_raises() -> None:
    """Language compiler not on PATH raises PrerequisiteError."""
    lang_with_bad_compiler = replace(SandboxConfig().language, compiler_path="/nonexistent/gcc")
    config = SandboxConfig(language=lang_with_bad_compiler)
    sandbox = PerfSandbox(config)
    with pytest.raises(PrerequisiteError):
        await sandbox.check_prerequisites()


@pytest.mark.asyncio
async def test_perf_not_found_raises() -> None:
    config = SandboxConfig(perf_path="/nonexistent/perf")
    sandbox = PerfSandbox(config)
    with pytest.raises(PerfNotFoundError):
        await sandbox.check_prerequisites()


@pytest.mark.asyncio
async def test_taskset_not_found_raises() -> None:
    config = SandboxConfig(taskset_path="/nonexistent/taskset")
    sandbox = PerfSandbox(config)
    with pytest.raises(TasksetNotFoundError):
        await sandbox.check_prerequisites()
