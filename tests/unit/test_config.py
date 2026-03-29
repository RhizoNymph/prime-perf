"""Tests for SandboxConfig defaults and environment variable loading."""

from __future__ import annotations

import dataclasses

import pytest

from perf_optimize.config import SandboxConfig
from perf_optimize.types import PerfCounter


class TestSandboxConfigDefaults:
    """Default config has sensible values and covers all fields."""

    def test_gcc_path(self) -> None:
        cfg = SandboxConfig()
        assert cfg.gcc_path == "gcc"

    def test_gcc_flags(self) -> None:
        cfg = SandboxConfig()
        assert cfg.gcc_flags == ("-O2", "-lm")

    def test_bwrap_path(self) -> None:
        cfg = SandboxConfig()
        assert cfg.bwrap_path == "bwrap"

    def test_perf_path(self) -> None:
        cfg = SandboxConfig()
        assert cfg.perf_path == "perf"

    def test_taskset_path(self) -> None:
        cfg = SandboxConfig()
        assert cfg.taskset_path == "taskset"

    def test_compile_timeout(self) -> None:
        cfg = SandboxConfig()
        assert cfg.compile_timeout_s == 30.0

    def test_test_timeout(self) -> None:
        cfg = SandboxConfig()
        assert cfg.test_timeout_s == 10.0

    def test_perf_timeout(self) -> None:
        cfg = SandboxConfig()
        assert cfg.perf_timeout_s == 60.0

    def test_perf_repeat(self) -> None:
        cfg = SandboxConfig()
        assert cfg.perf_repeat == 5

    def test_pin_cpu(self) -> None:
        cfg = SandboxConfig()
        assert cfg.pin_cpu == 0

    def test_ulimit_mem_kb(self) -> None:
        cfg = SandboxConfig()
        assert cfg.ulimit_mem_kb == 512_000

    def test_ulimit_procs(self) -> None:
        cfg = SandboxConfig()
        assert cfg.ulimit_procs == 32

    def test_ulimit_fsize_kb(self) -> None:
        cfg = SandboxConfig()
        assert cfg.ulimit_fsize_kb == 10_240

    def test_ro_bind_paths(self) -> None:
        cfg = SandboxConfig()
        assert "/usr" in cfg.ro_bind_paths
        assert "/lib" in cfg.ro_bind_paths
        assert "/lib64" in cfg.ro_bind_paths

    def test_perf_counters_contains_all(self) -> None:
        cfg = SandboxConfig()
        assert set(cfg.perf_counters) == set(PerfCounter)

    def test_cv_threshold_cycles(self) -> None:
        cfg = SandboxConfig()
        assert cfg.cv_threshold_cycles == pytest.approx(0.05)

    def test_cv_threshold_cache(self) -> None:
        cfg = SandboxConfig()
        assert cfg.cv_threshold_cache == pytest.approx(0.10)

    def test_frozen(self) -> None:
        cfg = SandboxConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.pin_cpu = 3  # type: ignore[misc]


class TestSandboxConfigFromEnv:
    """from_env reads PERF_OPT_ prefixed env vars."""

    def test_no_env_vars_returns_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no PERF_OPT_* vars, from_env returns the same as the default constructor."""
        # Remove any PERF_OPT_ vars that might be in the environment
        import os

        for key in list(os.environ):
            if key.startswith("PERF_OPT_"):
                monkeypatch.delenv(key)

        cfg = SandboxConfig.from_env()
        default = SandboxConfig()
        assert cfg == default

    def test_override_pin_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_PIN_CPU", "3")
        cfg = SandboxConfig.from_env()
        assert cfg.pin_cpu == 3

    def test_override_compile_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_COMPILE_TIMEOUT_S", "15.5")
        cfg = SandboxConfig.from_env()
        assert cfg.compile_timeout_s == pytest.approx(15.5)

    def test_override_test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_TEST_TIMEOUT_S", "5.0")
        cfg = SandboxConfig.from_env()
        assert cfg.test_timeout_s == pytest.approx(5.0)

    def test_override_perf_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_PERF_TIMEOUT_S", "120.0")
        cfg = SandboxConfig.from_env()
        assert cfg.perf_timeout_s == pytest.approx(120.0)

    def test_override_perf_repeat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_PERF_REPEAT", "10")
        cfg = SandboxConfig.from_env()
        assert cfg.perf_repeat == 10

    def test_override_gcc_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_GCC_PATH", "/usr/local/bin/gcc-13")
        cfg = SandboxConfig.from_env()
        assert cfg.gcc_path == "/usr/local/bin/gcc-13"

    def test_override_bwrap_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_BWRAP_PATH", "/custom/bwrap")
        cfg = SandboxConfig.from_env()
        assert cfg.bwrap_path == "/custom/bwrap"

    def test_override_perf_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_PERF_PATH", "/custom/perf")
        cfg = SandboxConfig.from_env()
        assert cfg.perf_path == "/custom/perf"

    def test_override_taskset_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_TASKSET_PATH", "/custom/taskset")
        cfg = SandboxConfig.from_env()
        assert cfg.taskset_path == "/custom/taskset"

    def test_override_ulimit_mem_kb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_ULIMIT_MEM_KB", "1024000")
        cfg = SandboxConfig.from_env()
        assert cfg.ulimit_mem_kb == 1_024_000

    def test_override_ulimit_procs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_ULIMIT_PROCS", "64")
        cfg = SandboxConfig.from_env()
        assert cfg.ulimit_procs == 64

    def test_override_ulimit_fsize_kb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_ULIMIT_FSIZE_KB", "20480")
        cfg = SandboxConfig.from_env()
        assert cfg.ulimit_fsize_kb == 20_480

    def test_override_cv_threshold_cycles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_CV_THRESHOLD_CYCLES", "0.02")
        cfg = SandboxConfig.from_env()
        assert cfg.cv_threshold_cycles == pytest.approx(0.02)

    def test_override_cv_threshold_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_CV_THRESHOLD_CACHE", "0.15")
        cfg = SandboxConfig.from_env()
        assert cfg.cv_threshold_cache == pytest.approx(0.15)

    def test_override_gcc_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_GCC_FLAGS", "-O3,-Wall,-lm")
        cfg = SandboxConfig.from_env()
        assert cfg.gcc_flags == ("-O3", "-Wall", "-lm")

    def test_multiple_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PERF_OPT_PIN_CPU", "2")
        monkeypatch.setenv("PERF_OPT_PERF_REPEAT", "3")
        monkeypatch.setenv("PERF_OPT_GCC_PATH", "/opt/gcc")
        cfg = SandboxConfig.from_env()
        assert cfg.pin_cpu == 2
        assert cfg.perf_repeat == 3
        assert cfg.gcc_path == "/opt/gcc"
        # Other fields remain default
        assert cfg.bwrap_path == "bwrap"
        assert cfg.compile_timeout_s == 30.0

    def test_unset_vars_keep_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the vars that are set should be overridden."""
        monkeypatch.setenv("PERF_OPT_PIN_CPU", "7")
        cfg = SandboxConfig.from_env()
        assert cfg.pin_cpu == 7
        assert cfg.gcc_path == "gcc"
        assert cfg.perf_repeat == 5
        assert cfg.compile_timeout_s == 30.0
