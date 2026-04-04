"""Configuration for the perf-optimize sandbox and measurement pipeline.

Configuration is loaded from environment variables with the PERF_OPT_ prefix,
falling back to sensible defaults.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import structlog

from .counters import HardwareProfile, detect_profile
from .languages import Language, LanguageConfig, resolve_language_config

logger = structlog.get_logger(__name__)


def _default_ro_bind_paths() -> tuple[str, ...]:
    candidates = (
        "/usr", "/bin", "/sbin",
        "/lib", "/lib64",
        "/etc/alternatives", "/etc/ld.so.cache",
    )
    return tuple(p for p in candidates if Path(p).exists())


def _default_hardware_profile() -> HardwareProfile:
    return detect_profile()


def _default_language() -> LanguageConfig:
    return resolve_language_config(Language.C)


@cache
def _detect_unshare_net(bwrap_path: str = "bwrap") -> bool:
    """Check whether ``bwrap --unshare-net`` works on this system.

    On some kernels/distros (e.g. Ubuntu 24.04+ with AppArmor restricting
    unprivileged user namespaces), ``--unshare-net`` fails with::

        bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted

    Returns False in that case so the sandbox can run without network isolation.
    """
    try:
        result = subprocess.run(
            [bwrap_path, "--unshare-net", "--ro-bind", "/usr", "/usr",
             "--", "/usr/bin/true"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return False


@dataclass(frozen=True)
class SandboxConfig:
    """Immutable configuration for the sandbox and measurement pipeline.

    All fields have sensible defaults. Use ``from_env()`` to create an
    instance with overrides read from PERF_OPT_* environment variables.
    """

    language: LanguageConfig = field(default_factory=_default_language)
    bwrap_path: str = "bwrap"
    perf_path: str = "perf"
    taskset_path: str = "taskset"
    compile_timeout_s: float = 30.0
    test_timeout_s: float = 10.0
    perf_timeout_s: float = 60.0
    perf_repeat: int = 5
    pin_cpu: int = 0
    ulimit_mem_kb: int = 512_000
    ulimit_procs: int = 32
    ulimit_fsize_kb: int = 10_240
    unshare_net: bool = True
    ro_bind_paths: tuple[str, ...] = field(default_factory=_default_ro_bind_paths)
    hardware_profile: HardwareProfile = field(default_factory=_default_hardware_profile)
    cv_threshold_cycles: float = 0.05
    cv_threshold_cache: float = 0.10

    @classmethod
    def from_env(cls, language: Language = Language.C) -> SandboxConfig:
        """Create a config, overriding defaults with PERF_OPT_* environment variables.

        Args:
            language: Which language to configure. Paths are resolved dynamically.

        Supported variables (all optional):
            PERF_OPT_BWRAP_PATH, PERF_OPT_PERF_PATH, PERF_OPT_TASKSET_PATH,
            PERF_OPT_COMPILE_TIMEOUT_S, PERF_OPT_TEST_TIMEOUT_S,
            PERF_OPT_PERF_TIMEOUT_S, PERF_OPT_PERF_REPEAT, PERF_OPT_PIN_CPU,
            PERF_OPT_ULIMIT_MEM_KB, PERF_OPT_ULIMIT_PROCS,
            PERF_OPT_ULIMIT_FSIZE_KB, PERF_OPT_CV_THRESHOLD_CYCLES,
            PERF_OPT_CV_THRESHOLD_CACHE
        """
        kwargs: dict[str, object] = {"language": resolve_language_config(language)}

        # String paths
        for attr in ("bwrap_path", "perf_path", "taskset_path"):
            env_val = os.environ.get(f"PERF_OPT_{attr.upper()}")
            if env_val is not None:
                kwargs[attr] = env_val

        # Float fields
        for attr in (
            "compile_timeout_s",
            "test_timeout_s",
            "perf_timeout_s",
            "cv_threshold_cycles",
            "cv_threshold_cache",
        ):
            env_val = os.environ.get(f"PERF_OPT_{attr.upper()}")
            if env_val is not None:
                kwargs[attr] = float(env_val)

        # Int fields
        for attr in ("perf_repeat", "pin_cpu", "ulimit_mem_kb", "ulimit_procs", "ulimit_fsize_kb"):
            env_val = os.environ.get(f"PERF_OPT_{attr.upper()}")
            if env_val is not None:
                kwargs[attr] = int(env_val)

        # Allow explicit override of --unshare-net via env var
        unshare_env = os.environ.get("PERF_OPT_UNSHARE_NET")
        if unshare_env is not None:
            kwargs["unshare_net"] = unshare_env.lower() in ("1", "true", "yes")

        return cls(**kwargs)
