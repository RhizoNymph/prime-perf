"""Language configurations for multi-language sandbox support.

Each language (C, Rust, Python, TypeScript) has a LanguageConfig that describes
how to compile/check, execute, and sandbox it. Predefined configs have empty
`extra_ro_binds`; use `resolve_language_config()` to detect actual runtime paths
for the current system.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, replace
from enum import StrEnum
from collections.abc import Callable
from functools import cache
from pathlib import Path


class Language(StrEnum):
    """Supported programming languages."""

    C = "c"
    RUST = "rust"
    PYTHON = "python"
    TYPESCRIPT = "typescript"


@dataclass(frozen=True)
class LanguageConfig:
    """How to compile, run, and sandbox a specific language.

    For compiled languages (C, Rust), `compiled=True` and the compiler produces
    a binary at `output_file`. For interpreted languages (Python, TS), `compiled=False`
    and the "compile" step is a syntax check; the source file is executed directly
    via `runtime_path`.
    """

    language: Language
    file_extension: str

    # Compilation / syntax check
    compiled: bool
    compiler_path: str | None
    compiler_flags: tuple[str, ...]
    linker_flags: tuple[str, ...]
    output_file: str

    # Execution (for interpreted languages)
    runtime_path: str | None
    runtime_flags: tuple[str, ...]

    # Additional ro-bind paths for bwrap (e.g., rustup sysroot, nvm dir)
    extra_ro_binds: tuple[str, ...]

    def run_command(self, work_dir_prefix: str = "/work") -> list[str]:
        """Build the command to execute the solution inside the sandbox."""
        if self.compiled:
            return [f"{work_dir_prefix}/{self.output_file}"]
        assert self.runtime_path is not None
        return [
            self.runtime_path,
            *self.runtime_flags,
            f"{work_dir_prefix}/{self.output_file}",
        ]


# ── Predefined language configs ──────────────────────────────────────────────

C_LANG = LanguageConfig(
    language=Language.C,
    file_extension=".c",
    compiled=True,
    compiler_path="gcc",
    compiler_flags=("-O2",),
    linker_flags=("-lm",),
    output_file="solution",
    runtime_path=None,
    runtime_flags=(),
    extra_ro_binds=(),
)

RUST_LANG = LanguageConfig(
    language=Language.RUST,
    file_extension=".rs",
    compiled=True,
    compiler_path="rustc",
    compiler_flags=("-O", "--edition", "2024"),
    linker_flags=(),
    output_file="solution",
    runtime_path=None,
    runtime_flags=(),
    extra_ro_binds=(),
)

PYTHON_LANG = LanguageConfig(
    language=Language.PYTHON,
    file_extension=".py",
    compiled=False,
    compiler_path="python3",
    compiler_flags=("-m", "py_compile"),
    linker_flags=(),
    output_file="solution.py",
    runtime_path="python3",
    runtime_flags=(),
    extra_ro_binds=(),
)

TYPESCRIPT_LANG = LanguageConfig(
    language=Language.TYPESCRIPT,
    file_extension=".ts",
    compiled=False,
    compiler_path="node",
    compiler_flags=("--experimental-strip-types", "--check"),
    linker_flags=(),
    output_file="solution.ts",
    runtime_path="node",
    runtime_flags=("--experimental-strip-types",),
    extra_ro_binds=(),
)

_BASE_CONFIGS: dict[Language, LanguageConfig] = {
    Language.C: C_LANG,
    Language.RUST: RUST_LANG,
    Language.PYTHON: PYTHON_LANG,
    Language.TYPESCRIPT: TYPESCRIPT_LANG,
}


# ── Dynamic path resolution ─────────────────────────────────────────────────


def _resolve_rust_ro_binds() -> tuple[str, ...]:
    """Find the rustc sysroot directory to ro-bind."""
    rustc = shutil.which("rustc")
    if rustc is None:
        return ()
    try:
        result = subprocess.run(
            [rustc, "--print", "sysroot"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            sysroot = result.stdout.strip()
            if sysroot and Path(sysroot).exists():
                return (sysroot,)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ()


def _resolve_python_ro_binds() -> tuple[str, ...]:
    """Find Python prefix and site-packages dirs to ro-bind.

    System python under /usr is already covered by the default ro-binds.
    For non-system pythons (pyenv, conda, venv), we use ``sys.prefix`` to
    discover the installation root, and only bind ``bin/`` and ``lib/``
    subdirectories when the prefix is under the user's home (to avoid
    exposing the entire home directory).
    """
    python = shutil.which("python3")
    if python is None:
        return ()

    binds: list[str] = []
    real_path = Path(python).resolve()

    if not str(real_path).startswith("/usr/"):
        try:
            result = subprocess.run(
                [str(real_path), "-c", "import sys; print(sys.prefix)"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                prefix = result.stdout.strip()
                if prefix and Path(prefix).exists():
                    home = str(Path.home())
                    if prefix.startswith(home):
                        prefix_path = Path(prefix)
                        for subdir in ("bin", "lib"):
                            sub = prefix_path / subdir
                            if sub.exists():
                                binds.append(str(sub))
                    else:
                        binds.append(prefix)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    usr_local_lib = Path("/usr/local/lib")
    if usr_local_lib.exists():
        python_dirs = list(usr_local_lib.glob("python3*"))
        if python_dirs:
            binds.append("/usr/local")

    return tuple(binds)


def _resolve_node_ro_binds() -> tuple[str, ...]:
    """Find the node installation directory to ro-bind.

    nvm/fnm install node under ~/.nvm/versions/node/vX.Y.Z/ or similar.
    We resolve the symlink chain to find the actual directory.
    """
    node = shutil.which("node")
    if node is None:
        return ()

    real_path = Path(node).resolve()

    # If node is under /usr, it's already covered by default ro-binds
    if str(real_path).startswith("/usr/"):
        return ()

    # Walk up to find the version root (e.g., ~/.nvm/versions/node/v22.12.0/)
    # The binary is typically at .../bin/node, so parent.parent gives the version dir
    version_dir = real_path.parent.parent
    if version_dir.exists():
        return (str(version_dir),)

    return ()


_RESOLVERS: dict[Language, Callable[[], tuple[str, ...]]] = {
    Language.RUST: _resolve_rust_ro_binds,
    Language.PYTHON: _resolve_python_ro_binds,
    Language.TYPESCRIPT: _resolve_node_ro_binds,
}


@cache
def resolve_language_config(language: Language) -> LanguageConfig:
    """Resolve a language config with system-specific ro-bind paths.

    This shells out once per language to discover runtime locations (e.g.,
    rustc sysroot, nvm node directory). Results are cached.
    """
    base = _BASE_CONFIGS[language]
    resolver = _RESOLVERS.get(language)
    if resolver is None:
        return base
    extra_binds = resolver()
    if not extra_binds:
        return base
    return replace(base, extra_ro_binds=base.extra_ro_binds + extra_binds)
