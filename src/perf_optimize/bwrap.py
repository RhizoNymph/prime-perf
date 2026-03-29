"""Pure functions for building bubblewrap and toolchain command lines.

This module has no side effects and performs no I/O. It constructs argument
lists suitable for ``asyncio.create_subprocess_exec``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from perf_optimize.config import SandboxConfig


def build_bwrap_command(
    config: SandboxConfig,
    work_dir: str,
    inner_command: list[str],
) -> list[str]:
    """Build the full ``taskset`` + ``bwrap`` + ``ulimit`` + command invocation.

    Merges language-specific ro-binds (e.g., rustup sysroot, nvm node dir)
    with the default ro-bind paths from config.

    Args:
        config: Sandbox configuration with paths, limits, and flags.
        work_dir: Host-side directory to bind-mount as ``/work`` inside the sandbox.
        inner_command: The command (and its arguments) to run inside the sandbox.

    Returns:
        A flat list of strings representing the full command.
    """
    cmd: list[str] = [
        config.taskset_path,
        "-c",
        str(config.pin_cpu),
        config.bwrap_path,
    ]

    # Read-only bind mounts from config
    for path in config.ro_bind_paths:
        cmd.extend(["--ro-bind", path, path])

    # Language-specific ro-binds (e.g., rustup sysroot, nvm node directory)
    for path in config.language.extra_ro_binds:
        cmd.extend(["--ro-bind", path, path])

    # Writable bind mount for the working directory
    cmd.extend(["--bind", work_dir, "/work"])

    # Virtual filesystems
    cmd.extend(["--proc", "/proc"])
    cmd.extend(["--dev", "/dev"])
    cmd.extend(["--tmpfs", "/tmp"])

    # Isolation flags
    cmd.append("--unshare-net")
    cmd.append("--unshare-pid")
    cmd.append("--new-session")
    cmd.append("--die-with-parent")

    # Working directory inside the sandbox
    cmd.extend(["--chdir", "/work"])

    # Separator between bwrap args and the inner command
    cmd.append("--")

    # Ulimit wrapper via bash -c, then exec into the real command.
    # We use bash (not sh/dash) because dash lacks ulimit -u support.
    # The '_' is the $0 placeholder for bash; "$@" expands to inner_command.
    ulimit_script = (
        f"ulimit -v {config.ulimit_mem_kb}"
        f" && ulimit -u {config.ulimit_procs}"
        f" && ulimit -f {config.ulimit_fsize_kb}"
        ' && exec "$@"'
    )
    cmd.extend(["/usr/bin/bash", "-c", ulimit_script, "_"])

    # The actual command to run inside the sandbox
    cmd.extend(inner_command)

    return cmd


def build_compile_command(config: SandboxConfig, source_file: str, output_file: str) -> list[str]:
    """Build a compilation or syntax-check command based on the language config.

    For compiled languages (C, Rust): ``[compiler, *flags, -o, output, source]``
    For interpreted languages (Python, TS): ``[compiler, *flags, source]`` (syntax check)

    Returns:
        A list of strings for the compile/check command.
    """
    lang = config.language
    assert lang.compiler_path is not None

    if lang.compiled:
        return [lang.compiler_path, *lang.compiler_flags, "-o", output_file, source_file]
    # Interpreted: syntax check only (e.g., python3 -m py_compile solution.py)
    return [lang.compiler_path, *lang.compiler_flags, source_file]


def build_run_command(config: SandboxConfig) -> list[str]:
    """Build the execution command for the solution inside the sandbox.

    For compiled languages: ``[/work/solution]``
    For interpreted languages: ``[runtime, *flags, /work/solution.py]``

    Returns:
        A list of strings for the run command.
    """
    return config.language.run_command()


def build_perf_command(config: SandboxConfig, binary_path: str | None = None) -> list[str]:
    """Build a ``perf stat`` measurement command.

    Args:
        config: Sandbox configuration.
        binary_path: Override the binary/script path. If None, uses the language
            config's default run command.

    Returns:
        A list of strings for ``perf stat -r N -x , -e <counters> -- <cmd>``.
    """
    counter_list = ",".join(config.hardware_profile.perf_events())

    run_cmd = [binary_path] if binary_path is not None else build_run_command(config)

    return [
        config.perf_path,
        "stat",
        "-r",
        str(config.perf_repeat),
        "-x",
        ",",
        "-e",
        counter_list,
        "--",
        *run_cmd,
    ]
