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
    stdin_file: str | None = None,
) -> list[str]:
    """Build the full ``taskset`` + ``bwrap`` + ``ulimit`` + command invocation.

    The returned list is suitable for passing to
    ``asyncio.create_subprocess_exec(*cmd)``.

    Args:
        config: Sandbox configuration with paths, limits, and flags.
        work_dir: Host-side directory to bind-mount as ``/work`` inside the sandbox.
        inner_command: The command (and its arguments) to run inside the sandbox.
        stdin_file: Optional path (inside the sandbox) to redirect as stdin.

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

    # Ulimit wrapper via sh -c, then exec into the real command.
    # The '_' is the $0 placeholder for sh; "$@" expands to inner_command.
    ulimit_script = (
        f"ulimit -v {config.ulimit_mem_kb}"
        f" && ulimit -u {config.ulimit_procs}"
        f" && ulimit -f {config.ulimit_fsize_kb}"
        ' && exec "$@"'
    )
    cmd.extend(["sh", "-c", ulimit_script, "_"])

    # The actual command to run inside the sandbox
    cmd.extend(inner_command)

    return cmd


def build_compile_command(
    config: SandboxConfig,
    source_file: str,
    output_file: str,
) -> list[str]:
    """Build a GCC compilation command.

    Returns:
        A list of strings: ``[gcc, *flags, -o, output_file, source_file]``.
    """
    return [config.gcc_path, *config.gcc_flags, "-o", output_file, source_file]


def build_perf_command(
    config: SandboxConfig,
    binary_path: str,
    stdin_file: str | None = None,
) -> list[str]:
    """Build a ``perf stat`` measurement command.

    Returns:
        A list of strings for ``perf stat -r N -x , -e <counters> -- <binary>``.
    """
    counter_list = ",".join(c.value for c in config.perf_counters)
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
        binary_path,
    ]
