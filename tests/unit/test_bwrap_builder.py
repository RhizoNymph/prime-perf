"""Tests for bwrap and toolchain command builders.

All tests exercise pure functions with no system dependencies.
"""

from __future__ import annotations

from perf_optimize.bwrap import (
    build_bwrap_command,
    build_compile_command,
    build_perf_command,
)
from perf_optimize.config import SandboxConfig
from perf_optimize.counters import CounterMapping, HardwareProfile

# ═══════════════════════════════════════════════════════════════════════════════
# build_bwrap_command
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildBwrapCommand:
    """Tests for the full bwrap command builder."""

    def setup_method(self) -> None:
        self.config = SandboxConfig()
        self.work_dir = "/tmp/perf-work-1234"
        self.inner_cmd = ["./solution"]

    def test_taskset_is_first_element(self) -> None:
        cmd = build_bwrap_command(self.config, self.work_dir, self.inner_cmd)
        assert cmd[0] == "taskset"
        assert cmd[1] == "-c"
        assert cmd[2] == str(self.config.pin_cpu)

    def test_bwrap_follows_taskset(self) -> None:
        cmd = build_bwrap_command(self.config, self.work_dir, self.inner_cmd)
        assert cmd[3] == "bwrap"

    def test_all_ro_bind_paths_present(self) -> None:
        cmd = build_bwrap_command(self.config, self.work_dir, self.inner_cmd)
        for path in self.config.ro_bind_paths:
            # Find the specific --ro-bind for this path
            found = False
            for i, arg in enumerate(cmd):
                if arg == "--ro-bind" and i + 2 < len(cmd) and cmd[i + 1] == path:
                    assert cmd[i + 2] == path, f"ro-bind dest mismatch for {path}"
                    found = True
                    break
            assert found, f"--ro-bind not found for path: {path}"

    def test_bind_work_dir(self) -> None:
        cmd = build_bwrap_command(self.config, self.work_dir, self.inner_cmd)
        bind_idx = cmd.index("--bind")
        assert cmd[bind_idx + 1] == self.work_dir
        assert cmd[bind_idx + 2] == "/work"

    def test_proc_dev_tmpfs_present(self) -> None:
        cmd = build_bwrap_command(self.config, self.work_dir, self.inner_cmd)
        assert "--proc" in cmd
        proc_idx = cmd.index("--proc")
        assert cmd[proc_idx + 1] == "/proc"

        assert "--dev" in cmd
        dev_idx = cmd.index("--dev")
        assert cmd[dev_idx + 1] == "/dev"

        assert "--tmpfs" in cmd
        tmpfs_idx = cmd.index("--tmpfs")
        assert cmd[tmpfs_idx + 1] == "/tmp"

    def test_isolation_flags_present(self) -> None:
        cmd = build_bwrap_command(self.config, self.work_dir, self.inner_cmd)
        assert "--unshare-net" in cmd
        assert "--unshare-pid" in cmd
        assert "--new-session" in cmd
        assert "--die-with-parent" in cmd

    def test_chdir_to_work(self) -> None:
        cmd = build_bwrap_command(self.config, self.work_dir, self.inner_cmd)
        chdir_idx = cmd.index("--chdir")
        assert cmd[chdir_idx + 1] == "/work"

    def test_separator_present(self) -> None:
        cmd = build_bwrap_command(self.config, self.work_dir, self.inner_cmd)
        assert "--" in cmd

    def test_ulimit_values_in_bash_script(self) -> None:
        cmd = build_bwrap_command(self.config, self.work_dir, self.inner_cmd)
        separator_idx = cmd.index("--")
        # After -- comes: /usr/bin/bash -c '<script>' _ <inner_cmd...>
        assert cmd[separator_idx + 1] == "/usr/bin/bash"
        assert cmd[separator_idx + 2] == "-c"
        script = cmd[separator_idx + 3]
        assert f"ulimit -v {self.config.ulimit_mem_kb}" in script
        assert f"ulimit -u {self.config.ulimit_procs}" in script
        assert f"ulimit -f {self.config.ulimit_fsize_kb}" in script
        assert 'exec "$@"' in script

    def test_placeholder_underscore_present(self) -> None:
        cmd = build_bwrap_command(self.config, self.work_dir, self.inner_cmd)
        separator_idx = cmd.index("--")
        # sh -c '<script>' _ <inner_cmd>
        assert cmd[separator_idx + 4] == "_"

    def test_inner_command_at_end(self) -> None:
        cmd = build_bwrap_command(self.config, self.work_dir, self.inner_cmd)
        assert cmd[-1] == "./solution"

    def test_multi_arg_inner_command(self) -> None:
        inner = ["./solution", "--flag", "value"]
        cmd = build_bwrap_command(self.config, self.work_dir, inner)
        assert cmd[-3:] == ["./solution", "--flag", "value"]

    def test_custom_pin_cpu(self) -> None:
        config = SandboxConfig(pin_cpu=3)
        cmd = build_bwrap_command(config, self.work_dir, self.inner_cmd)
        assert cmd[2] == "3"

    def test_custom_ulimit_values(self) -> None:
        config = SandboxConfig(
            ulimit_mem_kb=1_000_000,
            ulimit_procs=64,
            ulimit_fsize_kb=20_480,
        )
        cmd = build_bwrap_command(config, self.work_dir, self.inner_cmd)
        separator_idx = cmd.index("--")
        script = cmd[separator_idx + 3]
        assert "ulimit -v 1000000" in script
        assert "ulimit -u 64" in script
        assert "ulimit -f 20480" in script

    def test_custom_ro_bind_paths(self) -> None:
        config = SandboxConfig(ro_bind_paths=("/usr", "/custom/path"))
        cmd = build_bwrap_command(config, self.work_dir, self.inner_cmd)
        # Check /custom/path is bound
        found = False
        for i, arg in enumerate(cmd):
            if arg == "--ro-bind" and i + 1 < len(cmd) and cmd[i + 1] == "/custom/path":
                found = True
                break
        assert found

    def test_returns_list_of_strings(self) -> None:
        cmd = build_bwrap_command(self.config, self.work_dir, self.inner_cmd)
        assert isinstance(cmd, list)
        assert all(isinstance(s, str) for s in cmd)


# ═══════════════════════════════════════════════════════════════════════════════
# build_compile_command
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildCompileCommand:
    """Tests for compile command builder (language-aware)."""

    def test_c_default_config(self) -> None:
        config = SandboxConfig()  # default language is C
        cmd = build_compile_command(config, "solution.c", "solution")
        assert cmd == ["gcc", "-O2", "-lm", "-o", "solution", "solution.c"]

    def test_rust_config(self) -> None:
        from perf_optimize.languages import Language, resolve_language_config

        config = SandboxConfig(language=resolve_language_config(Language.RUST))
        cmd = build_compile_command(config, "solution.rs", "solution")
        assert cmd[0] == "rustc"
        assert "-O" in cmd
        assert cmd[-1] == "solution.rs"

    def test_python_syntax_check(self) -> None:
        from perf_optimize.languages import Language, resolve_language_config

        config = SandboxConfig(language=resolve_language_config(Language.PYTHON))
        cmd = build_compile_command(config, "solution.py", "solution.py")
        assert cmd[0] == "python3"
        assert "-m" in cmd
        assert "py_compile" in cmd
        assert cmd[-1] == "solution.py"
        assert "-o" not in cmd  # interpreted: no output binary

    def test_output_file_placement(self) -> None:
        config = SandboxConfig()
        cmd = build_compile_command(config, "input.c", "output_bin")
        o_idx = cmd.index("-o")
        assert cmd[o_idx + 1] == "output_bin"

    def test_source_file_is_last(self) -> None:
        config = SandboxConfig()
        cmd = build_compile_command(config, "main.c", "main")
        assert cmd[-1] == "main.c"


# ═══════════════════════════════════════════════════════════════════════════════
# build_perf_command
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildPerfCommand:
    """Tests for perf stat command builder."""

    def test_default_config(self) -> None:
        config = SandboxConfig()
        cmd = build_perf_command(config, "./solution")
        assert cmd[0] == "perf"
        assert cmd[1] == "stat"

    def test_repeat_flag(self) -> None:
        config = SandboxConfig(perf_repeat=5)
        cmd = build_perf_command(config, "./solution")
        r_idx = cmd.index("-r")
        assert cmd[r_idx + 1] == "5"

    def test_csv_separator_flag(self) -> None:
        config = SandboxConfig()
        cmd = build_perf_command(config, "./solution")
        x_idx = cmd.index("-x")
        assert cmd[x_idx + 1] == ","

    def test_events_flag_contains_all_profile_events(self) -> None:
        config = SandboxConfig()
        cmd = build_perf_command(config, "./solution")
        e_idx = cmd.index("-e")
        events_str = cmd[e_idx + 1]
        for event in config.hardware_profile.perf_events():
            assert event in events_str, f"Missing event: {event}"

    def test_events_comma_separated(self) -> None:
        config = SandboxConfig()
        cmd = build_perf_command(config, "./solution")
        e_idx = cmd.index("-e")
        events_str = cmd[e_idx + 1]
        events = events_str.split(",")
        assert len(events) == len(config.hardware_profile.perf_events())

    def test_double_dash_separator(self) -> None:
        config = SandboxConfig()
        cmd = build_perf_command(config, "./solution")
        assert "--" in cmd

    def test_binary_after_separator(self) -> None:
        config = SandboxConfig()
        cmd = build_perf_command(config, "./solution")
        sep_idx = cmd.index("--")
        assert cmd[sep_idx + 1] == "./solution"

    def test_custom_perf_path(self) -> None:
        config = SandboxConfig(perf_path="/usr/local/bin/perf")
        cmd = build_perf_command(config, "./binary")
        assert cmd[0] == "/usr/local/bin/perf"

    def test_custom_repeat_count(self) -> None:
        config = SandboxConfig(perf_repeat=10)
        cmd = build_perf_command(config, "./binary")
        r_idx = cmd.index("-r")
        assert cmd[r_idx + 1] == "10"

    def test_subset_of_counters(self) -> None:
        minimal_profile = HardwareProfile(
            name="minimal",
            vendor="TestVendor",
            counters=(
                CounterMapping("cycles", "cycles"),
                CounterMapping("instructions", "instructions"),
            ),
        )
        config = SandboxConfig(hardware_profile=minimal_profile)
        cmd = build_perf_command(config, "./binary")
        e_idx = cmd.index("-e")
        events_str = cmd[e_idx + 1]
        assert events_str == "cycles,instructions"

    def test_returns_list_of_strings(self) -> None:
        config = SandboxConfig()
        cmd = build_perf_command(config, "./solution")
        assert isinstance(cmd, list)
        assert all(isinstance(s, str) for s in cmd)
