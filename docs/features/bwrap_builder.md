# Feature: Bwrap Command Builder

## Scope

**In scope:** Building command-line argument lists (as `list[str]`) for bubblewrap sandbox invocation, GCC compilation, and perf stat measurement. Pure functions only.

**Not in scope:** Executing commands, managing subprocesses, handling stdout/stderr, or timeouts.

## Data/Control Flow

1. Caller provides a `SandboxConfig` and task-specific parameters.
2. Builder functions return `list[str]` suitable for `asyncio.create_subprocess_exec(*cmd)`.
3. No state is maintained between calls.

### `build_bwrap_command`

Constructs: `taskset -c {pin_cpu} bwrap [ro-binds] [bind work_dir] [vfs] [isolation] [chdir] -- sh -c 'ulimits && exec "$@"' _ [inner_command]`

Flow:
1. Start with `taskset -c {pin_cpu} bwrap`
2. Add `--ro-bind path path` for each path in `config.ro_bind_paths`
3. Add `--bind work_dir /work`
4. Add virtual filesystems: `--proc /proc`, `--dev /dev`, `--tmpfs /tmp`
5. Add isolation: `--unshare-net`, `--unshare-pid`, `--new-session`, `--die-with-parent`
6. Add `--chdir /work`
7. Add separator `--`
8. Add ulimit wrapper: `sh -c 'ulimit -v ... && ulimit -u ... && ulimit -f ... && exec "$@"' _`
9. Append inner_command arguments

### `build_compile_command`

Returns: `[gcc, *flags, -o, output_file, source_file]`

### `build_perf_command`

Returns: `[perf, stat, -r, N, -x, ',', -e, counter_list, --, binary_path]`

## Files

| File | Role | Key exports |
|------|------|-------------|
| `src/perf_optimize/bwrap.py` | Command builders | `build_bwrap_command`, `build_compile_command`, `build_perf_command` |
| `src/perf_optimize/config.py` | Configuration | `SandboxConfig` |
| `src/perf_optimize/types.py` | Counter enum | `PerfCounter` |
| `tests/unit/test_bwrap_builder.py` | Unit tests | 31 test cases |

## Invariants

- All three functions are pure (no I/O, no side effects).
- `build_bwrap_command` always starts with `taskset` as element [0].
- `bwrap` is always element [3] (after `taskset -c N`).
- The `_` placeholder always separates the sh -c script from the inner command.
- Inner command arguments are always the last elements of the returned list.
- All returned lists contain only `str` elements.
- The `--` separator always appears between bwrap flags and the sh -c wrapper.
