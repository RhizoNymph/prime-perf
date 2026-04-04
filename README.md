# perf-optimize

An RL environment for teaching LLMs to write performant code using hardware performance counters.

Built on [Prime Intellect's verifiers SDK](https://github.com/primeintellect-ai/verifiers), `perf-optimize` gives LLM agents structured feedback from Linux `perf` hardware counters (cycles, cache misses, branch mispredictions) as they iteratively optimize code. The agent receives a problem spec and a naive reference solution, then rewrites it to improve performance across multiple turns.

## How It Works

1. The agent receives a **problem specification** and a **naive reference solution** in C, Rust, Python, or TypeScript
2. Each turn, the agent submits optimized code inside `<code>` tags
3. The environment **compiles**, **tests for correctness**, and **measures hardware counters** via `perf stat` in a bubblewrap sandbox
4. The agent gets structured feedback: compilation errors, test failures, or labeled perf counter improvements vs. the reference
5. After up to 5 turns (or early `<submit/>`), reward is computed from weighted counter improvements, gated behind correctness

## Problem Set

| Problem | Bottleneck | Optimization Opportunities |
|---------|-----------|---------------------------|
| **matmul** | Cache/memory | Loop tiling, cache-oblivious, SIMD |
| **nbody** | Compute + memory | SoA layout, SIMD, algorithmic (Barnes-Hut) |
| **sort** | Branch-heavy | Branchless comparisons, cache-aware merge sort |
| **stencil** | Memory bandwidth | Tiling, prefetching, vectorization |
| **hash_table** | Cache/allocation | Open addressing, alignment, power-of-2 sizing |

Each problem includes a natural-language spec, reference solutions in 4 languages, binary test cases, and a larger perf input.

## Architecture

```
┌─────────────────────────────────────────────┐
│  PerfOptimizeEnv (verifiers MultiTurnEnv)    │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ TurnProcessor│  │   Reward Computation │  │
│  │  compile     │  │  weighted counters   │  │
│  │  test        │  │  correctness gate    │  │
│  │  measure     │  │  improvement vs ref  │  │
│  └──────┬──────┘  └──────────────────────┘  │
│         │                                    │
│  ┌──────▼──────────────────────────────────┐ │
│  │  PerfSandbox (async)                    │ │
│  │  bubblewrap isolation │ perf stat -r 5  │ │
│  │  CPU pinning          │ resource limits │ │
│  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Key design choices:**
- **Bubblewrap sandboxing** -- 8ms overhead (vs 200-600ms Docker), zero impact on perf measurement
- **Hardware-aware profiles** -- auto-detects AMD Zen vs Intel Core, limits events to PMU slot count to avoid multiplexing
- **Async throughout** -- all subprocess calls use asyncio for multi-rollout parallelism
- **Typed counters** -- `None` fields for unavailable counters; rewards renormalize weights over what's present

## Setup

### System Requirements

- **Linux** with `perf` and `bwrap` (bubblewrap) installed
- **Python** >= 3.13
- **Compilers**: `gcc` (C), `rustc` (Rust), `python3` (Python), `node`/`npx tsx` (TypeScript)
- **Recommended**: `kernel.perf_event_paranoid <= 1`, CPU frequency governor set to `performance`, turbo boost disabled

### Install

```bash
# Install with uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Verify Prerequisites

```bash
pytest tests/integration/test_prerequisites.py -v
```

## Usage

### Load the Environment

```python
from perf_optimize import load_environment

env = load_environment(
    language="c",
    max_turns=5,
    problems=["matmul", "sort"],  # optional filter
)
```

### Run Tests

```bash
# Unit tests (no system dependencies)
pytest

# Integration tests (requires bwrap + gcc + perf)
pytest -m integration

# Variance validation (slow, checks CV < 2%)
pytest -m variance
```

### Configuration

All settings have sensible defaults and can be overridden via `PERF_OPT_*` environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PERF_OPT_BWRAP_PATH` | `bwrap` | Path to bubblewrap binary |
| `PERF_OPT_PERF_PATH` | `perf` | Path to perf binary |
| `PERF_OPT_COMPILE_TIMEOUT_S` | `30` | Compilation timeout (seconds) |
| `PERF_OPT_TEST_TIMEOUT_S` | `10` | Per-test timeout (seconds) |
| `PERF_OPT_PERF_TIMEOUT_S` | `60` | Perf measurement timeout (seconds) |
| `PERF_OPT_PERF_REPEAT` | `5` | Number of perf stat repetitions |
| `PERF_OPT_PIN_CPU` | `0` | CPU core to pin measurements to |
| `PERF_OPT_ULIMIT_MEM_KB` | `512000` | Memory limit per process |
| `PERF_OPT_ULIMIT_PROCS` | `32` | Max processes in sandbox |

## Reward

Reward is gated behind correctness:
- **-1.0** -- never compiled successfully
- **-0.5** -- compiled but never passed all tests
- **0.0** -- correct but no improvement over reference
- **> 0.0** -- weighted improvement across hardware counters (cycles 0.5, L1-dcache 0.2, LLC 0.2, branch 0.1), floored at 0.0

## Project Structure

```
src/perf_optimize/
├── env.py          # PerfOptimizeEnv (verifiers MultiTurnEnv)
├── processor.py    # TurnProcessor: compile -> test -> measure
├── sandbox.py      # PerfSandbox: async bwrap + perf stat
├── reward.py       # Weighted counter improvement
├── types.py        # PerfCounters, ExecutionResult, etc.
├── counters.py     # AMD Zen / Intel Core hardware profiles
├── languages.py    # C, Rust, Python, TypeScript configs
├── perf_parser.py  # Parse perf stat CSV output
├── bwrap.py        # Bubblewrap command builders
├── config.py       # SandboxConfig with env var overrides
├── comparison.py   # Output comparison (exact/tolerance)
├── prompts.py      # System prompt and feedback formatters
└── problems.py     # Problem bank loader

problems/           # 5 problems x 4 languages
tests/
├── unit/           # No sandbox required
├── integration/    # Requires bwrap + gcc + perf
└── validation/     # Variance and signal detection
```

## Roadmap

- [x] **Phase 0** -- Measurement foundation (validated CV < 2% on AMD Zen4)
- [x] **Phase 1** -- Problem bank (5 problems x 4 languages)
- [x] **Phase 2** -- Verifiers environment with multi-turn feedback
- [ ] **Phase 3** -- Single-GPU training (Qwen3.5-27B + QLoRA)
- [ ] **Phase 4** -- Multi-GPU scaling (3x RTX 3090, FSDP2)
- [ ] **Phase 5** -- Ablations (labeled vs unlabeled vs hidden counters)
- [ ] **Phase 6** -- Expansion, cross-language analysis, Hub publication

## License

TBD
