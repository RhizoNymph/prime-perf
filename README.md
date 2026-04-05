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

### Run Evaluations

Use `prime eval run` to evaluate any model against the environment:

```bash
# Evaluate with an API model
prime eval run perf-optimize -e sonnet

# Filter to specific problems or languages
prime eval run perf-optimize -e haiku \
  --env.args '{"language": "c", "problems": ["matmul", "sort"]}'

# View results
prime eval tui
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

## Training with prime-rl

The environment integrates with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) for RL training. prime-rl uses a disaggregated architecture with three components: an **inference server** (vLLM), an **orchestrator** (rollout scheduling + environment), and a **trainer** (FSDP2 policy optimization).

### Prerequisites

```bash
# Set up a prime lab workspace (installs prime-rl and CLI tools)
prime lab setup --prime-rl

# Install the perf-optimize environment into prime-rl's venv
cd prime-rl && uv pip install -e ../
```

Ensure the training nodes have:
- `kernel.perf_event_paranoid` <= 1
- `bwrap` and `perf` installed
- CPU governor set to `performance`, turbo boost disabled (for stable measurements)
- Compilers for target language(s): `gcc` (C), `rustc` (Rust), `python3`, `node`/`npx tsx`

### Training Configs

Three ready-to-use configs live in `configs/prime-rl/`:

| Config | Model | Hardware | Purpose |
|--------|-------|----------|---------|
| `perf-optimize-local.toml` | Qwen3-1.7B (LoRA) | 1x 24GB GPU (shared) | Local iteration, sanity-check reward signal |
| `perf-optimize-multinode.toml` | Qwen3-4B (LoRA) | 2 nodes x 1 GPU | Validate multi-node path before hosted runs |
| `perf-optimize-hosted.toml` | Qwen3-14B | 2 nodes x 8 GPUs | Scaled run on Prime Intellect hosted training |

Each config is fully commented with launch instructions at the top.

### Local Single-GPU (small model)

Trainer + inference share one GPU by limiting vLLM's memory usage:

```bash
cd prime-rl

# tmux session layout
bash scripts/tmux.sh

uv run rl \
  @ ../configs/prime-rl/perf-optimize-local.toml \
  --trainer-gpu-ids 0 \
  --inference-gpu-ids 0 \
  --inference.gpu-memory-utilization 0.5
```

Or start each component manually in its own tmux pane:

```bash
# Inference pane
uv run inference @ ../configs/prime-rl/perf-optimize-local.toml --gpu-memory-utilization 0.5

# Orchestrator pane
uv run orchestrator @ ../configs/prime-rl/perf-optimize-local.toml

# Trainer pane
uv run trainer @ ../configs/prime-rl/perf-optimize-local.toml
```

### Two-Node (medium model, inference on separate node)

Inference on node 0, trainer on node 1. Requires a shared filesystem.

```bash
# On all nodes
export OUTPUT_DIR=/shared/perf-optimize-multinode
export INFERENCE_SERVER_IP=<private-ip-of-node-0>
export INFERENCE_SERVER_API_KEY=<shared-secret>

# Node 0: inference
cd prime-rl
uv run inference @ ../configs/prime-rl/perf-optimize-multinode.toml \
  --api-key $INFERENCE_SERVER_API_KEY

# Node 0 or 1: orchestrator (single instance)
uv run orchestrator @ ../configs/prime-rl/perf-optimize-multinode.toml \
  --client.base-url http://$INFERENCE_SERVER_IP:8000/v1 \
  --client.api-key-var INFERENCE_SERVER_API_KEY \
  --output-dir $OUTPUT_DIR

# Node 1: trainer
uv run torchrun \
  --nproc-per-node 1 \
  --local-rank-filter 0 \
  src/prime_rl/trainer/rl/train.py \
  @ ../configs/prime-rl/perf-optimize-multinode.toml \
  --output-dir $OUTPUT_DIR
```

### Hosted Scaled Run (Prime Intellect platform)

The hosted config uses `multi_node` deployment with dedicated inference and trainer pools. Push the environment first, then launch through the hosted dashboard or CLI:

```bash
# Auth
prime login

# Push the environment so worker nodes can install it
prime env push --path .

# Launch (also works as a multi-node SLURM job)
cd prime-rl
uv run rl @ ../configs/prime-rl/perf-optimize-hosted.toml
```

Metrics stream to wandb and to the Prime Intellect platform via `[orchestrator.prime_monitor]`. Note: hosted run registration is currently allowlist-only.

### SFT Warmup (Optional)

If the base model can't reliably produce compilable code in the expected XML format, do an SFT warmup first:

```bash
uv run sft @ ../configs/perf_optimize/sft.toml
```

Then point the RL config's `[model].name` at the SFT checkpoint.

### Environment Arguments

The environment accepts these arguments via the `args` field in the training config:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `language` | string | `"c"` | Target language: `"c"`, `"rust"`, `"python"`, `"typescript"` |
| `max_turns` | int | `5` | Max optimization turns per problem |
| `problems` | list | all | Filter to specific problems: `["matmul", "sort", ...]` |

### Monitoring

```bash
# Check wandb for reward curves, compilation success rate, test pass rate

# View eval results
prime eval tui

# Monitor per-counter improvements to understand what the model learns:
# - cycles (weight 0.5): overall compute reduction
# - L1-dcache-load-misses (0.2): cache optimization
# - cache-misses/LLC-load-misses (0.2): memory access patterns
# - branch-misses (0.1): branch prediction improvements
```

## Reward

Reward is gated behind correctness:
- **-1.0** -- never compiled successfully
- **-0.5** -- compiled but never passed all tests
- **0.0** -- correct but no improvement over reference
- **> 0.0** -- weighted improvement across hardware counters (cycles 0.5, L1-dcache 0.2, LLC 0.2, branch 0.1), floored at 0.0

Counter weights automatically renormalize when a counter is unavailable on the hardware (e.g. AMD lacks `LLC-load-misses`; its weight redistributes to the remaining counters).

## Project Structure

```
src/perf_optimize/
├── __init__.py     # load_environment() entry point
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
configs/            # Endpoint and training configs
prime-rl/           # prime-rl checkout (via prime lab setup)
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
