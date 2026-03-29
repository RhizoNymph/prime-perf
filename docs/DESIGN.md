# perf-optimize: RL Environment for Teaching LLMs to Write Performant Code

## Overview

`perf-optimize` is a reinforcement learning environment built on Prime Intellect's
verifiers SDK and prime-rl training framework. It teaches LLM agents to optimize code
for hardware performance by providing structured feedback from Linux `perf` hardware
performance counters.

The agent receives a problem specification and a naive reference solution, then
iteratively rewrites the solution to improve performance. Reward is derived from
hardware counter improvements (cycles, cache misses, branch mispredictions), gated
behind correctness verification via test suites.

## Core Design Decisions

### Task: Optimization from Reference (not from-scratch generation)

The agent is given a correct but naive solution and must produce a faster version.
This scopes the problem to *performance optimization* rather than simultaneously
learning correctness and performance, which would make reward extremely sparse.

The naive reference serves as both a correctness anchor (the agent can see what
correct output looks like) and a performance floor (reward is measured relative to
reference perf counters).

### Interaction: Multi-Turn with Labeled Perf Counters (Option A)

The agent gets structured feedback after each code submission: compilation status,
test results, and if all tests pass, labeled hardware performance counters. This is
the richest learning signal and the most sample-efficient starting point.

Each episode allows 5 turns. The agent can submit early by including a `<submit/>`
tag.

The labeled counters are: cycles, instructions, IPC (derived), L1-dcache-load-misses,
LLC-load-misses, and branch-misses. These are shown to the agent in a human-readable
format alongside the improvement percentage vs the reference solution.

### Ablation Variants (Future Work)

The labeled-counter design (Option A) enables a clean ablation ladder:

- **Option B (unlabeled counters):** Same numbers, stripped of names (metric_1,
  metric_2, ...). Tests whether the model can discover what each metric means from
  correlation with code changes.
- **Option D (hidden counters, single-shot):** The model never sees perf feedback at
  inference time. Hardware counters only shape the reward during training. Tests
  whether RL can bake performance intuitions directly into model weights.
- **A vs B:** Does the model need to understand what metrics mean?
- **A vs D:** How much improvement comes from inference-time feedback vs training-time
  internalization?
- **B vs D:** Does unlabeled multi-turn feedback help beyond what's in the weights?

### Reward: Multi-Stage Gated + Weighted Counter Improvements

Reward is terminal (computed once at episode end based on the best correct submission).
No intermediate per-turn rewards to avoid incentivizing mediocre early submissions.

**Gating:**
- Code does not compile → reward = -1.0
- Code compiles but fails tests → reward = -0.5 * (1 - tests_passed/tests_total)
- Code passes all tests → performance reward kicks in

**Performance reward (only when all tests pass):**
```
improvements = {
    "cycles":               (ref - agent) / ref,
    "L1-dcache-load-misses": (ref - agent) / ref,
    "LLC-load-misses":       (ref - agent) / ref,
    "branch-misses":         (ref - agent) / ref,
}

reward = (
    0.5 * improvements["cycles"] +
    0.2 * improvements["L1-dcache-load-misses"] +
    0.2 * improvements["LLC-load-misses"] +
    0.1 * improvements["branch-misses"]
)
reward = max(reward, 0.0)  # floor at 0 for correct but slower code
```

Cycles dominate because they're closest to wall-clock time. The decomposed component
signals provide denser gradient for specific optimization types (e.g., cache misses
respond to memory layout changes even when overall cycles haven't improved much yet).

**Wall-clock time** is tracked as a zero-weight metric for analysis and evaluation but
does not contribute to the training reward. This creates an interesting evaluation
dynamic: training on mechanistic hardware signals, evaluating on wall-clock.

### Framework: Prime Intellect verifiers + prime-rl

The environment is built as a verifiers environment module, publishable to the
Environments Hub. Training uses prime-rl's async GRPO implementation.

The environment is a custom `MultiTurnEnv` subclass (not `ToolEnv` or `SandboxEnv`).
The interaction protocol is text-based: the model writes code in `<code>` tags, the
environment compiles/runs/measures it, and returns structured text feedback. No
tool-calling schema is needed.

### Model: Qwen3.5-27B with 4-bit QLoRA

- **Base model:** Qwen3.5-27B (dense, released Feb 2026)
- **Quantization:** 4-bit NormalFloat (NF4) via bitsandbytes, ~14-15GB VRAM for weights
- **Adapter:** LoRA on all linear layers, r=64, α=16
- **Compute dtype:** bfloat16 for forward/backward pass
- **Training:** GRPO via prime-rl on 3x RTX 3090 (24GB each)
- **Learning rate:** ~1e-4 (higher than standard due to quantization noise)

QLoRA is well-suited here: the quantization noise may actually enhance exploration in
RL (per QeRL findings). The primary bottleneck is sandbox execution time, not model
generation speed, so the NF4 dequantization overhead is negligible.

### Sandboxing: Bubblewrap (bwrap)

All generated code executes inside a bubblewrap sandbox. bwrap provides filesystem
namespace isolation with ~8ms startup overhead (vs Docker's 200-600ms), zero runtime
overhead, and no effect on perf measurement variance.

**Isolation guarantees:**
- Filesystem: read-only access to system libs (/usr, /lib, /etc), writable only in
  the working directory
- Network: fully disabled (--unshare-net)
- Process: isolated PID namespace (--unshare-pid)
- Terminal: new session (--new-session) prevents injection
- Lifecycle: --die-with-parent ensures cleanup on orchestrator crash

**Resource limits (via ulimit inside bwrap):**
- Memory: 500MB (ulimit -v 512000)
- Processes: 32 max (ulimit -u 32), prevents fork bombs
- File writes: 10MB max (ulimit -f 10240)
- Wall clock: timeout per compilation (30s) and per test (10s)

**Perf access:** Requires `kernel.perf_event_paranoid <= 1` on the host. bwrap uses
the same kernel, so hardware counters are accessible directly.

### Measurement Methodology

Variance reduction for reliable reward signals:
- CPU pinning via `taskset -c 0`
- CPU frequency governor set to `performance`
- Turbo boost disabled
- `perf stat -r 5` for 5 statistical repetitions (medians preferred)
- `perf stat -x ','` for machine-parseable CSV output
- `isolcpus` kernel parameter recommended for dedicated measurement core

Counters collected:
- `cycles` — total CPU cycles
- `instructions` — total instructions retired
- `cache-references` — total cache accesses
- `cache-misses` — last-level cache misses
- `L1-dcache-load-misses` — L1 data cache misses
- `LLC-load-misses` — last-level cache load misses
- `branch-misses` — branch mispredictions

IPC (instructions per cycle) is derived as instructions/cycles.

## Architecture

### Environment Module Structure

```
environments/
  perf_optimize/
    perf_optimize.py       # Main environment: PerfOptimizeEnv class + load_environment()
    sandbox.py             # PerfSandbox: bwrap + compilation + perf measurement
    problems.py            # Problem bank loader and dataset construction
    pyproject.toml         # Package metadata and dependencies
    README.md              # Documentation
    problems/              # Problem definitions
      matmul/
        spec.md            # Problem specification (natural language)
        reference.c        # Naive reference solution
        tests/
          inputs/          # Test input files
          expected/        # Expected outputs
        perf_input.bin     # Larger input for performance measurement
      nbody/
        ...
```

### Environment Class: PerfOptimizeEnv

Subclasses `vf.MultiTurnEnv`. Key methods:

**`load_environment(language="c", max_turns=5, problems=None)`**
Entry point. Constructs the dataset from the problem bank, configures the rubric,
and returns the environment instance.

**`setup_state(state)`**
Per-rollout initialization: parses problem info from the dataset row, pre-computes
reference perf counters (or loads cached values), initializes tracking state
(best_cycles, best_perf, turn counter).

**`env_response(messages, state)`**
Core interaction loop:
1. Extract code from model response using XMLParser (`<code>` tags)
2. If no code found, return error message
3. Write code to temp directory, invoke PerfSandbox
4. If compilation fails, return compiler errors
5. If tests fail, return first failing test info
6. If tests pass, return labeled perf counters + improvement percentage
7. Update state with latest perf results

**`@vf.stop` methods:**
- `agent_submitted(state)` — model included `<submit/>` tag
- Built-in `max_turns` guard from MultiTurnEnv base class

### Sandbox: PerfSandbox

A helper class wrapping bubblewrap + gcc + perf stat. All methods are async-safe
(using `asyncio.create_subprocess_exec` or `asyncio.to_thread`).

**`compile_and_run(code, problem_info) -> ExecutionResult`**
1. Create tmpdir, write source file
2. Copy test inputs and perf input into tmpdir
3. Invoke bwrap with: ro-bind system libs, bind tmpdir as /work, unshare-net/pid
4. Inside bwrap: compile with timeout, run tests with timeout, run perf stat
5. Parse perf CSV output into structured dict
6. Clean up tmpdir
7. Return ExecutionResult dataclass

```python
@dataclass
class ExecutionResult:
    compiled: bool
    compiler_errors: str | None
    tests_passed: int
    tests_total: int
    test_errors: list[str]
    perf_counters: dict[str, float] | None   # None if tests didn't all pass
    wall_clock_ms: float | None
```

### Rubric

```python
rubric = vf.Rubric(
    funcs=[correctness_gate, perf_reward],
    weights=[1.0, 1.0]
)
rubric.add_metric(wall_clock_metric)
rubric.add_metric(best_cycles_metric)
rubric.add_metric(num_compilations_metric)
rubric.add_metric(num_correct_submissions_metric)
```

**`correctness_gate(state)`** — Returns -1.0 if the agent never produced correct code
in the entire episode, 0.0 otherwise. Ensures the dominant failure mode (never
compiling or never passing tests) is strongly penalized.

**`perf_reward(completion, info, state)`** — Computes the weighted counter improvement
score from the agent's best correct submission vs the reference. Only fires when
`state["best_perf"]` is set (i.e., at least one correct submission occurred).

**Zero-weight metrics** for observability: wall_clock_ms, best cycle count,
compilation count, correct submission count. These appear in eval/training logs but
don't affect reward.

### Dataset / Problem Bank

Each problem defines:
- **spec.md:** Natural language description of the problem, input/output format,
  constraints
- **reference.{c,rs,py}:** A correct but deliberately naive solution (e.g., textbook
  triple-nested-loop matrix multiply)
- **tests/:** Input/output pairs for correctness verification. Must cover edge cases
  but run fast (~seconds total)
- **perf_input:** A larger input sized to make performance differences measurable and
  exercise relevant hardware bottlenecks (caches, branch predictor)

**Initial problem set (5 problems, C language):**

| Problem | Bottleneck Type | Key Optimization Opportunities |
|---------|----------------|-------------------------------|
| matmul | Memory-bound (cache) | Loop tiling, cache-oblivious, SIMD |
| nbody | Compute + memory | SoA layout, SIMD, algorithmic (Barnes-Hut) |
| sort | Branch-heavy | Branchless comparisons, cache-aware merge |
| stencil | Memory bandwidth | Tiling, prefetching, vectorization |
| hash_table | Allocation + cache | Open addressing, cache-line alignment, power-of-2 sizing |

Problems are selected to cover distinct bottleneck types so the agent must learn
different optimization strategies rather than one universal trick.

### Prompt Construction

The initial prompt for each episode is constructed from the problem bank:

```
[system message]
You are a performance optimization expert. You will receive a problem
specification and a naive reference solution with its hardware performance
profile. Your goal is to write an optimized version that passes all tests
and achieves better performance.

Wrap your code in <code lang="c"> tags. When you are satisfied with your
solution, include <submit/> in your response. You have {max_turns} turns.

Think step by step about what bottlenecks exist in the reference solution
and how to address them.

[user message]
## Problem: {problem_name}

{spec.md contents}

## Reference Solution
```c
{reference.c contents}
```

## Reference Performance
  cycles:               {ref_cycles:,}
  instructions:         {ref_instructions:,}
  IPC:                  {ref_ipc:.2f}
  L1-dcache-load-misses: {ref_l1_misses:,}
  LLC-load-misses:      {ref_llc_misses:,}
  branch-misses:        {ref_branch_misses:,}

Write an optimized solution.
```

### Training Configuration

prime-rl TOML config (approximate):

```toml
model = "Qwen/Qwen3.5-27B"
max_steps = 500
batch_size = 128
rollouts_per_example = 8

[sampling]
max_tokens = 2048

[lora]
enabled = true
rank = 64
alpha = 16

[[env]]
id = "perf-optimize"
args = { language = "c", max_turns = 5 }

[wandb]
project = "perf-optimize"
```

## Key Technical Risks

1. **Measurement variance:** If perf counter readings are too noisy, the reward
   signal will be unreliable. Mitigation: aggressive variance reduction (CPU pinning,
   governor, turbo off, perf -r 5), and validate variance is acceptable before
   training.

2. **Problem difficulty calibration:** If the base model can't write compilable C
   code at all, the reward is -1.0 for every episode. Mitigation: evaluate baseline
   performance with `prime eval run` before training. If needed, do SFT warmup on
   correct optimization examples (similar to the Wordle example in prime-rl).

3. **Reward hacking:** The agent might find ways to improve counter values without
   meaningful optimization (e.g., reducing instruction count by removing
   functionality). Mitigation: correctness gating via test suites. Tests must be
   thorough enough to catch functional regressions.

4. **Sandbox + perf interaction:** perf_event_paranoid settings, capabilities, or
   kernel restrictions might interfere with perf inside bwrap. Mitigation: test this
   early and in isolation before building the environment.

5. **Generation quality vs training signal:** If the model produces non-compiling
   code >90% of the time, training signal is too sparse. Mitigation: SFT warmup,
   or start with a stronger code model (Qwen3-Coder-Next 80B-A3B as alternative).

## Related Work

- **CodeRL / RLTF:** RL on code with unit test pass rates as reward. Binary
  correctness only, no performance signal.
- **AlphaCode / AlphaCode 2:** Generate-and-filter approach. No RL on performance.
- **MLGO:** ML-guided compiler optimization (Google). Operates on compiler IR, not
  source code.
- **Halide / TVM autotuning:** Learned optimization of tensor programs. Compiler-level
  transformations, not source-level.
- **QeRL:** Quantization-enhanced RL. Demonstrates that NF4 quantization noise can
  improve RL exploration — directly relevant to our QLoRA training setup.

The key novelty of perf-optimize is using decomposed hardware performance counters as
a mechanistic reward signal for source-level code optimization. Existing work uses
either wall-clock time (noisy, non-diagnostic) or operates at the compiler IR level.