# perf-optimize: RL Environment for Teaching LLMs to Write Performant Code

## Overview

`perf-optimize` is a reinforcement learning environment built on Prime Intellect's
verifiers SDK and prime-rl training framework. It teaches LLM agents to optimize code
for hardware performance by providing structured feedback from Linux `perf` hardware
performance counters.

The agent receives a problem specification and a naive reference solution in one of
four languages (C, Rust, Python, TypeScript), then iteratively rewrites the solution
to improve performance. Reward is derived from hardware counter improvements (cycles,
cache misses, branch mispredictions), gated behind correctness verification via test
suites.

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

The reward is computed from a hardware-profile-aware weight map. Counters that are
`None` (unavailable on the current architecture) are skipped, and weights are
normalized over the available counters.

```
# Define weights per counter. Weights are normalized over available counters.
weight_map = {
    "cycles":               0.5,
    "l1_dcache_load_misses": 0.2,
    "llc_load_misses":       0.2,   # None on AMD → skipped
    "cache_misses":          0.2,   # AMD substitute for LLC signal
    "branch_misses":         0.1,
}

total_weight = 0
weighted_sum = 0
for counter, weight in weight_map.items():
    ref_val = getattr(ref_counters, counter)
    agent_val = getattr(agent_counters, counter)
    if ref_val is not None and agent_val is not None and ref_val > 0:
        improvement = (ref_val - agent_val) / ref_val
        weighted_sum += weight * improvement
        total_weight += weight

reward = max(weighted_sum / total_weight, 0.0) if total_weight > 0 else 0.0
```

On AMD, `llc_load_misses` is `None` (no separate LLC counter), so the effective
weights become cycles=0.5, l1_dcache=0.2, cache_misses=0.2, branch=0.1 normalized
over total_weight=1.0. On Intel, `llc_load_misses` is available and `cache_misses`
overlaps semantically — the weight map may need tuning per architecture to avoid
double-counting.

Cycles dominate because they're closest to wall-clock time. The decomposed component
signals provide denser gradient for specific optimization types (e.g., cache misses
respond to memory layout changes even when overall cycles haven't improved much yet).

**Wall-clock time** is tracked as a zero-weight metric for analysis and evaluation but
does not contribute to the training reward. This creates an interesting evaluation
dynamic: training on mechanistic hardware signals, evaluating on wall-clock.

### Multi-Language Support

The environment supports four languages: **C**, **Rust**, **Python**, and **TypeScript**.
Each problem has reference solutions in all four languages.

| Language | Compile | Run | Perf Signal | Optimization Space |
|----------|---------|-----|-------------|-------------------|
| C | gcc → binary | `./solution` | Best (direct HW) | Memory layout, SIMD, cache tiling |
| Rust | rustc → binary | `./solution` | Best (direct HW) | Same as C + iterator patterns, unsafe |
| Python | py_compile (syntax) | `python3 solution.py` | Good (NumPy = C under hood) | Loops → NumPy vectorization |
| TypeScript | node --check (syntax) | `node --experimental-strip-types solution.ts` | Moderate (V8 JIT noise) | Algorithm, data structures, typed arrays |

**Binary I/O format** is shared across all languages: int32 for N, then float32 arrays.
Each language has its native way to read/write binary (fread, std::io, sys.stdin.buffer,
Buffer/Float32Array). Expected test outputs are language-independent (same computation,
same result).

**LanguageConfig** defines how to compile, run, and sandbox each language. Runtime paths
(rustup sysroot, nvm node directory) are resolved dynamically from the system, not
hardcoded.

**Perf measurement for interpreted languages:** perf stat measures the full process
(interpreter + computation). For Python, NumPy-vectorized code shows dramatically
fewer cycles than loop-based code because NumPy calls optimized BLAS/LAPACK. For
TypeScript, V8's JIT adds warmup noise; `perf stat -r 5` helps average it out.

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
- **Host CPUs:** AMD (Zen4 architecture) — affects available perf counters
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

**Shell:** Uses `/usr/bin/bash` (not `sh`/`dash`) for the ulimit wrapper inside bwrap
because dash lacks `ulimit -u` (process limit) support.

**CSV format note:** With `-r` (repeat), perf stat inserts a variance field (with `%`
suffix) at position 3, shifting run_time and percentage right. The parser detects
this by checking if field[3] ends with `%`.

### Measurement Methodology

Variance reduction for reliable reward signals:
- CPU pinning via `taskset -c 0`
- CPU frequency governor set to `performance`
- Turbo boost disabled
- `perf stat -r 5` for 5 statistical repetitions (medians preferred)
- `perf stat -x ','` for machine-parseable CSV output
- `isolcpus` kernel parameter recommended for dedicated measurement core
- `kernel.perf_event_paranoid` must be ≤1 on all nodes

**Hardware profiles:** The counter set is architecture-dependent. A `HardwareProfile`
maps logical counter fields to hardware perf events and is auto-detected from the CPU
vendor. Only counters known to be supported are requested, and the count must fit
within the PMU's general-purpose counter slots to avoid multiplexing (which causes
`<not counted>` errors and unreliable measurements).

**AMD Zen4 profile (5 events, fits 6 PMU counters):**
- `cycles` — total CPU cycles
- `instructions` — total instructions retired
- `cache-misses` — L3 cache misses (generic event, = LLC misses on AMD)
- `L1-dcache-load-misses` — L1 data cache misses
- `branch-misses` — branch mispredictions
- `LLC-load-misses` — **not available** on AMD (field stays `None`)
- `cache-references` — **omitted** to avoid PMU contention

**Intel Core profile (7 events):**
- `cycles`, `instructions`, `cache-references`, `cache-misses`,
  `L1-dcache-load-misses`, `LLC-load-misses`, `branch-misses`

`PerfCounters` fields are `float | None`. `None` means the counter is not available
on the current hardware — distinct from `0` which means measured and zero events
occurred. Reward computation must skip `None` counters and redistribute weight.

IPC (instructions per cycle) is derived as instructions/cycles.

## Architecture

### Environment Module Structure

```
src/perf_optimize/
  __init__.py
  types.py               # PerfCounters (float|None), ExecutionResult, etc.
  exceptions.py          # Structured error hierarchy
  counters.py            # HardwareProfile, AMD_ZEN, INTEL_CORE, detect_profile()
  config.py              # SandboxConfig with hardware_profile
  perf_parser.py         # Parse perf stat CSV output using HardwareProfile
  bwrap.py               # Command builders for bwrap, gcc, perf stat
  sandbox.py             # PerfSandbox: async orchestration

environments/
  perf_optimize/
    perf_optimize.py     # PerfOptimizeEnv (Phase 2)
    problems.py          # Problem bank loader (Phase 1)
    problems/            # Problem definitions
      matmul/
        spec.md          # Problem specification (natural language)
        reference.c      # Naive reference solution
        tests/
          inputs/        # Binary test input files
          expected/      # Expected outputs (generated by running reference.c)
        perf_input.bin   # Larger input for performance measurement
      nbody/
        ...
```

**Note:** Expected test outputs must be generated by compiling and running the
reference C program, not computed independently (e.g., not via NumPy), because
float32 accumulation order differs between implementations.

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
    compilation: CompilationSuccess | CompilationFailure
    test_report: TestReport | None           # None if compilation failed
    perf_counters: PerfCounters | None       # None if tests didn't all pass
    wall_clock_ms: float | None

# PerfCounters fields are float | None (None = counter unavailable on this hardware)
@dataclass
class PerfCounters:
    cycles: float                            # Always present
    instructions: float                      # Always present
    cache_references: float | None           # None on AMD (omitted for PMU fit)
    cache_misses: float | None               # L3 misses on AMD
    l1_dcache_load_misses: float | None
    llc_load_misses: float | None            # None on AMD
    branch_misses: float | None
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

**`perf_reward(completion, info, state)`** — Computes the profile-aware weighted
counter improvement score from the agent's best correct submission vs the reference.
Only fires when `state["best_perf"]` is set. Skips `None` counters and normalizes
weights over available counters (see reward formula above).

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
  L1-dcache-load-misses: {ref_l1_misses:,}       (if available)
  LLC-load-misses:      {ref_llc_misses:,}        (if available, Intel only)
  cache-misses:         {ref_cache_misses:,}      (if available)
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
   training. **Phase 0 finding:** CV < 2% for cycles on AMD Zen4 with 5-event profile.
   Signal/noise ratio > 50x for tiled vs naive matmul.

2. **Problem difficulty calibration:** If the base model can't write compilable C
   code at all, the reward is -1.0 for every episode. Mitigation: evaluate baseline
   performance with `prime eval run` before training. If needed, do SFT warmup on
   correct optimization examples (similar to the Wordle example in prime-rl).

3. **Reward hacking:** The agent might find ways to improve counter values without
   meaningful optimization (e.g., reducing instruction count by removing
   functionality). Mitigation: correctness gating via test suites. Tests must be
   thorough enough to catch functional regressions.

4. **Sandbox + perf interaction:** ~~perf_event_paranoid settings, capabilities, or
   kernel restrictions might interfere with perf inside bwrap.~~ **Resolved in Phase
   0:** bwrap uses the host kernel so perf works directly. Requires
   `perf_event_paranoid ≤ 1`. Uses `/usr/bin/bash` (not dash) for ulimit wrapper.

5. **Generation quality vs training signal:** If the model produces non-compiling
   code >90% of the time, training signal is too sparse. Mitigation: SFT warmup,
   or start with a stronger code model (Qwen3-Coder-Next 80B-A3B as alternative).

6. **PMU counter multiplexing:** Requesting more events than the CPU has PMU counter
   slots causes perf to multiplex, producing `<not counted>` errors and inaccurate
   readings. **Resolved:** Hardware profiles limit event count to fit PMU slots
   (5 on AMD Zen4, 7 on Intel Core). `<not counted>` is now a hard error.

7. **Architecture-dependent counter availability:** AMD lacks `LLC-load-misses`;
   Intel lacks some AMD-specific events. **Resolved:** `HardwareProfile` auto-detects
   CPU vendor and selects the right counter set. `PerfCounters` uses `float | None`
   so unavailable counters are explicitly represented, not silently zeroed.

8. **Float32 precision across implementations:** NumPy's `a @ b` uses different
   accumulation order than C's triple-nested loop, producing bitwise-different results
   even with the same inputs. **Resolved:** Test expected outputs must be generated
   by running the reference C program, never computed independently in Python.

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