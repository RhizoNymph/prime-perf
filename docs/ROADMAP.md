# perf-optimize Roadmap

## Phase 0: Measurement Foundation

**Goal:** Prove that perf counters inside bwrap are reliable enough to serve as a
reward signal.

**Tasks:**

0.1. Install bubblewrap, verify `kernel.perf_event_paranoid` is set to ≤1 on all
     3090 nodes.

0.2. Write a minimal bwrap invocation script that: bind-mounts system libs read-only,
     bind-mounts a workdir read-write, unshares net/pid, sets ulimits, compiles a C
     file, runs it with `perf stat -r 5 -x ','`, and returns structured CSV output.

0.3. Write a `parse_perf_output(csv_text) -> dict[str, float]` function that extracts
     counter values from perf's CSV format.

0.4. **Variance validation:** Run the same binary (naive matmul, N=1024) 50 times
     through the full bwrap+perf pipeline. Compute coefficient of variation for each
     counter. Target: CV < 5% for cycles, < 10% for cache counters. If variance is
     too high, apply: isolcpus, disable turbo, set governor to performance, increase
     perf -r count.

0.5. Run the same binary *without* bwrap and compare counter distributions. Confirm
     bwrap adds no measurable variance.

0.6. Write a known-optimized matmul (tiled, SIMD) and confirm the counter differences
     vs naive are large enough to produce meaningful reward. Specifically: verify that
     the improvement signal (ref_cycles - opt_cycles)/ref_cycles is well above the
     measurement noise floor.

**Deliverable:** A standalone `sandbox.py` module with `PerfSandbox` class that
reliably compiles, tests, and measures C code. Variance report documenting measurement
stability.

**Exit criteria:** Measurement variance is low enough that a 10% cycle improvement
is distinguishable from noise with high confidence.

---

## Phase 1: Problem Bank (V1)

**Goal:** Create 5 well-calibrated problems with reference solutions, test suites,
and baseline perf measurements.

**Tasks:**

1.1. Define the problem specification format: spec.md template, directory structure
     conventions, test input/output format.

1.2. Implement the first problem: **matmul** (NxN float matrix multiply).
     - spec.md: problem description, I/O format (binary float arrays), constraints
     - reference.c: naive triple-nested loop, no optimizations, compiled with -O2
     - tests/: 5 input pairs (small N for correctness), expected outputs generated
       from reference solution
     - perf_input.bin: N=1024 input pair for performance measurement
     - Validate: reference compiles, passes tests, perf counters are stable

1.3. Implement **stencil** (2D 5-point stencil, iterated).
     - Memory bandwidth bound, benefits from tiling and prefetch
     - Different bottleneck profile from matmul

1.4. Implement **sort** (integer array sort, large N).
     - Branch-heavy, benefits from branchless techniques
     - Tests must verify ordering + stability if applicable

1.5. Implement **nbody** (gravitational N-body simulation, single timestep).
     - Compute + memory bound, benefits from SoA layout and SIMD
     - Float comparison tolerance in test validation

1.6. Implement **hash_table** (string key lookup benchmark).
     - Allocation + cache behavior, benefits from open addressing, cache-line
       alignment
     - Functional tests: insert N keys, look up all, verify all found

1.7. For each problem, measure and record the reference perf counters. Store these
     in a `reference_perf.json` alongside each problem. These become the baseline
     for reward computation.

1.8. Write `problems.py`: a loader that reads the problem directory structure and
     constructs a HuggingFace Dataset with prompt, answer (None), and info columns.

**Deliverable:** `problems/` directory with 5 complete problems. `problems.py` module
that produces a valid verifiers-compatible Dataset.

**Exit criteria:** All 5 problems have stable reference measurements and test suites
that catch meaningful functional regressions.

---

## Phase 2: Verifiers Environment

**Goal:** A working `PerfOptimizeEnv` that can run evaluations against API models.

**Tasks:**

2.1. Scaffold the verifiers environment module:
     ```
     environments/perf_optimize/
       perf_optimize.py
       sandbox.py       (from Phase 0)
       problems.py      (from Phase 1)
       pyproject.toml
       README.md
       problems/        (from Phase 1)
     ```

2.2. Implement `PerfOptimizeEnv(vf.MultiTurnEnv)`:
     - `__init__`: accepts language, max_turns, problem filter args
     - `setup_state`: loads problem info, initializes tracking (best_perf, turn
       counter)
     - `env_response`: extracts code via XMLParser, invokes PerfSandbox, formats
       feedback message, updates state
     - `@vf.stop agent_submitted`: detects `<submit/>` tag
     - `render_completion`: ensures final state includes best perf results

2.3. Implement the rubric:
     - `correctness_gate`: -1.0 if never correct, 0.0 otherwise
     - `perf_reward`: weighted counter improvement score
     - Zero-weight metrics: wall_clock, best_cycles, compile_count,
       correct_submission_count

2.4. Implement `load_environment(language="c", max_turns=5, problems=None)` entry
     point.

2.5. Write pyproject.toml with correct metadata, dependencies, and eval defaults.

2.6. Install and test locally:
     ```
     prime env install perf-optimize
     prime eval run perf-optimize -m gpt-5-nano  # or any API model
     ```
     Verify: multi-turn loop works, perf feedback is returned, reward is computed
     correctly, metrics are logged.

2.7. Run baseline evaluation against a capable API model (e.g., Claude Sonnet) to
     sanity-check that the environment is solvable and produces interesting reward
     variation. Record baseline scores.

2.8. **Critical validation:** Verify that async execution works correctly. The
     `env_response` must not block the event loop. Profile with multiple concurrent
     rollouts and confirm no serialization. The bwrap subprocess calls must use
     `asyncio.create_subprocess_exec`.

**Deliverable:** A complete verifiers environment module that passes `prime eval run`.

**Exit criteria:** Multi-turn loop executes end-to-end. Reward correctly reflects
perf improvements. A capable model achieves non-trivial positive reward on at least
3/5 problems.

---

## Phase 3: Local Training (Single GPU)

**Goal:** Train on a single 3090 and observe reward improvement.

**Tasks:**

3.1. Install prime-rl locally:
     ```
     prime lab setup --prime-rl
     ```

3.2. Evaluate baseline: run the Qwen3.5-27B base model (4-bit, via vLLM) against
     the environment. Record pass rate, average reward, per-problem breakdown. If
     the model can't compile code at all (>90% compile failures), proceed to 3.3.
     Otherwise skip to 3.4.

3.3. **(Conditional) SFT warmup:** Create a small dataset of (problem, naive solution,
     optimized solution) pairs. Fine-tune with SFT until the model reliably produces
     compilable code in the correct XML format. This mirrors the Wordle example's
     SFT warmup pattern. Evaluate again after SFT.

3.4. Configure prime-rl TOML for single-GPU training:
     - Model: Qwen3.5-27B with LoRA
     - Small batch size and rollouts_per_example (constrained by single GPU)
     - max_tokens: 2048
     - max_steps: 100 (initial short run)

3.5. Run training. Monitor:
     - Reward curve (should trend upward)
     - Compilation success rate (should increase or stay high)
     - Test pass rate (should increase)
     - Per-counter improvements (which counters improve first?)
     - Per-problem reward variance

3.6. Evaluate the trained checkpoint against the environment. Compare to baseline.

3.7. Analyze failure modes: what kinds of code does the model generate that fails?
     Are there systematic issues (wrong language features, format violations,
     algorithmic errors)?

**Deliverable:** A trained LoRA adapter that shows measurable reward improvement over
baseline. Training logs and analysis.

**Exit criteria:** Statistically significant reward improvement on at least 2/5
problems. The model generates compilable, correct, and measurably faster code more
often than baseline.

---

## Phase 4: Multi-GPU Training + Scaling

**Goal:** Scale training across the 3-node 3090 cluster and run a full training
campaign.

**Tasks:**

4.1. Configure prime-rl for multi-node: FSDP2 training across 3 GPUs, vLLM inference
     on one GPU, orchestrator managing the environment.

4.2. Increase batch_size and rollouts_per_example now that more compute is available.
     Target: batch_size=256, rollouts_per_example=8-16.

4.3. Run a longer training campaign (500+ steps). Experiment with:
     - Learning rate sweeps
     - Reward weight ablations (different counter weightings)
     - max_turns ablation (3 vs 5 vs 7)
     - Online difficulty filtering (if supported for custom environments)

4.4. Evaluate on held-out test inputs (different N values, different random seeds for
     input generation) to test generalization within problems.

4.5. Write up results: per-problem improvement curves, counter breakdowns, example
     optimizations the model discovers.

**Deliverable:** A well-trained model with comprehensive evaluation results.

**Exit criteria:** The trained model consistently outperforms the base model across
all 5 problems. Some generated optimizations are non-trivial (not just flag tweaks
or minor loop reordering).

---

## Phase 5: Ablations + Analysis

**Goal:** Run the ablation experiments that make this a research contribution.

**Tasks:**

5.1. **Option B (unlabeled counters):** Modify env_response to strip counter names.
     Retrain from the same base model. Compare learning curves to Option A.

5.2. **Option D (hidden counters, single-shot):** Remove multi-turn feedback entirely.
     Single-turn environment, reward computed from hidden counters. Retrain. Compare
     to Option A.

5.3. **Wall-clock evaluation:** For all trained models (A, B, D), evaluate using
     wall-clock time as the metric (not training reward). Does mechanistic counter
     reward transfer to wall-clock improvement?

5.4. **Counter weight ablation:** Train variants with different reward weight
     distributions (cycles-only, equal weights, cache-only) and compare which
     produces the best wall-clock improvements.

5.5. **Compiler flag extension:** Add compiler flags as part of the action space. The
     model can specify optimization flags (-O2, -O3, -march=native, -funroll-loops,
     -flto, -fprofile-generate/-fprofile-use) alongside source code changes. Run
     as a separate experiment to avoid confounding.

5.6. Analyze what optimizations the model discovers across ablations. Categorize:
     algorithmic changes, memory layout, SIMD, loop transformations, cache-aware
     tiling, branchless techniques. Does Option A learn different strategies than
     Option D?

5.7. Write up the ablation results into a blog post or short paper.

**Deliverable:** Ablation results with clear comparisons. Write-up suitable for a
blog post (portfolio) or workshop paper (research contribution).

**Exit criteria:** At least one ablation comparison (A vs D) produces a clear,
interpretable result about whether RL can internalize hardware performance models.

---

## Phase 6: Expansion + Publishing

**Goal:** Expand the problem bank, add languages, publish to the Environments Hub.

**Tasks:**

6.1. Add 10+ more problems covering: graph algorithms, string processing, image
     convolution, tree operations, compression, parsing.

6.2. Add Rust language support: separate reference solutions, rustc in the bwrap
     sandbox, same perf measurement pipeline. Rust is interesting because the
     optimizer is stronger (LLVM), so the gap between naive and optimal is different.

6.3. Add Python + NumPy/Cython support: tests whether the model learns to vectorize
     via NumPy or use Cython for hot loops. Different optimization vocabulary.

6.4. Publish the environment to the Prime Intellect Environments Hub:
     ```
     prime env push perf-optimize
     ```

6.5. Test on their hosted training platform. Contact Prime Intellect about perf
     access in their sandbox infrastructure (this is a great conversation starter).

6.6. Docker fallback for environments without perf access: measurement backend that
     uses wall-clock + instruction count (via `perf stat` where available, `time`
     where not). Degrades reward quality but enables broader deployment.

**Deliverable:** A polished, multi-language environment on the Environments Hub with
comprehensive documentation.

---

## Dependency Graph

```
Phase 0 (Measurement)
   │
   ▼
Phase 1 (Problems)
   │
   ▼
Phase 2 (Environment) ──────► Phase 5 (Ablations)
   │                              │
   ▼                              ▼
Phase 3 (Single-GPU Training)  Phase 6 (Expansion)
   │
   ▼
Phase 4 (Multi-GPU Scaling)
```

Phases 0-2 are sequential prerequisites. Phase 3 requires 0-2. Phase 4 requires 3.
Phase 5 can begin after Phase 3 (single-GPU ablation runs) and continues through
Phase 4. Phase 6 can begin after Phase 4 results are solid.

## Time Estimates (rough)

| Phase | Estimated Time | Notes |
|-------|---------------|-------|
| 0 | 2-3 days | Mostly validation and variance testing |
| 1 | 3-4 days | 5 problems with careful test suite design |
| 2 | 3-4 days | Verifiers integration, async correctness |
| 3 | 3-5 days | Includes possible SFT warmup |
| 4 | 1-2 weeks | Longer training runs, hyperparameter sweeps |
| 5 | 1-2 weeks | Multiple ablation runs + analysis |
| 6 | 2-3 weeks | Multi-language, polish, publishing |

Phases 0-3 (working trained model) are achievable in ~2 weeks of focused effort.
The full roadmap through Phase 6 is roughly 2 months.