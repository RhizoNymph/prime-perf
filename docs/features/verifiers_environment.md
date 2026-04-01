# Verifiers Environment

## Scope

Multi-turn RL environment (`PerfOptimizeEnv`) built on the verifiers SDK that lets
LLM agents iteratively optimize code using hardware performance counter feedback.

**In scope:** environment lifecycle, rollout loop, reward computation, feedback
formatting, state management, code extraction.

**Not in scope:** sandbox execution (see sandbox.md), problem loading (see
problem_bank.md), hardware profiles, training configuration.

## Data/Control Flow

```
load_environment(language, max_turns, problems_dir, problems)
│
├─ SandboxConfig.from_env(language)
├─ PerfSandbox(config)
├─ build_dataset_rows(problems_dir, language, profile)
│   └─ returns [{question, answer, info}, ...]
├─ format_system_prompt(language, max_turns)
├─ Rubric([correctness_gate, perf_reward], [1.0, 1.0])
│
└─ PerfOptimizeEnv.__init__()
    └─ super().__init__(dataset, system_prompt, rubric, max_turns)
        └─ format_dataset() wraps "question" → "prompt" messages

PerfOptimizeEnv.rollout(client, model, prompt, answer, info)
│
├─ setup_state(state)
│   ├─ decode base64: test_inputs, expected_outputs, perf_input
│   └─ init: best_perf_dict=None, submitted=False, counters=0
│
└─ LOOP while not is_completed and turn < max_turns:
    │
    ├─ get_model_response() → assistant message
    ├─ state["turn"] += 1
    │
    ├─ is_completed? (state["submitted"] or <submit/> in message)
    │   └─ if True: BREAK
    │
    └─ await _async_env_response(messages, state)
        │
        ├─ _extract_code(content) → source code or None
        │   └─ if None: format_no_code_found()
        │
        ├─ await sandbox.compile_and_run(code, tests, perf_input)
        │
        ├─ CompilationFailure?
        │   ├─ state["compile_failures"] += 1
        │   └─ format_compile_error(stderr)
        │
        ├─ Tests failed?
        │   ├─ state["test_failures"] += 1
        │   └─ format_test_failure(passed, total, errors)
        │
        └─ Tests passed:
            ├─ state["correct_submissions"] += 1
            ├─ update best_perf_dict if better
            └─ format_perf_feedback(agent_counters, ref_counters)
```

## Files and Key Exports

| File | Role | Key Exports |
|------|------|-------------|
| `src/perf_optimize/env.py` | Environment class | `PerfOptimizeEnv`, `_extract_code`, `_has_submit` |
| `src/perf_optimize/reward.py` | Reward functions | `compute_weighted_improvement`, `correctness_gate`, `perf_reward`, `PERF_WEIGHT_MAP` |
| `src/perf_optimize/prompts.py` | Feedback formatters | `format_system_prompt`, `format_compile_error`, `format_test_failure`, `format_perf_feedback`, `format_no_code_found` |
| `src/perf_optimize/__init__.py` | Entry point | `load_environment()` |

## Invariants and Constraints

- Problems without `reference_perf` baselines are loaded with a warning — `perf_reward()`
  returns 0.0 for them, degrading to correctness-only training.
- `<submit/>` must appear on its own line (with optional whitespace) to trigger episode
  termination. Inline mentions in prose or inside `<code>` blocks are ignored.
- `env_response()` is sync (ABC requirement) but raises `NotImplementedError` — the
  async `_async_env_response()` is called from the overridden `rollout()` instead.
- `rollout()` is a copy of `MultiTurnEnv.rollout()` with one change: `await
  self._async_env_response()` replaces `self.env_response()`.
- Only chat message format is supported (`message_type="chat"`).
- Code is extracted via regex `<code(?:\s+lang="[^"]*")?>\s*(.*?)\s*</code>` — not
  using `XMLParser` because our tag has attributes.
- Reward functions use `**_kwargs` to absorb extra arguments from verifiers' rubric
  signature introspection.
- `compute_weighted_improvement` automatically skips counters missing from either dict
  and renormalizes weights. On AMD (no `llc_load_misses`), the weight redistributes.
- `correctness_gate` returns -1.0 / -0.5 / 0.0; `perf_reward` returns 0.0+. Combined
  range is [-1.0, ~1.0].
- `best_perf_dict` tracks the best correct submission by lowest cycle count.
- Dataset rows use `"question"` key (not `"prompt"`) so verifiers' `format_dataset()`
  wraps them with the system prompt.
