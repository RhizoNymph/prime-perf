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

MultiTurnEnv.rollout(input, client, model, sampling_args)  [framework @final]
│
├─ init_state(input, client, model, sampling_args)
│   └─ creates State with client, model, trajectory=[], timing, etc.
├─ setup_state(state)  [PerfOptimizeEnv override]
│   ├─ decode base64: test_inputs, expected_outputs, perf_input
│   └─ init: best_perf_dict=None, submitted=False, counters=0
│
└─ LOOP while not is_completed(state):  [framework-managed]
    │
    ├─ get_prompt_messages(state)  [framework]
    │   ├─ turn 0: returns state["prompt"]
    │   └─ turn N>0: calls env_response(prev_messages, state)
    │
    ├─ env_response(messages, state)  [PerfOptimizeEnv override]
    │   │
    │   ├─ _extract_code(content) → source code or None
    │   │   └─ if None: format_no_code_found()
    │   │
    │   ├─ await _process_turn(content, state, turn, max_turns)
    │   │   ├─ delegates to TurnProcessor.process() (processor.py)
    │   │   │   ├─ await sandbox.compile_and_run(code, tests, perf_input)
    │   │   │   ├─ CompilationFailure? → TurnOutcome(compile_failures_delta=1)
    │   │   │   ├─ Tests failed? → TurnOutcome(test_failures_delta=1)
    │   │   │   └─ Tests passed → TurnOutcome(correct_submissions_delta=1,
    │   │   │                      best_perf_dict, format_perf_feedback)
    │   │   └─ applies state_updates from TurnOutcome (_delta suffix = increment)
    │   │
    │   └─ if <submit/> or at max turns:
    │       ├─ state["submitted"] = True
    │       └─ state["final_env_response"] = feedback_msgs
    │
    ├─ if final_env_response set: skip model call, loop to is_completed
    ├─ get_model_response(state, prompt_messages)  [framework]
    └─ add_model_response(state, prompt, response)  [framework]
```

## Files and Key Exports

| File | Role | Key Exports |
|------|------|-------------|
| `src/perf_optimize/env.py` | Environment class | `PerfOptimizeEnv`, `PerfOptimizeState`, `_extract_code`, `_has_submit` |
| `src/perf_optimize/processor.py` | Turn processing logic | `TurnProcessor`, `TurnOutcome`, `_REWARDED_COUNTERS` |
| `src/perf_optimize/reward.py` | Reward functions | `compute_weighted_improvement`, `correctness_gate`, `perf_reward`, `PERF_WEIGHT_MAP` |
| `src/perf_optimize/prompts.py` | Feedback formatters | `format_system_prompt`, `format_compile_error`, `format_test_failure`, `format_perf_feedback`, `format_no_code_found` |
| `src/perf_optimize/__init__.py` | Entry point | `load_environment()` |

## Invariants and Constraints

- Problems without `reference_perf` baselines are loaded with a warning — `perf_reward()`
  returns 0.0 for them, degrading to correctness-only training.
- `<submit/>` must appear on its own line (with optional whitespace) to trigger episode
  termination. Inline mentions in prose or inside `<code>` blocks are ignored.
- `env_response()` is async, called by the framework's `get_prompt_messages()` after
  each model turn. It returns `Messages` (feedback) and mutates state in-place.
- `max_turns_reached` stop condition is overridden to return `False` — termination is
  handled in `env_response()` via `state["final_env_response"]` to ensure the last
  model response is always compiled/tested/measured before the rubric scores.
- The framework's `@final rollout()` is not overridden; the standard `MultiTurnEnv`
  loop manages the turn cycle.
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
