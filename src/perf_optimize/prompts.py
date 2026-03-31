"""System prompt and feedback message formatters for the perf-optimize environment.

All functions are pure string formatters with no external dependencies.
"""

from __future__ import annotations

SYSTEM_PROMPT_TEMPLATE = """\
You are a performance optimization expert. Your task is to optimize the given \
reference solution to run as fast as possible while maintaining correctness.

**Language:** {language}
**Turns remaining:** {max_turns}

## Interaction Protocol

On each turn, submit your optimized code inside a `<code>` tag:

```
<code lang="{language}">
// your optimized solution here
</code>
```

Your code will be compiled, tested for correctness, and measured with hardware \
performance counters. You will receive feedback showing which counters improved.

When you are satisfied with your optimization, submit your final version with:

```
<submit/>
```

## Tips

- Focus on algorithmic improvements, memory access patterns, and cache efficiency.
- The reference solution is intentionally naive — there is significant room for improvement.
- Each turn gives you updated performance counter feedback. Use it to guide your next attempt.
- Only counters available on the current hardware are shown (varies by CPU architecture).\
"""


def format_system_prompt(language: str, max_turns: int) -> str:
    """Format the system prompt with language and turn count."""
    return SYSTEM_PROMPT_TEMPLATE.format(language=language, max_turns=max_turns)


def format_compile_error(stderr: str, turn: int, max_turns: int) -> str:
    """Format feedback for a compilation failure."""
    lines = [
        f"**Compilation failed** (turn {turn}/{max_turns})",
        "",
        "```",
        stderr.strip()[:2000],
        "```",
        "",
        "Fix the compilation errors and try again.",
    ]
    return "\n".join(lines)


def format_test_failure(
    passed: int, total: int, errors: list[str], turn: int, max_turns: int
) -> str:
    """Format feedback for test failures."""
    lines = [
        f"**Tests failed: {passed}/{total} passed** (turn {turn}/{max_turns})",
    ]
    if errors:
        lines.append("")
        for err in errors[:5]:
            lines.append(f"- {err[:300]}")
    lines.extend(["", "Fix the correctness issues and try again."])
    return "\n".join(lines)


def format_perf_feedback(
    agent_counters: dict[str, float],
    ref_counters: dict[str, float],
    turn: int,
    max_turns: int,
) -> str:
    """Format performance counter feedback.

    Only shows counters that are present (not None) in both dicts.
    Shows improvement percentage relative to the reference.
    """
    lines = [
        f"**All tests passed. Performance results** (turn {turn}/{max_turns})",
        "",
    ]

    for counter in sorted(agent_counters):
        agent_val = agent_counters[counter]
        ref_val = ref_counters.get(counter)
        if ref_val is None or ref_val == 0:
            lines.append(f"  {counter}: {agent_val:,.0f}")
        else:
            improvement = (ref_val - agent_val) / ref_val * 100
            sign = "+" if improvement >= 0 else ""
            lines.append(
                f"  {counter}: {agent_val:,.0f}  ({sign}{improvement:.1f}% vs reference)"
            )

    lines.extend([
        "",
        "You can submit your solution with `<submit/>` or try to optimize further.",
    ])
    return "\n".join(lines)


def format_no_code_found(turn: int, max_turns: int) -> str:
    """Format feedback when no code block was found in the response."""
    return (
        f"**No code found** (turn {turn}/{max_turns})\n\n"
        'Please wrap your solution in `<code lang="...">...</code>` tags.'
    )
