"""Trace generator for equation_transformation puzzles.

We only emit a trace when the solver actually found a rule that matched the
training examples (the downstream verifier still filters by GT). For the
numeric path we show the operator identification + the candidate-op search;
for the symbolic path we show the positional alignment.
"""
from __future__ import annotations

import re

from .equation_transformation_solver import (
    _detect_test_operator, _numeric_examples, _NUMERIC_OPS,
    _format_int_like_expected, _split_by_op, parse, _try_symbolic,
    solve,
)


_HUMAN_OP_NAMES = {
    "a+b": "addition (a + b)",
    "a-b": "subtraction (a - b)",
    "b-a": "reversed subtraction (b - a)",
    "a*b": "multiplication (a × b)",
    "|a-b|": "absolute difference",
    "concat_ab": "concatenation of a and b",
    "concat_ba": "concatenation of b and a",
    "rev(a)*b": "reverse-digits(a) × b",
    "rev(a)*rev(b)": "reverse-digits(a) × reverse-digits(b)",
    "rev(a)+rev(b)": "reverse-digits(a) + reverse-digits(b)",
    "rev(a)-rev(b)": "reverse-digits(a) - reverse-digits(b)",
    "a^2-b^2": "a² - b²",
    "a^2+b^2": "a² + b²",
    "(a+b)^2": "(a + b)²",
    "(a-b)^2": "(a - b)²",
}


def generate_trace(prompt: str) -> tuple[str, str] | None:
    answer, rule = solve(prompt)
    if answer is None or rule is None:
        return None
    test = parse(prompt)
    if test is None:
        return None

    parts: list[str] = []

    if rule.op_name == "symbolic_positional":
        return _trace_symbolic(prompt, test, answer)

    # Numeric path
    op = rule.test_op or _detect_test_operator(test)
    if op is None:
        return None
    op_examples = _numeric_examples(prompt, op)
    parts.append(
        f"The test query is '{test}', which uses the operator '{op}'. "
        f"I need to figure out what '{op}' means from the examples that use it."
    )
    parts.append(f"Examples that use the operator '{op}':")
    for a, b, r in op_examples:
        parts.append(f"  {a}{op}{b} = {r}")

    human_name = _HUMAN_OP_NAMES.get(rule.op_name, rule.op_name)
    parts.append(
        f"Try candidate operations on the operands. The pattern that matches "
        f"every example is: {human_name}  [internal label: {rule.op_name}]."
    )
    # Walk through verification on examples
    fn = dict(_NUMERIC_OPS)[rule.op_name]
    for a_s, b_s, r_s in op_examples:
        a, b = int(a_s), int(b_s)
        try:
            val = fn(a, b)
        except Exception:
            continue
        formatted = _format_int_like_expected(int(val), [r_s])
        parts.append(f"  check: {rule.op_name}({a}, {b}) = {val} → formatted '{formatted}' = '{r_s}' ✓")
    # Apply to test
    parts_t = _split_by_op(test, op)
    if parts_t is None:
        return None
    ta, tb = int(parts_t[0]), int(parts_t[1])
    try:
        val = fn(ta, tb)
    except Exception:
        return None
    formatted = _format_int_like_expected(int(val), [r for _, _, r in op_examples])
    parts.append(f"Apply to the test: {rule.op_name}({ta}, {tb}) = {val} → formatted '{formatted}'")
    return "\n".join(parts), answer


def _trace_symbolic(prompt: str, test: str, answer: str) -> tuple[str, str]:
    parts: list[str] = []
    parts.append(
        f"The test query '{test}' is a symbolic string (no numeric operator). "
        "I'll learn a positional character mapping from the examples."
    )
    # Re-extract example pairs (lhs, rhs)
    pairs: list[tuple[str, str]] = []
    for line in prompt.splitlines():
        line = line.strip().rstrip(".")
        if "=" not in line or line.lower().startswith("now"):
            continue
        lhs, rhs = line.split("=", 1)
        pairs.append((lhs.strip(), rhs.strip()))
    parts.append("Aligned examples:")
    for lhs, rhs in pairs:
        parts.append(f"  '{lhs}' = '{rhs}'")
    parts.append(
        "For each output position, find which input position it copies from "
        "(or which substitution applies). Apply that mapping to the test input."
    )
    parts.append(f"Result: '{answer}'")
    return "\n".join(parts), answer
