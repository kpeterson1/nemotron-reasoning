"""Trace generator for equation_transformation v2 solver.

Shows the model:
  1. Identifying the test operator.
  2. Listing the example equations that share that operator.
  3. Trying the chosen operation with reverse-operands and reverse-result flags.
  4. Verifying on each example.
  5. Applying to the test query.

Mirrors the structure of v1's trace generator but accounts for the new
rev_ops / rev_res / fmt fields on EqRule.
"""
from __future__ import annotations

from .equation_transformation_solver_v2 import (
    EqRule,
    _all_ops,
    _detect_test_operator,
    _digit_reverse,
    _extract_numeric_examples,
    _format_int_like_expected,
    _format_with_sign,
    _split_by_op,
    _strip_sign,
    parse,
    solve,
)


_HUMAN_OP_NAMES = {
    "a+b": "addition (a + b)",
    "a-b": "subtraction (a - b)",
    "b-a": "reversed subtraction (b - a)",
    "a*b": "multiplication (a × b)",
    "|a-b|": "absolute difference |a - b|",
    "-|a-b|": "negated absolute difference -(|a - b|)",
    "concat_ab": "concatenation of a and b",
    "concat_ba": "concatenation of b and a",
    "a+b+1":  "addition plus 1",
    "a+b-1":  "addition minus 1",
    "a-b+1":  "subtraction plus 1",
    "a-b-1":  "subtraction minus 1",
    "a*b+1":  "multiplication plus 1",
    "a*b-1":  "multiplication minus 1",
    "determinant": "2-digit determinant: a[0]*b[1] - a[1]*b[0]",
    "abs_det":     "absolute 2-digit determinant",
    "cross_mul":   "cross multiply: a[0]*b[0] + a[1]*b[1]",
    "cross_mul_rev": "cross multiply rev: a[0]*b[1] + a[1]*b[0]",
    "digit_mul":     "digit-wise multiply (concat)",
    "digit_mul_rev": "digit-wise multiply (cross-concat)",
    "digit_abs_diff": "digit-wise absolute differences (concat)",
    "digit_add_mod10": "digit-wise addition mod 10 (concat)",
    "digit_sub_mod10": "digit-wise subtraction mod 10 (concat)",
    "digit_sum_diff":  "(a's digit sum) - (b's digit sum)",
    "digit_sum_sum":   "(a's digit sum) + (b's digit sum)",
    "digit_prod_diff": "(a's digit product) - (b's digit product)",
    "digit_prod_sum":  "(a's digit product) + (b's digit product)",
    "ds(a*b)":         "digit sum of a*b",
    "dp(a*b)":         "digit product of a*b",
    "ds(a+b)":         "digit sum of a+b",
    "ds(|a-b|)":       "digit sum of |a-b|",
    "ds(a)*ds(b)":     "(digit sum of a) × (digit sum of b)",
    "ds(a)+ds(b)":     "(digit sum of a) + (digit sum of b)",
    "a%b":   "modulo a mod b",
    "b%a":   "modulo b mod a",
    "a//b":  "integer division a // b",
    "b//a":  "integer division b // a",
    "max%min": "max(a,b) mod min(a,b)",
    "a^2-b^2": "a² - b²",
    "a^2+b^2": "a² + b²",
    "(a+b)^2": "(a + b)²",
    "(a-b)^2": "(a - b)²",
    "|a^2-b^2|": "|a² - b²|",
    "a^b_xor": "bitwise XOR of a and b",
    "a&b":     "bitwise AND of a and b",
    "a|b":     "bitwise OR of a and b",
}


def _human(op_name: str) -> str:
    return _HUMAN_OP_NAMES.get(op_name, op_name)


def _trace_symbolic(prompt: str, test: str, answer: str) -> tuple[str, str]:
    parts: list[str] = []
    parts.append(
        f"The test query '{test}' is a symbolic string (no numeric operator). "
        "I'll learn a positional character mapping from the examples."
    )
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
        "For each output position, find which input position it copies "
        "from. Apply that mapping to the test input."
    )
    parts.append(f"Result: '{answer}'")
    return "\n".join(parts), answer


def generate_trace(prompt: str) -> tuple[str, str] | None:
    answer, rule = solve(prompt)
    if answer is None or rule is None:
        return None
    test = parse(prompt)
    if test is None:
        return None
    if rule.op_name == "symbolic_positional":
        return _trace_symbolic(prompt, test, answer)

    op = rule.test_op or _detect_test_operator(test)
    if op is None:
        return None

    op_examples_raw = _extract_numeric_examples(prompt, op)
    parts: list[str] = []
    parts.append(
        f"The test query is '{test}', which uses the operator '{op}'. "
        f"I need to figure out what '{op}' means from the examples that use it."
    )

    # Format observations.
    if rule.fmt != "num":
        parts.append(
            f"Notice some example results have a sign indicator "
            f"(fmt={rule.fmt}). I'll treat that as a negative-result marker."
        )

    parts.append(f"Examples that use the operator '{op}':")
    for a, b, r in op_examples_raw:
        parts.append(f"  {a}{op}{b} = {r}")

    # Build the canonicalised group (strip sign indicators back to signed-int).
    canon_group = [
        (a, b, _strip_sign(r, op, rule.fmt)) for a, b, r in op_examples_raw
    ]

    transform_flags = []
    if rule.rev_ops:
        transform_flags.append("reverse the digits of both operands first")
    if rule.rev_res:
        transform_flags.append("reverse the digits of the result")
    transform_desc = (
        ", ".join(transform_flags) if transform_flags else "no operand/result reversal"
    )
    parts.append(
        f"After trying candidate operations, the rule that matches every "
        f"example is: {_human(rule.op_name)} (with {transform_desc})."
    )

    # Verification per example.
    for a, b, expected_canon in canon_group:
        ra = a[::-1] if rule.rev_ops else a
        rb = b[::-1] if rule.rev_ops else b
        ai, bi = int(ra), int(rb)
        ops = _all_ops(ai, bi, ra, rb)
        raw = next((v for n, v in ops if n == rule.op_name), None)
        if raw is None:
            return None
        final = _digit_reverse(raw) if rule.rev_res else raw
        ok = "✓" if final == expected_canon else "✗"
        if rule.rev_ops:
            parts.append(
                f"  check: {a}->{ra}, {b}->{rb}; {rule.op_name}({ra}, {rb}) = "
                f"{raw}" + (f" -reverse-> {final}" if rule.rev_res else "")
                + f" → expected {expected_canon} {ok}"
            )
        else:
            parts.append(
                f"  check: {rule.op_name}({a}, {b}) = {raw}"
                + (f" -reverse-> {final}" if rule.rev_res else "")
                + f" → expected {expected_canon} {ok}"
            )

    # Apply to the test.
    ta_tb = _split_by_op(test, op)
    if ta_tb is None:
        return None
    ta, tb = ta_tb
    rta = ta[::-1] if rule.rev_ops else ta
    rtb = tb[::-1] if rule.rev_ops else tb
    ai, bi = int(rta), int(rtb)
    ops = _all_ops(ai, bi, rta, rtb)
    raw = next((v for n, v in ops if n == rule.op_name), None)
    if raw is None:
        return None
    final = _digit_reverse(raw) if rule.rev_res else raw
    sample_results = [r for _, _, r in op_examples_raw]
    final = _format_int_like_expected(final, sample_results)
    final = _format_with_sign(final, op, rule.fmt)
    parts.append(
        f"Apply to the test: {rule.op_name}({rta}, {rtb}) = {raw}"
        + (f" -reverse-> {final}" if rule.rev_res else "")
        + f" → final {final}"
    )
    if final != answer:
        # Should not happen; solver and trace must agree.
        return None
    return "\n".join(parts), answer
