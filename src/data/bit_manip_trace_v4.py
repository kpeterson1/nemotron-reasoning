"""Compact trace generator for v3 bit_manipulation solver.

Structural mirror of `equation_transformation_trace_v2`: same section
headers, same per-example check format, same "Apply to the test: ... →
final <x>" closer. Solver logic in `bit_manip_solver_v3` is unchanged —
this is a trace-format-only change to reduce LoRA cross-talk when bm and
eq traces are mixed in SFT.

Drops from v3:
  * The "Try NOT(input)" / "Try rotate-left-1" reject-then-accept scaffold.
  * The 8-row per-bit verification grid.

Replaces with:
  * Direct statement of the discovered rule.
  * Whole-string per-example verification (2-3 examples max), mirroring
    eq's "check: <op>(a, b) = <raw> → expected <r> ✓" pattern.

The trace is wrapped in `<think>...</think>` by `format_training.py`,
which also appends the boxed answer — so this function returns just the
think-block content + the bare answer string, same contract as v3.
"""
from __future__ import annotations

from .bit_manip_solver_v3 import (
    InvariantRule,
    PerBitRule,
    PerBitAtom,
    apply_rule,
    _ASYM_FAMILIES,
    _TRIPLE_FAMILIES,
)


# ---- Human-readable rule descriptions ----------------------------------

def _tt_name_2(tt: int) -> str:
    names = {
        0b1000: "AND",
        0b1110: "OR",
        0b0110: "XOR",
        0b0111: "NAND",
        0b0001: "NOR",
        0b1001: "XNOR",
        0b0010: "(A AND NOT B)",
        0b0100: "(NOT A AND B)",
        0b1011: "(A OR NOT B)",
        0b1101: "(NOT A OR B)",
    }
    return names.get(tt, f"f(TT=0x{tt:x})")


def _tt_name_3(tt: int) -> str:
    named = {
        sum(1 << i for i in range(8) if bin(i).count("1") >= 2): "MAJ",
        sum(1 << i for i in range(8) if bin(i).count("1") >= 1): "OR3",
        sum(1 << i for i in range(8) if bin(i).count("1") % 2 == 1): "XOR3",
        (1 << 7): "AND3",
    }
    return named.get(tt, f"f(TT3=0x{tt:02x})")


def _label_offset(d: int) -> str:
    if d == 0:
        return "input[i]"
    sign = "+" if d > 0 else "-"
    return f"input[i{sign}{abs(d)}]"


def _describe_invariant(rule: InvariantRule) -> str:
    edge = "wrap" if rule.wrap == "wrap" else "zero-pad edges"
    if rule.k == 1:
        prefix = "NOT " if rule.truth_table == 0b01 else ""
        return f"output[i] = {prefix}{_label_offset(rule.offsets[0])} ({edge})"
    if rule.k == 2:
        return (
            f"output[i] = {_label_offset(rule.offsets[0])} "
            f"{_tt_name_2(rule.truth_table)} "
            f"{_label_offset(rule.offsets[1])} ({edge})"
        )
    parts = ", ".join(_label_offset(d) for d in rule.offsets)
    return f"output[i] = {_tt_name_3(rule.truth_table)}({parts}) ({edge})"


def _describe_atom(atom: PerBitAtom) -> str:
    fam = atom.family
    if fam == "Constant":
        return f"{atom.operands[0]}"
    if fam == "Identity":
        return f"input[{atom.operands[0]}]"
    if fam == "NOT":
        return f"NOT input[{atom.operands[0]}]"
    if fam == "Parity":
        return "parity-of-all-inputs" + (" (inverted)" if atom.operands[0] else "")
    if fam == "Popcount":
        op, thr = atom.operands
        return f"(popcount(input) {op} {thr})"
    if fam in _TRIPLE_FAMILIES:
        i, j, k = atom.operands
        return f"{fam}(input[{i}], input[{j}], input[{k}])"
    i, j = atom.operands
    if fam in _ASYM_FAMILIES:
        base = fam.split("-")[0]
        return f"{base}(input[{i}], NOT input[{j}])"
    if fam == "NAND":
        return f"NOT(input[{i}] AND input[{j}])"
    if fam == "NOR":
        return f"NOT(input[{i}] OR input[{j}])"
    return f"{fam}(input[{i}], input[{j}])"


def _describe_per_bit(rule: PerBitRule) -> str:
    return "8 per-position rules: " + "; ".join(
        f"out[{p}]={_describe_atom(a)}" for p, a in enumerate(rule.atoms)
    )


# ---- Trace generation ---------------------------------------------------

def generate_trace(
    rule,
    pairs: list[tuple[str, str]],
    test_input: str,
) -> tuple[str, str]:
    """Build a compact trace mirroring equation_transformation_trace_v2.

    Returns (trace, final_8bit_answer)."""
    if isinstance(rule, InvariantRule):
        rule_desc = _describe_invariant(rule)
    elif isinstance(rule, PerBitRule):
        rule_desc = _describe_per_bit(rule)
    else:
        raise TypeError(f"unknown rule type: {type(rule).__name__}")

    pred = f"{apply_rule(rule, int(test_input, 2)):08b}"

    lines: list[str] = []
    lines.append(
        f"The test input is '{test_input}'. I need to figure out the rule from the examples."
    )
    lines.append("Examples:")
    for inp, out in pairs[:3]:
        lines.append(f"  {inp} → {out}")
    lines.append(
        f"After trying candidate operations, the rule that matches every example is: {rule_desc}."
    )
    for inp, out in pairs[:3]:
        applied = f"{apply_rule(rule, int(inp, 2)):08b}"
        ok = "✓" if applied == out else "✗"
        lines.append(f"  check: rule({inp}) = {applied} → expected {out} {ok}")
    lines.append(f"Apply to the test: rule({test_input}) = {pred} → final {pred}")
    return "\n".join(lines), pred
