"""Trace generator for v3 bit_manipulation solver.

Handles both rule types from `bit_manip_solver_v3`:
  - InvariantRule: a shift-invariant K=1..3 rule (one truth table applied
    uniformly to every output position via offsets).
  - PerBitRule: 8 independent per-position atoms.

Both branches emit:
  1. Preamble + 1–2 rejected naive hypotheses on example 0.
  2. The chosen rule (described in plain English).
  3. Verification on the first example, position by position.
  4. Application to the test input.
  5. Final 8-bit answer for `\\boxed{...}`.

The downstream verifier checks the boxed answer against GT; only correct
traces survive into the training mix.
"""
from __future__ import annotations

from .bit_manip_solver_v3 import (
    InvariantRule,
    PerBitRule,
    PerBitAtom,
    apply_rule,
    _evaluate_atom,
    N_BITS,
    _PAIR_FAMILIES,
    _ASYM_FAMILIES,
    _SYM_FAMILIES,
    _TRIPLE_FAMILIES,
    _TRIPLE_TT,
    _GLOBAL_FAMILIES,
)


def _bits_msb(x: int) -> list[int]:
    return [(x >> (N_BITS - 1 - i)) & 1 for i in range(N_BITS)]


# ---- Invariant rule trace -------------------------------------------------

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


def _read_input(rule: InvariantRule, x: int, i: int, k: int) -> int:
    bits = _bits_msb(x)
    d = rule.offsets[k]
    j = i + d
    if rule.wrap == "wrap":
        return bits[j % N_BITS]
    return bits[j] if 0 <= j < N_BITS else 0


def _describe_invariant(rule: InvariantRule) -> str:
    parts = ", ".join(_label_offset(d) for d in rule.offsets)
    if rule.k == 1:
        prefix = "NOT " if rule.truth_table == 0b01 else ""
        return f"output[i] = {prefix}{_label_offset(rule.offsets[0])}"
    if rule.k == 2:
        return (
            f"output[i] = {_label_offset(rule.offsets[0])} "
            f"{_tt_name_2(rule.truth_table)} "
            f"{_label_offset(rule.offsets[1])}"
        )
    return f"output[i] = {_tt_name_3(rule.truth_table)}({parts})"


def _verify_invariant_example(rule: InvariantRule, in_str: str, out_str: str) -> str:
    expected = _bits_msb(int(out_str, 2))
    lines = []
    for i in range(N_BITS):
        vals = []
        for ji, d in enumerate(rule.offsets):
            j = i + d
            if rule.wrap == "wrap":
                v = _bits_msb(int(in_str, 2))[j % N_BITS]
                ref = f"input[{j % N_BITS}]"
            else:
                bits = _bits_msb(int(in_str, 2))
                v = bits[j] if 0 <= j < N_BITS else 0
                ref = f"input[{j}]" if 0 <= j < N_BITS else "0"
            vals.append((ref, v))
        if rule.k == 1:
            v = vals[0][1]
            out_bit = v if rule.truth_table == 0b10 else 1 - v
            sym = "" if rule.truth_table == 0b10 else "NOT "
            calc = f"{sym}{v} = {out_bit}"
        elif rule.k == 2:
            a, b = vals[0][1], vals[1][1]
            out_bit = (rule.truth_table >> (a | (b << 1))) & 1
            calc = f"{a} {_tt_name_2(rule.truth_table)} {b} = {out_bit}"
        else:
            a, b, c = vals[0][1], vals[1][1], vals[2][1]
            out_bit = (rule.truth_table >> (a | (b << 1) | (c << 2))) & 1
            calc = f"{_tt_name_3(rule.truth_table)}({a},{b},{c}) = {out_bit}"
        ok = "✓" if out_bit == expected[i] else "✗"
        ref_str = ", ".join(f"{r}={v}" for r, v in vals)
        lines.append(f"  i={i}: {ref_str} → {calc}   (expected {expected[i]}) {ok}")
    return "\n".join(lines)


# ---- Per-bit rule trace ---------------------------------------------------

def _describe_atom(atom: PerBitAtom) -> str:
    fam = atom.family
    if fam == "Constant":
        return f"constant {atom.operands[0]}"
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
    """One line per output position describing its atom."""
    return "; ".join(
        f"out[{p}]={_describe_atom(atom)}" for p, atom in enumerate(rule.atoms)
    )


def _verify_per_bit_example(rule: PerBitRule, in_str: str, out_str: str) -> str:
    in_bits = tuple(_bits_msb(int(in_str, 2)))
    expected = _bits_msb(int(out_str, 2))
    lines = []
    for p, atom in enumerate(rule.atoms):
        out_bit = _evaluate_atom(atom, in_bits)
        ok = "✓" if out_bit == expected[p] else "✗"
        lines.append(
            f"  i={p}: {_describe_atom(atom)} = {out_bit}   "
            f"(expected {expected[p]}) {ok}"
        )
    return "\n".join(lines)


# ---- Public API -----------------------------------------------------------

def generate_trace(
    rule,
    pairs: list[tuple[str, str]],
    test_input: str,
) -> tuple[str, str]:
    """Build the reasoning trace and return (trace_str, final_8bit_answer)."""
    parts: list[str] = []
    parts.append(
        "I need to find an operation that maps each 8-bit input to its 8-bit "
        "output consistently across all examples."
    )

    first_in, first_out = pairs[0]
    # Reject 2 naive hypotheses on example 0 (NOT and ROL_1).
    not_first = f"{(~int(first_in, 2)) & 0xFF:08b}"
    parts.append(
        f"Try NOT(input): NOT({first_in}) = {not_first}. "
        f"Expected: {first_out}. "
        + ("Match." if not_first == first_out else "Doesn't match.")
    )
    rol1 = f"{((int(first_in, 2) << 1) | (int(first_in, 2) >> 7)) & 0xFF:08b}"
    parts.append(
        f"Try rotate-left-1: ROL1({first_in}) = {rol1}. "
        f"Expected: {first_out}. "
        + ("Match." if rol1 == first_out else "Doesn't match.")
    )

    # Describe the actual rule.
    if isinstance(rule, InvariantRule):
        edge = "zero-padding edge bits" if rule.wrap == "zero" else "wrapping at the edges"
        parts.append(
            f"Try a shift-invariant rule of the form {_describe_invariant(rule)}, {edge}. "
            "Verify on the first example:"
        )
        parts.append(f"Example: {first_in} -> {first_out}")
        parts.append(_verify_invariant_example(rule, first_in, first_out))
    elif isinstance(rule, PerBitRule):
        parts.append(
            "The pattern isn't a uniform shift across all 8 bits, so I'll "
            "fit each output bit position separately."
        )
        parts.append(
            "Rule (one expression per output bit, 0-indexed MSB-first):"
        )
        parts.append("  " + _describe_per_bit(rule))
        parts.append(f"Verify on the first example: {first_in} -> {first_out}")
        parts.append(_verify_per_bit_example(rule, first_in, first_out))
    else:
        raise TypeError(f"unknown rule type: {type(rule).__name__}")

    # Spot check on a second example.
    if len(pairs) >= 2:
        second_in, second_out = pairs[1]
        pred_second = f"{apply_rule(rule, int(second_in, 2)):08b}"
        parts.append(
            f"Spot check on a second example: rule({second_in}) = {pred_second} "
            f"(expected {second_out}). "
            + ("Match." if pred_second == second_out else "Mismatch.")
        )

    # Apply to test input.
    parts.append(f"Apply the rule to the test input {test_input}:")
    pred_int = apply_rule(rule, int(test_input, 2))
    pred_str = f"{pred_int:08b}"
    if isinstance(rule, InvariantRule):
        parts.append(_verify_invariant_example(rule, test_input, pred_str))
    else:
        parts.append(_verify_per_bit_example(rule, test_input, pred_str))
    parts.append(f"Output bits: {pred_str}")

    return "\n\n".join(parts), pred_str
