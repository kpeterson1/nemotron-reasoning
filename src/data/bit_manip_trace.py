"""Generate concise reasoning traces for bit_manipulation puzzles solved by
`bit_manip_solver`. The traces follow a consistent structure:

  1. Frame: state we're searching for a rule that fits all examples.
  2. Reject 2 simple wrong candidates (NOT, ROL_1) on the first example.
  3. Identify the winning shift-invariant rule and verify it on one example
     by computing each output bit position.
  4. Apply the rule to the test input position by position.
  5. Emit `\\boxed{<answer>}`.

Target trace length: under ~1500 tokens. Verification on a single 8-bit example
adds ~250 tokens; the full trace lands around 600–900 tokens depending on K.
"""
from __future__ import annotations

from .bit_manip_solver import Rule, apply_rule

def _bits(x: int) -> list[int]:
    return [(x >> (7 - i)) & 1 for i in range(8)]


def _read_input(rule: Rule, x: int, i: int, k: int) -> int:
    """Compute one of rule's K input bits at output position i for input x."""
    bits = _bits(x)
    d = rule.offsets[k]
    j = i + d
    if rule.wrap == "wrap":
        return bits[j % 8]
    return bits[j] if 0 <= j < 8 else 0


def _label_offset(d: int) -> str:
    if d == 0:
        return "input[i]"
    sign = "+" if d > 0 else "-"
    return f"input[i{sign}{abs(d)}]"


def _eval_per_position(rule: Rule, x: int, i: int) -> int:
    """Compute the rule's output bit at position i (used for trace verification)."""
    idx = 0
    for ji in range(rule.k):
        v = _read_input(rule, x, i, ji)
        idx |= v << ji
    return (rule.truth_table >> idx) & 1


def _fn_text_2(tt: int) -> str:
    names = {
        0b1000: "AND", 0b1110: "OR", 0b0110: "XOR",
        0b0111: "NAND", 0b0001: "NOR", 0b1001: "XNOR",
        0b0010: "(A AND NOT B)", 0b0100: "(NOT A AND B)",
        0b1011: "(A OR NOT B)", 0b1101: "(NOT A OR B)",
    }
    if tt in names:
        return names[tt]
    # Fallback: spell out the truth table
    rows = []
    for i in range(4):
        a = i & 1; b = (i >> 1) & 1
        rows.append(f"({a},{b})->{(tt>>i)&1}")
    return "fn[" + ", ".join(rows) + "]"


def _fn_text_3(tt: int) -> str:
    """Short label only. Use `_fn_text_3_full` separately to get the truth-table
    spelled out (used once at the top of the trace, not per-position)."""
    named = {
        sum(1 << i for i in range(8) if bin(i).count("1") >= 2): "MAJ",
        sum(1 << i for i in range(8) if bin(i).count("1") >= 1): "OR3",
        sum(1 << i for i in range(8) if bin(i).count("1") % 2 == 1): "XOR3",
        (1 << 7): "AND3",
    }
    if tt in named:
        return named[tt]
    return "f"  # short — full truth table is introduced once at the top


def _fn_text_3_full(tt: int) -> str:
    """Full truth-table description (spell out all 8 rows). Use only once."""
    named = {
        sum(1 << i for i in range(8) if bin(i).count("1") >= 2): "MAJ",
        sum(1 << i for i in range(8) if bin(i).count("1") >= 1): "OR3",
        sum(1 << i for i in range(8) if bin(i).count("1") % 2 == 1): "XOR3",
        (1 << 7): "AND3",
    }
    if tt in named:
        return named[tt]
    rows = []
    for i in range(8):
        a = i & 1; b = (i >> 1) & 1; c = (i >> 2) & 1
        rows.append(f"({a},{b},{c})->{(tt>>i)&1}")
    return "f where " + ", ".join(rows)


def _describe_rule_human(rule: Rule) -> str:
    parts = ", ".join(_label_offset(d) for d in rule.offsets)
    if rule.k == 1:
        prefix = "NOT " if rule.truth_table == 0b01 else ""
        return f"output[i] = {prefix}{_label_offset(rule.offsets[0])}"
    if rule.k == 2:
        return f"output[i] = {_label_offset(rule.offsets[0])} {_fn_text_2(rule.truth_table)} {_label_offset(rule.offsets[1])}"
    return f"output[i] = {_fn_text_3(rule.truth_table)}({parts})"


def _fmt_bits_input_view(rule: Rule, x: int, i: int) -> str:
    """Show the K input bits used at output position i, e.g. 'input[i+1]=1, input[i-1]=0'."""
    parts = []
    for ji, d in enumerate(rule.offsets):
        v = _read_input(rule, x, i, ji)
        parts.append(f"{_label_offset(d).replace('i', f'{i}+0').replace('+0+', '+').replace('+0-', '-').replace('+0]', ']').replace('+00', '0')}={v}")
    return ", ".join(parts)


def _bit_label(d: int, i: int) -> str:
    """e.g. 'input[3]' for d=0, i=3; 'input[5]' for d=2, i=3 (with wrap noted)."""
    j = i + d
    return f"input[{j}]"


def _verify_one_example(rule: Rule, in_str: str, out_str: str) -> str:
    """Produce the per-position verification text for a single example."""
    x = int(in_str, 2)
    expected = int(out_str, 2)
    in_bits = _bits(x)
    out_bits = _bits(expected)
    lines = []
    rule_desc = _describe_rule_human(rule)
    for i in range(8):
        # Build a per-position substitution showing actual values
        # e.g. "i=0: input[1]=0, input[2]=1 → 0 XOR 1 = 1   (output bit 0 = 1) ✓"
        actual_vals = []
        for ji, d in enumerate(rule.offsets):
            j = i + d
            if rule.wrap == "wrap":
                v = in_bits[j % 8]
                ref = f"input[{j%8}]"
            else:
                v = in_bits[j] if 0 <= j < 8 else 0
                ref = f"input[{j}]" if 0 <= j < 8 else "0"
            actual_vals.append((ref, v))
        if rule.k == 1:
            v = actual_vals[0][1]
            out_bit = v if rule.truth_table == 0b10 else 1 - v
            sym = "" if rule.truth_table == 0b10 else "NOT "
            calc = f"{sym}{actual_vals[0][1]} = {out_bit}"
        elif rule.k == 2:
            a, b = actual_vals[0][1], actual_vals[1][1]
            idx = a | (b << 1)
            out_bit = (rule.truth_table >> idx) & 1
            calc = f"{a} {_fn_text_2(rule.truth_table)} {b} = {out_bit}"
        else:
            a, b, c = (actual_vals[0][1], actual_vals[1][1], actual_vals[2][1])
            idx = a | (b << 1) | (c << 2)
            out_bit = (rule.truth_table >> idx) & 1
            calc = f"{_fn_text_3(rule.truth_table)}({a},{b},{c}) = {out_bit}"
        ok = "✓" if out_bit == out_bits[i] else "✗"
        ref_str = ", ".join(f"{r}={v}" for r, v in actual_vals)
        lines.append(f"  i={i}: {ref_str} → {calc}   (expected {out_bits[i]}) {ok}")
    return "\n".join(lines)


def generate_trace(rule: Rule, pairs: list[tuple[str, str]], test_input: str) -> tuple[str, str]:
    """Return (reasoning_trace, final_answer_8bits).

    The reasoning trace is the content for the `<think>...</think>` block.
    The final answer is the 8-bit binary string to put in `\\boxed{...}`.
    """
    rule_desc = _describe_rule_human(rule)
    edge = "zero-padding edge bits" if rule.wrap == "zero" else "wrapping at the edges"

    # ---- 1. preamble + 2 rejected hypotheses ----
    first_in, first_out = pairs[0]
    parts: list[str] = []
    parts.append(
        "I need to find an operation that maps each 8-bit input to its 8-bit "
        "output consistently across all examples."
    )

    # Rejected hypothesis 1: NOT
    not_first = f"{(~int(first_in, 2)) & 0xFF:08b}"
    parts.append(
        f"Try NOT(input): NOT({first_in}) = {not_first}. "
        f"Expected output: {first_out}. "
        + ("Match." if not_first == first_out else "Doesn't match.")
    )

    # Rejected hypothesis 2: rotate left by 1
    rol1 = f"{((int(first_in, 2) << 1) | (int(first_in, 2) >> 7)) & 0xFF:08b}"
    parts.append(
        f"Try rotate-left-by-1: ROL_1({first_in}) = {rol1}. "
        f"Expected: {first_out}. "
        + ("Match." if rol1 == first_out else "Doesn't match.")
    )

    # ---- 3. the winning rule ----
    if rule.k == 3:
        # introduce f once at the top to keep the trace concise
        full_def = _fn_text_3_full(rule.truth_table)
        parts.append(
            f"Try a rule of the form {rule_desc}, {edge}.  Define {full_def}.  "
            "Verify on the first example:"
        )
    else:
        parts.append(
            f"Try a rule of the form {rule_desc}, {edge}. "
            "Verify on the first example:"
        )
    parts.append("Example: " + first_in + " -> " + first_out)
    parts.append(_verify_one_example(rule, first_in, first_out))

    # Spot check on a second example
    if len(pairs) >= 2:
        second_in, second_out = pairs[1]
        pred_second = f"{apply_rule(rule, int(second_in, 2)):08b}"
        parts.append(
            f"Spot check on a second example: applying the rule to {second_in} "
            f"gives {pred_second} (expected {second_out}). "
            + ("Match." if pred_second == second_out else "Doesn't match — would need to retry.")
        )

    # ---- 4. apply to test input ----
    parts.append(f"Apply the rule to the test input {test_input}:")
    pred_int = apply_rule(rule, int(test_input, 2))
    pred_str = f"{pred_int:08b}"
    out_bits = _bits(pred_int)
    in_bits = _bits(int(test_input, 2))
    test_lines = []
    for i in range(8):
        actual_vals = []
        for ji, d in enumerate(rule.offsets):
            j = i + d
            if rule.wrap == "wrap":
                v = in_bits[j % 8]
                ref = f"input[{j%8}]={v}"
            else:
                v = in_bits[j] if 0 <= j < 8 else 0
                ref = f"input[{j}]={v}" if 0 <= j < 8 else f"0 (pad)"
            actual_vals.append((ref, v))
        if rule.k == 1:
            v = actual_vals[0][1]
            out_bit = v if rule.truth_table == 0b10 else 1 - v
            calc = ("NOT " if rule.truth_table == 0b01 else "") + f"{v} = {out_bit}"
        elif rule.k == 2:
            a, b = actual_vals[0][1], actual_vals[1][1]
            idx = a | (b << 1)
            out_bit = (rule.truth_table >> idx) & 1
            calc = f"{a} {_fn_text_2(rule.truth_table)} {b} = {out_bit}"
        else:
            a, b, c = actual_vals[0][1], actual_vals[1][1], actual_vals[2][1]
            idx = a | (b << 1) | (c << 2)
            out_bit = (rule.truth_table >> idx) & 1
            calc = f"{_fn_text_3(rule.truth_table)}({a},{b},{c}) = {out_bit}"
        test_lines.append(f"  i={i}: {', '.join(r for r,_ in actual_vals)} → {calc}")
    parts.append("\n".join(test_lines))
    parts.append(f"Output bits: {pred_str}")

    return "\n\n".join(parts), pred_str
