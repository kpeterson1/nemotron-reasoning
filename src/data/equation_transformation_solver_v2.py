"""Upgraded equation_transformation solver (v2).

Each puzzle defines one or more secret binary operators (non-alphanumeric
symbols). The test query uses a known operator and we must derive its
semantics from the examples that share it.

Improvements over v1:
  * Each numeric operation is tried in 4 transformations:
      (identity_operands, identity_result)
      (reversed_operands, identity_result)
      (identity_operands, reversed_result)
      (reversed_operands, reversed_result)
  * Output sign formatting is detected: results that end with the operator
    character (or "-") at length >1 indicate that negative results use a
    suffix instead of a leading minus. Mirror handling for prefix.
  * Wider rare-ops menu, including determinant, cross multiply, digit-wise
    operations on 2-digit operands, modulo variants.
  * Optional symbolic positional path for puzzles with non-numeric operands.

Public API matches v1: `solve(prompt) -> (answer_str | None, EqRule | None)`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


_TEST_RE = re.compile(
    r"Now,?\s+determine\s+the\s+result\s+for:\s*(\S.+?)\s*(?:\.\s*$|\s*$)",
    re.IGNORECASE | re.MULTILINE,
)
_NUMERIC_PAIR_RE = re.compile(r"^(\d+)(\D+)(\d+)$")


@dataclass(frozen=True)
class EqRule:
    op_name: str          # human-readable op, e.g. "a-b" or "digit_sum(|a-b|)"
    test_op: str          # operator char in the test query
    rev_ops: bool         # whether operands were reversed before applying op
    rev_res: bool         # whether result was digit-reversed
    fmt: str              # "num" | "neg_suffix" | "neg_prefix"


# ---- Core string and arithmetic helpers ----------------------------------

def _split_by_op(expr: str, op: str) -> tuple[str, str] | None:
    idx = expr.find(op)
    if idx == -1:
        return None
    return expr[:idx], expr[idx + len(op):]


def _digit_reverse(s: str) -> str:
    if s.startswith("-"):
        return "-" + s[1:][::-1]
    return s[::-1]


def _digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


def _digit_prod(n: int) -> int:
    p = 1
    for c in str(abs(n)):
        p *= int(c)
    return p


def _detect_test_operator(test: str) -> str | None:
    if not any(c.isalnum() for c in test):
        return None
    matches = list(re.finditer(r"[^\w\s]+", test))
    if not matches:
        return None
    for m in matches:
        s, e = m.start(), m.end()
        if s > 0 and e < len(test) and test[s - 1].isalnum() and test[e].isalnum():
            return m.group()
    return matches[0].group()


# ---- Operation menu ------------------------------------------------------

_COMMON_OP_NAMES: set[str] = {
    "a+b", "a-b", "b-a", "a*b", "|a-b|", "-|a-b|",
    "concat_ab", "concat_ba",
}


def _all_ops(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Return (op_name, result_string) candidates for the integer pair (a, b).
    The result is a string (not int) so we can preserve concatenation and
    leading-zero-aware formatting.

    Order: common ops first, then rare ones. Callers that try-common-first
    can stop on the first fit; only fall through to rare when no common fits.
    """
    out: list[tuple[str, str]] = []
    # Common
    out.append(("a+b", str(a + b)))
    out.append(("a-b", str(a - b)))
    out.append(("b-a", str(b - a)))
    out.append(("a*b", str(a * b)))
    out.append(("|a-b|", str(abs(a - b))))
    out.append(("-|a-b|", str(-abs(a - b))))
    out.append(("concat_ab", sa + sb))
    out.append(("concat_ba", sb + sa))
    # Offsets ±1 on basic ops (very common)
    out.append(("a+b+1", str(a + b + 1)))
    out.append(("a+b-1", str(a + b - 1)))
    out.append(("a-b+1", str(a - b + 1)))
    out.append(("a-b-1", str(a - b - 1)))
    out.append(("a*b+1", str(a * b + 1)))
    out.append(("a*b-1", str(a * b - 1)))
    # Multiplier-of-basic
    for k in (2, 3, 4, 5, 10):
        out.append((f"(a+b)*{k}", str((a + b) * k)))
        out.append((f"(a-b)*{k}", str((a - b) * k)))
        out.append((f"|a-b|*{k}", str(abs(a - b) * k)))
    # Division-like
    if b != 0:
        out.append(("a//b", str(a // b)))
        out.append(("a%b", str(a % b)))
    if a != 0:
        out.append(("b//a", str(b // a)))
        out.append(("b%a", str(b % a)))
    if a != 0 and b != 0:
        big, small = max(a, b), min(a, b)
        out.append(("max%min", str(big % small)))
    # Squares
    out.append(("a^2-b^2", str(a * a - b * b)))
    out.append(("a^2+b^2", str(a * a + b * b)))
    out.append(("(a+b)^2", str((a + b) ** 2)))
    out.append(("(a-b)^2", str((a - b) ** 2)))
    out.append(("|a^2-b^2|", str(abs(a * a - b * b))))
    # Bitwise (small int representation)
    out.append(("a^b_xor", str(a ^ b)))
    out.append(("a&b", str(a & b)))
    out.append(("a|b", str(a | b)))
    # Digit-aggregate
    out.append(("ds(a*b)", str(_digit_sum(a * b))))
    out.append(("dp(a*b)", str(_digit_prod(a * b))))
    out.append(("ds(a+b)", str(_digit_sum(a + b))))
    out.append(("ds(|a-b|)", str(_digit_sum(abs(a - b)))))
    out.append(("ds(a)*ds(b)", str(_digit_sum(a) * _digit_sum(b))))
    out.append(("ds(a)+ds(b)", str(_digit_sum(a) + _digit_sum(b))))
    # 2-digit pair-wise (reference's "rare" menu)
    if len(sa) == 2 and len(sb) == 2:
        d1, d2, d3, d4 = int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1])
        out.append(("digit_abs_diff", f"{abs(d1 - d3)}{abs(d2 - d4)}"))
        out.append(("digit_add_mod10", f"{(d1 + d3) % 10}{(d2 + d4) % 10}"))
        out.append(("digit_sub_mod10", f"{(d1 - d3) % 10}{(d2 - d4) % 10}"))
        out.append(("cross_mul", str(d1 * d3 + d2 * d4)))
        out.append(("cross_mul_rev", str(d1 * d4 + d2 * d3)))
        out.append(("digit_mul", f"{d1 * d3}{d2 * d4}"))
        out.append(("digit_mul_rev", f"{d1 * d4}{d2 * d3}"))
        out.append(("digit_sum_diff", str((d1 + d2) - (d3 + d4))))
        out.append(("digit_sum_sum", str((d1 + d2) + (d3 + d4))))
        out.append(("digit_prod_diff", str(d1 * d2 - d3 * d4)))
        out.append(("digit_prod_sum", str(d1 * d2 + d3 * d4)))
        det = d1 * d4 - d2 * d3
        out.append(("determinant", str(det)))
        out.append(("abs_det", str(abs(det))))
    return out


def _format_int_like_expected(value_str: str, samples: list[str]) -> str:
    """Match the format style of example results (e.g. zero-padded width)."""
    # Only pad if every numeric sample is zero-padded to the same width.
    numeric = [s for s in samples if s.lstrip("-").isdigit()]
    if not numeric:
        return value_str
    widths = {len(s) for s in numeric}
    if len(widths) != 1:
        return value_str
    w = widths.pop()
    if not all(s.startswith("0") and len(s) == w for s in numeric):
        return value_str
    if value_str.startswith("-"):
        return value_str
    return value_str.zfill(w)


# ---- Numeric solver -----------------------------------------------------

def _extract_numeric_examples(prompt: str, op: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for line in prompt.splitlines():
        line = line.strip().rstrip(".")
        if "=" not in line or line.lower().startswith("now"):
            continue
        lhs, rhs = line.split("=", 1)
        lhs, rhs = lhs.strip(), rhs.strip()
        if op not in lhs:
            continue
        parts = _split_by_op(lhs, op)
        if parts is None:
            continue
        a, b = parts
        if a.isdigit() and b.isdigit():
            out.append((a, b, rhs))
    return out


def _detect_format(group: list[tuple[str, str, str]], op_char: str) -> str:
    """Pick the result format ('num', 'neg_suffix', 'neg_prefix').

    Symbol-suffix: at least one result ends with `op_char` AND is length > 1.
    Symbol-prefix: at least one result begins with `op_char` AND is length > 1.
    Minus-suffix: at least one result ends with '-' AND is length > 1.
    Minus-prefix: at least one starts with '-' AND is length > 1.
    """
    has_neg_suffix = any(
        r.endswith("-") and len(r) > 1 for _, _, r in group
    )
    has_neg_prefix = any(
        r.startswith("-") and len(r) > 1 for _, _, r in group
    )
    has_op_suffix = op_char != "-" and any(
        r.endswith(op_char) and len(r) > 1 for _, _, r in group
    )
    has_op_prefix = op_char != "-" and any(
        r.startswith(op_char) and len(r) > 1 for _, _, r in group
    )
    if has_neg_suffix:
        return "neg_suffix"
    if has_neg_prefix:
        return "neg_prefix"
    if has_op_suffix:
        return "neg_suffix"  # we mirror reference: treat operator-suffix as a sign indicator
    if has_op_prefix:
        return "neg_prefix"
    return "num"


def _strip_sign(result: str, op_char: str, fmt: str) -> str:
    """Convert example display string back to a canonical signed-int string."""
    if fmt == "neg_suffix":
        if result.endswith("-") and len(result) > 1:
            return "-" + result[:-1]
        if op_char != "-" and result.endswith(op_char) and len(result) > 1:
            return "-" + result[: -len(op_char)]
        return result
    if fmt == "neg_prefix":
        if result.startswith("-") and len(result) > 1:
            return result
        if op_char != "-" and result.startswith(op_char) and len(result) > 1:
            return "-" + result[len(op_char):]
        return result
    return result


def _format_with_sign(value_str: str, op_char: str, fmt: str) -> str:
    """Inverse of `_strip_sign`. Apply sign-as-suffix/prefix formatting."""
    if not value_str.startswith("-"):
        return value_str
    magnitude = value_str[1:]
    if fmt == "neg_suffix":
        # Use op_char as the suffix marker.
        if op_char and op_char != "-":
            return magnitude + op_char
        return magnitude + "-"
    if fmt == "neg_prefix":
        if op_char and op_char != "-":
            return op_char + magnitude
        return "-" + magnitude
    return value_str


def _try_numeric(prompt: str, op: str, test: str) -> tuple[str | None, EqRule | None]:
    test_parts = _split_by_op(test, op)
    if test_parts is None:
        return None, None
    ta, tb = test_parts
    if not (ta.isdigit() and tb.isdigit()):
        return None, None
    raw_group = _extract_numeric_examples(prompt, op)
    if not raw_group:
        return None, None

    fmt = _detect_format(raw_group, op)
    # Canonicalize each example: convert displayed result to signed-int string.
    canon_group = [
        (a, b, _strip_sign(r, op, fmt)) for a, b, r in raw_group
    ]

    # Try each (rev_ops, rev_res, op_name) combination. Common ops first,
    # falling through to rare ops only if no common op fits.
    for tier in ("common", "rare"):
      for rev_ops in (False, True):
        for rev_res in (False, True):
            a0, b0, _ = canon_group[0]
            ta0 = a0[::-1] if rev_ops else a0
            tb0 = b0[::-1] if rev_ops else b0
            ordered_names = [n for n, _ in _all_ops(int(ta0), int(tb0), ta0, tb0)]
            if tier == "common":
                cand_iter = [n for n in ordered_names if n in _COMMON_OP_NAMES]
            else:
                cand_iter = [n for n in ordered_names if n not in _COMMON_OP_NAMES]

            for name in cand_iter:
                ok = True
                for a, b, r_canon in canon_group:
                    ra = a[::-1] if rev_ops else a
                    rb = b[::-1] if rev_ops else b
                    try:
                        ai, bi = int(ra), int(rb)
                    except ValueError:
                        ok = False
                        break
                    pairs = _all_ops(ai, bi, ra, rb)
                    matching = [v for n, v in pairs if n == name]
                    if not matching:
                        ok = False
                        break
                    raw = matching[0]
                    final = _digit_reverse(raw) if rev_res else raw
                    if final != r_canon:
                        ok = False
                        break
                if not ok:
                    continue
                # Found! Apply to the test.
                rta = ta[::-1] if rev_ops else ta
                rtb = tb[::-1] if rev_ops else tb
                try:
                    ai, bi = int(rta), int(rtb)
                except ValueError:
                    continue
                pairs = _all_ops(ai, bi, rta, rtb)
                matching = [v for n, v in pairs if n == name]
                if not matching:
                    continue
                raw = matching[0]
                final = _digit_reverse(raw) if rev_res else raw
                # Format integer width if examples imply zero-padding.
                stripped_samples = [r for _, _, r in raw_group]
                final = _format_int_like_expected(final, stripped_samples)
                # Re-apply sign-as-suffix/prefix.
                final = _format_with_sign(final, op, fmt)
                return final, EqRule(
                    op_name=name, test_op=op,
                    rev_ops=rev_ops, rev_res=rev_res, fmt=fmt,
                )
    return None, None


# ---- Symbolic positional path (fixed-length strings) --------------------

def _try_symbolic(prompt: str, test: str) -> tuple[str | None, EqRule | None]:
    """Symbolic positional puzzles where every output is a pure positional
    copy from the input. We deliberately do NOT fall back to substitution or
    constants — those guesses are <5% accurate in this dataset and contaminate
    downstream training data."""
    pairs: list[tuple[str, str]] = []
    for line in prompt.splitlines():
        line = line.strip().rstrip(".")
        if "=" not in line or line.lower().startswith("now"):
            continue
        lhs, rhs = line.split("=", 1)
        lhs, rhs = lhs.strip(), rhs.strip()
        if not lhs or not rhs:
            continue
        pairs.append((lhs, rhs))
    if not pairs:
        return None, None
    ll = {len(l) for l, _ in pairs}
    rl = {len(r) for _, r in pairs}
    if len(ll) != 1 or len(rl) != 1:
        return None, None
    L = ll.pop(); R = rl.pop()
    if len(test) != L:
        return None, None

    out_chars: list[str] = []
    for j in range(R):
        target = [r[j] for _, r in pairs]
        chosen_i: int | None = None
        for i in range(L):
            col = [l[i] for l, _ in pairs]
            if col == target:
                chosen_i = i
                break
        if chosen_i is None:
            return None, None
        out_chars.append(test[chosen_i])
    return "".join(out_chars), EqRule(
        op_name="symbolic_positional", test_op="",
        rev_ops=False, rev_res=False, fmt="num",
    )


# ---- Public API ---------------------------------------------------------

def parse(prompt: str) -> str | None:
    m = _TEST_RE.search(prompt)
    return m.group(1).strip() if m else None


def solve(prompt: str) -> tuple[str | None, EqRule | None]:
    test = parse(prompt)
    if test is None:
        return None, None
    op = _detect_test_operator(test)
    if op is not None:
        res, rule = _try_numeric(prompt, op, test)
        if res is not None:
            return res, rule
    return _try_symbolic(prompt, test)


def format_answer(value: str) -> str:
    return value
