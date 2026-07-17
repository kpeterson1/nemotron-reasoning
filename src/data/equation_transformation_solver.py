"""Solver for equation_transformation puzzles.

Each puzzle defines one or more "secret" operators (non-alphanumeric symbols)
and gives examples of how each operator transforms two operands. The test
query uses ONE operator and we must determine what THAT operator does from
its example uses, then apply it to the test operands.

Sub-types observed:
  * Numeric: operands are digit strings (e.g., `55{76`, `91%84`).
  * Symbolic: operands are short strings of arbitrary characters, the operator
    is a fixed separator (e.g., `@`), and the output is a positional remapping.

Strategy:
  1. Identify the operator in the test query.
  2. Collect example pairs that use the same operator.
  3. Numeric path: enumerate a wide set of binary arithmetic operations and
     pick the one matching every example exactly.
  4. Symbolic path: build a positional character map by aligning examples.
  5. Apply to the test operands.

Coverage will be partial — the equation_transformation rule space is large
and we cannot enumerate every possible operator semantics. Treat this as
best-effort: solver returns None when no candidate fits, and downstream
training code should skip those puzzles.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_TEST_RE = re.compile(
    r"Now,?\s+determine\s+the\s+result\s+for:\s*(\S.+?)\s*(?:\.\s*$|\s*$)",
    re.IGNORECASE | re.MULTILINE,
)
_EXAMPLE_RE = re.compile(r"^\s*([^\s=]+)\s*=\s*([^\s=]+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class EqRule:
    op_name: str
    test_op: str


def _split_by_op(expr: str, op: str) -> tuple[str, str] | None:
    """Split `lhs<op>rhs` only on the FIRST occurrence of `op`."""
    idx = expr.find(op)
    if idx == -1:
        return None
    return expr[:idx], expr[idx + len(op):]


def _detect_test_operator(test: str) -> str | None:
    """Find the operator in the test query. The operator is the first run of
    non-alphanumeric, non-space characters that is bracketed by alphanumeric
    or symbol operands."""
    # A test query like "44{77" has operator "{". A query like "#/&\":" is
    # all-symbol — no clear operator. For all-symbol queries, return None
    # (the symbolic path will handle them as positional puzzles).
    if not any(c.isalnum() for c in test):
        return None
    # Find the longest non-alphanumeric run NOT at the edges.
    matches = list(re.finditer(r"[^\w\s]+", test))
    if not matches:
        return None
    # Prefer the run with alnum on both sides; otherwise the first.
    for m in matches:
        s, e = m.start(), m.end()
        if s > 0 and e < len(test) and test[s - 1].isalnum() and test[e].isalnum():
            return m.group()
    return matches[0].group()


def _parse_int(s: str) -> int | None:
    s = s.strip()
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


# ---- Numeric operation enumeration ----------------------------------------

def _rev(n: int) -> int:
    """Digit-reverse an integer, preserving sign and leading-zero count is lost."""
    s = str(n)
    sign = -1 if s.startswith("-") else 1
    digits = s.lstrip("-")
    return sign * int(digits[::-1]) if digits else 0


def _digit_sum(n: int) -> int:
    return sum(int(c) for c in str(abs(n)))


def _digit_prod(n: int) -> int:
    p = 1
    for c in str(abs(n)):
        p *= int(c)
    return p


def _concat(a: int, b: int) -> int:
    return int(f"{a}{b}")


# A binary numeric operation receives (a, b) and returns a value or None.
def _build_numeric_ops():
    ops: list[tuple[str, callable]] = []
    # Basic
    ops.append(("a+b",    lambda a, b: a + b))
    ops.append(("a-b",    lambda a, b: a - b))
    ops.append(("b-a",    lambda a, b: b - a))
    ops.append(("a*b",    lambda a, b: a * b))
    ops.append(("a//b",   lambda a, b: a // b if b else None))
    ops.append(("b//a",   lambda a, b: b // a if a else None))
    ops.append(("a%b",    lambda a, b: a % b if b else None))
    ops.append(("b%a",    lambda a, b: b % a if a else None))
    ops.append(("a^b_xor", lambda a, b: a ^ b))
    ops.append(("a&b",    lambda a, b: a & b))
    ops.append(("a|b",    lambda a, b: a | b))
    ops.append(("|a-b|",  lambda a, b: abs(a - b)))
    # Offsets on basic ops (very common in this dataset)
    for k in range(-5, 6):
        if k == 0: continue
        ops.append((f"a+b+{k}",  (lambda a, b, k=k: a + b + k)))
        ops.append((f"a-b+{k}",  (lambda a, b, k=k: a - b + k)))
        ops.append((f"a*b+{k}",  (lambda a, b, k=k: a * b + k)))
        ops.append((f"|a-b|+{k}", (lambda a, b, k=k: abs(a - b) + k)))
    # Multiplier-of-basic-op
    for k in (2, 3, 4, 5, 10):
        ops.append((f"(a+b)*{k}",  (lambda a, b, k=k: (a + b) * k)))
        ops.append((f"(a-b)*{k}",  (lambda a, b, k=k: (a - b) * k)))
        ops.append((f"|a-b|*{k}",  (lambda a, b, k=k: abs(a - b) * k)))
    # Squares
    ops.append(("a^2-b^2",    lambda a, b: a * a - b * b))
    ops.append(("a^2+b^2",    lambda a, b: a * a + b * b))
    ops.append(("(a+b)^2",    lambda a, b: (a + b) * (a + b)))
    ops.append(("(a-b)^2",    lambda a, b: (a - b) * (a - b)))
    ops.append(("|a^2-b^2|",  lambda a, b: abs(a * a - b * b)))
    # Concatenation variants
    ops.append(("concat_ab", lambda a, b: _concat(a, b)))
    ops.append(("concat_ba", lambda a, b: _concat(b, a)))
    # Reversed-operand variants
    ops.append(("rev(a)*b", lambda a, b: _rev(a) * b))
    ops.append(("a*rev(b)", lambda a, b: a * _rev(b)))
    ops.append(("rev(a)*rev(b)", lambda a, b: _rev(a) * _rev(b)))
    ops.append(("rev(a)+rev(b)", lambda a, b: _rev(a) + _rev(b)))
    ops.append(("rev(a)-rev(b)", lambda a, b: _rev(a) - _rev(b)))
    ops.append(("rev(a+b)", lambda a, b: _rev(a + b)))
    ops.append(("rev(a*b)", lambda a, b: _rev(a * b)))
    ops.append(("rev(a-b)", lambda a, b: _rev(a - b)))
    ops.append(("rev(b-a)", lambda a, b: _rev(b - a)))
    ops.append(("rev(a)+b", lambda a, b: _rev(a) + b))
    ops.append(("rev(a)-b", lambda a, b: _rev(a) - b))
    ops.append(("a-rev(b)", lambda a, b: a - _rev(b)))
    ops.append(("a+rev(b)", lambda a, b: a + _rev(b)))
    # Digit-aggregate variants
    ops.append(("digit_sum(a*b)", lambda a, b: _digit_sum(a * b)))
    ops.append(("digit_prod(a*b)", lambda a, b: _digit_prod(a * b)))
    ops.append(("digit_sum(a+b)", lambda a, b: _digit_sum(a + b)))
    ops.append(("digit_sum(|a-b|)", lambda a, b: _digit_sum(abs(a - b))))
    ops.append(("digit_sum(a)*digit_sum(b)", lambda a, b: _digit_sum(a) * _digit_sum(b)))
    ops.append(("digit_sum(a)+digit_sum(b)", lambda a, b: _digit_sum(a) + _digit_sum(b)))
    return ops


_NUMERIC_OPS = _build_numeric_ops()


def _numeric_examples(prompt: str, op: str) -> list[tuple[str, str, str]]:
    """Return list of (lhs_str, rhs_str, result_str) for example lines whose
    LHS contains `op`. We carry strings (not ints) so we can preserve leading
    zeros in result formatting."""
    out: list[tuple[str, str, str]] = []
    for line in prompt.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        if line.lower().startswith("now"):
            continue
        # Drop trailing punctuation like "."
        line = line.rstrip(".")
        if "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        lhs, rhs = lhs.strip(), rhs.strip()
        if op not in lhs:
            continue
        parts = _split_by_op(lhs, op)
        if parts is None:
            continue
        a, b = parts
        # Numeric path: both sides must look integer-ish
        if a.isdigit() and b.isdigit():
            out.append((a, b, rhs))
    return out


def _format_int_like_expected(value: int, sample_results: list[str]) -> str:
    """Match the result's width style. If example results all have leading
    zeros to a fixed width, pad value to that width."""
    if value < 0:
        return str(value)
    widths = {len(s) for s in sample_results if s.lstrip("-").isdigit()}
    if len(widths) == 1:
        w = widths.pop()
        s = str(value)
        # Only zero-pad if every sample result was zero-padded.
        if all(rs.startswith("0") and len(rs) == w for rs in sample_results if rs.lstrip("-").isdigit()):
            return s.zfill(w)
    return str(value)


def _try_numeric(prompt: str, op: str, test: str) -> tuple[str | None, EqRule | None]:
    parts = _split_by_op(test, op)
    if parts is None:
        return None, None
    ta, tb = parts
    if not (ta.isdigit() and tb.isdigit()):
        return None, None
    examples = _numeric_examples(prompt, op)
    if not examples:
        return None, None
    a_ints = [_parse_int(a) for a, _, _ in examples]
    b_ints = [_parse_int(b) for _, b, _ in examples]
    if any(x is None for x in a_ints + b_ints):
        return None, None
    sample_results = [r for _, _, r in examples]
    for name, fn in _NUMERIC_OPS:
        try:
            preds = [fn(int(a), int(b)) for a, b in zip(a_ints, b_ints)]
        except Exception:
            continue
        if any(p is None for p in preds):
            continue
        # Compare. Need to match exactly the string form.
        formatted = [_format_int_like_expected(int(p), [sample_results[i]]) for i, p in enumerate(preds)]
        if formatted == sample_results:
            # Apply to test
            ta_i, tb_i = int(ta), int(tb)
            try:
                res = fn(ta_i, tb_i)
            except Exception:
                continue
            if res is None:
                continue
            res_str = _format_int_like_expected(int(res), sample_results)
            return res_str, EqRule(op_name=name, test_op=op)
    return None, None


# ---- Symbolic path --------------------------------------------------------

def _try_symbolic(prompt: str, test: str) -> tuple[str | None, EqRule | None]:
    """For puzzles like `]|@<| = ")` where the structure is fixed-length and
    output is a positional remapping. We learn a per-position character map.

    Approach: parse all example lines `lhs = rhs` where lhs and rhs are
    character strings. If all lhs have the same length L and all rhs have the
    same length R, learn output[j] = f_j(lhs[some_pos_j]) for each j.
    """
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
    # All same input length, all same output length?
    ll = {len(l) for l, _ in pairs}
    rl = {len(r) for _, r in pairs}
    if len(ll) != 1 or len(rl) != 1:
        return None, None
    L = ll.pop(); R = rl.pop()
    if len(test) != L:
        return None, None

    # For each output position j, find a source input position i such that
    # input[i] == output[j] across all examples (positional copy).
    out_chars: list[str] = []
    for j in range(R):
        target = [r[j] for _, r in pairs]
        # Try positional copy from input
        chosen_i = None
        for i in range(L):
            col = [l[i] for l, _ in pairs]
            if col == target:
                chosen_i = i
                break
        if chosen_i is not None:
            out_chars.append(test[chosen_i])
            continue
        # Try: target is a constant character
        if len(set(target)) == 1:
            out_chars.append(target[0])
            continue
        # Try: each output char is a known function of one input char (char-substitution)
        for i in range(L):
            col = [l[i] for l, _ in pairs]
            sub: dict[str, str] = {}
            ok = True
            for c, t in zip(col, target):
                if c in sub:
                    if sub[c] != t:
                        ok = False; break
                else:
                    sub[c] = t
            if ok and test[i] in sub:
                out_chars.append(sub[test[i]])
                chosen_i = i
                break
        if chosen_i is None:
            return None, None
    return "".join(out_chars), EqRule(op_name="symbolic_positional", test_op="")


# ---- Public API -----------------------------------------------------------

def parse(prompt: str) -> str | None:
    m = _TEST_RE.search(prompt)
    return m.group(1).strip() if m else None


def solve(prompt: str) -> tuple[str | None, EqRule | None]:
    test = parse(prompt)
    if test is None:
        return None, None
    op = _detect_test_operator(test)
    if op is not None:
        # Numeric attempt
        res, rule = _try_numeric(prompt, op, test)
        if res is not None:
            return res, rule
    # Fall back to symbolic positional
    return _try_symbolic(prompt, test)


def format_answer(value: str) -> str:
    return value
