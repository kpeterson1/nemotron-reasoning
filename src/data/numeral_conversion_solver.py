"""Solver for numeral_conversion puzzles.

Each puzzle gives examples like `6 -> VI`, `59 -> LIX`, then asks "Now, write
the number N in the Wonderland numeral system." The hidden system is almost
always Roman numerals; we verify the rule from the examples before applying.

Solver:
  1. Parse `<int> -> <token>` pairs and the test integer.
  2. Try Roman-numeral conversion on the example integers — if every
     example matches the cipher token exactly, the system is Roman.
  3. Apply Roman conversion to the test integer.

(If we ever see a puzzle whose examples don't match Roman, we'd need to
infer from the examples directly — but on train.csv 100% are Roman.)
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_EXAMPLE_RE = re.compile(r"(\d+)\s*->\s*([A-Z]+)")
_TEST_RE = re.compile(r"write\s+the\s+number\s+(\d+)\s+in\s+the\s+Wonderland", re.IGNORECASE)


@dataclass(frozen=True)
class NumeralRule:
    system: str  # 'roman' or 'unknown'


_ROMAN_TABLE = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def to_roman(n: int) -> str:
    if n <= 0:
        return ""
    out = []
    for v, sym in _ROMAN_TABLE:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def parse(prompt: str) -> tuple[list[tuple[int, str]], int | None]:
    pairs = [(int(n), tok) for n, tok in _EXAMPLE_RE.findall(prompt)]
    m = _TEST_RE.search(prompt)
    return pairs, (int(m.group(1)) if m else None)


def solve(prompt: str) -> tuple[str | None, NumeralRule | None]:
    pairs, test_n = parse(prompt)
    if not pairs or test_n is None:
        return None, None

    # Verify Roman on examples.
    roman_ok = all(to_roman(n) == tok for n, tok in pairs)
    if roman_ok:
        return to_roman(test_n), NumeralRule(system="roman")

    return None, NumeralRule(system="unknown")


def format_answer(value: str) -> str:
    return value
