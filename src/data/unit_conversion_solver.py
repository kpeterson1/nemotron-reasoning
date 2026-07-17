"""Solver for unit_conversion puzzles.

Each puzzle is of the form: "X.XX m becomes Y.YY" example lines followed by
"Now, convert the following measurement: T.TT m". The hidden rule is a linear
scaling Y = k * X for some constant k revealed by the examples.

Solver: parse pairs, compute k as the median ratio (robust to outliers),
apply to the test input.

`solve(prompt)` returns the predicted answer as a `float` (formatting handled
by the trace generator).
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass


_EXAMPLE_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*m\s+becomes\s+([-+]?\d+(?:\.\d+)?)")
_TEST_RE = re.compile(
    r"Now,\s*convert\s+the\s+following\s+measurement:\s*([-+]?\d+(?:\.\d+)?)\s*m"
)


@dataclass(frozen=True)
class UnitConversionRule:
    k: float
    ratios: tuple[float, ...]  # per-example ratios, for tracing


def parse(prompt: str) -> tuple[list[tuple[float, float]], float | None]:
    pairs = [
        (float(a), float(b)) for a, b in _EXAMPLE_RE.findall(prompt)
    ]
    m = _TEST_RE.search(prompt)
    return pairs, (float(m.group(1)) if m else None)


def solve(prompt: str) -> tuple[float | None, UnitConversionRule | None]:
    pairs, test_x = parse(prompt)
    if not pairs or test_x is None:
        return None, None
    ratios = tuple(y / x for x, y in pairs if x != 0)
    if not ratios:
        return None, None
    k = statistics.median(ratios)
    return k * test_x, UnitConversionRule(k=k, ratios=ratios)


def format_answer(value: float) -> str:
    """Round to 2 decimals, strip trailing zeros only when the result has a
    fractional part — match the ground-truth answer style seen in train.csv
    (e.g. 47.94, 50.9, 112.85, 27.11)."""
    s = f"{value:.2f}"
    return s
