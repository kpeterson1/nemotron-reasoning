"""Solver for gravitational_constant puzzles.

Each puzzle is of the form: "For t = X.XXs, distance = Y.YY m" example lines
followed by "Now, determine the falling distance for t = T.TTs given
d = 0.5*g*t^2.". The hidden rule is d = 0.5 * g * t^2 with a constant g
revealed by the examples (g = 2*d/t^2).

Solver: parse (t, d) pairs, compute g as the median of 2*d/t^2 across
examples, apply to the test t.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass


_EXAMPLE_RE = re.compile(
    r"For\s+t\s*=\s*([-+]?\d+(?:\.\d+)?)\s*s?\s*,\s*distance\s*=\s*([-+]?\d+(?:\.\d+)?)\s*m"
)
_TEST_RE = re.compile(
    r"determine\s+the\s+falling\s+distance\s+for\s+t\s*=\s*([-+]?\d+(?:\.\d+)?)\s*s",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GravityRule:
    g: float
    per_example_g: tuple[float, ...]


def parse(prompt: str) -> tuple[list[tuple[float, float]], float | None]:
    pairs = [(float(t), float(d)) for t, d in _EXAMPLE_RE.findall(prompt)]
    m = _TEST_RE.search(prompt)
    return pairs, (float(m.group(1)) if m else None)


def solve(prompt: str) -> tuple[float | None, GravityRule | None]:
    pairs, test_t = parse(prompt)
    if not pairs or test_t is None:
        return None, None
    gs = tuple(2 * d / (t * t) for t, d in pairs if t != 0)
    if not gs:
        return None, None
    g = statistics.median(gs)
    return 0.5 * g * test_t * test_t, GravityRule(g=g, per_example_g=gs)


def format_answer(value: float) -> str:
    """Match the ground-truth style (2 decimals)."""
    return f"{value:.2f}"
