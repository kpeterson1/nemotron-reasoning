"""Trace generator for gravitational_constant puzzles."""
from __future__ import annotations

import statistics

from .gravitational_constant_solver import parse, format_answer


def generate_trace(prompt: str) -> tuple[str, str] | None:
    pairs, test_t = parse(prompt)
    if not pairs or test_t is None:
        return None
    gs = [2 * d / (t * t) for t, d in pairs if t != 0]
    if not gs:
        return None
    g = statistics.median(gs)
    answer_value = 0.5 * g * test_t * test_t

    parts: list[str] = []
    parts.append(
        "The puzzle gives free-fall observations under a modified gravitational "
        "constant. The formula is d = 0.5 * g * t², so g = 2d / t²."
    )
    parts.append("Compute g from each example:")
    for t, d in pairs:
        if t == 0:
            continue
        g_i = 2 * d / (t * t)
        parts.append(
            f"  t={t}, d={d}:   g = 2 × {d} / {t}² = {2*d:.4f} / {t*t:.4f} = {g_i:.4f}"
        )
    parts.append(f"The values of g are consistent; take the median: g ≈ {g:.4f}.")
    parts.append(
        f"Apply the formula to t={test_t}:   d = 0.5 × {g:.4f} × {test_t}² "
        f"= 0.5 × {g:.4f} × {test_t*test_t:.4f} = {answer_value:.4f}"
    )
    parts.append(f"Rounded to 2 decimal places: {answer_value:.2f}")
    return "\n".join(parts), format_answer(answer_value)
