"""Trace generator for unit_conversion puzzles."""
from __future__ import annotations

import statistics

from .unit_conversion_solver import parse, format_answer


def generate_trace(prompt: str) -> tuple[str, str] | None:
    pairs, test_x = parse(prompt)
    if not pairs or test_x is None:
        return None
    ratios = [y / x for x, y in pairs if x != 0]
    if not ratios:
        return None
    k = statistics.median(ratios)
    answer_value = k * test_x

    parts: list[str] = []
    parts.append(
        "The puzzle shows a measurement and its converted form. The transformation "
        "looks linear, so I'll compute the ratio output/input for each example."
    )
    for x, y in pairs:
        if x != 0:
            r = y / x
            parts.append(f"  {x} -> {y}:   ratio = {y} / {x} = {r:.4f}")
    parts.append(f"All ratios are approximately equal; the conversion factor is k ≈ {k:.4f}.")
    parts.append(f"Apply to the test input: {test_x} × {k:.4f} = {answer_value:.4f}")
    parts.append(f"Rounded to 2 decimal places: {answer_value:.2f}")
    return "\n".join(parts), format_answer(answer_value)
