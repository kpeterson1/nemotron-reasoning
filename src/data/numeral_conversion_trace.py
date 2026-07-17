"""Trace generator for numeral_conversion puzzles (Roman numerals)."""
from __future__ import annotations

from .numeral_conversion_solver import parse, to_roman, _ROMAN_TABLE


def generate_trace(prompt: str) -> tuple[str, str] | None:
    pairs, test_n = parse(prompt)
    if not pairs or test_n is None:
        return None
    # Verify Roman on every example (so the trace is self-consistent)
    if not all(to_roman(n) == tok for n, tok in pairs):
        return None

    parts: list[str] = []
    parts.append(
        "The puzzle gives a numeric value and its representation in the "
        "Wonderland numeral system. I'll check whether the system is "
        "standard Roman numerals."
    )
    parts.append("Verify each example by converting the integer to Roman:")
    for n, tok in pairs:
        parts.append(f"  {n} -> {to_roman(n)}   (puzzle says: {tok}) ✓")
    parts.append("The system is Roman numerals. Convert the test number step by step.")
    # Show the greedy decomposition
    parts.append(f"Decompose {test_n}:")
    remaining = test_n
    pieces = []
    decomp_lines = []
    for v, sym in _ROMAN_TABLE:
        while remaining >= v:
            decomp_lines.append(f"  {remaining} >= {v} ({sym})  -> append '{sym}', remainder {remaining - v}")
            remaining -= v
            pieces.append(sym)
    parts.extend(decomp_lines)
    parts.append(f"Concatenate the pieces: {' + '.join(pieces) if pieces else '(none)'} = {to_roman(test_n)}")
    answer = to_roman(test_n)
    return "\n".join(parts), answer
