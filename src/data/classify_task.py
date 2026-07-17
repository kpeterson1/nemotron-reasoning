"""Classify a puzzle prompt into one of the six task types."""
from __future__ import annotations

_PATTERNS: tuple[tuple[str, str], ...] = (
    ("bit manipulation rule", "bit_manipulation"),
    ("secret encryption rules", "text_encryption"),
    ("numbers are secretly converted", "numeral_conversion"),
    ("secret unit conversion", "unit_conversion"),
    ("gravitational constant", "gravitational_constant"),
    ("secret set of transformation rules", "equation_transformation"),
)


def classify_task(prompt: str) -> str:
    """Return one of the six task type labels, or 'unknown' if no pattern matches."""
    if not prompt:
        return "unknown"
    p = prompt.lower()
    for needle, label in _PATTERNS:
        if needle in p:
            return label
    return "unknown"
