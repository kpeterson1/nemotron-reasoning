"""Lightweight answer normalization."""
from __future__ import annotations


def normalize_answer(raw: str) -> str:
    """Strip whitespace and common formatting noise from a raw answer."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if s.startswith("`") and s.endswith("`"):
        s = s[1:-1].strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s
