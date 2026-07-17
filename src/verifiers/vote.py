"""Majority/self-consistency voting across multiple samples — placeholder."""
from __future__ import annotations

from collections import Counter


def majority_vote(answers: list[str]) -> str:
    """Return the most common answer; ties broken by first occurrence."""
    if not answers:
        return ""
    counts = Counter(answers)
    return counts.most_common(1)[0][0]
