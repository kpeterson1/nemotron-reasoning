"""Filter generated samples by verifier-checkable criteria — placeholder."""
from __future__ import annotations

from src.evaluation.extract_answer import answers_match


def keep_if_correct(predicted: str, ground_truth: str) -> bool:
    return answers_match(predicted, ground_truth)
