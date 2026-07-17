"""Answer extraction and matching utilities.

CONTRACT: The boxed-extraction logic, fallback patterns, and verify()
semantics here MUST match docs/kaggle_metric_source.py exactly. The
Kaggle public score depends on this. If docs/kaggle_metric_source.py
changes (re-vendored from competition), re-run
tests/test_extract_answer_kaggle_parity.py and update this file.

Source: https://www.kaggle.com/code/metric/nvidia-nemotron-metric
(re-vendored locally at docs/kaggle_metric_source.py)

The two public functions in the metric are `extract_final_answer(text)` and
`verify(stored_answer, predicted)`. We expose `extract_boxed_answer` (a thin
alias for `extract_final_answer`) and `answers_match(predicted, ground_truth)`,
which calls `verify` with arguments swapped to read more naturally.
"""
from __future__ import annotations

import math
import re

_BOXED_START_RE = re.compile(r"\\boxed\{")
_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")
_BINARY_RE = re.compile(r"[01]+")

# Mirrors the Kaggle metric's fallback list. Note the full-width colon '：'
# alongside the ASCII colon, and the variants without "is".
_FINAL_ANSWER_PATTERNS = [
    re.compile(r"The final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
]


def extract_final_answer(text: str | None) -> str:
    r"""Extract the final answer from a model response.

    Strategy (mirrors Kaggle metric exactly):
      1. For each `\boxed{` occurrence, take everything up to the last `}`
         before the next `\boxed{` (or end of text). Handles answers that
         themselves contain `}` (e.g. `\boxed{}52}` -> `}52`) as well as
         nested LaTeX like `\boxed{\frac{1}{2}}` -> `\frac{1}{2}`. If any
         non-empty stripped segment exists, return the last one; otherwise
         return the last (empty) segment.
      2. Fall back to four "final answer" capture patterns (case-insensitive,
         supports ASCII and full-width colons).
      3. Fall back to the last numeric token.
      4. Fall back to the last non-empty line.
      5. Return "NOT_FOUND".
    """
    if text is None:
        return "NOT_FOUND"

    boxed_starts = list(_BOXED_START_RE.finditer(text))
    matches: list[str] = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    for pattern in _FINAL_ANSWER_PATTERNS:
        found = pattern.findall(text)
        if found:
            return found[-1].strip()

    nums = _NUMERIC_RE.findall(text)
    if nums:
        return nums[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


# Alias kept for older imports.
extract_boxed_answer = extract_final_answer


def verify(stored_answer: str, predicted: str) -> bool:
    """Match the Kaggle metric's `verify` signature and behavior.

    - If the ground truth is a pure binary string, compare strictly as strings
      (case-insensitive).
    - Else if both sides parse as floats, use `math.isclose(rel_tol=1e-2,
      abs_tol=1e-5)`.
    - Else case-insensitive string comparison.
    """
    if stored_answer is None or predicted is None:
        return False
    stored_answer = str(stored_answer).strip()
    predicted = str(predicted).strip()

    if _BINARY_RE.fullmatch(stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        return math.isclose(
            float(stored_answer), float(predicted), rel_tol=1e-2, abs_tol=1e-5
        )
    except (ValueError, TypeError):
        return predicted.lower() == stored_answer.lower()


def answers_match(predicted: str, ground_truth: str, tolerance: float = 1e-2) -> bool:
    """Convenience wrapper that calls `verify(ground_truth, predicted)`.

    The `tolerance` argument is accepted for API compatibility but ignored —
    the Kaggle metric hard-codes rel_tol=1e-2.
    """
    del tolerance
    return verify(ground_truth, predicted)


if __name__ == "__main__":
    cases = [
        (r"The answer is \boxed{42}", "42"),
        ("The final answer is: 3.14", "3.14"),
        ("Just a number 100 in text", "100"),
        (None, "NOT_FOUND"),
    ]
    for text, expected in cases:
        got = extract_final_answer(text)
        ok = "OK" if got == expected else "FAIL"
        print(f"[{ok}] extract({text!r}) -> {got!r} (expected {expected!r})")

    match_cases = [
        (("10011000", "10011000"), True),
        (("10011000", "10011001"), False),
        (("24.64", "24.6401"), True),
        (("XLVII", "xlvii"), True),
        (("11011", "00011011"), False),
    ]
    for (stored, pred), expected in match_cases:
        got = verify(stored, pred)
        ok = "OK" if got == expected else "FAIL"
        print(f"[{ok}] verify({stored!r}, {pred!r}) -> {got} (expected {expected})")
