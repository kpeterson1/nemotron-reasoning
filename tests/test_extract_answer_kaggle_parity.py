"""Parity contract between src/evaluation/extract_answer.py and docs/kaggle_metric_source.py.

The Kaggle public score is computed by the metric at
https://www.kaggle.com/code/metric/nvidia-nemotron-metric. A local snapshot
lives at docs/kaggle_metric_source.py. The functions in
src/evaluation/extract_answer.py MUST produce byte-identical output to that
snapshot for both extract_final_answer and verify. This test enforces that.

Hand-picked cases (known mismatches and docstring regressions) are sanity
checks. The property test at the bottom is the actual parity contract: it
loads the snapshot via importlib (with subprocess/external-dep imports
monkey-patched to no-op) and diffs the two implementations on a battery of
generated strings.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

from src.evaluation.extract_answer import (
    answers_match,
    extract_boxed_answer,
    extract_final_answer,
    verify,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_METRIC_PATH = REPO_ROOT / "docs" / "kaggle_metric_source.py"


@pytest.fixture(scope="module")
def kaggle_metric_module():
    """Load docs/kaggle_metric_source.py with bootstrap blocks neutered.

    The vendored Kaggle source executes `subprocess.run` and
    `kagglehub.model_download` at module import time. We stub out the
    external deps (kagglehub, pandas, tqdm) via sys.modules and patch
    subprocess.run to a no-op so importlib can load the module locally.
    """
    if not KAGGLE_METRIC_PATH.exists():
        pytest.skip(f"docs/kaggle_metric_source.py not present at {KAGGLE_METRIC_PATH}")

    class _PermissiveStub(types.ModuleType):
        def __getattr__(self, name):
            return _PermissiveCallable()

    class _PermissiveCallable:
        def __call__(self, *args, **kwargs):
            return "/tmp/stub"

        def __getattr__(self, name):
            return _PermissiveCallable()

    saved_modules: dict[str, types.ModuleType | None] = {}
    for name in ("kagglehub", "pandas", "tqdm"):
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = _PermissiveStub(name)

    saved_run = subprocess.run
    subprocess.run = lambda *args, **kwargs: None  # type: ignore[assignment]

    try:
        spec = importlib.util.spec_from_file_location(
            "_kaggle_metric_under_test", KAGGLE_METRIC_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        subprocess.run = saved_run  # type: ignore[assignment]
        for name, prev in saved_modules.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev

    return module


# --- Sanity: the three known-mismatch cases must now PASS ---------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"\boxed{}52}", "}52"),
        (r"\boxed{\frac{a+b}{c}}", r"\frac{a+b}{c}"),
    ],
)
def test_known_mismatch_cases_fixed(text: str, expected: str) -> None:
    assert extract_final_answer(text) == expected


# --- Sanity: all four examples from the original docstring (regression) -------


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"The answer is \boxed{42}", "42"),
        ("The final answer is: 3.14", "3.14"),
        ("Just a number 100 in text", "100"),
        (None, "NOT_FOUND"),
    ],
)
def test_docstring_examples(text: str | None, expected: str) -> None:
    assert extract_final_answer(text) == expected


# --- Sanity: verify() examples from Kaggle metric docstring -------------------


@pytest.mark.parametrize(
    "stored,predicted,expected",
    [
        ("10011000", "10011000", True),
        ("10011000", "10011001", False),
        ("24.64", "24.6401", True),
        ("XLVII", "xlvii", True),
        ("11011", "00011011", False),
    ],
)
def test_verify_docstring_examples(stored: str, predicted: str, expected: bool) -> None:
    assert verify(stored, predicted) is expected


# --- Sanity: aliases and convenience wrappers ---------------------------------


def test_extract_boxed_answer_is_alias() -> None:
    assert extract_boxed_answer is extract_final_answer


def test_answers_match_argument_order() -> None:
    # answers_match(predicted, ground_truth) == verify(ground_truth, predicted)
    assert answers_match("xlvii", "XLVII") is True
    assert answers_match("10011001", "10011000") is False


# --- Property test: 30+ generated strings, byte-identical output --------------


def _parity_corpus() -> list[str]:
    """Battery of strings spanning LaTeX, numerics, prose, edge cases.

    Cases are crafted to stress: nested braces, multiple boxes, empty boxes,
    answers containing literal '}', unclosed boxes at EOS, all four fallback
    patterns, full-width colon variants, numeric fallback, last-line fallback,
    pure-whitespace text, and degenerate inputs.
    """
    return [
        # Boxed: nested LaTeX
        r"\boxed{\frac{1}{2}}",
        r"\boxed{\frac{a+b}{c}}",
        r"\boxed{\sqrt{x^2 + y^2}}",
        r"\boxed{\frac{\sqrt{2}}{\pi}}",
        # Boxed: degenerate / pathological
        r"\boxed{}",
        r"\boxed{}52}",
        r"\boxed{}",
        r"\boxed{ }",
        r"\boxed{ } \boxed{}",
        r"\boxed{a} \boxed{}",
        r"\boxed{} \boxed{b}",
        r"\boxed{a}\boxed{b}",
        r"\boxed{first} junk \boxed{second}",
        r"trailing \boxed{42",
        r"\boxed{multi\nline}",
        # Boxed: with answers that contain '}'
        r"answer: \boxed{}52}",
        r"\boxed{f(x)} done",
        # Boxed: simple
        r"The answer is \boxed{42}",
        r"\boxed{  hello  }",
        r"\boxed{-3.14}",
        # No-box fallback: "The final answer is:"
        "The final answer is: 3.14",
        "The final answer is: hello world",
        "Final answer is: 99",
        # No-box fallback: "Final answer:" without "is"
        "Final answer: 9.81",
        "final answer: xlvii",
        # No-box fallback: full-width colon
        "Final answer：9.81",
        "final answer：xlvii",
        # No-box fallback: numeric
        "intermediate 1, 2, 3 then 99 done",
        "the value is -3.14 and that is it",
        "only one number: 42",
        # No-box fallback: last line
        "alpha\nbeta\ngamma",
        "first line\nsecond line",
        # Edge: empty / whitespace
        "",
        "   ",
        "\n\n",
        # Edge: no boxed, no patterns, no numbers
        "just some prose",
        "abc def ghi",
        # Mixed: boxed and "final answer"
        r"Final answer: 5 \boxed{6}",
        # Multiple boxes where first is non-empty, second empty
        r"\boxed{real} junk \boxed{}",
        # Boxed where content is just whitespace
        r"\boxed{   } \boxed{actual}",
    ]


@pytest.mark.parametrize("text", _parity_corpus())
def test_parity_with_kaggle_source(kaggle_metric_module, text: str) -> None:
    """Each input must produce byte-identical output across implementations."""
    local = extract_final_answer(text)
    kaggle = kaggle_metric_module.extract_final_answer(text)
    assert local == kaggle, (
        f"Parity break on input={text!r}: "
        f"local={local!r} kaggle={kaggle!r}"
    )


def test_parity_corpus_size() -> None:
    """Guard against the corpus shrinking below the contracted floor (30+)."""
    assert len(_parity_corpus()) >= 30
