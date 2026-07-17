"""Tests for src.evaluation.extract_answer.extract_boxed_answer."""
from __future__ import annotations

from src.evaluation.extract_answer import extract_boxed_answer


def test_simple_boxed() -> None:
    assert extract_boxed_answer("The answer is \\boxed{42}") == "42"


def test_boxed_with_inner_parens() -> None:
    assert extract_boxed_answer("Result: \\boxed{f(x)}") == "f(x)"


def test_multiple_boxed_returns_last_non_empty() -> None:
    assert extract_boxed_answer("\\boxed{first} then \\boxed{second}") == "second"


def test_multiple_boxed_skips_empty() -> None:
    assert extract_boxed_answer("\\boxed{real} junk \\boxed{}") == "real"


def test_no_boxed_falls_back_to_final_answer() -> None:
    assert extract_boxed_answer("The final answer is: 3.14") == "3.14"


def test_no_boxed_falls_back_to_last_numeric() -> None:
    out = extract_boxed_answer("intermediate 1, 2, 3 then 99 done")
    assert out == "99"


def test_no_boxed_falls_back_to_last_line() -> None:
    out = extract_boxed_answer("alpha\nbeta\ngamma")
    assert out == "gamma"


def test_unclosed_boxed_at_end_of_string() -> None:
    assert extract_boxed_answer("trailing \\boxed{42") == "42"


def test_boxed_with_surrounding_spaces() -> None:
    assert extract_boxed_answer("\\boxed{  hello  }") == "hello"


def test_empty_input_returns_not_found() -> None:
    assert extract_boxed_answer("") == "NOT_FOUND"


def test_none_input_returns_not_found() -> None:
    assert extract_boxed_answer(None) == "NOT_FOUND"


def test_only_empty_boxed_returns_empty_string() -> None:
    # Kaggle behavior: if matches exist but all are empty, return ''.
    assert extract_boxed_answer("\\boxed{}") == ""
    assert extract_boxed_answer("\\boxed{} \\boxed{}") == ""


def test_empty_boxed_with_real_match_returns_real() -> None:
    assert extract_boxed_answer("\\boxed{} and \\boxed{real}") == "real"


def test_final_answer_pattern_without_is() -> None:
    assert extract_boxed_answer("Final answer: 9.81") == "9.81"


def test_final_answer_pattern_full_width_colon() -> None:
    assert extract_boxed_answer("Final answer：9.81") == "9.81"
    assert extract_boxed_answer("final answer：xlvii") == "xlvii"


def test_no_boxed_falls_back_to_negative_numeric() -> None:
    assert extract_boxed_answer("the value is -3.14 and that is it") == "-3.14"
