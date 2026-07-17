"""Tests for src.evaluation.extract_answer.answers_match.

Cases come from the Kaggle metric docstring.
"""
from __future__ import annotations

from src.evaluation.extract_answer import answers_match


def test_exact_string_match() -> None:
    assert answers_match("hello", "hello")


def test_case_insensitive_text() -> None:
    assert answers_match("HELLO", "hello")
    assert answers_match("Hello World", "hello world")


def test_numeric_within_tolerance() -> None:
    assert answers_match("24.64", "24.6401")
    assert answers_match("3.14", "3.14159")


def test_numeric_outside_tolerance() -> None:
    assert not answers_match("3.14", "3.5")
    assert not answers_match("100", "120")


def test_roman_numeral_case_insensitive() -> None:
    assert answers_match("XLVII", "xlvii")
    assert answers_match("iv", "IV")


def test_binary_string_exact_match() -> None:
    assert answers_match("10011000", "10011000")
    assert not answers_match("10011000", "10011001")


def test_binary_string_not_parsed_as_number() -> None:
    # 00011011 != 11011 as strings; numeric parsing would say they're equal.
    assert not answers_match("00011011", "11011")
    assert not answers_match("11011", "00011011")


def test_near_zero_uses_abs_tolerance() -> None:
    # 0 vs 0.0001 — outside rel_tol but inside abs_tol=1e-5? abs diff is 1e-4,
    # which is > abs_tol=1e-5, so this should be False (matching Kaggle).
    assert not answers_match("0.0001", "0")


def test_within_relative_tolerance_one_percent() -> None:
    # 9.81 vs 9.83: |.02|/9.81 ≈ 0.002 < rel_tol=1e-2 -> True.
    assert answers_match("9.83", "9.81")
