"""Tests for src.data.format_training."""
from __future__ import annotations

from src.data.format_training import HARNESS_SUFFIX, format_training_example


class _StubTokenizer:
    def apply_chat_template(self, messages, tokenize: bool = False) -> str:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


def test_chat_template_tokens_present() -> None:
    rec = format_training_example(
        prompt="puzzle here",
        answer="42",
        reasoning_trace="step",
        tokenizer=_StubTokenizer(),
    )
    assert "<|im_start|>" in rec["text"]
    assert "<|im_end|>" in rec["text"]


def test_think_tags_present() -> None:
    rec = format_training_example(
        prompt="puzzle here",
        answer="42",
        reasoning_trace="reason here",
        tokenizer=_StubTokenizer(),
    )
    assert "<think>" in rec["text"]
    assert "</think>" in rec["text"]
    assert "reason here" in rec["text"]


def test_boxed_answer_in_assistant() -> None:
    rec = format_training_example(
        prompt="puzzle here",
        answer="42",
        reasoning_trace="step",
        tokenizer=_StubTokenizer(),
    )
    assert "\\boxed{42}" in rec["text"]
    assert "\\boxed{42}" in rec["completion_text"]


def test_harness_suffix_appended_to_user() -> None:
    rec = format_training_example(
        prompt="puzzle here",
        answer="42",
        reasoning_trace="step",
        tokenizer=_StubTokenizer(),
    )
    user_msg = rec["messages"][1]
    assert user_msg["role"] == "user"
    assert user_msg["content"].endswith(HARNESS_SUFFIX)
    assert "puzzle here" in user_msg["content"]
