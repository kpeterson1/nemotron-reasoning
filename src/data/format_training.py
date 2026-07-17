"""Format puzzle examples into chat-template training records.

The Kaggle harness appends a `\\boxed{}` instruction suffix to every prompt and
runs inference with `enable_thinking=True`, producing a `<think>...</think>`
trace before the final answer. Training examples must mirror that exact shape so
the LoRA adapter learns to use the same format at inference time.
"""
from __future__ import annotations

from typing import Any

HARNESS_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def _user_message(prompt: str) -> str:
    return prompt + HARNESS_SUFFIX


def _assistant_text(reasoning_trace: str, answer: str) -> str:
    return f"<think>\n{reasoning_trace}\n</think>\n\\boxed{{{answer}}}"


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def format_training_example(
    prompt: str,
    answer: str,
    reasoning_trace: str,
    tokenizer: Any,
) -> dict[str, Any]:
    """Format one example as a chat-templated record with a completion mask.

    Returns a dict with:
      - text: the full chat-templated string
      - prompt_text: the chat-templated string up to (but not including) the
        assistant turn — used to compute the completion mask boundary
      - completion_text: the assistant turn (think + boxed answer)
      - messages: the raw chat messages
    """
    user = _user_message(prompt)
    assistant = _assistant_text(reasoning_trace, answer)
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    text = _apply_chat_template(tokenizer, messages)
    prompt_messages = messages[:-1]
    prompt_text = _apply_chat_template(tokenizer, prompt_messages)
    return {
        "text": text,
        "prompt_text": prompt_text,
        "completion_text": assistant,
        "messages": messages,
    }


def format_simple_example(prompt: str, answer: str, tokenizer: Any) -> dict[str, Any]:
    """Format an example with no reasoning trace — empty `<think>` block."""
    return format_training_example(
        prompt=prompt, answer=answer, reasoning_trace="", tokenizer=tokenizer
    )


class _StubTokenizer:
    """Fallback tokenizer used by `__main__` so this module is runnable without
    HuggingFace installed."""

    def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool = False) -> str:
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


if __name__ == "__main__":
    tok = _StubTokenizer()
    rec = format_training_example(
        prompt="In Alice's Wonderland, a secret bit manipulation rule applies. "
        "Examples: 1010 -> 0101. Now: 1100 -> ?",
        answer="0011",
        reasoning_trace="Each bit is flipped (NOT). 1100 -> 0011.",
        tokenizer=tok,
    )
    print(rec["text"])
    print("---")
    print("completion:", rec["completion_text"])
