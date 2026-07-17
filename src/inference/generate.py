"""vLLM-backed generation that mirrors the Kaggle harness exactly.

The harness:
  - Loads `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` with vLLM, max_model_len=4096,
    gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=32.
  - Builds prompts as: tokenizer.apply_chat_template([{role:'user', content:
    PROMPT + HARNESS_SUFFIX}], add_generation_prompt=True, enable_thinking=True).
  - Samples with temperature=1.0, top_p=1.0, max_tokens=3584.
  - Optionally attaches a LoRA adapter as LoRARequest.

This file matches that recipe so local runs are scoring-equivalent to leaderboard.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

HARNESS_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# Cache the LLM/tokenizer between calls in the same process. vLLM startup is
# expensive; we don't want to pay it per prompt batch.
_LLM = None
_TOKENIZER = None


def _ensure_llm(
    model: str,
    *,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_lora_rank: int,
    max_num_seqs: int,
):
    global _LLM, _TOKENIZER
    if _LLM is not None:
        return _LLM, _TOKENIZER

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

    from vllm import LLM  # type: ignore

    _LLM = LLM(
        model=model,
        dtype="auto",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    _TOKENIZER = _LLM.get_tokenizer()
    return _LLM, _TOKENIZER


def generate(
    prompts: list[str],
    model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    adapter_dir: str | None = None,
    *,
    system_message: str | None = None,
    max_tokens: int = 3584,
    temperature: float = 1.0,
    top_p: float = 1.0,
    enable_thinking: bool = True,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    max_lora_rank: int = 32,
    max_num_seqs: int = 16,
    append_harness_suffix: bool = True,
) -> list[str]:
    """Generate one completion per prompt. Each prompt is wrapped with the
    chat template (optional system, user=prompt[+suffix]) and sampled per the
    harness parameters. The Kaggle harness itself uses NO system message; pass
    `system_message=None` to match the harness exactly."""
    if not prompts:
        return []

    from vllm import SamplingParams  # type: ignore

    llm, tokenizer = _ensure_llm(
        model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_lora_rank=max_lora_rank,
        max_num_seqs=max_num_seqs,
    )

    chat_inputs = []
    for p in prompts:
        user_content = p + HARNESS_SUFFIX if append_harness_suffix else p
        messages: list[dict[str, str]] = []
        if system_message is not None:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_content})
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except Exception:
            text = user_content
        chat_inputs.append(text)

    sampling = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )

    lora_request = None
    if adapter_dir:
        from vllm.lora.request import LoRARequest  # type: ignore

        lora_request = LoRARequest("adapter", 1, adapter_dir)

    outputs = llm.generate(chat_inputs, sampling_params=sampling, lora_request=lora_request)
    return [o.outputs[0].text for o in outputs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--adapter-dir", default=None)
    args = parser.parse_args()
    prompts = [p for p in args.prompt_file.read_text().splitlines() if p.strip()]
    outputs = generate(prompts, model=args.model, adapter_dir=args.adapter_dir)
    for out in outputs:
        print(out)
        print("---")


if __name__ == "__main__":
    main()
