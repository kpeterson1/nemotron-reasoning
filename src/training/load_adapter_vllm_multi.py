"""Diagnostic vLLM run: boot once, test base + multiple LoRA adapters on multiple prompts.

Per LoRA request a separate `LoRARequest` (with a unique int id) is sent.
"""
from __future__ import annotations

import argparse
from pathlib import Path


_PROMPTS = [
    ("short_arith", "<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n"),
    # Longer, in-distribution prompt (matches v9 training data shape: a bit_manip puzzle).
    ("bit_manip", "<|im_start|>user\nGiven the following 8-bit bit manipulation puzzle, find the rule and output the answer for the test input.\n\nInput  -> Output\n10110011 -> 01100111\n00001111 -> 00011110\n11000011 -> 10000111\nTest: 01010101 -> ?\nPlease put your final answer inside `\\boxed{}`.<|im_end|>\n<|im_start|>assistant\n"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    parser.add_argument(
        "--adapter-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more adapter directories to compare.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"[multi] booting LLM (~5 min)...", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=len(args.adapter_dirs) + 1,
        max_model_len=2048,
        enforce_eager=True,
    )
    print(f"[multi] LLM ready.", flush=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    results: dict[str, dict[str, list[int]]] = {}  # prompt_name -> adapter_name -> token_ids

    for prompt_name, prompt in _PROMPTS:
        print(f"\n[multi] === prompt {prompt_name!r} ===", flush=True)
        # Base
        out_base = llm.generate([prompt], sampling_params=sampling, lora_request=None)
        base_ids = list(out_base[0].outputs[0].token_ids)
        base_text = out_base[0].outputs[0].text
        results.setdefault(prompt_name, {})["base"] = base_ids
        print(f"[multi] base       len={len(base_ids):2d}  ids[:16]={base_ids[:16]}  text={base_text[:80]!r}", flush=True)

        # Each adapter
        for i, d in enumerate(args.adapter_dirs, start=1):
            lora_req = LoRARequest(d.name, i, str(d))
            outs = llm.generate([prompt], sampling_params=sampling, lora_request=lora_req)
            ids = list(outs[0].outputs[0].token_ids)
            text = outs[0].outputs[0].text
            results[prompt_name][d.name] = ids
            same = ids == base_ids
            print(f"[multi] {d.name:50s}  same_as_base={same}  len={len(ids):2d}  ids[:16]={ids[:16]}  text={text[:80]!r}", flush=True)

    # Summary
    print(f"\n[multi] SUMMARY")
    for prompt_name, by_adapter in results.items():
        print(f"  prompt: {prompt_name}")
        base = by_adapter["base"]
        for name, ids in by_adapter.items():
            if name == "base":
                continue
            print(f"    {name:50s}  diff_from_base={ids != base}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[multi] FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise
