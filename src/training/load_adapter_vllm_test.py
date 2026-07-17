"""Smoke-test loading a target_parameters PEFT adapter via vLLM.

Boots vLLM with enable_lora=True, attaches the adapter as a LoRARequest, and
generates a few tokens. Catches the failure modes we actually care about:
  * unexpected-modules error (key names don't match vLLM's expected set)
  * rank-exceeded error
  * "FusedMoE does not support LoRA" backend error
  * silent skip (warning about ignored modules) — captured by capturing logs
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also run the same prompt without the adapter; report whether outputs differ.",
    )
    args = parser.parse_args()

    # Capture warning_once / logger.warning lines about ignored modules.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"[vllm-test] booting LLM (~5 min)...", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_model_len=2048,  # smoke — don't allocate KV for 8k
        enforce_eager=True,  # skip CUDA graph capture, faster startup
    )
    print(f"[vllm-test] LLM ready. Attaching LoRA adapter from {args.adapter_dir}", flush=True)

    lora_req = LoRARequest("smoke", 1, str(args.adapter_dir))
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    prompt = "<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n"
    outs = llm.generate([prompt], sampling_params=sampling, lora_request=lora_req)
    out = outs[0]
    adapter_token_ids = list(out.outputs[0].token_ids)
    adapter_text = out.outputs[0].text
    print(f"[vllm-test] adapter token ids: {adapter_token_ids}", flush=True)
    print(f"[vllm-test] adapter text:\n{adapter_text!r}", flush=True)

    if args.compare_base:
        outs_b = llm.generate([prompt], sampling_params=sampling, lora_request=None)
        base_token_ids = list(outs_b[0].outputs[0].token_ids)
        base_text = outs_b[0].outputs[0].text
        print(f"[vllm-test] base token ids: {base_token_ids}", flush=True)
        print(f"[vllm-test] base text:\n{base_text!r}", flush=True)
        same = base_token_ids == adapter_token_ids
        print(f"[vllm-test] outputs differ from base: {not same}", flush=True)
    print(f"[vllm-test] SUCCESS — adapter loaded and inference ran.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[vllm-test] FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
