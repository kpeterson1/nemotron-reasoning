"""Run dev_frozen evaluation using THE EXACT Kaggle metric prompt builder.

Per the user-supplied Kaggle metric source:

    tokenizer.apply_chat_template(
        [{'role': 'user', 'content': user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

with `user_content = problem_prompt + HARNESS_SUFFIX`.

This script boots vLLM once, runs both the old adapter and the new adapter at
greedy T=0.0 on n=N, optionally also at T=1.0 for diagnostic comparison, and
writes a single JSON with all results.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from src.evaluation.extract_answer import answers_match, extract_boxed_answer


# Verbatim from the user's Kaggle metric description.
HARNESS_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def _load_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def kaggle_prompt(user_problem: str, tokenizer) -> str:
    """Exact Kaggle metric prompt builder — user-only chat template."""
    user_content = user_problem + HARNESS_SUFFIX
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def _score(rows: list[dict], texts: list[str]) -> dict:
    correct = 0
    trunc = 0
    by_t_total: dict[str, int] = defaultdict(int)
    by_t_correct: dict[str, int] = defaultdict(int)
    preds = []
    trace_format_ok = 0   # response contained '<think>' AND '</think>' AND '\boxed{'
    boxed_present = 0
    think_open = 0
    think_close = 0
    for row, raw in zip(rows, texts):
        pred = extract_boxed_answer(raw)
        ok = answers_match(pred, row["answer"])
        truncated = "\\boxed{" not in raw
        t = row.get("task_type", "unknown")
        by_t_total[t] += 1
        if ok:
            correct += 1
            by_t_correct[t] += 1
        if truncated:
            trunc += 1
        has_box = "\\boxed{" in raw
        has_to = "<think>" in raw
        has_tc = "</think>" in raw
        if has_box: boxed_present += 1
        if has_to: think_open += 1
        if has_tc: think_close += 1
        if has_box and has_to and has_tc:
            trace_format_ok += 1
        preds.append({
            "id": row.get("id"),
            "task_type": t,
            "predicted": pred,
            "answer": row["answer"],
            "correct": ok,
            "truncated": truncated,
            "has_think_open": has_to,
            "has_think_close": has_tc,
            "has_boxed": has_box,
            "raw_len": len(raw),
            "raw_head": raw[:160],
        })
    n = len(rows) or 1
    return {
        "n": n,
        "accuracy": correct / n,
        "trunc_rate": trunc / n,
        "per_task": {t: by_t_correct[t] / by_t_total[t] for t in by_t_total},
        "per_task_n": dict(by_t_total),
        "trace_format_ok_rate": trace_format_ok / n,
        "has_boxed_rate": boxed_present / n,
        "has_think_open_rate": think_open / n,
        "has_think_close_rate": think_close / n,
        "predictions": preds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--split-file", type=Path, default=Path("datasets/splits/dev_frozen.jsonl"))
    parser.add_argument("--n-list", type=int, nargs="+", default=[50, 500])
    # Defaults match Kaggle competition Overview tab values exactly. Do not
    # change without re-verifying against the leaderboard runtime.
    parser.add_argument("--max-tokens", type=int, default=7680)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--old-adapter", type=Path, required=True)
    parser.add_argument("--new-adapter", type=Path, required=True)
    parser.add_argument("--old-label", default="old_v9")
    parser.add_argument("--new-label", default="warm_300")
    parser.add_argument("--include-t1-n", type=int, default=None,
                        help="Also run T=1.0 sampled on the first N rows (diagnostic). 0/None to skip.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows_all = _load_jsonl(args.split_file)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"[kaggle-eval] booting LLM (~5 min)...", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=2,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    print(f"[kaggle-eval] LLM ready.", flush=True)
    tokenizer = llm.get_tokenizer()

    # Sanity print: show the first prompt so the user can confirm format.
    sample_prompt = kaggle_prompt(rows_all[0]["prompt"], tokenizer)
    print(f"[kaggle-eval] sample prompt (first dev_frozen row, id={rows_all[0].get('id')}, length={len(sample_prompt)}):", flush=True)
    print(repr(sample_prompt[:240]) + " …", flush=True)
    print(f"  endswith assistant prefix?  {sample_prompt.endswith('<|im_start|>assistant\\n<think>\\n')}", flush=True)

    summary: dict = {
        "model": args.model,
        "split_file": str(args.split_file),
        "old_adapter": str(args.old_adapter),
        "new_adapter": str(args.new_adapter),
        "harness_suffix": HARNESS_SUFFIX,
        "results_by_n_temp": {},
    }

    adapters = [
        (args.old_label, args.old_adapter, 1),
        (args.new_label, args.new_adapter, 2),
    ]

    for n in args.n_list:
        rows = rows_all[:n]
        prompts = [kaggle_prompt(r["prompt"], tokenizer) for r in rows]

        # Greedy T=0
        for label, adapter_path, lora_id in adapters:
            lora_req = LoRARequest(label, lora_id, str(adapter_path))
            sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
            t0 = time.time()
            outs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_req)
            elapsed = time.time() - t0
            texts = [o.outputs[0].text for o in outs]
            score = _score(rows, texts)
            score["elapsed_sec"] = elapsed
            score["temperature"] = 0.0
            score["n_requested"] = n
            key = f"n{n}_T0.0_{label}"
            summary["results_by_n_temp"][key] = score
            print(
                f"[kaggle-eval] {key:42s}  acc={score['accuracy']:.4f}  trunc={score['trunc_rate']:.2f}  "
                f"trace_ok={score['trace_format_ok_rate']:.2f}  elapsed={elapsed:.1f}s",
                flush=True,
            )

        # T=1.0 diagnostic
        if args.include_t1_n and args.include_t1_n > 0 and n <= args.include_t1_n:
            for label, adapter_path, lora_id in adapters:
                lora_req = LoRARequest(label, lora_id, str(adapter_path))
                sampling = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=0)
                t0 = time.time()
                outs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_req)
                elapsed = time.time() - t0
                texts = [o.outputs[0].text for o in outs]
                score = _score(rows, texts)
                score["elapsed_sec"] = elapsed
                score["temperature"] = 1.0
                score["n_requested"] = n
                score["seed"] = 0
                key = f"n{n}_T1.0_{label}"
                summary["results_by_n_temp"][key] = score
                print(
                    f"[kaggle-eval] {key:42s}  acc={score['accuracy']:.4f}  trunc={score['trunc_rate']:.2f}  "
                    f"trace_ok={score['trace_format_ok_rate']:.2f}  elapsed={elapsed:.1f}s",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"[kaggle-eval] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
