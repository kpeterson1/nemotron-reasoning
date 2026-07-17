"""Single-adapter eval on the same fixed n=50 IDs used by eval_3way_vllm.py.

We reuse the cached base / old_v9 / scratch results from runs/eval/3way_smoke50.json,
and only run the warm-start adapter here — saves re-booting vLLM for three
already-evaluated conditions on identical prompts.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from src.evaluation.extract_answer import answers_match, extract_boxed_answer


_HARNESS_SUFFIX = (
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


def _build_chat_prompts(rows: list[dict], tokenizer) -> list[str]:
    out = []
    for r in rows:
        user_content = r["prompt"] + _HARNESS_SUFFIX
        messages = [{"role": "user", "content": user_content}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        except Exception:
            text = user_content
        out.append(text)
    return out


def _score(rows: list[dict], texts: list[str]) -> dict:
    correct = 0
    trunc = 0
    by_t_total: dict[str, int] = defaultdict(int)
    by_t_correct: dict[str, int] = defaultdict(int)
    preds = []
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
        preds.append({
            "id": row.get("id"),
            "task_type": t,
            "predicted": pred,
            "answer": row["answer"],
            "correct": ok,
            "truncated": truncated,
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
        "predictions": preds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--split-file", type=Path, default=Path("datasets/splits/dev_frozen_shuffled.jsonl"))
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=3584)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--label", default="warm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_jsonl(args.split_file)[: args.n]
    ids = [r.get("id") for r in rows]
    print(f"[eval] split={args.split_file.name}  n={len(rows)}  ids[:5]={ids[:5]}", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"[eval] booting LLM (~5 min)...", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=1,
        max_model_len=args.max_model_len,
        max_num_seqs=16,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    print(f"[eval] LLM ready.", flush=True)
    tokenizer = llm.get_tokenizer()
    prompts = _build_chat_prompts(rows, tokenizer)
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    lora_req = LoRARequest(args.label, 1, str(args.adapter))
    t0 = time.time()
    outs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_req)
    elapsed = time.time() - t0
    texts = [o.outputs[0].text for o in outs]
    score = _score(rows, texts)
    score["elapsed_sec"] = elapsed
    print(
        f"[eval] {args.label:8s}  acc={score['accuracy']:.4f}  trunc={score['trunc_rate']:.2f}  elapsed={elapsed:.1f}s",
        flush=True,
    )

    summary = {
        "n": len(rows),
        "ids": ids,
        "model": args.model,
        "adapter": str(args.adapter),
        "label": args.label,
        "results": {args.label: score},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"[eval] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
