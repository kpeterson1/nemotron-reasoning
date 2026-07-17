"""Multi-adapter Kaggle-exact eval with configurable context/budget.

Same prompt builder as the user's Kaggle metric description:
    tokenizer.apply_chat_template(
        [{'role': 'user', 'content': user_content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )

with user_content = problem + HARNESS_SUFFIX.

Loads vLLM once, attaches up to --max-loras adapters, runs greedy generation
(and optionally T=1 diagnostic) for each requested (n, T) pair across every
adapter, writes a single JSON.

Usage:
    python -m src.training.eval_kaggle_exact_multi \\
        --split-file datasets/splits/dev_frozen.jsonl \\
        --n-list 50 500 \\
        --max-tokens 7680 --max-model-len 8192 \\
        --adapter v9 runs/train/lora_v9 \\
        --adapter warm300 runs/train/lora_v9_warm_start_300step_vllm \\
        --adapter warm300_r32padded runs/train/lora_v9_warm_start_300step_vllm_r32padded \\
        --output runs/eval/kaggle_exact_multi.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from src.evaluation.extract_answer import answers_match, extract_boxed_answer


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
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_problem + HARNESS_SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


def _score(rows: list[dict], texts: list[str]) -> dict:
    correct = 0
    trunc = 0
    by_t_total: dict[str, int] = defaultdict(int)
    by_t_correct: dict[str, int] = defaultdict(int)
    preds = []
    boxed_present = 0
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
        has_tc = "</think>" in raw
        if has_box: boxed_present += 1
        if has_tc: think_close += 1
        preds.append({
            "id": row.get("id"),
            "task_type": t,
            "predicted": pred,
            "answer": row["answer"],
            "correct": ok,
            "truncated": truncated,
            "has_boxed": has_box,
            "has_think_close": has_tc,
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
        "has_boxed_rate": boxed_present / n,
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
    parser.add_argument("--max-loras", type=int, default=3)
    parser.add_argument("--include-t1-n", type=int, default=None,
                        help="Also run T=1.0 sampled on the first N rows (diagnostic).")
    parser.add_argument("--adapter", action="append", nargs=2, metavar=("LABEL", "PATH"),
                        required=True, help="Adapter to evaluate. Repeat for multiple adapters.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows_all = _load_jsonl(args.split_file)
    adapters = [(label, Path(path), i + 1) for i, (label, path) in enumerate(args.adapter)]
    print(f"[kme] adapters ({len(adapters)}):", flush=True)
    for label, path, lid in adapters:
        print(f"        ({lid}) {label}  ->  {path}", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"[kme] booting LLM (max_model_len={args.max_model_len})...", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=max(args.max_loras, len(adapters)),
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    print(f"[kme] LLM ready.", flush=True)
    tokenizer = llm.get_tokenizer()

    # Render prompts once per n (the prompt content doesn't depend on adapter).
    summary: dict = {
        "model": args.model,
        "split_file": str(args.split_file),
        "harness_suffix": HARNESS_SUFFIX,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "adapters": [{"label": label, "path": str(path)} for label, path, _ in adapters],
        "results_by_n_temp_adapter": {},
    }

    # Sanity print the first prompt
    sample = kaggle_prompt(rows_all[0]["prompt"], tokenizer)
    print(f"[kme] sample prompt for id={rows_all[0].get('id')}: len={len(sample)}", flush=True)
    print(f"      last 40 chars: {sample[-40:]!r}", flush=True)

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
            score.update({"elapsed_sec": elapsed, "temperature": 0.0,
                          "n_requested": n, "adapter": label})
            key = f"n{n}_T0.0_{label}"
            summary["results_by_n_temp_adapter"][key] = score
            print(
                f"[kme] {key:50s}  acc={score['accuracy']:.4f}  trunc={score['trunc_rate']:.3f}  "
                f"boxed={score['has_boxed_rate']:.3f}  tc={score['has_think_close_rate']:.3f}  elapsed={elapsed:.1f}s",
                flush=True,
            )

        # Optional T=1 diagnostic
        if args.include_t1_n and args.include_t1_n > 0 and n <= args.include_t1_n:
            for label, adapter_path, lora_id in adapters:
                lora_req = LoRARequest(label, lora_id, str(adapter_path))
                sampling = SamplingParams(temperature=1.0, top_p=1.0,
                                          max_tokens=args.max_tokens, seed=0)
                t0 = time.time()
                outs = llm.generate(prompts, sampling_params=sampling, lora_request=lora_req)
                elapsed = time.time() - t0
                texts = [o.outputs[0].text for o in outs]
                score = _score(rows, texts)
                score.update({"elapsed_sec": elapsed, "temperature": 1.0,
                              "n_requested": n, "adapter": label, "seed": 0})
                key = f"n{n}_T1.0_{label}"
                summary["results_by_n_temp_adapter"][key] = score
                print(
                    f"[kme] {key:50s}  acc={score['accuracy']:.4f}  trunc={score['trunc_rate']:.3f}  "
                    f"boxed={score['has_boxed_rate']:.3f}  tc={score['has_think_close_rate']:.3f}  elapsed={elapsed:.1f}s",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"[kme] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
