"""Evaluation runner: scores a split using the Kaggle harness conventions."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.extract_answer import answers_match, extract_boxed_answer


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _wrap_with_template(problem: str, template: str | None) -> str:
    """Apply a local prompt template (does NOT add the harness suffix —
    `generate()` handles that)."""
    if template:
        return template.format(problem=problem)
    return problem


def run_eval(
    split: str,
    config_path: Path,
    model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    prompt_family: str | None = None,
    adapter_dir: str | None = None,
    *,
    limit: int | None = None,
    max_tokens: int = 3584,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 16,
    save_raw: bool = True,
    system_message: str | None = None,
    user_instruction: str | None = None,
    arm_label: str | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    splits_dir = Path(config["splits_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(splits_dir / f"{split}.jsonl")
    if limit:
        rows = rows[:limit]

    template: str | None = None
    if prompt_family:
        prompt_path = Path("prompts") / prompt_family / "v1.yaml"
        if prompt_path.exists():
            template = yaml.safe_load(prompt_path.read_text()).get("template")

    prompts = [_wrap_with_template(r["prompt"], template) for r in rows]
    if user_instruction:
        prompts = [f"{user_instruction}\n\n{p}" for p in prompts]

    # Single batched generate call — vLLM handles continuous batching internally.
    from src.inference.generate import generate

    t0 = time.time()
    raws = generate(
        prompts,
        model=model,
        adapter_dir=adapter_dir,
        system_message=system_message,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        temperature=temperature,
        top_p=top_p,
    )
    elapsed = time.time() - t0

    correct = 0
    by_type_correct: dict[str, int] = defaultdict(int)
    by_type_total: dict[str, int] = defaultdict(int)
    by_type_truncated: dict[str, int] = defaultdict(int)
    truncated_total = 0
    predictions: list[dict[str, Any]] = []

    for row, raw in zip(rows, raws):
        pred = extract_boxed_answer(raw)
        is_correct = answers_match(pred, row["answer"])
        # "Truncated" = model never wrote \boxed{} in its output. Either the
        # trace ran past max_tokens or the model emitted a final answer outside
        # the boxed format (treated equivalently for budget-tracking purposes).
        is_truncated = "\\boxed{" not in raw
        task_type = row.get("task_type", "unknown")
        by_type_total[task_type] += 1
        if is_correct:
            correct += 1
            by_type_correct[task_type] += 1
        if is_truncated:
            truncated_total += 1
            by_type_truncated[task_type] += 1
        rec = {
            "id": row.get("id"),
            "task_type": task_type,
            "predicted": pred,
            "answer": row["answer"],
            "correct": is_correct,
            "truncated": is_truncated,
            "raw_len": len(raw),
        }
        if save_raw:
            rec["raw"] = raw
        predictions.append(rec)

    total = len(rows) or 1
    per_task = {
        t: by_type_correct[t] / by_type_total[t]
        for t in by_type_total
        if by_type_total[t] > 0
    }
    per_task_trunc = {
        t: by_type_truncated[t] / by_type_total[t]
        for t in by_type_total
        if by_type_total[t] > 0
    }
    report = {
        "split": split,
        "model": model,
        "adapter_dir": adapter_dir,
        "prompt_family": prompt_family,
        "arm_label": arm_label,
        "system_message": system_message,
        "user_instruction": user_instruction,
        "n": len(rows),
        "elapsed_sec": elapsed,
        "accuracy": correct / total,
        "truncation_rate": truncated_total / total,
        "per_task_type": per_task,
        "per_task_truncation": per_task_trunc,
        "per_task_total": dict(by_type_total),
        "predictions": predictions,
    }

    suffix = adapter_dir.replace("/", "_") if adapter_dir else "base"
    pf = prompt_family or "raw"
    arm = f"-{arm_label}" if arm_label else ""
    run_id = f"{split}-{pf}-{suffix}{arm}-{int(time.time())}"
    out = output_dir / f"{run_id}.json"
    out.write_text(json.dumps(report, indent=2))
    print(
        f"[run_eval] wrote {out}  n={len(rows)}  acc={report['accuracy']:.4f}  "
        f"trunc={report['truncation_rate']:.2f}  elapsed={elapsed:.1f}s"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--prompt-family", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=3584)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()
    run_eval(
        split=args.split,
        config_path=args.config,
        model=args.model,
        prompt_family=args.prompt_family,
        adapter_dir=args.adapter_dir,
        limit=args.limit,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
