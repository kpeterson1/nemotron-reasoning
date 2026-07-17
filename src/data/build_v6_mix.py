"""Build train_formatted_v6.jsonl from train_remaining using the upgraded
solvers (bit_manip v3, equation_transformation v2).

Strategy:
  1. For each task type, run that task's solver-trace pipeline on the
     train_remaining rows and keep only traces whose predicted answer
     matches the GT answer (`correct=True`).
  2. sqrt-rebalance across task types so the per-task contribution scales
     with sqrt(available_traces) — this matches the v4/v5 strategy and
     prevents easy tasks from dominating.
  3. Chat-template format the surviving records with `format_training.py`.
  4. Write to `datasets/processed/train_formatted_v6.jsonl`.

Usage:
  python -m src.data.build_v6_mix --target-total 3000 \\
    --train-split datasets/splits/train_remaining.jsonl \\
    --out datasets/processed/train_formatted_v6.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

from .bit_manip_solver_v3 import parse_examples as bm_parse, solve as bm_solve
from .bit_manip_trace_v3 import generate_trace as bm_trace
from .equation_transformation_trace_v2 import generate_trace as eq_trace
from .format_training import format_training_example
from .gravitational_constant_trace import generate_trace as gravity_trace
from .numeral_conversion_trace import generate_trace as numeral_trace
from .text_encryption_trace import generate_trace as textenc_trace
from .unit_conversion_trace import generate_trace as unit_trace


_TRACE_FN = {
    "numeral_conversion":     lambda p: numeral_trace(p),
    "unit_conversion":        lambda p: unit_trace(p),
    "gravitational_constant": lambda p: gravity_trace(p),
    "text_encryption":        lambda p: textenc_trace(p),
    "equation_transformation": lambda p: eq_trace(p),
}


def _bit_manip_trace(prompt: str) -> tuple[str, str] | None:
    pairs, q = bm_parse(prompt)
    if not pairs or q is None:
        return None
    rule = bm_solve(pairs)
    if rule is None:
        return None
    return bm_trace(rule, pairs, q)


_TRACE_FN["bit_manipulation"] = _bit_manip_trace


def _gather(train_split: Path) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    with train_split.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_task[row["task_type"]].append(row)
    return by_task


def _trace_and_verify(rows: list[dict], task: str) -> list[dict]:
    """Apply task's trace generator + filter to correct ones."""
    fn = _TRACE_FN[task]
    kept: list[dict] = []
    for row in rows:
        try:
            res = fn(row["prompt"])
        except Exception:
            continue
        if res is None:
            continue
        trace, answer = res
        if answer != str(row["answer"]).strip():
            continue
        kept.append({
            "id": row["id"],
            "task_type": task,
            "prompt": row["prompt"],
            "answer": row["answer"],
            "reasoning_trace": trace,
            "source": "solver",
        })
    return kept


def _sqrt_rebalance(by_task: dict[str, list[dict]], target_total: int, seed: int) -> list[dict]:
    """Allocate per-task counts proportional to sqrt(available), normalised
    so the total equals `target_total` (capped per-task by the available pool).
    Sample without replacement using a fixed RNG seed."""
    rng = random.Random(seed)
    weights = {t: math.sqrt(len(rs)) for t, rs in by_task.items()}
    z = sum(weights.values())
    target_counts: dict[str, int] = {}
    for task, w in weights.items():
        target_counts[task] = min(len(by_task[task]), int(round(target_total * w / z)))
    out: list[dict] = []
    for task, count in target_counts.items():
        pool = list(by_task[task])
        rng.shuffle(pool)
        out.extend(pool[:count])
    rng.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-total", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument(
        "--exclude-tasks",
        nargs="*",
        default=[],
        help="Task types to drop from the mix (e.g. --exclude-tasks bit_manipulation).",
    )
    args = parser.parse_args()
    excluded = set(args.exclude_tasks)
    if excluded:
        print(f"[v6] excluding task types: {sorted(excluded)}", flush=True)

    print(f"[v6] gathering rows from {args.train_split}", flush=True)
    by_task = _gather(args.train_split)
    for t, rs in by_task.items():
        print(f"  {t}: {len(rs)} rows", flush=True)

    print("[v6] tracing + verifying per task", flush=True)
    verified_by_task: dict[str, list[dict]] = {}
    for task, rs in by_task.items():
        if task in excluded:
            print(f"  {task}: EXCLUDED — skip", flush=True)
            continue
        if task not in _TRACE_FN:
            print(f"  {task}: no trace generator — skip", flush=True)
            continue
        kept = _trace_and_verify(rs, task)
        verified_by_task[task] = kept
        print(f"  {task}: {len(kept)}/{len(rs)} verified-correct", flush=True)

    print(f"[v6] sqrt-rebalance to target_total={args.target_total}", flush=True)
    sampled = _sqrt_rebalance(verified_by_task, args.target_total, args.seed)
    counts = collections.Counter(r["task_type"] for r in sampled)
    print(f"[v6] sampled {len(sampled)} total", flush=True)
    for t, c in counts.most_common():
        print(f"  {t}: {c}", flush=True)

    print(f"[v6] loading tokenizer {args.tokenizer}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print(f"[v6] formatting and writing → {args.out}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.out.open("w") as f:
        for r in sampled:
            rec = format_training_example(
                prompt=r["prompt"],
                answer=str(r["answer"]),
                reasoning_trace=r["reasoning_trace"],
                tokenizer=tok,
            )
            # Compute token counts for the trainer's per-token-loss boundary.
            tot = len(tok.encode(rec["text"]))
            p_tok = len(tok.encode(rec["prompt_text"]))
            c_tok = tot - p_tok
            out_rec = {
                "id": r["id"],
                "task_type": r["task_type"],
                "prompt": r["prompt"],
                "answer": r["answer"],
                "reasoning_trace": r["reasoning_trace"],
                "source": r["source"],
                "text": rec["text"],
                "prompt_text": rec["prompt_text"],
                "completion_text": rec["completion_text"],
                "n_tokens_total": tot,
                "n_tokens_prompt": p_tok,
                "n_tokens_completion": c_tok,
            }
            f.write(json.dumps(out_rec) + "\n")
            n_written += 1
    print(f"[v6] wrote {n_written} records", flush=True)


if __name__ == "__main__":
    main()
