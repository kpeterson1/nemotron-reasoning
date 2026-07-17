"""Build train_formatted_v9.jsonl.

v9 differs from v6 in two ways:
  1. bit_manipulation traces use bit_manip_trace_v4 (compact, eq-shaped
     scaffold) instead of bit_manip_trace_v3.
  2. Per-category counts are NOT sqrt-rebalanced. Instead:
       - bit_manipulation:        include ALL verified-correct traces
       - equation_transformation: include ALL verified-correct traces
       - text_encryption + 4 ceiling categories: sample ~v5 counts (~688 each)

This tests whether the v4 trace reformat eliminates the cross-task
interference that crashed text_encryption in v6 (which had v3-format
bit_manip traces).

Hypothesis the mix is designed to test: same training data category set
as v6, but with bm traces matching eq's structural skeleton — should
retain v5's text_encryption ceiling while still teaching bit_manipulation.

v12 wiring (2026-06-11, two orthogonal levers; built as one dataset):
  - text_encryption: now uses text_encryption_trace_v5 (character-by-character
    decode) — targets the FREE-RUN map-application failure (Q9: model writes a
    correct map then ignores it; emit==mechanical 0/47 misses). Wired here
    (the textenc_trace import below).
  - bit_manipulation: Spark 1 owns the shortened bit_manip v5 lever (targets
    the assembled_map-construction regression the logprob probe localized).
    Swap the `bm_trace` import below to that generator before the v12 build.
The probe is structurally blind to the text_enc free-run failure (teacher
forcing pins the final-decode region to gold), so it neither confirms nor
refutes the char-by-char premise; the free-run decomposition is the evidence.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from .bit_manip_solver_v3 import parse_examples as bm_parse, solve as bm_solve
from .bit_manip_trace_v5_cut1 import generate_trace as bm_trace  # v12: shortened v5 (Cut 1)
from .equation_transformation_trace_v2 import generate_trace as eq_trace
from .format_training import format_training_example
from .gravitational_constant_trace import generate_trace as gravity_trace
from .numeral_conversion_trace import generate_trace as numeral_trace
from .text_encryption_trace_v5 import generate_trace as textenc_trace  # v12: char-by-char decode (Q9 map-application fix)
from .unit_conversion_trace import generate_trace as unit_trace


_TRACE_FN = {
    "numeral_conversion":     numeral_trace,
    "unit_conversion":        unit_trace,
    "gravitational_constant": gravity_trace,
    "text_encryption":        textenc_trace,
    "equation_transformation": eq_trace,
}


def _bit_manip_trace(prompt: str):
    pairs, q = bm_parse(prompt)
    if not pairs or q is None:
        return None
    rule = bm_solve(pairs)
    if rule is None:
        return None
    return bm_trace(rule, pairs, q)


_TRACE_FN["bit_manipulation"] = _bit_manip_trace


# v5's per-task counts in train_formatted_v5.jsonl:
#   numeral_conversion 688, text_encryption 688, unit_conversion 692,
#   gravitational_constant 693, equation_transformation 191, bit_manipulation 0
# Target for ceiling + text_encryption in v9: match those numbers (capped by
# the verified-correct pool size).
_V5_TARGET_COUNT = {
    "numeral_conversion":     688,
    "text_encryption":        688,
    "unit_conversion":        692,
    "gravitational_constant": 693,
}
# bit_manipulation and equation_transformation: include ALL verified-correct.
_INCLUDE_ALL = {"bit_manipulation", "equation_transformation"}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    args = parser.parse_args()

    print(f"[v9] gathering rows from {args.train_split}", flush=True)
    by_task = _gather(args.train_split)
    for t, rs in by_task.items():
        print(f"  {t}: {len(rs)} rows", flush=True)

    print("[v9] tracing + verifying per task", flush=True)
    verified_by_task: dict[str, list[dict]] = {}
    for task, rs in by_task.items():
        if task not in _TRACE_FN:
            print(f"  {task}: no trace generator — skip", flush=True)
            continue
        kept = _trace_and_verify(rs, task)
        verified_by_task[task] = kept
        print(f"  {task}: {len(kept)}/{len(rs)} verified-correct", flush=True)

    rng = random.Random(args.seed)
    sampled: list[dict] = []
    print("[v9] sampling per-task", flush=True)
    for task, kept in verified_by_task.items():
        if task in _INCLUDE_ALL:
            chosen = list(kept)
            why = "ALL"
        else:
            target = _V5_TARGET_COUNT.get(task, len(kept))
            target = min(target, len(kept))
            pool = list(kept)
            rng.shuffle(pool)
            chosen = pool[:target]
            why = f"sampled {target}"
        sampled.extend(chosen)
        print(f"  {task}: {len(chosen)}  ({why})", flush=True)

    rng.shuffle(sampled)
    counts = collections.Counter(r["task_type"] for r in sampled)
    print(f"[v9] final mix: {len(sampled)} total", flush=True)
    for t, c in counts.most_common():
        print(f"  {t}: {c}", flush=True)

    print(f"[v9] loading tokenizer {args.tokenizer}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print(f"[v9] formatting and writing → {args.out}", flush=True)
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
    print(f"[v9] wrote {n_written} records", flush=True)


if __name__ == "__main__":
    main()
