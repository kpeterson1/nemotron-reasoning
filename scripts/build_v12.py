"""Build train_formatted_v12.jsonl = v9 with TWO categories regenerated:
  - bit_manipulation -> bit_manip_trace_v5_cut1 (shortened v5, Cut 1)
  - text_encryption   -> text_encryption_trace_v5 (Spark 2 char-by-char decode)
All other categories (numeral/unit/gravity/equation_transformation) are copied
BYTE-IDENTICAL from v9.

Record-replacement (NOT build_v9_mix.py): the shared builder rng-samples
text_encryption and then does a final rng.shuffle, so changing the text_enc
trace perturbs the sampled selections AND the record order of every category —
which would break the "non-changed categories byte-identical to v9" guardrail.
Record-replacement guarantees it and keeps the problem set fixed (same v9
problems, only the two trace formats change) — the cleanest two-variable run.

A regenerated trace is kept only if its computed answer == the v9 record's
answer (same gate as v9 / v11).

Usage:
    python -m scripts.build_v12 \
        --v9 datasets/processed/train_formatted_v9.jsonl \
        --out datasets/processed/train_formatted_v12.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.data.bit_manip_solver_v3 import parse_examples as bm_parse, solve as bm_solve
from src.data.bit_manip_trace_v5_cut1 import generate_trace as bm_trace
from src.data.text_encryption_trace_v5 import generate_trace as te_trace
from src.data.format_training import format_training_example

TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
REGEN = {"bit_manipulation", "text_encryption"}


def _bm(prompt):
    pairs, q = bm_parse(prompt)
    if not pairs or q is None:
        return None
    rule = bm_solve(pairs)
    if rule is None:
        return None
    return bm_trace(rule, pairs, q)


def _regen(task, prompt):
    if task == "bit_manipulation":
        return _bm(prompt)
    return te_trace(prompt)  # text_encryption -> (trace, answer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v9", type=Path, default=Path("datasets/processed/train_formatted_v9.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("datasets/processed/train_formatted_v12.jsonl"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

    raw_lines = args.v9.read_text().splitlines()
    print(f"[v12] read {len(raw_lines)} v9 records", flush=True)

    out_lines: list[str] = []
    kept = collections.Counter()
    dropped = collections.Counter()
    unchanged_hash = hashlib.sha256()
    counts = collections.Counter()

    for line in raw_lines:
        rec = json.loads(line)
        task = rec["task_type"]
        if task not in REGEN:
            out_lines.append(line)                      # verbatim -> byte-identical
            unchanged_hash.update((line + "\n").encode())
            counts[task] += 1
            continue
        res = _regen(task, rec["prompt"])
        if res is None:
            dropped[task] += 1
            continue
        trace, answer = res
        if str(answer).strip() != str(rec["answer"]).strip():
            dropped[task] += 1
            continue
        fmt = format_training_example(
            prompt=rec["prompt"], answer=str(rec["answer"]),
            reasoning_trace=trace, tokenizer=tok)
        tot = len(tok.encode(fmt["text"]))
        p_tok = len(tok.encode(fmt["prompt_text"]))
        out_lines.append(json.dumps({
            "id": rec["id"], "task_type": task, "prompt": rec["prompt"],
            "answer": rec["answer"], "reasoning_trace": trace,
            "source": rec.get("source", "solver"),
            "text": fmt["text"], "prompt_text": fmt["prompt_text"],
            "completion_text": fmt["completion_text"],
            "n_tokens_total": tot, "n_tokens_prompt": p_tok,
            "n_tokens_completion": tot - p_tok,
        }))
        kept[task] += 1
        counts[task] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n")

    print(f"[v12] wrote {len(out_lines)} records -> {args.out}")
    for t in REGEN:
        print(f"[v12] {t}: kept {kept[t]}, dropped {dropped[t]}")
    print("[v12] per-category counts:")
    for t, c in counts.most_common():
        print(f"  {c:5d}  {t}")
    print(f"[v12] non-changed lines sha256 {unchanged_hash.hexdigest()[:16]} "
          f"(verify vs v9 separately)")
    for t in REGEN:
        toks = sorted(json.loads(l)["n_tokens_completion"] for l in out_lines
                      if json.loads(l)["task_type"] == t)
        print(f"[v12] {t} completion tokens: min={toks[0]} med={toks[len(toks)//2]} "
              f"max={toks[-1]} over2000={sum(1 for x in toks if x>2000)}")


if __name__ == "__main__":
    main()
