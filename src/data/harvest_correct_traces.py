"""Harvest base-model brief-arm traces that are correct under the Kaggle
verifier. Designed to run for ~30 hours; checkpoints by appending one JSONL
record per prompt as it finishes, and skips any IDs already present on resume.

For each row of `--split-file`, the model is queried with the harness suffix +
brief-arm system message + brief-arm user instruction (the same setup that
got 48% on n=50). After each chunk of prompts we run extract_boxed_answer +
verify; the full record (including raw output, prediction, correctness) is
appended to `--out`. Filter by `correct=True` downstream to get the SFT corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJ = Path(__file__).resolve().parents[2]  # repo root (src/data/<file> -> repo)
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.evaluation.extract_answer import answers_match, extract_boxed_answer
from src.inference.generate import generate

BRIEF_SYSTEM = (
    "Solve the puzzle concisely. Show your reasoning in under 1000 tokens, "
    "then give your final answer."
)
BRIEF_USER_PREFIX = (
    "Be concise. Do not enumerate all examples — identify the pattern "
    "quickly and apply it."
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _seen_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    seen: set[str] = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "id" in rec:
                    seen.add(rec["id"])
            except json.JSONDecodeError:
                # corrupt last line from an interrupted write; ignore
                pass
    return seen


def _build_prompt(problem: str) -> str:
    return f"{BRIEF_USER_PREFIX}\n\n{problem}"


def harvest(
    split_file: Path,
    out_path: Path,
    chunk_size: int = 500,
    max_num_seqs: int = 64,
    max_tokens: int = 3584,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = _load_jsonl(split_file)
    if limit:
        rows = rows[:limit]

    seen = _seen_ids(out_path)
    todo = [r for r in rows if r["id"] not in seen]
    print(
        f"[harvest] {len(rows)} total rows, {len(seen)} already done, "
        f"{len(todo)} remaining",
        flush=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_correct_total = 0
    t_start = time.time()
    for chunk_start in range(0, len(todo), chunk_size):
        chunk = todo[chunk_start : chunk_start + chunk_size]
        prompts = [_build_prompt(r["prompt"]) for r in chunk]

        t0 = time.time()
        raws = generate(
            prompts,
            system_message=BRIEF_SYSTEM,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
        )
        elapsed = time.time() - t0

        n_correct_chunk = 0
        with out_path.open("a") as f:
            for r, raw in zip(chunk, raws):
                pred = extract_boxed_answer(raw)
                ok = answers_match(pred, r["answer"])
                if ok:
                    n_correct_chunk += 1
                rec = {
                    "id": r["id"],
                    "task_type": r["task_type"],
                    "prompt": r["prompt"],
                    "answer": r["answer"],
                    "raw_output": raw,
                    "predicted": pred,
                    "correct": ok,
                    "raw_len": len(raw),
                }
                f.write(json.dumps(rec) + "\n")

        n_correct_total += n_correct_chunk
        done_so_far = chunk_start + len(chunk)
        rate = (chunk_start + len(chunk)) / max(1, time.time() - t_start)
        eta = (len(todo) - done_so_far) / max(rate, 1e-9)
        print(
            f"[harvest]  chunk {chunk_start:>5}-{chunk_start+len(chunk):<5}  "
            f"chunk_correct={n_correct_chunk:>3}/{len(chunk)}  "
            f"running_total={n_correct_total + len(seen.intersection({r['id'] for r in rows[: chunk_start + len(chunk)]}))} "
            f"elapsed={time.time()-t_start:.0f}s  rate={rate*60:.1f}/min  ETA={eta/3600:.1f}h",
            flush=True,
        )

    return {
        "total_processed": len(todo),
        "correct_in_this_run": n_correct_total,
        "elapsed_sec": time.time() - t_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=3584)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = harvest(
        split_file=args.split_file,
        out_path=args.out,
        chunk_size=args.chunk_size,
        max_num_seqs=args.max_num_seqs,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit=args.limit,
    )
    print(f"[harvest] DONE  {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
