"""Carve evaluation/training splits from the raw competition CSV."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.data.classify_task import classify_task


def _row_to_dict(row: pd.Series, idx: int) -> dict[str, str]:
    prompt = str(row.get("prompt") or row.get("question") or row.get("input") or "")
    answer = str(row.get("answer") or row.get("output") or row.get("label") or "")
    rid = str(row.get("id", idx))
    return {
        "id": rid,
        "prompt": prompt,
        "answer": answer,
        "task_type": classify_task(prompt),
    }


def carve_splits(
    csv_path: Path,
    output_dir: Path,
    dev_size: int,
    seed: int,
    stratify: bool = True,
) -> None:
    df = pd.read_csv(csv_path)
    rows = [_row_to_dict(df.iloc[i], i) for i in range(len(df))]

    rng = random.Random(seed)

    if stratify:
        by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in rows:
            by_type[r["task_type"]].append(r)
        n_types = max(1, len(by_type))
        base = dev_size // n_types
        extra = dev_size - base * n_types
        dev: list[dict[str, str]] = []
        # Allocate `extra` additional examples to the first buckets in sorted order
        # for deterministic behavior across runs with the same seed.
        for i, tt in enumerate(sorted(by_type)):
            bucket = by_type[tt]
            rng.shuffle(bucket)
            take = base + (1 if i < extra else 0)
            dev.extend(bucket[:take])
        dev_ids = {r["id"] for r in dev}
        remaining = [r for r in rows if r["id"] not in dev_ids]
    else:
        rng.shuffle(rows)
        dev = rows[:dev_size]
        remaining = rows[dev_size:]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "dev_frozen.jsonl", dev)
    _write_jsonl(output_dir / "train_remaining.jsonl", remaining)

    hard200 = sorted(dev, key=lambda r: len(r["prompt"]), reverse=True)[:200]
    speed_bench = sorted(dev, key=lambda r: len(r["prompt"]))[:50]
    _write_jsonl(output_dir / "hard200.jsonl", hard200)
    _write_jsonl(output_dir / "speed_bench.jsonl", speed_bench)

    print(f"[split] dev={len(dev)} remaining={len(remaining)} hard200={len(hard200)} speed_bench={len(speed_bench)}")
    _print_distribution("dev_frozen", dev)
    _print_distribution("train_remaining", remaining)


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _print_distribution(name: str, rows: list[dict[str, str]]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["task_type"]] += 1
    print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/splits"))
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify", action="store_true", default=True)
    args = parser.parse_args()
    carve_splits(args.csv, args.output_dir, args.dev_size, args.seed, args.stratify)


if __name__ == "__main__":
    main()
