"""Re-score historical eval outputs with the fixed brace-walking extractor.

Walks runs/eval/*-raw-*.json files (each contains per-example `raw` text,
`answer`, and the old-extractor `predicted`/`correct` fields) and scores
each with two extractor implementations:

  - OLD: the pre-fix `[^}]*` regex behavior (snapshot inline below).
  - NEW: the current src/evaluation/extract_answer.py (brace-walking).

For each file: prints old_score, new_score, delta, and the first 5
disagreement examples (where old/new differ on either extracted value
or correctness).

Usage: python scripts/rescore_with_fixed_extractor.py [json_path ...]
If no args, scans runs/eval/*-raw-*.json.

NOT committed — depends on raw eval outputs that live in runs/eval/
(gitignored). Re-create from this file if needed.
"""
from __future__ import annotations

import glob
import json
import math
import re
import sys
from pathlib import Path

# Make src.* importable when run from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.extract_answer import extract_final_answer as new_extract
from src.evaluation.extract_answer import verify as new_verify


# === OLD extractor: pre-fix `[^}]*` regex behavior, snapshot inline =========
_OLD_BOXED_RE = re.compile(r"\\boxed\{([^}]*)(?:\}|$)")
_OLD_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")
_OLD_FINAL_ANSWER_PATTERNS = [
    re.compile(r"The final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer is:\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"Final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"final answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
]


def old_extract(text):
    if text is None:
        return "NOT_FOUND"
    matches = re.findall(_OLD_BOXED_RE, text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    for pattern in _OLD_FINAL_ANSWER_PATTERNS:
        found = pattern.findall(text)
        if found:
            return found[-1].strip()
    nums = _OLD_NUMERIC_RE.findall(text)
    if nums:
        return nums[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


_BINARY_RE = re.compile(r"[01]+")


def old_verify(stored, predicted):
    """verify() is unchanged by the fix — both versions agree. Keeping a copy
    here so this script is self-contained for the old-vs-new comparison."""
    if stored is None or predicted is None:
        return False
    stored = str(stored).strip()
    predicted = str(predicted).strip()
    if _BINARY_RE.fullmatch(stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        return predicted.lower() == stored.lower()


# === Re-scoring ==============================================================


def rescore_file(path: Path) -> dict | None:
    with open(path) as f:
        data = json.load(f)
    preds = data.get("predictions")
    if not preds or not isinstance(preds, list):
        return None
    if not isinstance(preds[0], dict) or "raw" not in preds[0]:
        return None

    n = 0
    old_correct = 0
    new_correct = 0
    disagreements = []
    per_task_old: dict[str, list[int]] = {}
    per_task_new: dict[str, list[int]] = {}

    for p in preds:
        raw = p.get("raw")
        gt = p.get("answer")
        if raw is None or gt is None:
            continue
        task = p.get("task_type", "?")

        old_pred = old_extract(raw)
        new_pred = new_extract(raw)
        old_match = old_verify(gt, old_pred)
        new_match = new_verify(gt, new_pred)

        n += 1
        old_correct += int(old_match)
        new_correct += int(new_match)
        per_task_old.setdefault(task, [0, 0])
        per_task_new.setdefault(task, [0, 0])
        per_task_old[task][0] += int(old_match)
        per_task_old[task][1] += 1
        per_task_new[task][0] += int(new_match)
        per_task_new[task][1] += 1

        if old_pred != new_pred or old_match != new_match:
            disagreements.append(
                {
                    "id": p.get("id"),
                    "task_type": task,
                    "raw_tail": raw[-300:],
                    "ground_truth": gt,
                    "old_extracted": old_pred,
                    "new_extracted": new_pred,
                    "old_correct": old_match,
                    "new_correct": new_match,
                }
            )

    if n == 0:
        return None
    return {
        "path": str(path),
        "n": n,
        "old_score": old_correct / n,
        "new_score": new_correct / n,
        "delta": (new_correct - old_correct) / n,
        "stored_accuracy": data.get("accuracy"),
        "split": data.get("split"),
        "adapter_dir": data.get("adapter_dir"),
        "disagreements": disagreements,
        "per_task_old": {k: (c / m if m else None, m) for k, (c, m) in per_task_old.items()},
        "per_task_new": {k: (c / m if m else None, m) for k, (c, m) in per_task_new.items()},
    }


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        paths = [Path(p) for p in sorted(glob.glob("runs/eval/*-raw-*.json"))]

    results = []
    for p in paths:
        r = rescore_file(p)
        if r is not None:
            results.append(r)
            short = p.name
            stored = r["stored_accuracy"]
            stored_str = f"{stored:.3f}" if isinstance(stored, (int, float)) else "?"
            print(
                f"{short:80s} n={r['n']:>4d} "
                f"stored={stored_str:>6s} "
                f"old={r['old_score']:.3f} new={r['new_score']:.3f} "
                f"delta={r['delta']:+.4f} "
                f"disagree={len(r['disagreements'])}"
            )

    # Save full result with disagreement samples to a working file.
    out = Path("/tmp/rescore_summary.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results (with disagreement samples) written to {out}")


if __name__ == "__main__":
    main()
