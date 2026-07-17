"""Compare two eval run JSON reports side by side."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def compare(run1_path: Path, run2_path: Path) -> None:
    r1 = _load(run1_path)
    r2 = _load(run2_path)

    print(f"{'metric':<28} {run1_path.name:>20} {run2_path.name:>20} {'delta':>10}")
    print("-" * 82)

    a1 = r1.get("accuracy", 0.0)
    a2 = r2.get("accuracy", 0.0)
    print(f"{'overall_accuracy':<28} {a1:>20.4f} {a2:>20.4f} {a2 - a1:>+10.4f}")

    types = sorted(set(r1.get("per_task_type", {})) | set(r2.get("per_task_type", {})))
    for t in types:
        v1 = r1.get("per_task_type", {}).get(t, 0.0)
        v2 = r2.get("per_task_type", {}).get(t, 0.0)
        print(f"{t:<28} {v1:>20.4f} {v2:>20.4f} {v2 - v1:>+10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1", type=Path, required=True)
    parser.add_argument("--run2", type=Path, required=True)
    args = parser.parse_args()
    compare(args.run1, args.run2)


if __name__ == "__main__":
    main()
