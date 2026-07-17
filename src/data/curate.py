"""Curate training data — placeholder for filtering and deduping pipelines."""
from __future__ import annotations

import argparse
from pathlib import Path


def curate(input_path: Path, output_path: Path) -> None:
    """Pass-through stub. Real curation logic (verifier filtering, dedup, length
    thresholds) will live here."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(input_path.read_bytes())
    print(f"[curate] copied {input_path} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    curate(args.input, args.output)


if __name__ == "__main__":
    main()
