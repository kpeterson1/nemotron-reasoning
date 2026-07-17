#!/usr/bin/env python3
"""
Create small, git-safe metadata files from a LoRA adapter_model.safetensors.

This does NOT copy or modify the large adapter weights. It only reads tensor
names, shapes, dtypes, and simple classification fields, then writes text files.

Example:
  python scripts/make_adapter_manifest.py \
    --adapter /kaggle/working/reference/adapter_model.safetensors \
    --outdir references/huikang/reference_adapter
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from safetensors import safe_open


LORA_SUFFIX_RE = re.compile(r"\.lora_([AB])\.weight$")


def parse_shape(raw_shape: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in raw_shape)


def shape_to_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(x) for x in shape)


def infer_lora_side(key: str) -> str:
    match = LORA_SUFFIX_RE.search(key)
    return match.group(1) if match else ""


def infer_lora_base(key: str) -> str:
    return LORA_SUFFIX_RE.sub("", key)


def infer_projection(key: str) -> str:
    """
    Pull out the projection/module name from common LoRA tensor keys.

    Example:
      ...self_attn.q_proj.lora_A.weight -> q_proj
      ...experts.12.up_proj.lora_B.weight -> up_proj
      ...mamba.in_proj.lora_A.weight -> in_proj
    """
    base = infer_lora_base(key)
    parts = base.split(".")
    return parts[-1] if parts else ""


def infer_layer_index(key: str) -> str:
    """
    Extract layer index from keys containing `.layers.<n>.`.
    """
    match = re.search(r"\.layers\.(\d+)\.", key)
    return match.group(1) if match else ""


def infer_expert_index(key: str) -> str:
    """
    Extract expert index from submission-style keys:
      ...experts.0.up_proj...
      ...experts.127.down_proj...

    Tinker fused keys like experts.w1 / experts.w2 do not have a numeric expert.
    """
    match = re.search(r"\.experts\.(\d+)\.", key)
    return match.group(1) if match else ""


def infer_expert_kind(key: str) -> str:
    """
    Classify whether a tensor looks like:
      - fused Tinker expert: experts.w1 / experts.w2 / experts.w3
      - per-expert submission tensor: experts.<number>.up_proj/down_proj
      - not expert
    """
    if ".experts.w1" in key:
        return "fused_w1"
    if ".experts.w2" in key:
        return "fused_w2"
    if ".experts.w3" in key:
        return "fused_w3"
    if re.search(r"\.experts\.\d+\.", key):
        return "per_expert"
    return ""


def infer_mamba_kind(key: str) -> str:
    """
    Classify Mamba-related projection names relevant to the Tinker -> Kaggle conversion.
    """
    if ".gate_proj." in key:
        return "gate_proj"
    if ".x_proj." in key:
        return "x_proj"
    if ".in_proj." in key:
        return "in_proj"
    if ".out_proj." in key:
        return "out_proj"
    return ""


def infer_prefix_kind(key: str) -> str:
    if key.startswith("base_model.model.backbone."):
        return "backbone"
    if key.startswith("base_model.model.model."):
        return "model"
    return ""


def infer_rank(key: str, shape: tuple[int, ...]) -> str:
    """
    Infer LoRA rank from lora_A/lora_B shape.

    Common LoRA:
      lora_A: rank x input_dim
      lora_B: output_dim x rank

    Some fused expert tensors may be 3D, e.g.:
      lora_A: experts x rank x input_dim
      lora_B: experts x output_dim x rank
    """
    side = infer_lora_side(key)
    if not side:
        return ""

    if len(shape) == 2:
        if side == "A":
            return str(shape[0])
        if side == "B":
            return str(shape[1])

    if len(shape) == 3:
        if side == "A":
            return str(shape[1])
        if side == "B":
            return str(shape[2])

    return ""


def read_tensors(adapter_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with safe_open(str(adapter_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor_slice = f.get_slice(key)
            shape = parse_shape(tensor_slice.get_shape())
            dtype = str(tensor_slice.get_dtype())

            rows.append(
                {
                    "key": key,
                    "shape": shape_to_text(shape),
                    "dtype": dtype,
                    "rank": infer_rank(key, shape),
                    "lora_side": infer_lora_side(key),
                    "lora_base": infer_lora_base(key),
                    "layer_index": infer_layer_index(key),
                    "projection": infer_projection(key),
                    "prefix_kind": infer_prefix_kind(key),
                    "expert_kind": infer_expert_kind(key),
                    "expert_index": infer_expert_index(key),
                    "mamba_kind": infer_mamba_kind(key),
                }
            )

    rows.sort(key=lambda r: r["key"])
    return rows


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "key",
        "shape",
        "dtype",
        "rank",
        "lora_side",
        "lora_base",
        "layer_index",
        "projection",
        "prefix_kind",
        "expert_kind",
        "expert_index",
        "mamba_kind",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_first_keys(path: Path, rows: list[dict[str, str]], limit: int) -> None:
    with path.open("w") as f:
        for row in rows[:limit]:
            f.write(
                f"{row['key']}\tshape={row['shape']}\tdtype={row['dtype']}\trank={row['rank']}\n"
            )


def write_summary(path: Path, rows: list[dict[str, str]], adapter_path: Path) -> None:
    def count_where(field: str, value: str) -> int:
        return sum(1 for r in rows if r[field] == value)

    projections: dict[str, int] = {}
    ranks: dict[str, int] = {}
    prefix_kinds: dict[str, int] = {}
    expert_kinds: dict[str, int] = {}
    mamba_kinds: dict[str, int] = {}

    for r in rows:
        for field, bucket in [
            ("projection", projections),
            ("rank", ranks),
            ("prefix_kind", prefix_kinds),
            ("expert_kind", expert_kinds),
            ("mamba_kind", mamba_kinds),
        ]:
            value = r[field] or "<blank>"
            bucket[value] = bucket.get(value, 0) + 1

    with path.open("w") as f:
        f.write("# Adapter manifest summary\n\n")
        f.write(f"Adapter path: `{adapter_path}`\n\n")
        f.write(f"Total tensors: {len(rows)}\n\n")

        f.write("## Prefix kind counts\n\n")
        for k, v in sorted(prefix_kinds.items()):
            f.write(f"- {k}: {v}\n")

        f.write("\n## Expert kind counts\n\n")
        for k, v in sorted(expert_kinds.items()):
            f.write(f"- {k}: {v}\n")

        f.write("\n## Mamba/projection conversion counts\n\n")
        for k, v in sorted(mamba_kinds.items()):
            f.write(f"- {k}: {v}\n")

        f.write("\n## Rank counts\n\n")
        for k, v in sorted(ranks.items()):
            f.write(f"- {k}: {v}\n")

        f.write("\n## Projection counts\n\n")
        for k, v in sorted(projections.items()):
            f.write(f"- {k}: {v}\n")

        f.write("\n## Quick red flags\n\n")
        f.write(f"- Tinker-style prefix `base_model.model.model`: {count_where('prefix_kind', 'model')}\n")
        f.write(f"- Kaggle-style prefix `base_model.model.backbone`: {count_where('prefix_kind', 'backbone')}\n")
        f.write(f"- Fused expert `experts.w1`: {count_where('expert_kind', 'fused_w1')}\n")
        f.write(f"- Fused expert `experts.w2`: {count_where('expert_kind', 'fused_w2')}\n")
        f.write(f"- Per-expert tensors: {count_where('expert_kind', 'per_expert')}\n")
        f.write(f"- `gate_proj` tensors: {count_where('mamba_kind', 'gate_proj')}\n")
        f.write(f"- `x_proj` tensors: {count_where('mamba_kind', 'x_proj')}\n")
        f.write(f"- `in_proj` tensors: {count_where('mamba_kind', 'in_proj')}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        required=True,
        type=Path,
        help="Path to adapter_model.safetensors",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Directory where text metadata files will be written",
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=200,
        help="Number of keys to write to safetensors_keys_first_N.txt",
    )
    args = parser.parse_args()

    if not args.adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = read_tensors(args.adapter)

    manifest_path = args.outdir / "tensor_manifest.tsv"
    first_keys_path = args.outdir / f"safetensors_keys_first_{args.first_n}.txt"
    summary_path = args.outdir / "manifest_summary.md"

    write_tsv(manifest_path, rows)
    write_first_keys(first_keys_path, rows, args.first_n)
    write_summary(summary_path, rows, args.adapter)

    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {first_keys_path}")
    print(f"Wrote: {summary_path}")
    print(f"Total tensors: {len(rows)}")


if __name__ == "__main__":
    main()
