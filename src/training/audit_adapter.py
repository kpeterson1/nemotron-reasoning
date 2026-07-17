"""Inspect a saved PEFT LoRA adapter for routed-expert coverage and submission size.

Reports:
- adapter_config.json contents (target_modules + target_parameters fields)
- total safetensors entries
- lora_A / lora_B count split
- expert up_proj / down_proj entry count (3D-parameter ParamWrapper outputs)
- key prefix format (first few keys)
- tensor shapes (one sample per "kind")
- unzipped vs zipped (ZIP_DEFLATED) adapter size
- projected submission.zip size + headroom under the 1.5 GB Kaggle limit
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from safetensors import safe_open


_LIMIT_GB = 1.5
_KAGGLE_LIMIT_BYTES = int(_LIMIT_GB * 1024**3)


def audit(adapter_dir: Path) -> None:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    print(f"=== adapter_config ===")
    print(f"  r                  = {cfg.get('r')}")
    print(f"  lora_alpha         = {cfg.get('lora_alpha')}")
    print(f"  target_modules     = {cfg.get('target_modules')}")
    print(f"  target_parameters  = {cfg.get('target_parameters')}")
    print(f"  base_model         = {cfg.get('base_model_name_or_path')}")

    weights_path = adapter_dir / "adapter_model.safetensors"
    keys: list[str] = []
    shapes: dict[str, tuple] = {}
    dtypes: dict[str, str] = {}
    with safe_open(weights_path, framework="pt") as f:
        for k in f.keys():
            keys.append(k)
            t = f.get_tensor(k)
            shapes[k] = tuple(t.shape)
            dtypes[k] = str(t.dtype).replace("torch.", "")

    print(f"\n=== safetensors summary ===")
    print(f"  total entries: {len(keys)}")

    lora_A = [k for k in keys if "lora_A" in k]
    lora_B = [k for k in keys if "lora_B" in k]
    print(f"  lora_A entries: {len(lora_A)}")
    print(f"  lora_B entries: {len(lora_B)}")
    print(f"  other entries: {len(keys) - len(lora_A) - len(lora_B)}")

    expert_up = [k for k in keys if "mixer.experts.up_proj" in k]
    expert_down = [k for k in keys if "mixer.experts.down_proj" in k]
    print(f"\n=== routed-expert coverage ===")
    print(f"  keys touching mixer.experts.up_proj   : {len(expert_up)}")
    print(f"  keys touching mixer.experts.down_proj : {len(expert_down)}")
    expected_layers = 23  # 23 MoE layers in NemotronH-30B-A3B
    expected_per_param = 2  # lora_A + lora_B per ParamWrapper
    expected_each = expected_layers * expected_per_param
    expert_total = len(expert_up) + len(expert_down)
    print(f"  expected per param (lora_A + lora_B per MoE layer): {expected_each}")
    print(f"  observed up_proj / down_proj entries match expected: "
          f"{len(expert_up) == expected_each} / {len(expert_down) == expected_each}")
    print(f"  total expert LoRA entries: {expert_total} "
          f"(expected {expected_each * 2})")

    # Sample shapes per "kind"
    print(f"\n=== key prefix format (first 8) ===")
    for k in keys[:8]:
        print(f"  {k}  shape={shapes[k]}  dtype={dtypes[k]}")

    # Per-kind sample
    print(f"\n=== sample by kind ===")
    seen_kinds: set[str] = set()
    for k in keys:
        # build a 'kind' label by stripping layer number
        parts = k.split(".")
        kind_parts = [
            "<L>" if (p.isdigit() and parts[i - 1] == "layers") else p
            for i, p in enumerate(parts)
        ]
        kind = ".".join(kind_parts)
        if kind in seen_kinds:
            continue
        seen_kinds.add(kind)
        print(f"  {kind}  -> shape {shapes[k]}  ({dtypes[k]})")
        if len(seen_kinds) >= 18:
            break

    # Sum bytes from shapes/dtypes
    def _bytes_per_dtype(d: str) -> int:
        return {
            "bfloat16": 2, "float16": 2, "float32": 4,
            "int64": 8, "int32": 4, "int8": 1, "uint8": 1,
        }.get(d, 4)
    total_param_bytes = 0
    total_numel = 0
    for k in keys:
        n = 1
        for s in shapes[k]:
            n *= int(s)
        total_numel += n
        total_param_bytes += n * _bytes_per_dtype(dtypes[k])
    print(f"\n=== size ===")
    print(f"  total LoRA params (numel sum):  {total_numel:,}")
    print(f"  total param bytes (from shapes): {total_param_bytes / 1024**2:,.2f} MiB")
    file_size = weights_path.stat().st_size
    print(f"  safetensors file size on disk:  {file_size / 1024**2:,.2f} MiB")

    # Estimate zipped submission size by zipping the adapter contents in-memory.
    buf = io.BytesIO()
    submit_files = ("adapter_config.json", "adapter_model.safetensors", "chat_template.jinja")
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in submit_files:
            p = adapter_dir / name
            if p.is_file():
                z.write(p, name)
    zipped = buf.tell()
    print(f"  projected submission.zip size:  {zipped / 1024**2:,.2f} MiB "
          f"({zipped / 1024**3:.3f} GiB)")
    headroom = _KAGGLE_LIMIT_BYTES - zipped
    print(f"  under Kaggle {_LIMIT_GB:.1f} GB limit: "
          f"{'YES' if zipped < _KAGGLE_LIMIT_BYTES else 'NO'}  "
          f"(headroom {headroom / 1024**3:+.3f} GiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_dir", type=Path)
    args = parser.parse_args()
    audit(args.adapter_dir)


if __name__ == "__main__":
    main()
