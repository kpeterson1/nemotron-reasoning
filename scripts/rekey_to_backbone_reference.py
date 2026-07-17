"""Backbone-rekey + reference-match: map a naive (model.model-prefixed) vLLM
adapter to the Huikang/Kaggle reference structure that scored 0.58.

Reverse-engineered by diffing the 0.58 ground-truth artifact
(runs/train/lora_v9_warm_start_300step_vllm_r32padded_backbone) against its
padded predecessor (..._vllm_r32padded). The full diff is EXACTLY two changes
(tensor values/shapes/dtypes are byte-identical — verified max-abs-diff 0.0):

  1. Key prefix rename: `base_model.model.model.` -> `base_model.model.backbone.`
  2. adapter_config `target_modules`: regex string -> the reference list
     ["k_proj","o_proj","in_proj","q_proj","up_proj","v_proj","down_proj",
      "out_proj"]  (Huikang format; note: no gate_proj — NemotronH MoE is
      non-gated, so gate_proj never matched any module).

Apply AFTER pad_lora_to_uniform_rank.py (experts must already be r32 uniform).
Full Kaggle-packaging path:
    convert_peft_to_vllm_moe.py (HEAD/buggy) -> _vllm
    pad_lora_to_uniform_rank.py --target-rank 32 -> _vllm_r32padded
    rekey_to_backbone_reference.py -> _vllm_r32padded_backbone   (this script)

Usage:
    python -m scripts.rekey_to_backbone_reference \
        --src runs/train/lora_v13_ws300_vllm_r32padded \
        --dst runs/train/lora_v13_ws300_vllm_r32padded_backbone
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

# Single source of truth: the converter owns the Kaggle reference target_modules list.
from src.training.convert_peft_to_vllm_moe import REFERENCE_TARGET_MODULES

OLD_PREFIX = "base_model.model.model."
NEW_PREFIX = "base_model.model.backbone."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="padded vLLM adapter dir (model.model prefix, r32 experts)")
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)

    # 1) rename tensor keys (values unchanged)
    renamed, untouched = 0, 0
    out: dict = {}
    with safe_open(str(args.src / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith(OLD_PREFIX):
                nk = NEW_PREFIX + k[len(OLD_PREFIX):]
                renamed += 1
            else:
                nk = k
                untouched += 1
            out[nk] = f.get_tensor(k)
    save_file(out, str(args.dst / "adapter_model.safetensors"))
    print(f"[rekey] renamed {renamed} keys ({OLD_PREFIX} -> {NEW_PREFIX}); "
          f"{untouched} untouched; {len(out)} total", flush=True)

    # 2) adapter_config: target_modules regex -> reference list
    cfg = json.loads((args.src / "adapter_config.json").read_text())
    cfg["target_modules"] = list(REFERENCE_TARGET_MODULES)
    (args.dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[rekey] adapter_config target_modules -> reference list "
          f"({len(REFERENCE_TARGET_MODULES)} modules)", flush=True)

    # 3) carry the chat template (and tokenizer if present) for submission
    for fn in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
        src = args.src / fn
        if src.is_file():
            shutil.copy2(src, args.dst / fn)
            print(f"[rekey] copied {fn}", flush=True)
    print(f"[rekey] wrote {args.dst}", flush=True)


if __name__ == "__main__":
    main()
