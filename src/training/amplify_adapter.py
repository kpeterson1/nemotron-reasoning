"""Make a copy of an adapter with lora_B scaled by a constant — diagnostic only.

If the amplified adapter changes vLLM output and the original doesn't, the wiring
is live and the original is just under-trained. If even a 50× lora_B amplification
yields identical output to base, the LoRA path is not being applied at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=50.0)
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    tensors = {}
    with safe_open(args.src / "adapter_model.safetensors", framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k.endswith("lora_B.weight"):
                t = t * args.scale
            tensors[k] = t
    save_file(tensors, str(args.dst / "adapter_model.safetensors"))
    cfg = json.loads((args.src / "adapter_config.json").read_text())
    (args.dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    for fname in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
        p = args.src / fname
        if p.is_file():
            (args.dst / fname).write_bytes(p.read_bytes())
    print(f"[amplify] scaled lora_B by {args.scale}× -> {args.dst}")


if __name__ == "__main__":
    main()
