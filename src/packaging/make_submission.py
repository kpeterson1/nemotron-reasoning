"""Package a LoRA adapter directory as submission.zip.

Only the files that vLLM needs to load the adapter at inference time are
included. Trainer checkpoints, optimizer state, run reports, and any
training artifacts under `checkpoint-*/` are skipped — these can make the
zip 4× larger than necessary without affecting inference.

Required: adapter_config.json, adapter_model.safetensors.
Optional but harmless if present: chat_template.jinja.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

_REQUIRED = ("adapter_config.json", "adapter_model.safetensors")
_OPTIONAL = ("chat_template.jinja",)


def make_submission(
    adapter_dir: Path,
    output: Path,
    *,
    max_lora_rank: int = 32,
    max_model_len: int = 8192,
) -> None:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter dir not found: {adapter_dir}")

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json missing in {adapter_dir}")
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors missing in {adapter_dir}")

    # Validate the adapter config against the competition's vLLM constraints.
    config = json.loads(config_path.read_text())
    rank = config.get("r")
    if rank is None:
        raise ValueError("adapter_config.json missing 'r' (rank)")
    if rank > max_lora_rank:
        raise ValueError(
            f"adapter rank {rank} exceeds competition max_lora_rank={max_lora_rank}"
        )
    if not config.get("inference_mode", True):
        print("[package] warning: inference_mode=False — flipping for submission")
        config["inference_mode"] = True
        config_path.write_text(json.dumps(config, indent=2))
    base_model = config.get("base_model_name_or_path", "")
    print(f"[package] adapter rank r={rank}  (max allowed {max_lora_rank}): OK")
    print(f"[package] target_modules: {config.get('target_modules')}")
    print(f"[package] base_model: {base_model}")
    print(f"[package] max_model_len constraint: {max_model_len} (informational; the")
    print(f"          adapter itself imposes no sequence-length limit — this is enforced")
    print(f"          by the harness)")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    to_pack: list[Path] = []
    for name in _REQUIRED:
        p = adapter_dir / name
        if p.is_file():
            to_pack.append(p)
    for name in _OPTIONAL:
        p = adapter_dir / name
        if p.is_file():
            to_pack.append(p)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in to_pack:
            arcname = path.relative_to(adapter_dir)
            z.write(path, arcname)

    print(f"[package] wrote {output}")
    with zipfile.ZipFile(output) as z:
        for info in z.infolist():
            print(f"  {info.filename:<60} {info.file_size:>12} bytes")
    print(f"  total size: {output.stat().st_size} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()
    make_submission(
        args.adapter_dir,
        args.output,
        max_lora_rank=args.max_lora_rank,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
