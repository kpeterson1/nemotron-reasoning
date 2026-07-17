"""Pad every below-r_global LoRA tensor with zeros so the safetensors holds
uniformly-ranked tensors that match the global ``r`` in ``adapter_config.json``.

Why: stock vLLM 0.20.1 reads the global ``r`` from ``adapter_config.json`` and
assumes every ``lora_A``/``lora_B`` tensor matches it. With a mixed-rank
PEFT adapter (e.g. r=32 non-expert + r=8 expert per-expert keys), the file is
internally inconsistent with the config. Local vLLM happens to load it
(possibly with silent rank-mismatch behavior), but Kaggle's stock harness
appears not to — observed: local greedy n=500 = 69.8%, Kaggle = 0.56.

Padding strategy is exactly equivalent in the math:

  delta = B @ A   shape (out, in), rank-r decomposition
  Pad A from (r=r_local, in) to (r=r_global, in) by appending zero rows.
  Pad B from (out, r=r_local) to (out, r=r_global) by appending zero cols.

  B_padded @ A_padded
   = sum_{r=0..r_global-1} B_padded[:, r] outer A_padded[r, :]
   = sum_{r=0..r_local-1} B[:, r] outer A[r, :]  +  sum_{r=r_local..r_global-1} 0 * 0
   = original B @ A.

Scaling: vLLM applies ``scaling = lora_alpha / r`` from the global config.
We preserve the global alpha/r unchanged, so scaling is unchanged. The
original PEFT expert tensors were trained with per-module alpha=r (so per-
module scaling = 1.0), and the converter set ``bake_expert_lora_B_by = 1.0``
already, so the saved tensors are already in the units vLLM expects under
the global scaling. No re-baking is needed for the padded version either.

Memory: the padded tensors are mostly zero in the new positions, so the
safetensors raw size grows linearly with the rank ratio (~4x for r=8→r=32),
but DEFLATE compresses zero runs efficiently — empirically the zip size
grows much less than the raw size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def pad(
    src_dir: Path,
    dst_dir: Path,
    *,
    target_rank: int | None = None,
    audit_samples: int = 6,
) -> dict:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = src_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    r_global = int(cfg["r"])
    if target_rank is None:
        target_rank = r_global
    if target_rank < r_global:
        raise ValueError(f"target_rank={target_rank} cannot be smaller than global r={r_global}")
    print(f"[pad] src={src_dir}  dst={dst_dir}  target_rank={target_rank}  (global r={r_global})", flush=True)

    sf = src_dir / "adapter_model.safetensors"
    new_state: dict[str, torch.Tensor] = {}
    sample_audits: list[dict] = []
    n_padded_A = 0
    n_padded_B = 0
    n_kept = 0
    paired_for_audit: dict[str, dict] = {}

    with safe_open(str(sf), framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            if k.endswith(".lora_A.weight"):
                r_here, in_dim = t.shape
                if r_here < target_rank:
                    pad_rows = torch.zeros(target_rank - r_here, in_dim, dtype=t.dtype, device=t.device)
                    new_state[k] = torch.cat([t, pad_rows], dim=0)
                    n_padded_A += 1
                    if len(sample_audits) < audit_samples:
                        base = k[: -len(".lora_A.weight")]
                        paired_for_audit.setdefault(base, {})["A_orig"] = t
                        paired_for_audit[base]["A_pad"] = new_state[k]
                elif r_here == target_rank:
                    new_state[k] = t
                    n_kept += 1
                else:
                    raise ValueError(f"unexpected lora_A rank {r_here} > target {target_rank} at {k}")
            elif k.endswith(".lora_B.weight"):
                out_dim, r_here = t.shape
                if r_here < target_rank:
                    pad_cols = torch.zeros(out_dim, target_rank - r_here, dtype=t.dtype, device=t.device)
                    new_state[k] = torch.cat([t, pad_cols], dim=1)
                    n_padded_B += 1
                    base = k[: -len(".lora_B.weight")]
                    if base in paired_for_audit:
                        paired_for_audit[base]["B_orig"] = t
                        paired_for_audit[base]["B_pad"] = new_state[k]
                elif r_here == target_rank:
                    new_state[k] = t
                    n_kept += 1
                else:
                    raise ValueError(f"unexpected lora_B rank {r_here} > target {target_rank} at {k}")
            else:
                new_state[k] = t
                n_kept += 1
        # Audit a few B@A products for numeric equivalence pre/post pad.
        equiv_max_diff = 0.0
        equiv_samples = []
        for base, pair in list(paired_for_audit.items())[:audit_samples]:
            if "A_orig" not in pair or "B_orig" not in pair:
                continue
            with torch.no_grad():
                delta_orig = pair["B_orig"].float() @ pair["A_orig"].float()
                delta_pad = pair["B_pad"].float() @ pair["A_pad"].float()
                diff = (delta_orig - delta_pad).abs().max().item()
                equiv_samples.append({"key_base": base, "orig_shape": list(pair["A_orig"].shape), "pad_shape": list(pair["A_pad"].shape), "max_abs_diff": diff})
                equiv_max_diff = max(equiv_max_diff, diff)

    print(f"[pad] padded lora_A: {n_padded_A}    padded lora_B: {n_padded_B}    untouched: {n_kept}", flush=True)
    print(f"[pad] B@A numeric equivalence on {len(equiv_samples)} samples: max abs diff = {equiv_max_diff:.6e}", flush=True)
    for ea in equiv_samples:
        print(f"      base={ea['key_base'][-60:]:60s}  A {ea['orig_shape']} -> {ea['pad_shape']}  Δmax={ea['max_abs_diff']:.3e}")

    # Write the padded safetensors.
    out_sf = dst_dir / "adapter_model.safetensors"
    save_file(new_state, str(out_sf))

    # Copy adapter_config.json verbatim, then copy chat_template + tokenizer files if present.
    (dst_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    for opt in ("chat_template.jinja", "tokenizer_config.json", "tokenizer.json"):
        src = src_dir / opt
        if src.exists():
            (dst_dir / opt).write_bytes(src.read_bytes())

    raw_size = out_sf.stat().st_size
    summary = {
        "src_dir": str(src_dir),
        "dst_dir": str(dst_dir),
        "target_rank": target_rank,
        "global_r": r_global,
        "n_tensors": len(new_state),
        "n_padded_A": n_padded_A,
        "n_padded_B": n_padded_B,
        "n_untouched": n_kept,
        "numeric_equivalence_max_abs_diff": equiv_max_diff,
        "audit_samples": equiv_samples,
        "padded_safetensors_size_bytes": raw_size,
        "padded_safetensors_size_MiB": round(raw_size / 2**20, 2),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="Converted vLLM-format adapter dir")
    parser.add_argument("--dst", type=Path, required=True, help="Output dir for padded adapter")
    parser.add_argument("--target-rank", type=int, default=None, help="Pad all lora_A/lora_B to this rank (default: global r from adapter_config.json)")
    args = parser.parse_args()
    pad(args.src, args.dst, target_rank=args.target_rank)


if __name__ == "__main__":
    main()
