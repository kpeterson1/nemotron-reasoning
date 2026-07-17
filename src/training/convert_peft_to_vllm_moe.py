"""Convert a PEFT ``target_parameters`` LoRA on NemotronH routed experts into
a vLLM-compatible per-expert LoRA directory structure.

See the docstring at the top for background. This script unpacks PEFT's 3D
``experts.base_layer.lora_A`` / ``experts.lora_A`` etc. (packed with one LoRA
delta per 128 experts) into 128 separate 2D LoRA tensors keyed by expert index,
which vLLM's ``_create_merged_loras_inplace`` expects.

``_stack_moe_lora_weights`` never unpacks the PEFT-saved ``experts.base_layer``
/ ``experts`` tensor pair. We do the unpack offline, splitting each PEFT 3D
ParamWrapper LoRA into 128 standard per-expert 2D LoRAs whose keys are the ones
vLLM expects.
"""
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Kaggle reference target_modules — a LIST, not a regex string. Kaggle's vLLM 0.17.1
# LoRA loader under-applies the adapter when target_modules is a regex string
# (C9/C10: regex scored 0.55 vs list 0.57 on byte-identical weights). These are the
# module-name suffixes the converted per-expert keys use. This is the single source
# of truth — scripts/rekey_to_backbone_reference.py imports it from here.
REFERENCE_TARGET_MODULES = [
    "k_proj", "o_proj", "in_proj", "q_proj",
    "up_proj", "v_proj", "down_proj", "out_proj",
]

# Regex to match PEFT's expert LoRA keys. Captures layer and A/B suffix.
_PEFT_EXPERT_KEY = re.compile(
    r"^(?P<prefix>base_model\.model\.model\.layers\.(?P<layer>\d+)\.mixer\.experts)"
    r"(?P<inner>\.base_layer)?\.lora_(?P<ab>[AB])\.weight$"
)


def _expert_r_alpha(src_cfg: dict) -> tuple[int, int]:
    """Resolve the (rank, alpha) PEFT actually used for the routed-expert ParamWrappers.

    PEFT's ``rank_pattern`` / ``alpha_pattern`` regex-match against the ``current_key``
    (e.g. ``model.layers.<L>.mixer.experts.up_proj``). We pull the value(s) for the two
    expert ParamWrappers, verify they agree, and return them. Falls back to the global
    ``r`` / ``lora_alpha`` if no pattern matched.
    """
    rp = src_cfg.get("rank_pattern") or {}
    ap = src_cfg.get("alpha_pattern") or {}
    rs: set[int] = set()
    alphas: set[int] = set()
    for key in ("mixer.experts.up_proj", "mixer.experts.down_proj"):
        if key in rp:
            rs.add(int(rp[key]))
        if key in ap:
            alphas.add(int(ap[key]))
    if len(rs) > 1:
        raise RuntimeError(
            f"rank_pattern values disagree between up_proj and down_proj experts: {rp} — "
            "this converter assumes a single expert rank."
        )
    if len(alphas) > 1:
        raise RuntimeError(
            f"alpha_pattern values disagree between up_proj and down_proj experts: {ap}"
        )
    r = next(iter(rs)) if rs else int(src_cfg["r"])
    alpha = next(iter(alphas)) if alphas else int(src_cfg["lora_alpha"])
    return r, alpha


def convert(
    src_dir: Path,
    dst_dir: Path,
    num_experts: int = 128,
) -> dict:
    """Unpack PEFT's 3D packed expert LoRA into per-expert 2D tensors for vLLM.
    
    Args:
        src_dir: Path to PEFT adapter (contains adapter_model.safetensors, adapter_config.json)
        dst_dir: Output directory for converted vLLM-format adapter
        num_experts: Number of experts per MoE layer (default 128 for NemotronH)
    
    Returns:
        Summary dict with conversion metadata.
    """
    src_cfg = json.loads((src_dir / "adapter_config.json").read_text())
    r_global = int(src_cfg["r"])
    alpha_global = int(src_cfg["lora_alpha"])
    r_expert, alpha_expert = _expert_r_alpha(src_cfg)
    
    # vLLM applies one uniform scaling = alpha_global / r_global to every LoRA. If the per-module
    # training scaling differs, we have to bake the offset into lora_B before saving so that the
    # downstream vLLM forward reproduces the PEFT-computed delta exactly.
    train_scale_expert = alpha_expert / r_expert
    vllm_scale_global = alpha_global / r_global
    bake_expert = train_scale_expert / vllm_scale_global  # multiply expert lora_B by this
    print(
        f"[convert] r_global={r_global}  alpha_global={alpha_global}  "
        f"r_expert={r_expert}  alpha_expert={alpha_expert}  "
        f"vllm_scale={vllm_scale_global:.4f}  train_scale_expert={train_scale_expert:.4f}  "
        f"bake_expert_lora_B_by={bake_expert:.4f}"
    )
    # Keep r as a single value here — used only for slicing expert tensors.
    r = r_expert
    lora_alpha = alpha_global  # only used for echoing into the destination config

    src_st = src_dir / "adapter_model.safetensors"
    with safe_open(src_st, framework="pt") as f:
        peft_keys = list(f.keys())
        peft_tensors = {k: f.get_tensor(k) for k in peft_keys}

    # Index PEFT expert tensors by (layer, slot, A/B). slot: "up" (=base_layer) or "down" (=outer).
    expert_by_layer: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    passthrough: dict[str, torch.Tensor] = {}
    for key, t in peft_tensors.items():
        m = _PEFT_EXPERT_KEY.match(key)
        if m is None:
            passthrough[key] = t
            continue
        layer = int(m["layer"])
        slot = "up" if m["inner"] else "down"  # base_layer = up_proj, outer = down_proj
        ab = m["ab"]  # 'A' or 'B'
        expert_by_layer.setdefault(layer, {}).setdefault(slot, {})[ab] = t

    if not expert_by_layer:
        raise RuntimeError("No PEFT expert tensors found — check adapter structure.")

    # Guardrail (restored): the layer set must exactly match NemotronH-30B-A3B's 23 MoE
    # layers. A silently missing/extra layer = a silently-wrong conversion — the exact
    # failure mode this investigation showed matters.
    expected_moe_layers = {
        1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34, 36, 38, 40, 43, 45, 47, 49, 51
    }
    seen = set(expert_by_layer)
    if seen != expected_moe_layers:
        raise RuntimeError(
            f"PEFT MoE-layer coverage mismatch: missing={sorted(expected_moe_layers - seen)} "
            f"extra={sorted(seen - expected_moe_layers)}"
        )

    # Validate packed shapes. All expert A/B tensors must have the first or last dimension = r*E.
    sample = next(iter(expert_by_layer.values()))
    up_A_shape = tuple(sample["up"]["A"].shape)
    up_B_shape = tuple(sample["up"]["B"].shape)
    down_A_shape = tuple(sample["down"]["A"].shape)
    down_B_shape = tuple(sample["down"]["B"].shape)
    rE = r * num_experts
    expected = {
        "up.A": (rE, up_A_shape[1]),
        "up.B": (up_B_shape[0], rE),
        "down.A": (rE, down_A_shape[1]),
        "down.B": (down_B_shape[0], rE),
    }
    observed = {
        "up.A": up_A_shape, "up.B": up_B_shape, "down.A": down_A_shape, "down.B": down_B_shape,
    }
    for k in expected:
        if expected[k] != observed[k]:
            raise RuntimeError(
                f"shape mismatch for {k}: expected {expected[k]}, observed {observed[k]} "
                f"(num_experts={num_experts}, r={r})"
            )

    # in/out dims (consistent across all MoE layers in NemotronH-30B-A3B)
    hidden = up_A_shape[1]         # 2688
    intermediate = up_B_shape[0]   # 1856
    # Guardrail (restored): up/down projections must share the inner dims.
    assert intermediate == down_A_shape[1], "expected intermediate to match down_proj A"
    assert hidden == down_B_shape[0], "expected hidden to match down_proj B"

    print(f"[convert] r={r}  num_experts={num_experts}  hidden={hidden}  intermediate={intermediate}")

    # Unpack PEFT's 3D packed expert LoRA into per-expert 2D tensors.
    out_tensors: dict[str, torch.Tensor] = {}
    seen_layers: set[int] = set()

    for layer, slots in expert_by_layer.items():
        seen_layers.add(layer)
        for slot_name, slot_data in slots.items():
            # Map slot name to vLLM's target module name.
            hf_proj_name = "up_proj" if slot_name == "up" else "down_proj"
            A_packed = slot_data["A"]  # (r*E, in) for up, (r*E, in) for down
            B_packed = slot_data["B"]  # (out, r*E) for both up and down

            # === UNPACK EXPERT A (expert-major layout) ===
            # PEFT packs A as (E*r, in) with expert-major ordering: rows [e*r:(e+1)*r] belong to expert e.
            # Reshape to (E, r, in) and extract [e, :, :] for each expert e.
            A_per_expert = A_packed.reshape(num_experts, r, A_packed.shape[1])  # (E, r, in)

            # === UNPACK EXPERT B (FIXED: expert in the middle, not innermost) ===
            # PEFT packs B as (out, E*r), NOT (out, r*E) as the old docstring claimed.
            # The correct unpacking reshape is (out, E, r), with expert as the MIDDLE axis.
            # Then extract [:, e, :] for each expert e to get the (out, r) LoRA factors.
            #
            # BEFORE (WRONG):
            #   B_per_expert = B_packed.reshape(B_packed.shape[0], r, num_experts)  # (out, r, E)
            #   b_e = B_per_expert[:, :, e].contiguous()  # [:, :, e] → wrong axis
            #
            # AFTER (CORRECT):
            #   B_per_expert = B_packed.reshape(B_packed.shape[0], num_experts, r)  # (out, E, r)
            #   b_e = B_per_expert[:, e, :].contiguous()  # [:, e, :] → expert axis
            B_per_expert = B_packed.reshape(B_packed.shape[0], num_experts, r)  # (out, E, r)

            for e in range(num_experts):
                prefix = f"base_model.model.model.layers.{layer}.mixer.experts.{e}.{hf_proj_name}"
                # Standard 2D LoRA shapes: A = (r, in), B = (out, r). Contiguous to keep
                # safetensors happy and downstream torch.stack identical to native.
                a_e = A_per_expert[e].contiguous()
                b_e = B_per_expert[:, e, :].contiguous()  # FIXED: [:, e, :] not [:, :, e]
                if bake_expert != 1.0:
                    b_e = b_e * bake_expert
                out_tensors[f"{prefix}.lora_A.weight"] = a_e
                out_tensors[f"{prefix}.lora_B.weight"] = b_e

    # Add non-expert tensors (in/out_proj, shared_experts, attention q/k/v/o).
    print(f"[convert] non-expert passthrough tensors: {len(passthrough)}")
    out_tensors.update(passthrough)

    # Write converted adapter to disk.
    dst_dir.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, dst_dir / "adapter_model.safetensors")

    # adapter_config for vLLM. Two corrections over a naive src_cfg passthrough:
    #  (1) drop the PEFT-3D training fields (target_parameters / rank_pattern / alpha_pattern):
    #      vLLM loads the per-expert keys directly with one uniform scaling = alpha_global /
    #      r_global (baked above); leaving target_parameters in a per-expert adapter is wrong.
    #  (2) emit target_modules as a LIST (REFERENCE_TARGET_MODULES), not src_cfg's regex string —
    #      Kaggle's vLLM 0.17.1 under-applies a regex target_modules (C9/C10). This makes the
    #      converter output Kaggle-valid standalone (no downstream rekey needed for the config).
    dst_config = {
        **src_cfg,
        "r": r_global,
        "lora_alpha": lora_alpha,
        "target_modules": list(REFERENCE_TARGET_MODULES),
        "target_parameters": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "inference_mode": True,
    }
    (dst_dir / "adapter_config.json").write_text(json.dumps(dst_config, indent=2))

    # Copy any other files (e.g., README).
    for src in src_dir.iterdir():
        if src.name not in ("adapter_model.safetensors", "adapter_config.json"):
            dst = dst_dir / src.name
            if src.is_file() and not dst.exists():
                dst.write_bytes(src.read_bytes())

    summary = {
        "src_dir": str(src_dir),
        "dst_dir": str(dst_dir),
        "r_global": r_global,
        "alpha_global": alpha_global,
        "r_expert": r_expert,
        "alpha_expert": alpha_expert,
        "bake_expert_lora_B": bake_expert,
        "num_experts": num_experts,
        "moe_layers": sorted(seen_layers),
        "passthrough_tensors": len(passthrough),
        "expert_tensors_written": 4 * num_experts * len(seen_layers),
        "total_tensors": len(out_tensors),
    }
    print(f"[convert] wrote {len(out_tensors)} tensors to {dst_dir / 'adapter_model.safetensors'}")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="PEFT target_parameters adapter dir")
    parser.add_argument("--dst", type=Path, required=True, help="Output dir for converted adapter")
    parser.add_argument("--num-experts", type=int, default=128)
    args = parser.parse_args()
    convert(args.src, args.dst, num_experts=args.num_experts)


if __name__ == "__main__":
    main()
