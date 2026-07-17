#!/usr/bin/env python3
"""
Round-trip check for the routed-expert unpack in convert_peft_to_vllm_moe.py.

Question: when the converter reshapes PEFT's packed expert LoRA into 128
per-expert tensors, does it assign the CORRECT delta to each expert, or a
scrambled (transposed) one? A scrambled-but-valid reshape produces 128
well-formed tensors that individually look fine but carry the wrong
per-expert content — the textbook cause of "experts apply but at reduced,
environment-dependent fidelity."

Observed packed shapes (layer 1):
    experts.base_layer.lora_A.weight : (1024, 2688)   # 1024 = 128 * 8
    experts.base_layer.lora_B.weight : (1856, 1024)   # 1024 = 128 * 8
So effective packed per-expert rank here is 8 (E*r = 1024).

Strategy:
  1. Recover E and r from the packed first axis (E*r) and num_experts=128.
  2. From the RAW packed tensor, slice expert e two ways:
        expert-major: rows [e*r : (e+1)*r]
        rank-major  : rows [e :: E]            (stride across experts)
  3. From the CONVERTED artifact, read the converter's actual per-expert
     up_proj.lora_A for the same layer/expert.
  4. Whichever slicing reproduces the converted tensor for ALL sampled
     experts (test e=0,1,E-1) is the layout the converter used.
        - matches expert-major  -> CORRECT, conversion is clean on this axis
        - matches rank-major     -> SCRAMBLED, this is the fidelity bug
        - matches only e=0       -> partial/transposed (still a bug)
  5. Repeat independently for lora_B (its expert axis is laid out
     differently: packed (out, E*r) vs (E*r, in) for A), so check both.

Run on Spark 1 (single line or via this file):
    python check_expert_unpack_roundtrip.py \
        --raw  runs/train/lora_v9_warm_start_300step/adapter_model.safetensors \
        --conv runs/train/lora_v9_warm_start_300step_vllm/adapter_model.safetensors \
        --layer 1 --num-experts 128
"""
import argparse, re
import torch
from safetensors import safe_open

def load(path):
    f = safe_open(path, framework="pt", device="cpu")
    return {k: f.get_tensor(k) for k in f.keys()}

def find(keys_or_dict, *needles):
    for k in keys_or_dict:
        if all(n in k for n in needles):
            return k
    return None

def try_match(packed, conv_expert, E, r, axis_first, label):
    """
    packed: the raw packed tensor.
    conv_expert: dict e -> converted per-expert tensor (2D).
    axis_first True  => expert axis is the FIRST (row) axis of packed (A case).
    axis_first False => expert axis is the LAST (col) axis of packed (B case).
    Tests expert-major vs rank-major slicing along that axis.
    Returns (layout_str, per_expert_results).
    """
    results = {}
    for e in sorted(conv_expert):
        target = conv_expert[e].float()
        cands = {}
        if axis_first:
            # packed shape (E*r, dim). expert-major: contiguous block.
            cands["expert_major"] = packed[e*r:(e+1)*r, :].float()
            # rank-major: stride
            cands["rank_major"] = packed[e::E, :].float()
        else:
            # packed shape (dim, E*r). expert axis is columns.
            cands["expert_major"] = packed[:, e*r:(e+1)*r].float()
            cands["rank_major"] = packed[:, e::E].float()
        match = None
        for name, c in cands.items():
            if c.shape == target.shape and torch.allclose(c, target, atol=1e-5, rtol=1e-4):
                match = name
                break
        results[e] = match
    layouts = set(results.values())
    if layouts == {"expert_major"}:
        verdict = "EXPERT-MAJOR (CORRECT)"
    elif layouts == {"rank_major"}:
        verdict = "RANK-MAJOR (SCRAMBLED — fidelity bug)"
    elif None in layouts and len(layouts) > 1:
        verdict = "MIXED/PARTIAL (transposed — fidelity bug)"
    else:
        verdict = f"UNRESOLVED layouts={layouts}"
    print(f"  [{label}] {verdict}  per-expert={results}")
    return verdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--conv", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--num-experts", type=int, default=128)
    a = ap.parse_args()
    E = a.num_experts
    L = a.layer

    raw = load(a.raw)
    conv = load(a.conv)

    # --- locate the packed A/B for this layer in RAW ---
    # base_layer.lora_A/B = up_proj source; experts.lora_A/B (no base_layer) = down_proj source
    pA = find(raw, f"layers.{L}.", "experts.base_layer.lora_A")
    pB = find(raw, f"layers.{L}.", "experts.base_layer.lora_B")
    print(f"packed up_proj A: {pA} {tuple(raw[pA].shape)}")
    print(f"packed up_proj B: {pB} {tuple(raw[pB].shape)}")

    # infer r from first axis of A: E*r = rows
    rowsA = raw[pA].shape[0]
    assert rowsA % E == 0, f"A rows {rowsA} not divisible by E={E}"
    r = rowsA // E
    print(f"inferred per-expert packed rank r = {r}  (E*r = {rowsA})")

    # --- gather converted per-expert up_proj A/B for sampled experts ---
    sample = sorted({0, 1, E-1})
    convA = {}
    convB = {}
    for e in sample:
        ka = find(conv, f"layers.{L}.", f"experts.{e}.up_proj.lora_A")
        kb = find(conv, f"layers.{L}.", f"experts.{e}.up_proj.lora_B")
        if ka is None or kb is None:
            print(f"  [WARN] missing converted up_proj for expert {e}: A={ka} B={kb}")
            continue
        convA[e] = conv[ka]
        convB[e] = conv[kb]
        print(f"  conv e={e:3d} up_proj.A {tuple(conv[ka].shape)}  up_proj.B {tuple(conv[kb].shape)}")

    print("\n=== up_proj.lora_A unpack ===")
    # packed A is (E*r, in); expert axis is rows -> axis_first=True
    vA = try_match(raw[pA], convA, E, r, axis_first=True, label="up_proj.A")

    print("\n=== up_proj.lora_B unpack ===")
    # packed B observed as (out, E*r): expert axis is columns -> axis_first=False
    # (verify against the printed shape; if B is (E*r, out) flip axis_first)
    B_axis_first = (raw[pB].shape[0] % E == 0 and raw[pB].shape[0] // E == r
                    and raw[pB].shape[1] != r*E)
    print(f"  (B axis_first={B_axis_first} from shape {tuple(raw[pB].shape)})")
    vB = try_match(raw[pB], convB, E, r, axis_first=B_axis_first, label="up_proj.B")

    print("\n================ VERDICT ================")
    bad = lambda v: ("SCRAMBLED" in v) or ("PARTIAL" in v) or ("UNRESOLVED" in v)
    if bad(vA) or bad(vB):
        print("Expert unpack is WRONG on >=1 axis. This is the prime")
        print("candidate for Kaggle's ~1/3 expert fidelity. The conversion")
        print("assigns scrambled/transposed deltas per expert. FIX THE")
        print("RESHAPE before any vLLM-version hunt — local eval may mask it.")
        raise SystemExit(2)
    print("Expert unpack is EXPERT-MAJOR / CORRECT on A and B.")
    print("Conversion is clean on the per-expert assignment axis too.")
    print("Q3 is genuinely a Kaggle-runtime question: proceed to pin")
    print("Kaggle's vLLM version / moe_backend.")
    raise SystemExit(0)

if __name__ == "__main__":
    main()
