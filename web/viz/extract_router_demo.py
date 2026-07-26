#!/usr/bin/env python
"""Capture REAL MoE router decisions for web/viz/nemotron-explorer.html.

Runs one forward pass of nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 over a
short demo prompt and records, for every MoE layer and every prompt token,
the exact (topk_indices, topk_weights) returned by the model's own
NemotronHTopkRouter.forward (sigmoid scores -> +e_score_correction_bias for
selection -> top-6 -> renormalize -> x routed_scaling_factor). Nothing is
recomputed outside the model; the hook captures the router's actual output.

Output: web/viz/router_demo_data.json (embedded into the HTML by hand/script).

Usage: python web/viz/extract_router_demo.py [--prompt "..."] [--device auto]
"""
import argparse
import json
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import torch
import torch.nn.functional as F


def _install_mamba_ssm_stub() -> None:
    """modeling_nemotron_h.py hard-imports rmsnorm_fn from mamba_ssm (no wheel
    for torch 2.13/cu130/aarch64). Provide a pure-torch implementation of the
    kernel's documented semantics (norm_before_gate=False: norm(x * silu(z)),
    RMS taken per channel group). The fast-path scan kernels stay unavailable,
    so the model uses its own torch_forward path — routing math is untouched.
    """

    def rmsnorm_fn(x, weight, bias, z=None, eps=1e-5, group_size=None, norm_before_gate=False):
        dtype = x.dtype
        if z is not None and not norm_before_gate:
            x = x * F.silu(z)
        gs = group_size or x.shape[-1]
        xg = x.float().view(*x.shape[:-1], x.shape[-1] // gs, gs)
        xg = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
        out = xg.view(x.shape) * weight.float()
        if bias is not None:
            out = out + bias.float()
        out = out.to(dtype)
        if z is not None and norm_before_gate:
            out = out * F.silu(z)
        return out

    names = ["mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton", "mamba_ssm.ops.triton.layernorm_gated"]
    for n in names:
        mod = types.ModuleType(n)
        mod.__spec__ = ModuleSpec(n, loader=None)
        mod.__version__ = "0.0.0"  # < 2.0.4 → is_mamba_2_ssm_available() stays False
        sys.modules[n] = mod
    sys.modules[names[-1]].rmsnorm_fn = rmsnorm_fn


_install_mamba_ssm_stub()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    / "snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848"
)
DEFAULT_PROMPT = "The derivative of x^2 is 2x."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(Path(__file__).parent / "router_demo_data.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(SNAPSHOT, trust_remote_code=True, local_files_only=True)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    tokens = [tok.decode([i]) for i in ids[0].tolist()]
    print(f"prompt tokens ({len(tokens)}): {tokens}", flush=True)

    print("loading model (BF16, ~63 GB)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    captured: dict[int, list] = {}
    hooks = []
    for li, layer in enumerate(model.backbone.layers):
        gate = getattr(layer.mixer, "gate", None)
        if gate is None or not hasattr(gate, "e_score_correction_bias"):
            continue

        def make_hook(layer_idx: int):
            def hook(_mod, _inp, out):
                topk_indices, topk_weights = out
                captured[layer_idx] = [
                    [[int(i), round(float(w), 4)] for i, w in zip(idx_row, w_row)]
                    for idx_row, w_row in zip(
                        topk_indices.detach().cpu().tolist(),
                        topk_weights.detach().float().cpu().tolist(),
                    )
                ]
            return hook

        hooks.append(gate.register_forward_hook(make_hook(li)))
    print(f"hooked {len(hooks)} MoE routers", flush=True)

    with torch.no_grad():
        model(ids.to(model.device))
    for h in hooks:
        h.remove()

    moe_layers = sorted(captured)
    n_tok = len(tokens)
    assert all(len(captured[l]) == n_tok for l in moe_layers), "token-count mismatch in capture"
    # sort each token's experts by descending weight for display
    routing = {
        str(l): [sorted(row, key=lambda p: -p[1]) for row in captured[l]] for l in moe_layers
    }
    out = {
        "prompt": args.prompt,
        "tokens": tokens,
        "moe_layers": moe_layers,
        "routing": routing,
        "provenance": {
            "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "snapshot": "cbd3fa9f933d55ef16a84236559f4ee2a0526848",
            "mechanism": "forward hook on backbone.layers[i].mixer.gate; weights are the "
            "router's returned topk_weights (sigmoid, renormalized, x2.5 routed_scaling_factor)",
            "torch": torch.__version__,
            "caveat": "captured on the model's PyTorch fallback path (fused "
            "mamba-ssm/causal-conv1d kernels unavailable for torch 2.13/cu130/aarch64; "
            "gated RMSNorm was a torch reimplementation of the kernel's documented "
            "semantics, not numerically diffed against it). bf16 numerics can flip "
            "near-tie top-6 picks versus a kernels build.",
        },
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    ws = [w for _, w in routing[str(moe_layers[0])][0]]
    print(f"wrote {args.out}; layer {moe_layers[0]} token0 weight sum = {sum(ws):.3f} (expect 2.5)")


if __name__ == "__main__":
    sys.exit(main())
