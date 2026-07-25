# Nemotron-3-Nano Architecture Explorer

`nemotron-explorer.html` is an interactive 3D walkthrough of
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16: the full 52-layer hybrid stack in
literal `hybrid_override_pattern` order (Mamba-2 / MoE / attention), per-layer
parameter counts reconciled against the checkpoint's exact 31.5779B safetensors
total, and a router demo showing which 6 of 128 experts each prompt token is
actually routed to at every MoE layer.

## No build step

It is a single self-contained HTML file — open it in a browser. The only
external dependency is Three.js r160, loaded from the jsdelivr CDN via an
import map, so viewing it requires network access but no install, bundler, or
server.

## Where the data comes from

- **Architecture numbers** are derived at runtime from the public HuggingFace
  `config.json` of nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16, snapshot
  `cbd3fa9f933d55ef16a84236559f4ee2a0526848` — nothing is hand-entered except
  the config constants themselves.
- **Router-demo weights are real**: captured 2026-07-20 by
  `extract_router_demo.py` (this directory) — one BF16 forward pass of the full
  model over an 11-token prompt, with a forward hook on every
  `mixer.gate` (all 23 MoE layers). The capture is embedded in the HTML and
  also saved as `router_demo_data.json` (provenance block inside).

## Caveat: capture ran on the PyTorch fallback path

The fused mamba-ssm/causal-conv1d kernels have no build for the capture stack
(torch 2.13/cu130/aarch64), and the gated RMSNorm was a torch reimplementation
of the kernel's documented semantics, not numerically diffed against it. bf16
numerics can flip near-tie top-6 picks versus a kernels build, so treat expert
identities as representative rather than bit-exact; weight magnitudes and the
routing behavior shown are unaffected.
