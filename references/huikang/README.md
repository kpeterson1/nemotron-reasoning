# Huikang Reference Materials

This directory contains small reference materials from Tong Hui Kang’s public NVIDIA Nemotron Model Reasoning Challenge work.

These references are used to debug adapter format/runtime mismatch, especially the local validation vs Kaggle submission gap where a local score can look better than the Kaggle score.

## Why this reference exists

Huikang’s published workflow documents a practical path from a Tinker-trained LoRA adapter to a Kaggle/vLLM-compatible submission adapter.

The important adapter-format issues are:

- Tinker adapter keys use `base_model.model.model`, while the Kaggle/vLLM submission format uses `base_model.model.backbone`.
- Tinker stores fused expert LoRA weights such as `experts.w1` and `experts.w2`.
- The submission-format adapter uses per-expert weights such as `experts.{i}.up_proj` and `experts.{i}.down_proj`.
- Tinker stores separate Mamba `gate_proj` and `x_proj` LoRA weights.
- The submission-format adapter expects a combined Mamba `in_proj`.
- The `gate_proj + x_proj -> in_proj` conversion uses SVD and can be lossy.
- Kaggle evaluates with vLLM using `max_lora_rank=32`, `temperature=0.0`, `top_p=1.0`, `max_tokens=7680`, and `max_model_len=8192`.

## Contents

> **Removed 2026-07-17:** the vendored copies of Huikang's own files
> (`tinker-submission-notebook.md`, `notebook_tinker.py`,
> `kaggle_kernel/tinker-submission-notebook.ipynb`,
> `kaggle_kernel/kernel-metadata.json`, `reference_adapter/adapter_config.json`)
> — no license was located, so redistribution defaults to all-rights-reserved.
> Recover them from his pinned public commit; see `../README.md` (Cited sources).

- `reference_adapter/`  
  Small metadata files **extracted by our tooling** from Huikang’s reference submission-format adapter (key/shape/dtype manifests — our artifacts, still present). This is used as a practical known-reference layout for comparing adapter keys, shapes, dtypes, target modules, and conversion behavior.

- `tinker_adapter/`  
  Optional metadata files extracted from Huikang’s raw Tinker adapter. This is useful for seeing the before-conversion layout.

## What is intentionally not committed

Large adapter artifacts are intentionally excluded from git:

- `adapter_model.safetensors`
- `submission.zip`
- model cache files
- full Kaggle output folders

These files are too large for normal git and should only be stored outside the repo or with Git LFS if explicitly needed.

## Intended use

Use these references to answer questions like:

- Does our adapter still contain Tinker-style keys?
- Did conversion rename `model` to `backbone`?
- Did conversion unfuse `experts.w1/w2` into per-expert `up_proj/down_proj` tensors?
- Did conversion merge Mamba `gate_proj/x_proj` into `in_proj`?
- Does our `adapter_config.json` match the expected rank and target modules?
- Are our tensor shapes and ranks compatible with the reference adapter?
- Is a low Kaggle score caused by packaging/runtime mismatch rather than training quality?

## Guardrails

Do not train from this directory.

Do not submit anything from this directory directly.

Do not commit large `.safetensors` or `.zip` files unless Git LFS is intentionally configured.
