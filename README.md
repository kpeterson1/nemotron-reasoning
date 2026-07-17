# Nemotron Reasoning Challenge

Competition entry for NVIDIA Nemotron Reasoning Challenge. The benchmark consists of 9,500 rule-induction puzzles across 6 task types. Submission is a LoRA adapter (rank ≤ 32) for Nemotron-3-Nano-30B-A3B, a hybrid Mamba/attention + sparse Mixture-of-Experts model (52 layers — per the model's `hybrid_override_pattern`: 23 Mamba SSM + 23 MoE-FFN + 6 attention; each MoE layer routes top-6 of 128 routed experts per token plus 1 shared expert; ~30B total / ~3B active params per token). Strategy: data-centric approach with diagnostic inference, targeted synthetic data generation, and LoRA fine-tuning. The Kaggle evaluation harness is **greedy**: `temperature=0.0`, `top_p=1.0`, `max_tokens=7680`, `max_model_len=8192`, `enable_thinking=True` (confirmed from the competition Overview tab).

**Writeup:** [`docs/writeup.md`](docs/writeup.md) — the canonical retrospective (merged + repo-audited 2026-07-06).
**Data:** competition data is not redistributed here — regenerate the frozen eval splits from Kaggle's `train.csv` via `src/data/split.py` (seed-42, deterministic); see [`datasets/splits/README.md`](datasets/splits/README.md).
**Results:** see [`RESULTS.md`](RESULTS.md) — local + Kaggle, all variants.
**Investigation & methodology:** [`docs/investigations/OPEN_QUESTIONS.md`](docs/investigations/OPEN_QUESTIONS.md) — the single-variable experiments, negative results, and corrections (C1–C11) are the core of this work.
**Publication status:** this repo is **private**; [`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md) lists privacy/licensing tasks (Huikang reference material, competition data) that **must** be resolved before any public release.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in credentials
```

## Quick Start

```bash
make eval SPLIT=dev_frozen
```

## Project Structure

```
nemotron-reasoning/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── Makefile
├── configs/
│   ├── inference/
│   │   └── baseline.yaml
│   ├── data/
│   ├── train/
│   │   └── lora_baseline.yaml
│   ├── eval/
│   │   └── default.yaml
│   └── openclaw/
├── prompts/
│   ├── direct/
│   │   └── v1.yaml
│   ├── rule_induction/
│   │   └── v1.yaml
│   ├── verify_then_answer/
│   │   └── v1.yaml
│   ├── few_shot/
│   │   └── v1.yaml
│   └── task_typed/
│       └── v1.yaml
├── src/
│   ├── __init__.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── generate.py
│   │   ├── normalize.py
│   │   └── prompt_loader.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── run_eval.py
│   │   ├── compare.py
│   │   └── extract_answer.py
│   ├── verifiers/
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   ├── filter.py
│   │   └── vote.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── split.py
│   │   ├── curate.py
│   │   ├── classify_task.py
│   │   └── format_training.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_lora.py
│   ├── orchestration/
│   │   └── __init__.py
│   └── packaging/
│       ├── __init__.py
│       └── make_submission.py
├── datasets/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   │   └── README.md
│   └── synthetic/
├── reports/
│   ├── daily/
│   ├── ablations/
│   └── leaderboard/
│       └── submission_log.csv
├── runs/
│   ├── inference/
│   ├── train/
│   ├── eval/
│   └── registry.csv
├── notebooks/
│   ├── analysis/
│   └── kaggle_submission/
├── tests/
│   ├── unit/
│   │   ├── test_eval.py
│   │   ├── test_extract_answer.py
│   │   └── test_format_training.py
│   ├── regression/
│   └── notebook_smoke/
└── submission/
    └── README.md
```

## Task Types

| Task type                | Example answer format         |
|--------------------------|-------------------------------|
| bit_manipulation         | binary string, e.g. `10011000` |
| gravitational_constant   | numeric, e.g. `9.81`          |
| unit_conversion          | numeric, e.g. `24.64`         |
| text_encryption          | string, e.g. `khoor`          |
| numeral_conversion       | string/number, e.g. `XLVII`   |
| equation_transformation  | symbolic string, e.g. `a+b`   |

## Eval Parameters

Confirmed Kaggle harness parameters (competition Overview tab). The eval is **greedy** (temperature 0.0):

| Parameter                | Value        |
|--------------------------|--------------|
| temperature              | 0.0 (greedy) |
| top_p                    | 1.0          |
| max_tokens               | 7680         |
| max_model_len            | 8192         |
| max_num_seqs             | 64           |
| max_lora_rank            | 32           |
| gpu_memory_utilization   | 0.85         |
| enable_thinking          | True         |

> **Reproducing locally:** pass these explicitly — the `run_eval` / `make eval` defaults are *not* these values, so a default local run does not reproduce Kaggle conditions.

> **⚠️ Runtime / vLLM-version caveat.** The same byte-identical adapter can score very differently depending on the vLLM version running it. We observed a ~25pp swing on `text_encryption` for one adapter across local vLLM 0.20.1 vs 0.22.1, and Kaggle runs vLLM **0.17.1** (distinct from both, not reproducible on our hosts). Consequences: (1) a strong *local* score does not guarantee the Kaggle score — validate against the target runtime; (2) packaging must match what that runtime's LoRA loader expects (see "Packaging an adapter for Kaggle" below). Full diagnosis in `docs/investigations/` (corrections C7, C9–C11).

The harness appends to every prompt:

```
Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
```

## Submission Format

A LoRA adapter (rank ≤ 32) for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, packaged as `submission.zip` containing `adapter_config.json` and adapter weights. Build with:

```bash
make package ADAPTER=runs/train/lora_baseline
```

## Packaging an adapter for Kaggle (convert → pad → rekey)

A PEFT `target_parameters` LoRA on NemotronH's routed experts is **not** directly loadable by Kaggle's vLLM — the 3D packed expert weights must be unpacked and the config canonicalized. Three CPU-only steps:

1. **Convert** — unpack the 3D packed expert LoRA into per-expert 2D tensors:
   ```bash
   python -m src.training.convert_peft_to_vllm_moe --src <peft_adapter> --dst <out>_vllm
   ```
   The converter emits `target_modules` as a **list** (not a regex string). This is load-bearing: Kaggle's vLLM 0.17.1 silently under-applies the LoRA when `target_modules` is a regex (≈ −2pp; corrections C9/C10). It also nulls the PEFT-3D training fields and validates the 23-MoE-layer coverage.

2. **Pad** — make every expert rank uniform (e.g. r8 → r32) so it matches the global rank (`rank_pattern={}` declares uniform):
   ```bash
   python -m src.training.pad_lora_to_uniform_rank --src <out>_vllm --dst <out>_r32padded --target-rank 32
   ```

3. **Rekey (optional)** — rename keys to the `base_model.model.backbone.` prefix to match the reference layout:
   ```bash
   python -m scripts.rekey_to_backbone_reference --src <out>_r32padded --dst <out>_r32padded_backbone
   ```
   Empirically the prefix is **noise** on Kaggle (≤1pp, sign-inconsistent); the load-bearing fix is the list `target_modules` from step 1.

Then zip the three files — `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja` — **flat at the archive root**, under the 1.5 GB submission cap. Full packaging investigation (mixed-rank mis-load, target_modules format, prefix) in `docs/investigations/` (corrections C8–C11) and `docs/investigations/OPEN_QUESTIONS.md`.
