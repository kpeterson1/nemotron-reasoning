# Reproducing this work

The operational guide: environment setup, local evaluation, the first-divergence
diagnostic probe, and packaging an adapter for Kaggle's runtime. What was found
and why is the [writeup](writeup.md); repository state, key scripts, and compute
are documented in [writeup §10](writeup.md#10-reproducibility). This file is the
how-to.

## Environment setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in credentials
```

## Quick start (local eval)

```bash
make eval SPLIT=dev_frozen
```

**Data:** competition data is not redistributed here — regenerate the frozen eval splits from Kaggle's `train.csv` via `src/data/split.py` (seed-42, deterministic); see [`datasets/splits/README.md`](../datasets/splits/README.md).

## Quickstart: one diagnostic path (first-divergence probe)

The fastest way to see the project's core instrument working: teacher-force the gold `bit_manipulation` traces through a trained adapter and locate where the model's greedy prediction first leaves the trace ([writeup §4](writeup.md#4-diagnostic-method-teacher-forced-first-divergence-probing)).

**Prerequisites:**

- The environment above, on a GPU host that can serve `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` under vLLM.
- The regenerated `dev_frozen` split at `datasets/splits/dev_frozen.jsonl` (see Quick start above).
- A trained, vLLM-converted LoRA adapter directory (trained weights are not redistributed; train per `configs/` or supply your own). To reproduce the reported diagnostic this must be the **v9 baseline adapter** — the 44-of-47 finding is defined against its failures.
- **Required:** a baseline eval artifact of that same v9 adapter on `dev_frozen`. The probe reads `--baseline-eval` unconditionally (it defines which rows count as "previously failing"); this is not optional. **`make eval` alone cannot produce it** — the Makefile target runs `run_eval` without `--adapter-dir`, so it evaluates the *base model*, whose failure set is not the one behind the 44-of-47 claim.

**Step 1 — produce the v9 baseline eval artifact** (illustrative — flags match `run_eval`'s CLI, but the command is not re-run here; you must supply the v9 adapter yourself):

```bash
python -m src.evaluation.run_eval \
  --split dev_frozen --config configs/eval/default.yaml \
  --adapter-dir <path-to-v9-baseline-adapter> \
  --temperature 0.0 --max-tokens 7680 --max-model-len 8192
```

`--temperature 0.0` must be passed explicitly (`run_eval`'s own default is 1.0; the canonical baselines are greedy — see [Kaggle eval parameters](#kaggle-eval-parameters) for the remaining harness values). The artifact lands in `runs/eval/` (the config's `output_dir`) as `dev_frozen-raw-<adapter>-<timestamp>.json`. The original diagnostic used the canonical artifact `dev_frozen-raw-submissions_extracted_lora_v9_warm_start_300step-1780923962.json` (gitignored, not redistributed); a re-run baseline is subject to the ±0.5pp greedy noise floor, so your previously-failing set may differ slightly from the original 47.

**Step 2 — run the probe against that artifact** (illustrative, same caveat):

```bash
python -m scripts.bitmanip_logprob_probe \
  --adapter <path-to-v9-baseline-adapter> \
  --trace-version v4 \
  --baseline-eval runs/eval/<artifact-from-step-1>.json \
  --restrict-failing \
  --out runs/eval/bitmanip_logprob_probe.json
```

**Expected output:** a JSON report at the `--out` path — per example, the first-divergence record (`pos`, trace `region`, gold token, its logprob and rank, and the token the model preferred), plus `clean`/mismatch counts — and a printed first-divergence-by-region histogram. On the v9 baseline adapter this procedure is what produced the "44 of 47 failures first diverge at the RULE_STATEMENT line" finding ([writeup §5](writeup.md#5-case-study-bit-manipulation)).

This quickstart reproduces the *diagnostic*, not the leaderboard. No full Kaggle-score reproduction is claimed anywhere in this repository — see the runtime caveat under [Kaggle eval parameters](#kaggle-eval-parameters) below.

## Repository layout

```
src/           solvers, trace generators, training, evaluation, packaging (src/data, src/evaluation, src/training, …)
scripts/       one-off diagnostics and pipeline drivers (logprob probes, converters' round-trip check, build scripts)
configs/       training / eval / inference YAML configs
prompts/       prompt templates by strategy
docs/          the canonical writeup, session log, and docs/investigations/ (corrections, parity reports)
datasets/      training mixes and eval splits (competition-derived files not redistributed; regenerate via src/data/split.py)
runs/          training and eval artifacts (large files gitignored)
references/    third-party reference material — provenance documented in references/README.md
reports/       observation and progress logs kept during the competition
tests/         unit and regression tests
```

## Task types

| Task type                | Example answer format         |
|--------------------------|-------------------------------|
| bit_manipulation         | binary string, e.g. `10011000` |
| gravitational_constant   | numeric, e.g. `9.81`          |
| unit_conversion          | numeric, e.g. `24.64`         |
| text_encryption          | string, e.g. `khoor`          |
| numeral_conversion       | string/number, e.g. `XLVII`   |
| equation_transformation  | symbolic string, e.g. `a+b`   |

## Kaggle eval parameters

Confirmed Kaggle harness parameters (competition Overview tab; the rows marked * are additionally confirmed from the competition metric's deployed code, which is not redistributed here). The eval is **greedy** (temperature 0.0):

| Parameter                | Value        |
|--------------------------|--------------|
| temperature              | 0.0 (greedy) |
| top_p                    | 1.0          |
| max_tokens               | 7680         |
| max_model_len            | 8192         |
| max_num_seqs             | 64           |
| max_lora_rank            | 32           |
| gpu_memory_utilization   | 0.85         |
| enable_thinking *        | True         |
| trust_remote_code *      | True         |
| enable_prefix_caching *  | True         |
| enable_chunked_prefill * | True         |
| dtype *                  | 'auto'       |

> **Reproducing locally:** pass these explicitly — the `run_eval` / `make eval` defaults are *not* these values, so a default local run does not reproduce Kaggle conditions.

> **⚠️ Runtime / vLLM-version caveat.** The same byte-identical adapter can score very differently depending on the vLLM version running it. We observed a ~25pp swing on `text_encryption` for one adapter across local vLLM 0.20.1 vs 0.22.1, and Kaggle runs vLLM **0.17.1** (user-reported; distinct from both, not reproducible on our hosts). Consequences: (1) a strong *local* score does not guarantee the Kaggle score — validate against the target runtime; (2) packaging must match what that runtime's LoRA loader expects (see "Packaging an adapter for Kaggle" below). Full diagnosis in [writeup §6](writeup.md#6-failure-to-translate-and-the-runtime-investigation) and `docs/investigations/` (corrections C7, C9–C11).

The harness appends to every prompt:

```
Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
```

## Submission format

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

Then zip the three files — `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja` — **flat at the archive root**, under the 1.5 GB submission cap. Full packaging investigation (mixed-rank mis-load, target_modules format, prefix) in `docs/investigations/` (corrections C8–C11) and [`OPEN_QUESTIONS.md`](investigations/OPEN_QUESTIONS.md).
