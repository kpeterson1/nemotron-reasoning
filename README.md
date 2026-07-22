# Nemotron Reasoning Challenge — Diagnosing Reasoning-SFT Bottlenecks

A case study in finding out *why* a fine-tuned reasoning model fails — entered in the NVIDIA Nemotron Model Reasoning Challenge (Kaggle, closed June 15, 2026). Final result: public LB **0.58** · private LB **0.604** · rank **3488/4182**. This is not a winning solution; it is a documented investigation of the bottlenecks that determine whether reasoning SFT works, with the strongest experiment being a complete diagnosis → intervention → measurement loop on one failing category.

**The canonical detailed account is [`docs/writeup.md`](docs/writeup.md).** This README is the short version.

## What this project investigated

The benchmark consists of 9,500 rule-induction puzzles across 6 task types. The submission is a LoRA adapter — a small, low-rank set of weight updates (rank ≤ 32) — for Nemotron-3-Nano-30B-A3B, a hybrid Mamba/attention + sparse Mixture-of-Experts model (52 layers — per the model's `hybrid_override_pattern`: 23 Mamba SSM + 23 MoE-FFN + 6 attention; each MoE layer routes top-6 of 128 routed experts per token plus 1 shared expert; ~30B total parameters, with the "A3B" suffix denoting its ~3B-class active-parameter count per token). Evaluation is **greedy decoding**: the model must commit to exactly one token at every position, with no sampling to hide near-misses.

The standard approach — ours included — is to write deterministic Python solvers that produce verified step-by-step reasoning traces, then fine-tune the model to reproduce them. The question this project ended up answering is: **when that pipeline underperforms, where exactly is the ceiling?**

## The three ceilings

The project's central finding is that correct traces are necessary but not sufficient. Performance is constrained by three distinct ceilings, and progress against one says nothing about the others ([writeup §3](docs/writeup.md#3-the-three-ceilings)):

1. **Solver coverage** — can a correct trace be produced for the problem at all?
2. **Trace learnability** — can the model reproduce the trace token by token under greedy decoding? A trace token that isn't predictable from what came before it is a guaranteed divergence point.
3. **Runtime validity** — does the inference environment preserve the behavior you measured locally?

## Case study: bit manipulation

On the `bit_manipulation` category, our solver produced correct traces for **88.1%** of the dev split — but the trained model reproduced only **33.3%** of gold answers. The data existed; the model couldn't learn it.

A teacher-forced probe (one forward pass over each gold trace, finding the first position where the model's greedy prediction leaves the trace — a technique adopted from tonghuikang's public work) showed that **44 of 47** examined failures diverged at the same place: the line that *declares* the transformation rule. Those tokens weren't derivable from anything before them; the trace simply announced the answer to its own sub-problem.

Redesigning the trace to *derive* the rule before stating it — no solver change, no new data — moved local accuracy from **33.3% to 53.6%**. The full case study, with before/after trace excerpts, is [writeup §5](docs/writeup.md#5-case-study-bit-manipulation).

## Why the Kaggle result did not reflect the local gain

The redesigned adapter scored **0.56** on the public leaderboard against the **0.58** baseline — a local +20pp category gain arrived as a small leaderboard regression.

The investigation that followed found a bigger problem than the gain itself: the **same byte-identical adapter** scored **68.7%** vs **43.4%** on text encryption across two of our own machines — a ~25 percentage-point swing with identical weights, prompts, and settings. The identified difference between the machines is the vLLM version (one run's artifact doesn't record its engine version, so that attribution is inferred, not confirmed). Kaggle reportedly runs a third, older vLLM version that our hardware cannot run at all.

The conclusion, stated carefully: runtime version divergence is *a* dominant contributor to the local-vs-Kaggle gap — not the sole cause, and packaging issues contributed too. But it means a local eval number is a claim about one runtime, not about the model. Details and the environment table: [writeup §6](docs/writeup.md#6-failure-to-translate-and-the-runtime-investigation).

## What survived scrutiny

Several early conclusions turned out to be confounded experiments; catching and correcting them is half the story ([writeup §8](docs/writeup.md#8-confounds-and-corrected-interpretations)). What held up ([writeup §7](docs/writeup.md#7-findings-that-survived-scrutiny)):

- **Trace learnability is a real, separate ceiling** — measurable, diagnosable, and fixable by format redesign.
- **First-divergence probing works** — it located a format failure precisely enough to fix it.
- **Runtime version is a first-class experimental variable** — record it in every eval artifact.
- **`target_modules` must be a list, not a regex,** in the adapter config for Kaggle's vLLM — the one reproducible packaging lever we found (~2pp on the public LB, byte-identical weights).
- **Renaming the adapter's key prefix does nothing** — a suspected lever that a clean A/B showed to be noise.

## Where to go deeper

- [`docs/writeup.md`](docs/writeup.md) — the canonical retrospective: method, case study, runtime investigation, corrections, reproducibility.
- [`RESULTS.md`](RESULTS.md) — every local eval and Kaggle submission, with the packaging ladder.
- [`docs/investigations/OPEN_QUESTIONS.md`](docs/investigations/OPEN_QUESTIONS.md) — the single-variable experiments, negative results, and corrections (C1–C11) behind the writeup.
- Probe scripts: [`scripts/bitmanip_logprob_probe.py`](scripts/bitmanip_logprob_probe.py), [`scripts/text_enc_logprob_probe_vllm.py`](scripts/text_enc_logprob_probe_vllm.py).
- Trace generators and solvers: [`src/data/`](src/data/) (the case study's before/after formats are `bit_manip_trace_v4.py` and `bit_manip_trace_v5.py`).
- MoE adapter converter: [`src/training/convert_peft_to_vllm_moe.py`](src/training/convert_peft_to_vllm_moe.py).

**Publication status:** this repository is a public snapshot of a private working repo, which retains the full investigation history — see the repository-state note in [`docs/writeup.md` §10](docs/writeup.md#10-reproducibility). Third-party provenance and what was removed before publication (competition data, reference material) are documented in [`references/README.md`](references/README.md).

---

## Using this repository

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # fill in credentials
```

### Quick start

```bash
make eval SPLIT=dev_frozen
```

**Data:** competition data is not redistributed here — regenerate the frozen eval splits from Kaggle's `train.csv` via `src/data/split.py` (seed-42, deterministic); see [`datasets/splits/README.md`](datasets/splits/README.md).

### Layout

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

### Task types

| Task type                | Example answer format         |
|--------------------------|-------------------------------|
| bit_manipulation         | binary string, e.g. `10011000` |
| gravitational_constant   | numeric, e.g. `9.81`          |
| unit_conversion          | numeric, e.g. `24.64`         |
| text_encryption          | string, e.g. `khoor`          |
| numeral_conversion       | string/number, e.g. `XLVII`   |
| equation_transformation  | symbolic string, e.g. `a+b`   |

### Eval parameters

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

> **⚠️ Runtime / vLLM-version caveat.** The same byte-identical adapter can score very differently depending on the vLLM version running it. We observed a ~25pp swing on `text_encryption` for one adapter across local vLLM 0.20.1 vs 0.22.1, and Kaggle runs vLLM **0.17.1** (user-reported; distinct from both, not reproducible on our hosts). Consequences: (1) a strong *local* score does not guarantee the Kaggle score — validate against the target runtime; (2) packaging must match what that runtime's LoRA loader expects (see "Packaging an adapter for Kaggle" below). Full diagnosis in [`docs/writeup.md` §6](docs/writeup.md#6-failure-to-translate-and-the-runtime-investigation) and `docs/investigations/` (corrections C7, C9–C11).

The harness appends to every prompt:

```
Please put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`
```

### Submission format

A LoRA adapter (rank ≤ 32) for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, packaged as `submission.zip` containing `adapter_config.json` and adapter weights. Build with:

```bash
make package ADAPTER=runs/train/lora_baseline
```

### Packaging an adapter for Kaggle (convert → pad → rekey)

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

Then zip the three files — `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja` — **flat at the archive root**, under the 1.5 GB submission cap. Full packaging investigation (mixed-rank mis-load, target_modules format, prefix) in `docs/investigations/` (corrections C8–C11) and [`docs/investigations/OPEN_QUESTIONS.md`](docs/investigations/OPEN_QUESTIONS.md).
