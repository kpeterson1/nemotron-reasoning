# Nemotron Reasoning Challenge — Diagnosing Reasoning-SFT Bottlenecks

A case study in finding out *why* a fine-tuned reasoning model fails. The central finding: **correct training data and learnable training data are different ceilings.** On one category, our solver produced correct reasoning traces for 88.1% of problems while the trained model reproduced only 33.3% of gold answers; a teacher-forced probe traced 44 of 47 examined failures to a single line of the trace, and redesigning that line — no solver change, no new data — lifted local accuracy to 53.6%. Entered in the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) (Kaggle, closed June 15, 2026); final result: public LB **0.58** · private LB **0.604** · rank **3488/4182**.

> **What this is:** a case study in debugging deterministic reasoning SFT; a record of controlled trace, packaging, and runtime experiments; reusable tooling for locating token-level failure points in a trace. It is a documented investigation of the bottlenecks that determine whether reasoning SFT works, with the strongest experiment being a complete diagnosis → intervention → measurement loop on one failing category.
>
> **What this is not:** a winning competition recipe; evidence that local gains automatically transfer to Kaggle; a claim that one trace format solves every reasoning category.

**The canonical detailed account is [`docs/writeup.md`](docs/writeup.md).** This README is the short version.

A web explainer of this work is live at [nemotron-reasoning.pages.dev](https://nemotron-reasoning.pages.dev/), including an interactive [3D architecture explorer](https://nemotron-reasoning.pages.dev/viz/nemotron-explorer) of the Nemotron-3-Nano MoE/Mamba stack — both built from this repository's [`web/`](web/) directory.

## Choose your path

- **Read in 5 minutes** — this README: [the three ceilings](#the-three-ceilings) and [the case study](#case-study-bit-manipulation).
- **Full evidence trail** — [`docs/writeup.md`](docs/writeup.md): method, case study, runtime investigation, confounds and corrections.
- **Reproduce** — [`docs/REPRODUCING.md`](docs/REPRODUCING.md): environment, local eval, the first-divergence probe quickstart, Kaggle packaging.
- **Explore the model** — the interactive [3D architecture explorer](https://nemotron-reasoning.pages.dev/viz/nemotron-explorer) of the Nemotron-3-Nano MoE/Mamba stack.

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
- [`docs/REPRODUCING.md`](docs/REPRODUCING.md) — the operational guide: environment, eval parity, the probe quickstart, adapter packaging.
- [`RESULTS.md`](RESULTS.md) — every local eval and Kaggle submission, with the packaging ladder.
- [`docs/investigations/OPEN_QUESTIONS.md`](docs/investigations/OPEN_QUESTIONS.md) — the single-variable experiments, negative results, and corrections (C1–C11) behind the writeup.
- Probe scripts: [`scripts/bitmanip_logprob_probe.py`](scripts/bitmanip_logprob_probe.py), [`scripts/text_enc_logprob_probe_vllm.py`](scripts/text_enc_logprob_probe_vllm.py).
- Trace generators and solvers: [`src/data/`](src/data/) (the case study's before/after formats are `bit_manip_trace_v4.py` and `bit_manip_trace_v5.py`).
- MoE adapter converter: [`src/training/convert_peft_to_vllm_moe.py`](src/training/convert_peft_to_vllm_moe.py).

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make eval SPLIT=dev_frozen
```

That's the local eval on its defaults. The full operational guide — credentials/data setup, eval parameters that match the Kaggle harness (the local defaults do **not**), the first-divergence probe quickstart, and adapter packaging (convert → pad → rekey) with its runtime caveats — is [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

**Publication status:** this repository is a public snapshot of a private working repo, which retains the full investigation history — see the repository-state note in [`docs/writeup.md` §10](docs/writeup.md#10-reproducibility). Third-party provenance and what was removed before publication (competition data, reference material) are documented in [`references/README.md`](references/README.md).
