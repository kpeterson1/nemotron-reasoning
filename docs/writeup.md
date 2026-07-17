# Learnability Is Necessary, Not Sufficient: A Confound-Catching Retrospective on the NVIDIA Nemotron Reasoning Challenge

**NVIDIA Nemotron Model Reasoning Challenge (Kaggle, closed June 15, 2026)**
Public LB: 0.58 · Private LB (final): 0.604 · Rank: 3488/4182 · Not prize-eligible

---

## 1. Introduction

The NVIDIA Nemotron Model Reasoning Challenge asked competitors to fine-tune a LoRA adapter (rank ≤ 32) on Nemotron-3-Nano-30B-A3B-BF16, a hybrid Mamba2/Mixture-of-Experts architecture with 30 billion total parameters and ~3.5 billion active per token. Evaluation was greedy decoding (temperature=0.0) across six reasoning categories. The winners finished around 0.92 on the private leaderboard; the 0.86–0.87 cluster we tracked during the competition was the public-LB pack.

Our best public-leaderboard score was 0.58; the final private-leaderboard result was 0.604 at rank 3488/4182 — the private split held slightly above the public one. The two are different metrics over different held-out splits, so every per-submission score quoted in this writeup (the 0.55–0.58 band) is a **public-LB** value unless explicitly labeled otherwise. We did not place in the top 10%, and we are not eligible for the Open Contribution Awards. This writeup is a retrospective — a public accounting of what we found, what we got wrong, and what turned out to be the dominant constraint. It is written for two audiences: the Nemotron community (particularly around vLLM version sensitivity, which we believe is under-discussed), and anyone else running SFT on reasoning traces under greedy decoding.

The retrospective centers on three findings:

- **A diagnostic method that worked locally.** Teacher-forced logprob probing of gold traces identified format-level learnability failures at the token position where the model's greedy prediction diverges from the trace. Applied to `bit_manipulation`, this diagnosis motivated a trace redesign that produced **a +20pp local accuracy gain** (33.3% → 53.6% on the submitted mix) — validating the method under our reference eval environment.
- **A runtime wall that ate the gain.** The local improvements did not translate to Kaggle. The controlled investigation that followed identified vLLM version divergence as a dominant contributor to our local-vs-Kaggle gap. Two of our own machines, same weights, same greedy settings, produced a ~25pp swing on `text_encryption` accuracy purely from the vLLM version. **We are explicitly reversing an earlier claim** — made in a preliminary draft of this writeup — that the gap "was never a runtime divergence problem." It was.
- **A confound-catching methodology.** Several of our headline conclusions turned out to be confounded. A "load-bearing bug" story about the MoE converter reshape fix was invalidated by a subsequent packaging discovery. A text_encryption learnability diagnosis was invalidated when the motivating accuracy number turned out to be a vLLM version artifact. The single-variable discipline that produced each finding was also the discipline that overturned it. That reversal pattern is, in our view, the more general contribution.

We present the negative results without apology. The competition score reflects where we ended, not what the investigation was worth.

---

## 2. Premise: The Learnability Gap

Our approach followed the competition's dominant paradigm: deterministic Python solvers generate verified chain-of-thought traces, and the model is SFT-trained to reproduce them. This worked well for four of the six categories. For two — `bit_manipulation` and (we initially thought) `text_encryption` — it did not.

The pattern was striking. On `bit_manipulation`, our solver correctly produces traces for 88.1% (74/84) of the dev set's bit_manipulation split. The trained model reproduces only 33.3% of gold answers under greedy decoding. That is a 55pp gap between what our data contains and what the model learns. A naive reading blames solver coverage; the actual issue is downstream.

The full per-category picture, from the canonical local eval of the v9 baseline (dev_frozen n=500, greedy, vLLM 0.20.1 — Kaggle provides no per-category output, so no leaderboard equivalent exists):

| Category | Solver coverage | v9 model accuracy (local) | Gap |
|---|---|---|---|
| gravitational_constant | n/a (model at 100%) | 100% | 0pp |
| numeral_conversion | n/a (model at 100%) | 100% | 0pp |
| unit_conversion | n/a (model at 100%) | 100% | 0pp |
| text_encryption | 100% (83/83) | 68.7% | ~31pp |
| bit_manipulation | 88.1% (74/84) | 33.3% | ~55pp |
| eq_transformation | 28.6% (24/84) | 15.5% | ~13pp |

An earlier draft of this table showed `text_encryption` at 43.4% (a ~57pp gap). That number came from the artifact eval run the C7 investigation (§5) later explained: the canonical accuracy for the same weights is 68.7%. At the time, the picture read as two categories hitting the same learnability wall; post-C7, only `bit_manipulation`'s gap survives as a learnability finding (§6b covers the text_encryption phantom). `eq_transformation`'s gap is bounded by solver coverage — the solver, not learnability, is the ceiling there.

The question is: **why does a model trained on verified correct traces fail to reproduce them?** Under greedy decoding, the model must commit to exactly one token at every position. If a token in the gold trace is not the argmax of the model's distribution — because nothing in the preceding context determines it — the trace diverges, and the remainder is unrecoverable. Correctness of the trace is a prerequisite; **reproducibility under greedy decoding is the binding constraint**.

Solver coverage measures whether the trace exists. Learnability measures whether the model can produce it.

---

## 3. Method: Teacher-Forced Logprob Probing

The diagnostic instrument is straightforward. Given a gold trace, we run a single forward pass through the model with the trace as the continuation, extract per-token log-probabilities and ranks, and identify the first position where the ground-truth token is no longer the model's argmax (rank > 1). At the divergence points we observed, the gold token's log-probability sat between −0.7 and −2.3 — near-ties that the model resolves in favor of a different token. Aggregating first-divergence positions across failing examples reveals whether failures cluster at a specific trace *region* (a format problem) or scatter randomly (a capacity problem).

The technique is not novel — we adopted it from tonghuikang's public work — but the way we applied it is worth naming clearly, because it distinguishes what we did from what he did:

- **Our usage:** post-hoc diagnostic. Run the probe on a failing category, identify the region where the model can't predict, redesign the trace format, retrain.
- **Huikang's usage:** continuous pre-submission gate. Run the probe after every training run, on every trace in the training set, and only submit when the worst traces cleared the threshold.

His workflow closed the loop *inside* training. Ours opened the loop long enough to diagnose a specific failure, then closed it manually with a format change. Both are valid; his produces higher signal per submission. We adopted the technique too late in the competition to convert it into the tight loop, and that gap is our clearest retrospective regret.

---

## 4. Local Validation: +20pp on Bit_Manipulation

The `bit_manipulation` category asks the model to apply bitwise operations (AND, OR, XOR, NOT) to binary strings. Our v4 trace format used a compact rule statement — e.g. `bit 0: A[7] XOR B[7]` — replacing a prior derivation-grid format (v3) that spent an 8-row per-bit verification grid deriving each bit before stating it. The v3→v4 change had shipped as an improvement: it cut trace length sharply (v4 traces max out at ~377 tokens) and eliminated truncation.

Teacher-forced probing showed why v4 was worse than it looked. Of 47 failures examined, 44 diverged at the same region: the RULE_STATEMENT tokens declaring per-bit operator and operand indices. These tokens are not derivable from the preceding context — the model has no basis to predict *which* operator or *which* index comes next. The v4 format was measurably shorter and measurably less learnable; its earlier win came from eliminating truncation, not from being easier to learn. Truncation had been the binding constraint until the traces got short enough to expose learnability as the next one.

The v5 redesign inverts the structure: derivation-first, where every token in the rule statement is preceded by the computation that produces it. The bit index, the operator, and the operand mapping are all derived from prior context before being stated. No token requires the model to guess.

On our canonical local eval (Spark 1, vLLM 0.20.1, dev_frozen n=500, greedy), `bit_manipulation` accuracy moved from **33.3% → 53.6%** with the v5 traces (v13, the submitted mix; the intermediate v12 mix reached 54.8%). A gain of roughly +20pp on the target category. The method worked.

Then we submitted it.

---

## 5. The Failure to Translate — and What It Revealed (C7)

The v13 adapter — v9 with only bit_manip swapped to the shortened v5 traces, everything else byte-identical — scored 0.56 on the Kaggle public LB. Our v9 baseline scored 0.58. A +20pp local category gain translated to a *regression* on the leaderboard.

This was the point at which we should have concluded, and did initially conclude, that the local-vs-Kaggle gap could not be a runtime problem. Local eval and Kaggle eval were pointing in opposite directions on the same intervention. Our preliminary writeup stated this explicitly: *"the local-Kaggle gap was never a runtime divergence problem."*

That conclusion was wrong. We are reversing it here.

The investigation that overturned it began with a parity check on `text_encryption`. Our Spark 1 canonical eval showed v9 text_encryption at 68.7%. A second run of the same v9 adapter showed 43.4%. Same weights (byte-identical, `cmp`-verified), same prompts, 0% truncation in both runs, same greedy settings. Twenty-six problems flipped between the two runs, with raw outputs starting identical and diverging mid-generation.

The 43.4% run's artifact does not record its engine version; we attribute it to Spark 2 (vLLM 0.22.1) **by inference** — our corrections log marks this attribution explicitly as inferred, not artifact-confirmed. Under that attribution, the two machines differ in one variable that mattered: vLLM version.

| Environment | vLLM version | Text_encryption accuracy on v9 |
|---|---|---|
| Spark 1 (canonical) | 0.20.1 (confirmed from engine logs) | 68.7% |
| Spark 2 | 0.22.1 (attribution of the 43.4% run inferred — artifact records no version) | 43.4% |
| Kaggle | 0.17.1 (user-reported from a competitor notebook; not independently verifiable) | unknown (no per-category output) |

The ~25pp swing is a **local-vs-local** finding, on two machines we controlled, with identical weights. Kaggle's 0.17.1 is a third point further out. Long traces (text_encryption's typical output shape) appear to amplify kernel-level greedy divergence into large accuracy swings — >50× the ±0.5pp noise floor we had measured for greedy on 0.20.1.

The implication for the bit_manip validation is uncomfortable. Our +20pp local gain was measured on Spark 1 (vLLM 0.20.1). Kaggle reportedly runs vLLM 0.17.1. **We have no way to know what the same weights would have scored under Kaggle's runtime**, and the Nemotron-DGX-Spark hardware combination cannot run vLLM 0.17.1 (CUDA driver requirements). The runtime is a structural constraint of the competition, not something we could have engineered around locally.

To be precise about what we're claiming: vLLM version divergence is *a* dominant contributor to the local-vs-Kaggle gap, not the sole cause. Packaging (§7), stage-2 recipe differences, and other factors also contribute. But the ~25pp same-weights-different-runtime swing establishes that the runtime is *material* — enough that any local optimization landing in the same numerical range should be treated as unverified on Kaggle until submitted.

---

## 6. Confounds Caught

The methodological through-line of the competition was, in retrospect, catching our own false conclusions. Three cases stand out.

### 6a. The load-bearing converter bug (C2 / C11)

Early in the competition we discovered a real bug in our PEFT→vLLM MoE weight converter: `lora_B` expert tensors were reshaped assuming PEFT packed them as `(out, r, E)` when the actual packing is `(out, E, r)`. This scrambled per-expert deltas across all 23 MoE layers × 128 experts. A two-character fix corrected it, and round-trip verification confirmed the bug was real.

The mechanics, for reproducers: PEFT stores the expert LoRA weights as packed 3D `nn.Parameter` tensors (`lora_A`: `(num_experts, rank, in_features)`, `lora_B`: `(out_features, num_experts, rank)`), which the converter must unpack into per-expert 2D matrices for vLLM. A correct conversion produces **12,008 tensors** from the adapter; we validated this count and spot-checked individual expert deltas via round-trip extraction after every conversion.

When we submitted the fix to Kaggle, the public-LB score dropped from ~0.58 to 0.55. We concluded — plausibly at the time — that the scrambled deltas had been acting as accidental stochastic regularization, and that the correctly-assigned deltas contributed nothing. This became a "load-bearing bug" narrative in an earlier draft of this writeup.

The conclusion was itself a confound. The B-buggy and B-fixed submissions did not differ only in the reshape; they also differed in `target_modules` format (regex vs list). On vLLM 0.17.1, regex-mode `target_modules` silently causes partial LoRA application — a different subset of layers is active in each submission. The 0.58 → 0.55 drop was measuring the packaging difference, not the reshape fix. **The true Kaggle effect of correctly assigning expert deltas remains unmeasured.**

The reshape bug was real. The story we told about its consequences was not.

### 6b. The text_encryption phantom

Independently, we had diagnosed a suspected learnability failure in `text_encryption`. On the low-scoring eval frame (the 43.4% run, Spark 2-attributed — see §5), the model appeared to build a correct cipher-to-plaintext map and then ignore it during decryption, generating fluent English instead of doing mechanical substitution. To structurally prevent the LM prior from taking over, we adopted a character-by-character trace redesign — interleaving lookup and emit for each character — following tonghuikang's cipher trace design: his reference reasoner emits a per-character `cipher→plain` lookup step immediately before each decoded word.

The forensic detail behind that diagnosis — gathered before C7 explained it — is worth preserving, because the observations were real even though the conclusion drawn from them was not. (These counts come from an uncommitted, contemporaneous CPU analysis; no committed artifact backs them, though the arithmetic reconciles with the run: 43.4% = 36/83 correct, leaving 47 misses.) Decomposing the 47 misses: 26 of 47 differed from the gold answer by exactly one plausible English word (`treasure` decrypted as `studies`, `dreams` as `reads`, `curious` as `crystal` — cryptographically wrong, defensible as English continuations). For 0 of 47 misses did the emitted plaintext equal the result of mechanically applying the model's own cipher map; on correct predictions, the mechanical-map match held 81% of the time (29/36). Solver coverage for the category was 100% (83/83), with traces of at most 536 tokens (min 384 / median 470; re-audited 2026-07-09 via `scripts/audit_solver_coverage.py --tasks text_encryption`, CPU-only) — so the gap could not be blamed on data. At the time, we read all of this as the LM prior hijacking mechanical substitution at the decryption step.

What those numbers actually characterize is the failure mode of the **artifact eval run** (Spark 2 / vLLM 0.22.1 by inference) — the miss set they were computed on was substantially manufactured by the eval environment. They describe how that runtime's divergences look when they land (one plausible English word off), not a live learnability wall in the trained model.

On the canonical Spark 1 frame, the redesign did not help — and the full chain matters, because the drop from 68.7% was not single-variable. v9's 68.7% fell to 56.6% in v11, which changed *only* the bit_manipulation traces (cross-task interference through the shared LoRA — no text_encryption change at all). v12, which added the char-by-char redesign bundled with a shortened bit_manip format (a deliberate two-variable step), fell further to 53.0%. v13, which reverted char-by-char, recovered only to 54.2%. The cost isolable to char-by-char is at most ~3.6pp; most of the 68.7% → 53% drop was interference from the longer bit_manip traces, not the text_encryption redesign itself.

The deeper problem: the motivating 43.4% number was itself the vLLM-version artifact (the C7 finding above). The canonical Spark 1 baseline was already 68.7% — much closer to the solver coverage ceiling than we had believed. There was no learnability wall to break through; there was an eval environment producing misleadingly bad numbers on a category with long output traces. The redesign was chasing a phantom.

### 6c. The backbone prefix that wasn't (C9 / C10)

Two variants of the same v9 adapter — one with `backbone.` prefix, one with `model.model.` prefix, both with list-format `target_modules` — differed by ~1pp on the Kaggle public LB (0.58 vs 0.57). We initially attributed this to the prefix rename. Ablating cleanly showed the prefix rename is noise near our submission-level floor. The real variable was elsewhere.

The common thread across all three: single-variable discipline is what caught the confound *and* what invalidated the earlier conclusion drawn from confounded data. The method is symmetric. You do not get to keep the findings when the discipline that produced them also overturns them.

---

## 7. What Held (C9 / C10)

Not everything was a confound. The real ~2pp Kaggle lever we identified is:

- **`target_modules` must be a list, not a regex.** On vLLM 0.17.1, regex-format `target_modules` in the adapter config silently causes partial LoRA application. Converting to explicit list format restored the missing layers and produced a clean ~2pp gain on the Kaggle public LB.
- **Backbone prefix rename is noise.** The difference between `model.model.` and `backbone.` prefixes is within our submission-level noise floor (~1pp) and is not a reproducible lever.

Both of these apply specifically to vLLM 0.17.1. They may or may not matter on newer versions. And both deltas — like every packaging A/B in this writeup — were measured on the public leaderboard; a public-LB delta does not necessarily predict private-LB standing.

---

## 8. Takeaways

**Test trace learnability before scaling data.** Solver coverage measures whether your data exists; teacher-forced probing measures whether the model can reproduce it. A 55pp gap between them means the trace format is fighting greedy decoding. This is the single most useful thing we learned, and it generalizes to any SFT pipeline evaluated under deterministic generation.

**Adopt the probe as a loop, not a diagnostic.** Huikang used min-logprob inspection after every training run as a pre-submission gate. We used it as a post-hoc diagnostic on failing categories. His loop produces higher signal per submission and would have changed our trajectory if we had adopted it earlier. If you take one thing from this writeup, take that.

**vLLM version is a first-class variable.** Two of our own machines, same weights, same greedy settings, produced a ~25pp swing on `text_encryption` accuracy — attributable to vLLM version under the inference documented in §5. Long-output categories appear disproportionately affected. Any SFT pipeline where local and production inference environments differ by even a minor vLLM version should treat their eval numbers as unverified across environments. We suspect this is broadly under-documented and would be worth surfacing in the NVIDIA Nemotron community.

**Coherent stories are not necessarily true stories.** Our "load-bearing bug" narrative was coherent, publishable, and wrong. Our "text_encryption learnability wall" was coherent, actionable, and chasing a phantom. Both were overturned by applying, retroactively, the same single-variable discipline that produced them. **The discipline that catches confounds is the discipline that catches you drawing false conclusions from confounded data.** That symmetry is uncomfortable but load-bearing.

**Honest negative results are part of the contribution.** The competition score reflects where we ended, not what the investigation was worth. We finished at 0.58 public / 0.604 private (rank 3488/4182), outside the top 10%, and the trace-learnability method that worked locally did not translate to Kaggle. Documenting why is worth more than optimizing for another 0.02 in retrospect.

---

## 9. Reproducibility

**Repository:** `github.com/kpeterson1/nemotron-reasoning` (public with this writeup)

**Repository state:** This public repository is published as a single snapshot of the final tree — the code, traces, and diagnostics behind the 0.58 public-LB / 0.604 private-LB result. The full investigation history (per-version ablations, the reversal trail, and single-variable branches described in this writeup) is retained privately; the reasoning and outcomes are documented in §5–§6 and in `docs/investigations/`.

**Key scripts:**
- `src/training/convert_peft_to_vllm_moe.py` — MoE adapter conversion pipeline. The shipped version applies the correct `(out, E, r)` expert-tensor reshape; the earlier transposed-reshape bug and its fix are described in §6a.
- `src/data/` — per-category deterministic solvers and trace generators
- `src/data/split.py` — regenerates the evaluation splits (dev_frozen n=500, seed-42 stratified) from Kaggle's `train.csv`; competition data is not redistributed with this repository (see `datasets/splits/README.md`)
- `src/evaluation/run_eval.py` — vLLM inference harness (greedy, max_tokens=7680, max_model_len=8192)
- `scripts/bitmanip_logprob_probe.py`, `scripts/text_enc_logprob_probe_vllm.py` — teacher-forced logprob probes

**Evaluation parameters (matching Kaggle runtime as closely as possible):**

| Parameter | Value |
|---|---|
| max_lora_rank | 32 |
| max_tokens | 7680 |
| top_p | 1.0 |
| temperature | 0.0 |
| max_model_len | 8192 |

**Compute:** Two NVIDIA DGX Spark nodes (GB10 Grace Blackwell, 128 GB unified memory). Spark 1 ran vLLM 0.20.1 (canonical local eval); Spark 2 ran vLLM 0.22.1 (used to surface C7). Neither can run vLLM 0.17.1 due to CUDA driver requirements — Kaggle-runtime reproduction was not possible on our hardware.

**Session logs and corrections log:** `docs/investigations/` and `docs/SESSION_LOG.md` preserve the trail of findings, reversals, and single-variable ablations described in this writeup.

**Acknowledgments:** tonghuikang's public writeup and midpoint Open Prize–winning repository were the structural benchmark for this work. The min-logprob diagnostic technique and the character-by-character cipher decode are his contributions; our application of them to diagnose format-level learnability and — separately — to name our own confounds is original, though we adopted the technique too late to close the loop before the deadline. The first-place team's writeup is cited in `references/README.md` as corroboration of the evaluation parameters. Provenance for the reference copies of tonghuikang's implementation we worked from is documented in `references/README.md`.
