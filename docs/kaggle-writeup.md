# Solution Writeup (Solo Team): Deterministic Solvers, a Trace-Learnability Probe, and a Runtime Trap

**Final result: public LB 0.58 · private LB 0.604 · rank 3488/4182.** This is not a prize writeup — it's a methodology post. What I think is worth sharing is a diagnostic loop that located exactly *where* a trained model stopped being able to follow its own training traces, a trace redesign that recovered +20.3pp locally on the strength of that diagnosis, and the runtime trap that likely explains much of why the gain didn't transfer to the leaderboard.

The approach: deterministic solvers per category → verified reasoning traces → LoRA SFT (rank 32) → local greedy evaluation → probe failures → redesign traces → retrain → submit.

A few evaluation conventions used throughout, so the numbers below are interpretable:

- **Canonical local frame:** DGX Spark ("Spark 1"), vLLM 0.20.1, a frozen dev split (n=500, seed-42 stratified from `train.csv`), greedy decoding. Every local number is on this frame unless labeled otherwise.
- Local greedy eval carries a measured **±0.5pp run-to-run noise floor** (greedy on this architecture is not strictly deterministic across batch/context-length configs); I treat any sub-1pp delta as meaningless.
- All per-submission scores quoted are **public-LB** values. Public-LB deltas do not necessarily predict private standing.

---

## 1. The gap that started it

On `bit_manipulation`, my solver produced correct, verified traces for **88.1%** of dev problems (74/84). The model trained on those traces reproduced the answer on **33.3%**.

That ~55pp gap is the finding that shaped the rest of the project. The data existed and was correct; the model could not reproduce it. Solver coverage and model accuracy turned out to be two different ceilings, and the second one — whether a trace is *learnable* under greedy decoding — was the binding one for this category.

## 2. Finding where the trace stopped being learnable

To find out *where* generation left the reference trace — the solver's verified trace the model was trained to reproduce — I used a teacher-forced first-divergence probe (adapted from tonghuikang's public min-logprob diagnostic):

1. Teacher-force the reference trace through the model — one forward pass, no generation.
2. Extract per-token log-probabilities and ranks for every reference token.
3. Find the first position where the reference token is no longer the argmax. Under greedy decoding, that position is exactly where generation will diverge.
4. Aggregate those positions across a category's failures. Clustered divergences → format problem (cheap fix); scattered → capacity or solver problem.

The result was about as clean as these things get: **44 of 47 examined failures first diverged at the same region** — the rule-statement line, where the trace declares which per-bit operator and operand indices constitute the rule. At those points the reference token's log-probability sat between −0.7 and −2.3: near-ties the model resolved in favor of a different token.

The mechanism is obvious in hindsight. Nothing in the preceding context *determines* those tokens — the trace simply announces the rule. The highest-information tokens in the trace were the least predictable ones.

## 3. Redesigning the line: derive before you declare

The fix changed no solver logic and added no data — only the trace format. The baseline (v4) format states the rule in one breath (this and the next excerpt are produced by the repo's trace generators on a **synthetic** example — no competition data — in the identical format as the shipped training traces):

```
After trying candidate operations, the rule that matches every example is:
output[i] = input[i-1] XOR input[i+3] (zero-pad edges).
  check: rule(10100101) = 01111010 → expected 01111010 ✓
```

The redesigned (v5) format derives every component of the rule before stating it:

```
Bit 1: target O1=1101000110.
  No input column or its complement equals T, and T is not constant -> not unary.
  AND: no operand pair reproduces T.
  OR: no operand pair reproduces T.
  XOR: the partner is forced by Ij=T XOR Ii. Scan first operand:
    i=0: T XOR I0=0111001110 = I4 -> XOR(I0,I4)=1101000110=T. accepted.
  => out[1] = XOR(input[0], input[4]).
```

In v5, the bit index, operator, and operand mapping are all computed in-trace before being declared. No token requires the model to guess.

On the canonical local eval, `bit_manipulation` moved from **33.3% → 53.6% (+20.3pp)** in the submitted mix (v13); an intermediate mix (v12) reached 54.8%. Same solver, same problems, format-only change — precisely targeted by the probe.

## 4. The turn: it didn't transfer

The v13 adapter — the baseline (v9) with only `bit_manipulation` swapped to the redesigned traces, packaged identically to my best submission — scored **0.56** public-LB. The v9 baseline scored **0.58**. A +20pp local category gain arrived on the leaderboard as a regression.

Chasing why led to the most consequential measurement of the project: a **same-weights comparison** on `text_encryption`. The canonical Spark 1 eval showed the v9 adapter at **68.7%**; a second run of the *byte-identical* (`cmp`-verified) adapter showed **43.4%**. Same prompts, same greedy settings, 0% truncation in both runs. Twenty-six problems flipped, with raw outputs starting identical and diverging mid-generation.

| Environment | vLLM version | text_encryption on v9 |
|---|---|---|
| Spark 1 (canonical) | 0.20.1 (confirmed from engine logs) | 68.7% |
| Spark 2 | 0.22.1 — the 43.4% run's attribution to this host is **inferred** from the machine and session trail, not artifact-confirmed (the artifact records no engine version) | 43.4% |
| Kaggle | 0.17.1 — **user-reported** from a competitor notebook, not independently verifiable | unknown (no per-category output) |

That ~25pp swing is a local-vs-local finding on two machines I controlled, with identical weights — more than 50× the ±0.5pp noise floor. Long output traces (text_encryption's typical shape) appear to amplify kernel-level greedy divergence into large accuracy swings.

Two hedges belong right next to this result. First, my hardware cannot run vLLM 0.17.1 at all (CUDA driver requirements), so I have no way to know what the same weights would have scored under Kaggle's runtime — the gap is unmeasurable from my side, not merely unmeasured. Second, runtime version divergence is *a dominant contributor* to the local-vs-Kaggle gap, **not the sole cause**: packaging and recipe differences also contribute, and the ≈11pp residual between my best submission's 0.58 and the local overall 0.694 is attributed to the same runtime divergence by inference, not direct measurement.

What the same-weights swing does establish is enough: runtime differences were material enough to overwhelm ordinary adapter-level deltas. A local gain is environment-specific until tested under deployment parity.

## 5. Conclusions I reached and then retracted

This is the section I most wish someone had written for me before the competition. Three interpretations that each looked solid failed a later single-variable check:

| What I initially concluded | The confound | What actually survived |
|---|---|---|
| **"The bug was load-bearing."** Fixing a transposed expert-tensor reshape in my MoE adapter converter dropped the public LB from ~0.58 to 0.55 — so the scrambled deltas must have been *helping* (accidental regularization). | The buggy and fixed submissions also differed in `target_modules` format (regex string vs explicit list). The A/B was measuring packaging, not the reshape. | The reshape fix is mathematically correct (round-trip verified). **Its true Kaggle effect through correct packaging remains untested.** |
| **"text_encryption hit a learnability wall."** On a 43.4% eval frame, the model appeared to build a correct cipher map and then ignore it, so I adopted a char-by-char trace redesign to structurally prevent that. | The motivating 43.4% was the runtime artifact of §4 — the canonical baseline was already 68.7%, near the category's solver ceiling. And the redesign shipped bundled with the bit_manipulation trace change (a **deliberate two-variable step**, v12), so its effect was never isolated. | There was no wall. The drop from 68.7% was mostly cross-task interference from the longer bit_manipulation traces: 68.7 (v9) → 56.6 (bit_manip-only change, v11) → 53.0 (two-variable v12) → 54.2 (redesign reverted, v13 — did not recover). At most ~3.6pp is isolable to the char-by-char redesign. |
| **"The checkpoint key prefix is load-bearing."** A submission with renamed key prefixes scored 0.55 vs 0.58 on byte-identical weights. | That submission changed two fields: the key prefix *and* the `target_modules` format. | A clean prefix-only A/B (byte-identical weights, per-tensor SHA verified) shows the prefix is neutral-to-slightly-negative — noise at the submission-level floor. The real, reproducible lever was `target_modules` as an explicit **list** instead of a regex: a clean ~2pp public-LB gain (0.55 → 0.57) on byte-identical weights. The proposed mechanism (the runtime silently under-applying a regex) is inferred, not confirmed locally on that runtime version. |

The through-line: in every case, score movement was not evidence of mechanism until the changed variables were isolated. All three confounds were two-variable experiments wearing single-variable conclusions, and single-variable discipline — applied retroactively, with byte-identical weights and per-tensor hashes — is what caught them.

## 6. What transferred

The takeaways I'd actually carry into another competition of this shape:

- **A correct trace can be unlearnable under greedy decoding.** Measure solver coverage and model accuracy separately; a gap between them is a learnability signal, not a data-volume signal. Tokens the model can't predict from prior context are divergence points waiting to happen — derive consequential tokens in-trace before declaring them.
- **Runtime parity is a prerequisite for trusting local gains.** Byte-identical weights swung ~25pp across two inference-engine versions I controlled. Pin the engine version before extrapolating local → leaderboard; if you can't run the deployment runtime locally, treat the gap as unmeasurable rather than assuming it away. Record the engine version in every eval artifact (the single most consequential missing field in mine — it turned a clean comparison into an inferred attribution) and store raw generations, not just scores (the mid-generation flip pattern was only visible because raw outputs were kept).
- **Evaluate per category, because a shared adapter trades skills.** The leaderboard exposes one overall number, but the longer bit_manipulation traces cost text_encryption accuracy through cross-task interference in the shared adapter. Without per-category local eval, that trade is invisible.
- **What I'd do differently, stated plainly: gate, don't autopsy.** I used the first-divergence probe post hoc — diagnose one failing category, redesign, retrain. The stronger use is as an in-loop gate: run the probe after every training run, on the training traces themselves, and only spend a submission when the worst traces clear a threshold. I adopted the technique too late to close that loop before the deadline.

## 7. Where the ceiling actually was

One honest paragraph: trace redesign only helps where learnability is the ceiling, and my weakest category's ceiling was elsewhere. Equation transformation was **solver-coverage-bound** — my solver produced verified traces for only 28.6% of dev problems (24/84), and the model sat at 15.5%. No amount of trace-format work moves a category whose ceiling is the data pipeline itself. Three categories were fully solved locally (100%), one was learnability-bound (bit_manipulation), one was runtime-confounded (text_encryption) — and this one was simply out of solver reach.

## Links

- **Repo** (solvers, trace generators, probes, full retrospective): [github.com/kpeterson1/nemotron-reasoning](https://github.com/kpeterson1/nemotron-reasoning) — the full-length writeup with appendices is `docs/writeup.md`.
- **Live explainer:** [nemotron-reasoning.pages.dev](https://nemotron-reasoning.pages.dev/)
- **3D architecture explorer** (the base model's hybrid Mamba2/MoE stack, browsable): [nemotron-reasoning.pages.dev/viz/nemotron-explorer](https://nemotron-reasoning.pages.dev/viz/nemotron-explorer)

Acknowledgment: the min-logprob diagnostic and the char-by-char cipher-decode trace design are tonghuikang's contributions, adopted from his public work; the application of them here — to localize format-level learnability failures and, separately, to name my own confounds — is my own.
