# Results

Consolidated results for the NVIDIA Nemotron Reasoning Challenge work. Raw eval
JSONs (`runs/eval/*.json`) and trained adapters are gitignored (large / derived
from competition data); this file is the committed summary. Full reasoning is in
[`docs/investigations/OPEN_QUESTIONS.md`](docs/investigations/OPEN_QUESTIONS.md)
and `docs/investigations/kaggle_lora_mismatch/`.

> **Local ≠ Kaggle.** Local numbers are vLLM **0.20.1**, greedy (temp 0.0),
> `dev_frozen` (n=500). Kaggle runs vLLM **0.17.1** (not reproducible on our
> hosts) — the same adapter scores differently across runtimes (correction C7).
> Treat sub-1pp deltas as noise (±0.5pp greedy non-determinism, C5).

## Legend / glossary

Definitions pulled from `docs/SESSION_LOG.md`, `docs/investigations/OPEN_QUESTIONS.md`
(corrections C1–C11), and the `kaggle_lora_mismatch/` reports. Full correction
text lives in OPEN_QUESTIONS.md; one-liners here.

**Adapter versions (each is single-variable vs the one before unless noted):**
- **v9** — baseline adapter/dataset; bit_manip uses the compact "v4" trace format. The reference point every row compares against.
- **v11** (a.k.a. v11bm) — v9 with *only* bit_manip retraced in the "v5" derivation-first format.
- **v12** — v11 + shortened v5 traces ("Cut 1") + a char-by-char text_enc redesign.
- **v13** — shortened-v5 bit_manip + text_enc **reverted to v9's format** (drops char-by-char); isolates the bit_manip gain.
- **v13+rank** — v13 with the MoE expert LoRA rank raised 8→32 (too large to submit: 3.0 GiB > 1.5 GB cap).

**Trace formats:**
- **v4 / v5 (bit_manip traces)** — v4 *states* the per-bit rule; v5 *derives* every operator/bit-index from a prior self-verifying line.
- **Cut 1** — the shortened v5 variant (fewer derivation lines).
- **char-by-char (text_enc)** — a text_enc trace redesign that decodes one output character at a time from an explicit cipher→plain map.

**Named Kaggle submissions (mostly the *same v9 weights*, different packaging;
all per-submission scores are public-LB values):**
- **Adapter A** — v9, experts padded r8→r32, `backbone` key prefix, `target_modules` as a list. Best public LB = 0.58.
- **the squeeze** — Adapter A's weights but `model.model` prefix + **regex** target_modules. 0.55.
- **B_FIXED** — adapter rebuilt with the corrected expert-B reshape. 0.55.

**Packaging shorthand:**
- **dev_frozen** — the local held-out eval split (n=500). Local v9 ≈ **0.686–0.694** depending on the eval run — within the ±0.5pp noise floor (C5), so the two numbers are the same baseline.
- **expert rank / r8 / r32** — LoRA rank of the MoE expert tensors (trained at r8; padded or declared to r32).
- **mixed-rank / `rank_pattern`** — experts at r8 under a global r32; `rank_pattern` is the adapter_config field that declares per-module rank. If left undeclared, Kaggle's loader mis-applies the experts.
- **`target_modules` list vs regex** — adapter_config field. Must be a **list** of module names; a regex string makes Kaggle's vLLM silently under-apply the LoRA (~−2pp). The dominant packaging lever (C9/C10).
- **backbone prefix / backbone-rekey** — renaming keys `base_model.model.model.` → `…backbone.` to match the reference layout. Empirically ≈ noise on Kaggle.
- **noise floor (±0.5pp)** — vLLM greedy run-to-run non-determinism; sub-1pp deltas aren't meaningful (C5).

**Corrections referenced here** (full text in OPEN_QUESTIONS.md):
- **C5** noise floor; **C7** the same adapter scores differently across vLLM versions (Kaggle = 0.17.1); **C8** v13's 0.55 was a packaging confound, not a trace regression; **C9/C10** target_modules-as-list is the load-bearing packaging lever, prefix is noise; **C11** the expert-B reshape fix is mathematically correct but its Kaggle effect is untested (reconciles the older "the fix regressed the score" claim).

## Local — dev_frozen accuracy by category (n=500, greedy, vLLM 0.20.1)

| adapter | what changed vs v9 | overall | bit_manip | text_enc | eq_trans | numeral | unit | gravity |
|---|---|---|---|---|---|---|---|---|
| **v9** (baseline) | v4 traces | 0.694 | 0.333 | **0.687** | 0.155 | 1.000 | 1.000 | 1.000 |
| **v11** | bit_manip → derivation-first v5 | 0.704 | 0.500 | 0.566 | 0.167 | 1.000 | 1.000 | 1.000 |
| **v12** | + shortened v5 (Cut 1) + char-by-char text_enc | 0.710 | **0.548** | 0.530 | 0.190 | 1.000 | 1.000 | 1.000 |
| **v13** | shortened-v5 bit_manip + **v9 text_enc reverted** | 0.704 | 0.536 | 0.542 | 0.155 | 1.000 | 1.000 | 1.000 |
| **v13+rank** | v13 + MoE expert rank 8→32 | **0.716** | 0.512 | 0.602 | 0.190 | 1.000 | 1.000 | 1.000 |

Notes:
- **bit_manip +20.3pp** (33.3% → 53.6% in v13, the submitted mix; v12 peaked at
  54.8%, +21.4pp) from the derivation-first trace (Q8): v4 *stated* the per-bit
  rule; v5 *derives* it, where the model was choking.
- **text_enc regressed** under any v5 bit_manip (68.7 → 54–60%): cross-task
  interference from the longer bit_manip traces, **not** the text_enc format
  (Q9). Reverting text_enc (v13) did not recover it; expert-rank (v13+rank)
  partially did (+6pp) — but see Kaggle caveat below.
- `gravitational_constant` / `numeral_conversion` / `unit_conversion` are solved
  (100%) by deterministic solvers in every version.

## Kaggle — public leaderboard (the packaging ladder)

Most rows are the **same v9 weights** with different *packaging* — the local→Kaggle
gap turned out to be largely a packaging/runtime issue, not model quality
(corrections C8–C11).

| submission | expert rank | target_modules | key prefix | Kaggle (public LB) |
|---|---|---|---|---|
| v9 warm-start (naive) | 8 (mixed, undeclared) | regex | model.model | 0.56 |
| v9 rankpattern | 8 (declared) | regex | model.model | 0.57 |
| v9 rankpattern+backbone | 8 (declared) | regex | backbone | 0.56 |
| **v9 r32-padded+backbone ("Adapter A")** | **32 (padded)** | **list** | backbone | **0.58 (best)** |
| v9 r32-padded, model.model, **list** | 32 | list | model.model | 0.57 |
| v9 r32-padded, model.model, **regex** ("squeeze") | 32 | regex | model.model | 0.55 |
| B_FIXED (fixed expert-B reshape) | 8 (declared via rank_pattern) | regex | model.model | 0.55 |
| **v13** (shortened-v5 traces, A-packaging) | 32 | list | backbone | 0.56 |
| v13+rank (expert r32) | 32 | list | backbone | **not submittable** (3.0 GiB > 1.5 GB cap) |

> Corrected 2026-07-06: the B_FIXED row previously read "32 | list | backbone" —
> the archived submission zip (`submissions/extracted/lora_v9_ws300_B_FIXED.zip`,
> adapter_config + tensor header) shows **regex** `target_modules`, **model.model**
> prefix, and expert **r8** declared via `rank_pattern`. This strengthens C11: the
> B_FIXED 0.55 was never packaging-isolated from Adapter A's 0.58.

Packaging findings (byte-identical-weights A/Bs):
- **Mixed-rank experts must be fixed** (pad to uniform r32, or declare `rank_pattern`): naive 0.56 → fixed ~0.57–0.58. Kaggle's loader mis-applies undeclared mixed-rank experts.
- **`target_modules` must be a LIST, not a regex** (C9/C10): 0.55 → 0.57 on byte-identical weights (public-LB delta; does not necessarily predict private-LB standing). vLLM 0.17.1 silently under-applies a regex `target_modules`. **This is the dominant packaging lever** and is now baked into the converter.
- **Backbone prefix is noise** (≤1pp, sign-inconsistent across rank states) — *not* load-bearing (C9/C10), despite earlier belief.
- **v13 = 0.56 < A's 0.58**: with correct, identical packaging, the v5 trace gains do **not** translate to Kaggle (C8). Trace iteration closed; **Adapter A (0.58 public LB) is the best submission**.
- Even fully-applied, **public LB 0.58 vs local 0.694 ≈ 11pp residual** = runtime-version parity (C7), separate from packaging and not locally reproducible.

## Headline

- Best **public LB**: **0.58** (Adapter A), vs the local **0.694** — the gap is packaging + runtime-version, now diagnosed.
- Final **private LB: 0.604**, rank **3488/4182** — the private split held slightly above the public one. Public and private are different metrics over different held-out splits; the ladder above is public-LB throughout, and its A/B deltas do not necessarily predict private standing.
- The strongest result is methodological: a chain of single-variable experiments that **isolated confounds and refuted several of our own hypotheses** (C2 reshape, C7 vLLM parity, C8 packaging confound, C9–C10 target_modules lever, C11 reconciliation). See the corrections in `OPEN_QUESTIONS.md`.
