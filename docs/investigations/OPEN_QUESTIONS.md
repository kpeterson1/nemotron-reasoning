# Open Questions — living investigation tracker

This file tracks open investigation questions, their hypotheses, the specific
test that resolves each, and the evidence. It is the single source of truth for
"what don't we know yet, and how will we find out."

## Convention

- Every question has the exact structure shown by the entries below: **Status**,
  **Opened** (`YYYY-MM-DD (git short SHA)`), **Why it matters**, **Hypotheses**
  (each with the prediction that confirms/refutes it), **Test**, **Evidence**
  (run-artifact filenames / log paths / SHAs), **Resolution**.
- `Status` is one of: `open`, `investigating`, `resolved`.
- When a question is resolved, **move** the whole entry from `## Open` to
  `## Resolved` (do not delete it) and fill in **Resolution** with the answer,
  the evidence that settled it, and the date/SHA.
- **Never edit a Resolution after the fact.** If a resolution is later
  overturned, open a *new* question that supersedes it and cross-link both ways.
- New facts that overturn a prior belief go in
  `## Corrections to project memory` with the superseded belief, the correct
  fact, and the evidence.
- Update this file with `/question <ID> <update>` rather than hand-editing.

## Glossary / legend

A one-line index of the corrections and questions below, plus recurring
shorthand. The authoritative text is each entry's own section; this is a
reader's map. (Definitions drawn from this file, `docs/SESSION_LOG.md`, and
`kaggle_lora_mismatch/`.)

**Corrections (C#) — one-liner each:**
- **C1** — the experts *were* trained (packed 3D ParamWrapper tensors), not absent; the early "0 expert keys" count was an artifact of naive key-counting.
- **C2** — *[later reconciled by C11]* claimed the converter's expert-B reshape bug was load-bearing and that fixing it regressed Kaggle 0.58→0.55, and that the gap is not a runtime difference.
- **C3** — eval is greedy (temp=0.0); canonical local v9 baseline acc ≈ 0.686.
- **C4** — submitted artifacts *do* contain the MoE experts (~12,008 tensors).
- **C5** — vLLM greedy is non-deterministic at ±0.5pp; sub-1pp deltas are noise (the "noise floor").
- **C6** — the weak "bit_manip 9.5% / eq_trans 11.9%" figures were *base-model* (no-adapter) numbers; the trained adapter is far higher.
- **C7** — the "text_enc 43%" figure was an eval-*environment* (vLLM-version) artifact, not a real ceiling; canonical = 68.7%. Kaggle runs vLLM 0.17.1 (user-reported), distinct from both local hosts → the Kaggle-true text_enc is irreducible to [43%, 69%] locally.
- **C8** — v13's Kaggle 0.55 was a *packaging* confound (shipped via naive packaging), not a v5-trace regression.
- **C9** — the "squeeze" 3pp drop is confounded (it changed prefix *and* target_modules); a clean prefix-only A/B shows the prefix ≈ neutral, so the lever is likely the target_modules format.
- **C10** — disambiguator resolved C9: `target_modules` must be a **list** (regex under-applies, ~+2pp real); the backbone prefix is noise.
- **C11** — reconciles C2 ↔ C8/C10: C2's "the fix regressed the score" is unsafe (the 0.55 was a packaging confound); the reshape fix is mathematically correct (round-trip) but its Kaggle effect through correct packaging is **untested**.

**Questions (Q#) — one-liner each:**
- **Q1** *(resolved)* — does local vLLM apply the MoE expert LoRA keys? Yes (+10.4pp).
- **Q3** *(investigating)* — why do the trained experts contribute weakly on Kaggle?
- **Q3a** *(resolved)* — is the gap a key-prefix mismatch? No (within noise locally).
- **Q3b** *(open)* — the ~4pp non-expert transfer loss (experts-zeroed 0.590 local → 0.55 Kaggle).
- **Q3c** *(resolved)* — is the converter's expert-B reshape correct? It was transposed (real bug); see C11 for the reconciliation of its score effect.
- **Q4** *(open)* — does Kaggle accept a raw (pre-conversion) PEFT adapter?
- **Q5** *(open)* — the recipe gap to the leaders (86–87%).
- **Q6** *(open)* — solver/rule coverage on the weak categories.
- **Q7** *(open)* — recover a valid Huikang submitted adapter for layout comparison.
- **Q8** *(resolved)* — bit_manip learning gap: the model couldn't reproduce a *stated* rule; a *derivation-first* trace fixed it (+16.7pp).
- **Q9** *(open)* — v5 bit_manip traces regress text_encryption −12pp (cross-task interference).

> *2026-07-17: commit-SHA citations in this file were replaced with
> dates/descriptions — they pointed into the private development history,
> which the public snapshot does not carry.*

**Recurring shorthand:**
- **dev_frozen** — local held-out eval split (n=500). Local v9 ≈ **0.686–0.694** depending on the eval run — within the ±0.5pp noise floor (C5); the two numbers are the same baseline.
- **ws300 / warm_start_300step** — two-stage training: a stage-1 reasoning LoRA, then a 300-step stage-2 that warm-starts the MoE-expert LoRA with stage-1 frozen.
- **the converter** — `src/training/convert_peft_to_vllm_moe.py` (unpacks PEFT 3D expert weights into per-expert vLLM tensors).
- **ParamWrapper** — PEFT's representation of the routed experts as 3D packed tensors (invisible to a naive key count; the root of C1).
- **expert rank / r8 / r32, `rank_pattern`, mixed-rank** — see RESULTS.md legend.
- **`target_modules` list vs regex, backbone-rekey, noise floor (±0.5pp)** — see RESULTS.md legend.
- **Adapter A / the squeeze / B_FIXED** — named submissions; see RESULTS.md legend. Plus **rankpattern / rankpattern_backbone** — the prefix-only A/B pair (byte-identical weights, only the key prefix differs).
- **teacher-forced / logprob probe** — feed the gold trace token-by-token and measure where the model's argmax first diverges.
- **RULE_STATEMENT / assembled_map / final_decode** — named regions inside a trace, used to localize where divergence happens.
- **Huikang** — Tong Hui Kang, winner of the midpoint **Open Prize** (not the overall competition — where older entries below say "winner" they mean him in that capacity); his reference adapter/notebook (`references/huikang/`) is the layout/packaging comparison point.
- **Spark 1 / Spark 2** — the two dev hosts; Spark 2 ran vLLM 0.22.1 (relevant to the C7 version swing).

## Corrections to project memory

**C1.** SUPERSEDED: "raw v9 PEFT adapter has 0 MoE expert keys; only 116 of
6,004 modules trained." CORRECT: experts WERE trained — stored as packed 3D
ParamWrapper tensors (PEFT's 3D packed-expert representation) invisible to a naive key count. `runs/train/lora_v9` has 92
expert keys; `lora_v9_warm_start_300step` has 184; the converter unpacks these
to 11,776 per-expert tensors. EVIDENCE: safetensors key counts (this session,
2026-06-09).

**C2.** *[SUPERSEDED — see C11 and C7. Both of C2's own conclusions below — "the
bug was load-bearing / fixing it regressed the score" and "the gap is NOT a
runtime difference" — were later overturned: C11 shows the 0.55 was a packaging
confound (the reshape effect is untested), and C7 establishes a real runtime /
vLLM-version difference.]* SUPERSEDED: "the local→Kaggle gap is a Kaggle runtime / vLLM-version /
moe_backend difference (Q2/Q3 framing)." CORRECT: the converter had a real
expert-B reshape bug (packed `(out,E,r)` read as `(out,r,E)`), AND fixing it
REGRESSED Kaggle 0.58→0.55. The bug was load-bearing. The gap is NOT a runtime
difference; the expert training itself is weak and the scramble was accidentally
masking it. EVIDENCE: round-trip check (this session) + Kaggle B_FIXED
submission = 0.55.

**C3.** RETAINED from prior memory (still true): eval is GREEDY (temp=0.0),
confirmed by Kaggle Overview + organizer. Canonical local baseline:
`dev_frozen-raw-runs_train_lora_v9-v1.1-1780297968.json` (acc=0.686).

**C4.** RETAINED: submitted artifacts DO contain MoE experts (12,008 tensors /
11,868 expert keys). Confirmed again this session.

**C5.** RETAINED: vLLM greedy is non-deterministic at ±0.5pp; sub-1pp deltas are
noise. (Relevant: B_FIXED local 0.686 vs buggy 0.694 is within noise.)

**C6.** SUPERSEDED: "bit_manip 9.5% (89% truncation), eq_trans 11.9% — the weak
categories Phase-2 solver expansion will fix" (the framing carried into Q6).
CORRECT: those are **base-model (no-adapter)** numbers. The **trained** v9
ws300 adapter scores bit_manip **33.3% / 0% truncation** and eq_trans **15.5% /
6% truncation**. bit_manip is **coverage-saturated** (88.1% solver coverage on
dev_frozen, max trace 377 tokens — no truncation problem); its gap is learning,
not solver coverage (see [Q8](#q8)). eq_trans solver coverage is the real ceiling
(28.6%) but the 49/84 uncovered are variable-length symbolic transforms
Huikang's reference does not solve either, and the 11 numeric misses are
underivable/ambiguous/inconsistent, not bugs. EVIDENCE: base
`runs/eval/dev_frozen-raw-base-1780830752.json` (bm 0.095/0.893, eq 0.119/0.548);
trained `runs/eval/dev_frozen-raw-submissions_extracted_lora_v9_warm_start_300step-1780923962.json`
(bm 0.333/0.0, eq 0.155/0.06); `scripts/audit_solver_coverage.py`
(2026-06-10).

**C7.** SUPERSEDED: "v9 text_encryption baseline is 43% (47 misses) — a real
ceiling that char-by-char decode addresses." CORRECT: the 43% is an
eval-ENVIRONMENT parity artifact, not the canonical baseline. The 43% comes from
`runs/eval/dev_frozen-raw-runs_train_lora_v9_warm_start_300step_vllm-1779355462.json`;
the canonical `...submissions_extracted...-1780923962.json` shows text_enc
**68.7%**. CONFIRMED from artifacts: the two adapters are BYTE-IDENTICAL
(`cmp` → identical), same model, same prompt (all prompt fields None), and
text_enc truncation = 0.0 in BOTH (so NOT a truncation/max_tokens artifact —
refutes that hypothesis). The two runs' raw outputs even begin identically then
diverge mid-generation; 26 text_enc rows flip between them. INFERRED (not in the
artifact): the environment delta is the vLLM version (Spark 2 host 0.22.1 vs
this host 0.20.1) — text_enc's long multi-step traces amplify kernel-level
greedy-decode differences into a **26pp** swing (>50× the ±0.5pp noise floor in
C5; this is a version-level effect, not a config-permutation one). IMPLICATION:
char-by-char (v12) was motivated by the parity-inflated 43%; on the canonical
baseline text_enc was 68.7% and char-by-char regressed it to 53.0%. RESIDUAL
RISK (ASSUMED unknown): Kaggle's vLLM env is unconfirmed, so the Kaggle-true
text_enc ∈ [43%, 69%]. EVIDENCE: this session (2026-06-12).
KAGGLE vLLM VERSION: reported as **0.17.1** (user, from Huikang's submission
notebook EngineCore init line) — distinct from BOTH local frames (this host
0.20.1, Spark 2 0.22.1), so neither local eval matches Kaggle and [43%,69%] is
IRREDUCIBLE locally without a 0.17.1 env. PARITY DISCIPLINE: this 0.17.1 is
(c) user-reported, NOT independently verifiable in the committed notebook copy —
`references/huikang/kaggle_kernel/tinker-submission-notebook.ipynb` has cleared
outputs (0 `EngineCore`/`0.17` hits); the only version token locally is
`reference_adapter/adapter_config.json` `"peft_version": "0.18.1"` (a PEFT
version, not vLLM). DECISION: reinforces the v13 revert — v9 text_enc is the
cross-frame-robust, originally-Kaggle-validated trace (68.7% on 0.20.1), vs
char-by-char which is now known version-fragile. The Kaggle leaderboard score of
v13 is the ONLY way to resolve [43%,69%] (Kaggle runs the version we can't
reproduce locally).

**C8.** SUPERSEDED: "v13 scored 0.55 on Kaggle (vs adapter A's 0.58), so the
v4→v5 bit_manip trace change regressed Kaggle." CORRECT: v13's 0.55 is a
PACKAGING confound, NOT a trace regression. v13 was shipped via the NAIVE
converter output (`runs/train/lora_v13_ws300_vllm`): experts at rank **8**
unpadded, `adapter_config` `rank_pattern={}`/`r=32`, `model.model` key prefix —
artifact-identical packaging to the ORIGINAL 0.56 submission. Adapter A (0.58)
added three steps v13 skipped: r32-PAD (experts r8→r32 uniform), backbone-rekey
(`model.model`→`backbone`), and reference-match (keys/shapes/dtypes/expert-layout/
Mamba in_proj to Huikang). CONFIRMED from the safetensors: A experts (32,1856)
backbone-prefixed; v13 experts (8,1856) model.model-prefixed; both fp32, both
buggy/HEAD converter, both `rank_pattern={}`. v13's 0.55 lands in the
naive-packaging band (orig 0.56, within ±0.5pp) where Kaggle's runtime
mis-loads the mixed-rank (r8≠declared-r32) experts — exactly the local/Kaggle
gap (local vLLM 0.20.1 loads mixed-rank fine → v13 local 0.704 is real; Kaggle
mis-loads → ~0.55). TRAINING was single-variable (v13 vs A differ only in the
dataset: v4→cut1-v5 bit_manip + text_enc reverted to v9; identical stages/
hyperparams/converter). The trace change is UNTESTED on Kaggle until v13 is
repackaged through pad→backbone→reference-match. EVIDENCE: this session
(2026-06-12); side-by-side build paths reconstructed from
submissions/*.md + docs/investigations/kaggle_lora_mismatch/*.

**C9.** PARTIALLY SUPERSEDES the ladder's "reference-match is neutral" call
(and refines C8/Q3a). The pad+model.model "squeeze" on Adapter A scored Kaggle
**0.55** vs A's **0.58** — a clean 3pp drop on BYTE-IDENTICAL weights (max-abs-
diff 0.0). Initial read: "backbone prefix is load-bearing." CORRECT: the squeeze
changed TWO fields vs A, so it is NOT a clean prefix test — prefix
(backbone→model.model) AND `target_modules` (list→regex). The CLEAN prefix-only
A/B (`rankpattern` 0.57 vs `rankpattern_backbone` 0.56 — byte-identical weights,
ONLY prefix renamed, target_modules=regex held; per-tensor SHA verified in
BACKBONE_DIAGNOSTIC §2) shows the **prefix is neutral-to-slightly-negative**, NOT
load-bearing. So the 3pp most likely comes from the **`target_modules` regex→list
format**, not the prefix. MECHANISM (inferred): Kaggle vLLM 0.17.1 LoRA loader
may not parse a regex `target_modules` (expects a list) → modules unrecognized →
partial LoRA apply → ~0.55; the list form (Huikang uses it) → full apply → 0.58.
This REVERSES the earlier "reference-match neutral" finding: the LIST component is
load-bearing (~+3pp); the BACKBONE-prefix component is the neutral part. Q3a's
local backbone null stands but is local-only (vLLM 0.20.1 is lenient to the
prefix; does not bind Kaggle 0.17.1). RESIDUAL: prefix vs target_modules not
fully isolated (rank×prefix interaction not excluded). DISAMBIGUATOR (one
submission): model.model + LIST at r32 — ~0.58 ⇒ list is the lever; ~0.55 ⇒
prefix is. CONFIRMED from artifacts: the 4-row prefix/target_modules/rank matrix
(this session, 2026-06-13); Huikang ref adapter 12008/12008 backbone + list
(`references/huikang/reference_adapter/`).

**C10.** RESOLVES C9 disambiguator. The model.model+LIST r32 cell scored Kaggle **0.57**. Full byte-identical-weights matrix: A(backbone+list) 0.58, this(model.model+list) 0.57, squeeze(model.model+regex) 0.55. CLEAN A/Bs: (a) target_modules regex->list (prefix held model.model) = 0.55->0.57 = **+2pp, real** (~4x noise floor); (b) prefix model.model->backbone (target_modules held list) = 0.57->0.58 = +1pp, BUT the r8 prefix-only A/B was -1pp (rankpattern 0.57 vs rankpattern_backbone 0.56) — OPPOSITE signs, both ~1pp => **prefix is noise, NOT load-bearing**. CONCLUSION: the packaging lever is **target_modules must be a LIST** (regex silently under-applies the LoRA on Kaggle vLLM 0.17.1) — a config-parsing/"does the adapter fully apply" detail, not the backbone prefix, not expert rank semantics. Confirms C9; refutes "backbone load-bearing" (C9 user-hyp). A (backbone+list) 0.58 stays best-recorded (model.model+list 0.57 within 1pp/noise). RESIDUAL: even fully-applied, Kaggle 0.58 vs local 0.69 ~= 11pp = runtime-version parity (C7), a separate larger problem. EVIDENCE: 3-cell Kaggle matrix (this session, 2026-06-13).

**C11.** RECONCILES C2 ↔ C8/C10 on the expert-B reshape fix. C2 recorded: "the expert-B reshape bug was load-bearing; fixing it regressed Kaggle 0.58→0.55." C8/C10 later showed the 0.55-band scores were PACKAGING confounds (mixed-rank experts / regex target_modules silently under-applying the LoRA), NOT the reshape. So C2's "fixing it regressed" conclusion is UNSAFE — the B_FIXED 0.55 submission was never isolated from packaging. The reshape fix is MATHEMATICALLY CORRECT (round-trip: PEFT packs B as (out,E,r); scripts/check_expert_unpack_roundtrip.py) and is now the shipped default in `src/training/convert_peft_to_vllm_moe.py`. Its Kaggle effect vs the buggy reshape THROUGH CORRECT PACKAGING (r32 pad + list target_modules) is UNTESTED: Adapter A's 0.58 used the BUGGY reshape, and a clean A/B (fixed reshape, identical packaging) has not been run. STATUS: shipped as correct code; Kaggle-clean-test pending; no score-win claimed. EVIDENCE: this session (2026-06-14); the fix and its refactor cleanup are retained in the private repository's history.

## Open

### Q3: Why do the trained MoE experts contribute weakly on Kaggle?

- **Status:** investigating
- **Opened:** 2026-06-09
- **Why it matters:** Gates whether expert-recipe / solver work can lift the
  Kaggle score, and whether local eval predicts Kaggle expert behavior at all.
- **Hypotheses:**
  - Expert training signal is weak (traces under-determine the routed-expert
    deltas) → predict: a clean expert assignment scores no better, possibly
    worse, than the scrambled one. CONFIRMED by Q3c below (but see C11 — Q3c's
    score-effect conclusion was later reconciled as a packaging confound).
  - Distribution shift: Kaggle test fires expert indices the trained subset
    didn't cover → predict: per-expert activation histogram on Kaggle test
    diverges from dev_frozen.
- **Test:** compare expert-activation coverage on dev_frozen vs a Kaggle-like
  held-out set; inspect which of the 128 experts/layer carry meaningful trained
  norm vs near-zero.
- **Evidence:** B_FIXED Kaggle 0.55; buggy Kaggle ~0.58; local both ~0.686.
- **Resolution:** (open — reframed from the old "Kaggle runtime" Q3)

### Q3c: Is the expert-B reshape in the converter correct?

- **Status:** resolved → SEE RESOLVED (kept here only as cross-link)
- Cross-links to RESOLVED Q3c below. → but see **C11**: Q3c's score-effect
  conclusion ("fixing the reshape regressed Kaggle 0.58→0.55") was later
  reconciled as a packaging confound (the reshape's Kaggle effect is untested).

<a id="q5"></a>
### Q5: Recipe gap to leaders (86–87%).

- **Status:** open
- **Opened:** 2026-06-09
- **Why it matters:** Defines the path to a competitive score by June 15.
- **Hypotheses:** min-logprob/focal-CE loss unbuilt; example count ~4,468 vs
  winner ~7,830; no lm_head; 1 epoch; LR 3e-4.
- **Test:** ablations holding one variable at a time.
- **Evidence:** —
- **Resolution:** (open) NOTE: previously GATED on Q2/Q3 runtime repro; that
  gate is REMOVED (C2). Recipe/solver work is now the primary lever.

<a id="q6"></a>
### Q6: Solver/rule-coverage on the weak three categories.

- **Status:** open → ELEVATED to primary priority
- **Opened:** 2026-06-09
- **Why it matters:** Now the most likely lever for Kaggle gains (per C2, the
  gap is in expert/trace quality, not runtime). bit_manip base 9.5% (89%
  truncation unaided); eq_trans ~29% solver ceiling; text_enc Form 3 refuted in
  v10 (revert to v9 format).
  <!-- TODO(reader-clarity): "Form 3" and "v10" are not defined anywhere in the
  committed docs (grep-clean across docs/, submissions/, reports/). They appear
  to reference an earlier text_encryption trace-format experiment that lost,
  motivating the v9-format revert — but that is unverified. Define or excise. -->
- **Test:** per-category solver expansion, single-variable, eval on dev_frozen.
- **Evidence:** per-category v9 accuracy in handoff.
  - 2026-06-10: audited solver coverage on dev_frozen via
    `scripts/audit_solver_coverage.py`. bit_manip **88.1%** covered (74/84,
    max trace 377 tok, 0 truncation) — NOT a coverage problem (see [Q8](#q8)).
    eq_trans **28.6%** covered (24/84); 49 misses are variable-length symbolic
    transforms (Huikang's `equation_numeric.py` is numeric-only too), 11 numeric
    misses are underivable/ambiguous/inconsistent (not bugs). The "base 9.5% /
    89% truncation / 11.9%" premise in **Why it matters** above is base-model,
    superseded by **C6** — bit_manip solver expansion is NOT the lever.
- **Resolution:** (open — reframed: bit_manip is a learning problem [Q8](#q8), not
  solver coverage; eq_trans numeric misses left unchanged by design.)

### Q3b: ~4pp non-expert transfer loss (experts-zeroed 0.590 local → 0.55 Kaggle).

- **Status:** open
- **Opened:** 2026-06-09
- **Why it matters:** A separate, smaller gap on the base/non-expert path.
- **Hypotheses:** distribution shift dev_frozen vs Kaggle test; minor numerics.
- **Test:** —
- **Evidence:** Kaggle ws300_noexpert = 0.55; local zeroed = 0.590.
- **Resolution:** (open)

### Q4: Does Kaggle accept a raw 232-tensor PEFT adapter (pre-conversion)?

- **Status:** open
- **Opened:** 2026-06-09
- **Why it matters:** Tests whether conversion is net-positive vs raw PEFT.
- **Test:** submit raw adapter to Kaggle.
- **Evidence:** —
- **Resolution:** (open)

### Q7: Recover a valid Huikang submitted adapter for key-layout comparison.

- **Status:** open
- **Opened:** 2026-06-09
- **Why it matters:** Direct comparison of expert-key layout against the
  winner's working submission.
- **Evidence:** `references/huikang/notebook_output/submission.zip` is
  BadZipFile.
- **Resolution:** (open)

<a id="q8"></a>
### Q8: bit_manip learning gap — model can't reproduce the trace's rule statement.

- **Status:** resolved (hypothesis confirmed; cross-task cost spun out to [Q9](#q9))
- **Opened:** 2026-06-10
- **Why it matters:** bit_manip is coverage-saturated (88.1%, see [Q6](#q6) / C6)
  yet the trained model scores only 33.3%. This is the largest single-category
  lever left, and it is a trace-format/learning problem, not a solver one. Gates
  whether a trace redesign (vs recipe changes [Q5](#q5)) is the right next move.
- **Hypotheses:**
  - The compact `bit_manip_trace_v4` STATES the per-bit rule without DERIVING it
    (its docstring: "Drops the 8-row per-bit verification grid") → predict the
    model diverges from gold exactly at the rule-statement tokens (operators +
    bit indices), since it was never shown how to compute them. **CONFIRMED**
    (teacher-forced probe below).
  - A trace that reintroduces a compact per-output-bit derivation will shrink
    rule-statement divergence and raise bit_manip accuracy → **UNTESTED**
    (requires a v5 trace + dataset rebuild + train; deferred, do not start).
- **Test:** (a) teacher-force the 74 covered gold traces under the deployed
  adapter, locate first argmax-divergence by trace region [DONE];
  (b) build `bit_manip_trace_v5` with per-bit derivation, single-variable
  rebuild + train + re-eval, predict divergence↓ and acc↑
  [GENERATOR + v11 DONE 2026-06-10; TRAIN + RE-EVAL PENDING].
- **Evidence:** `scripts/bitmanip_logprob_probe.py` →
  `runs/eval/bitmanip_logprob_probe.json` (LOCAL ONLY — `runs/eval/` is
  gitignored). 74 covered: 27 model-correct, 47 model-wrong. Of the 47 wrong,
  **44 (94%) first diverge at the RULE_STATEMENT line**; divergence tokens are
  bit-index digits (16), boolean operators (7), atom-shape words (19). Gold
  logprobs at divergence −0.7…−2.3 (near-ties, model prefers a different rule).
  By rule kind: invariant_k1 69% correct, invariant_k2 12%, per_bit(7–8 specs)
  35% — failure tracks un-derived operator/index commitment, not trace length.
  - 2026-06-10: `src/data/bit_manip_trace_v5.py` (derivation-first,
    every operator/index forced by a prior self-verifying line) +
    `scripts/build_v11_bitmanip.py`. v5 retains 1190/1226 train bit_manip
    (97.1%; 36 dropped use families v5 does not derive — NAND/NOR/MAJ/XOR3/
    Popcount/invariant-k3); 0 internally-inconsistent traces; trace tokens
    dev max 1692 / train max 1935 (v4 was ~376). v11 dataset
    `datasets/processed/train_formatted_v11.jsonl` (LOCAL ONLY, gitignored):
    bit_manip regenerated, all 5 other categories byte-identical to v9
    (verified), line order preserved.
  - 2026-06-10: TRAINED v11bm (two-stage, single-variable vs v9 —
    only bit_manip traces differ; buggy/load-bearing converter; eff batch 8).
    Canonical dev_frozen eval `runs/eval/dev_frozen-raw-runs_train_lora_v11bm_ws300_vllm-1781134253.json`:
    **bit_manip 33.3%→50.0% (+16.7pp, 28→42/84)**; 22 of the 47 previously-
    failing now correct. Probe `runs/eval/bitmanip_logprob_probe_v11bm.json`
    (v5 traces, 47 set): teacher-forced **clean 1/47 → 17/47**, rule-region
    first-divergence **44/47 → 30/47**. Cost: bit_manip truncation 0→4.8%
    (v5 traces ~3× longer, median completion 691→2035 tok).
- **Resolution:** CONFIRMED. A derivation-first trace where every operator/
  index is forced by a prior self-verifying line raises bit_manip accuracy
  (+16.7pp) and reduces rule-region teacher-forced divergence (44→30 of 47;
  clean 1→17). The mechanism — v4 stated the rule without deriving it, the
  model diverged exactly there — is validated. CAVEAT: v11bm regressed
  text_encryption −12pp via cross-task interference (overall only +1.0pp); that
  ship/mitigate question is [Q9](#q9), NOT part of this resolution. 2026-06-10.
  Cross-links: [Q5](#q5) (recipe), [Q6](#q6) (coverage framing), C6.

<a id="q9"></a>
### Q9: v5 bit_manip traces regress text_encryption −12pp (cross-task interference).

- **Status:** open
- **Opened:** 2026-06-10
- **Why it matters:** Gates whether v11 (v5 bit_manip traces) can SHIP. The v5
  format won bit_manip +16.7pp but the shared LoRA lost text_encryption −12pp,
  netting overall only +1.0pp (0.694→0.704). Decides ship vs mitigate.
- **Hypotheses:**
  - Shared LoRA capacity: the longer/denser v5 bit_manip traces (median
    completion 691→2035 tok) crowd out text_enc → predict shortening v5 and/or
    raising LoRA rank recovers text_enc with most of the bit_manip gain.
  - It is the same interference the v9 build guarded against ("v6 v3-format
    bit_manip crashed text_encryption") → predict it scales with bit_manip
    trace length/token-share in the mix.
- **Test:** ablate v5 trace length (cap derivation lines) and/or LoRA rank,
  single-variable, re-eval dev_frozen text_enc + bit_manip.
- **Evidence:** text_encryption 0.687→0.566 (57→47/83, 16 flipped right→wrong),
  byte-identical training data to v9. NOT contamination (0/83 outputs contain
  bit_manip markers) and NOT truncation (1/83; raw_len median 1031 vs 1030) —
  flips are normal cipher reasoning with small letter errors. `runs/eval/
  dev_frozen-raw-runs_train_lora_v11bm_ws300_vllm-1781134253.json`.
  - 2026-06-10 (text_enc teacher-forced probe, v9 vs v11bm, all 83 + 16 flipped):
    the interference landed in **assembled_map** (cipher→plain map construction),
    NOT final_decode. Flipped-16 first-divergence: assembled_map 5→10, clean
    6→1, final_decode 0→0; assembled_map div-rate 0.0066→0.0138 (~2.1×).
    `runs/eval/text_enc_logprob_probe_{v9,v11bm}.json` via
    `scripts/text_enc_logprob_probe_vllm.py` (vLLM-equivalent; the pulled HF
    `scripts/text_enc_logprob_probe.py` can't load PEFT in this env —
    WeightConverter/distributed_operation bug). CAVEAT: teacher-forcing pins
    final_decode to the correct gold decode, so it CANNOT observe the Spark 2
    free-run "ignore-the-map" behavior; final_decode=0 is partly an artifact.
    CONTRADICTS the Spark 2 prediction that interference would hit final_decode.
    IMPLICATION: shorten-v5 (capacity) targets this regression directly;
    char-by-char text_enc decode targets a different (free-run decode) failure.
  - 2026-06-12 (v12 trained: shortened bit_manip v5 [Cut 1] + char-by-char
    text_enc): bit_manip 50.0%→**54.8%** (Cut 1 win; truncation 4.8%→0%); but
    text_enc 56.6%→**53.0%** — char-by-char did NOT recover, it regressed
    FURTHER. Free-run (eval outputs): the fluent-English leap IS gone — the
    model now decodes char-by-char per word (e.g. 6cc637ea: g→s,o→t,…=studies;
    "_ook"→book), recovering 11/26 v9-misses. BUT net −13 (24 broken, 11 fixed):
    the regressions decode char-by-char FAITHFULLY to a WRONG map (aac9bfab:
    "wizard" for "bright"). So char-by-char fixed decode APPLICATION but the real
    bottleneck is map CONSTRUCTION (matches the teacher-forced probe), which
    char-by-char propagates faithfully; v9's word-level decode + Wonderland-vocab
    unknown-filling sometimes rescued wrong maps. See C7: the 43% that motivated
    char-by-char was an eval-environment parity artifact (canonical v9 text_enc
    = 68.7%), so char-by-char regressed a working category.
    `runs/eval/dev_frozen-raw-runs_train_lora_v12_ws300_vllm-1781197217.json`.
- **Resolution:** (open) DECISION: REVERT char-by-char (v13 = shortened
  bit_manip v5 + v9 original text_enc), isolating the clean +21.4pp bit_manip
  delta *[correction 2026-07-16: +21.4pp was the expected carry from v12
  (54.8%); v13's eval landed at 53.6% = +20.3pp]*. The real text_enc lever is map construction (capacity, e.g. +rank) —
  held as a separate single-variable test on the clean v13 baseline, NOT stacked
  onto the revert. char-by-char is NOT a canonical-frame win (C7). Probe-
  grounded (v11bm): regression is map-construction confidence (assembled_map).

## Resolved

### Q1: Does local vLLM apply MoE expert LoRA keys?

- **Status:** resolved
- **Resolution:** YES — +10.4pp (0.694 live → 0.590 zeroed). Evidence:
  `dev_frozen-raw-submissions_extracted_lora_v9_ws300_noexpert-1780949611.json`
  (0.5900) vs `..._lora_v9_warm_start_300step-1780923962.json` (0.6940).

### Q3a: Is the gap a key-prefix mismatch (model. vs backbone.)?

- **Status:** resolved
- **Resolution:** NO. Backbone-prefix A/B: local 0.6900 vs original 0.6940
  (within noise). Evidence:
  `dev_frozen-raw-submissions_extracted_lora_v9_warm_start_300step_backbone_prefix-1780958643.json`

### Q3c: Is the converter's expert-B reshape correct?

- **Status:** resolved
- **Opened:** 2026-06-09
- **Why it matters:** A transposed reshape would scramble per-expert deltas and
  could explain the local→Kaggle expert-fidelity gap.
- **Hypotheses:**
  - Converter reshapes B as `(out, r, E)` per its docstring → predict the packed
    tensor's first-8 sample for expert 0 matches `reshape(out,r,E)[:,:,0]`.
  - PEFT actually packs B as `(out, E, r)` → predict the sample matches
    `reshape(out,E,r)[:,0,:]` instead.
- **Test:** round-trip `check_expert_unpack_roundtrip.py` + two-reshape sample
  print on layer 1, expert 0.
- **Evidence:** `reshape(out,r,E)[:,:,0] != reshape(out,E,r)[:,0,:]` for expert
  0 (samples diverged after element 0). `up_proj.A` matched expert-major
  (correct); `up_proj.B` matched rank-major (transposed). Converter docstring
  claim of `(out,r,E)` is WRONG; actual PEFT packing is `(out,E,r)`.
- **Resolution:** The reshape WAS transposed (real bug). BUT correcting it
  REGRESSED Kaggle 0.58→0.55 (B_FIXED submission) and was noise-neutral locally
  (0.686 vs 0.694). CONCLUSION: the bug is real but load-bearing — the scrambled
  B was accidentally masking weak expert training. Reverted to the buggy
  artifact as the working baseline. This SUPERSEDES the prior "Kaggle runtime
  difference" framing of Q3 (see C2); the reframed Q3 (weak expert training)
  remains open. 2026-06-09.
