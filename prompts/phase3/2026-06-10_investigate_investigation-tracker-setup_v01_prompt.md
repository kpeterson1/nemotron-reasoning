# prompts/phase3/2026-06-09_investigate_investigation-tracker-setup_v01_prompt.md

Create a living investigation tracker at docs/investigations/OPEN_QUESTIONS.md
and a convention for maintaining it. Requirements:

1. The file has two sections: "## Open" and "## Resolved". Each question is an
   entry with this exact structure:
   ### Q<N>: <one-line question>
   - **Status:** open | investigating | resolved
   - **Opened:** <YYYY-MM-DD> (<git short SHA at time of opening>)
   - **Why it matters:** <1-2 sentences — what decision this gates>
   - **Hypotheses:** <bulleted, each with the prediction that would
     confirm/refute it>
   - **Test:** <the specific experiment/command that resolves it>
   - **Evidence:** <links to run artifacts: eval JSON filenames, log paths,
     SHAs>
   - **Resolution:** <empty until resolved; then the answer + the evidence that
     settled it + date/SHA>

2. Resolved questions MOVE from "## Open" to "## Resolved" (don't delete) with
   Resolution filled in. Never edit a Resolution after the fact — if it's
   overturned, open a new question that supersedes it and cross-link.

3. Add a "## Corrections to project memory" section at the top for facts that
   overturned a prior belief, each with the superseded belief, the correct fact,
   and the evidence.

4. Seed the file with the questions and corrections below. Use the real artifact
   filenames given.

5. Add a slash command at .claude/commands/question.md that, given a question ID
   and an update, appends evidence or moves the entry to Resolved following the
   structure above — so updating the tracker is one command, not manual editing.

6. Commit the seeded file and the slash command. Show me the diff before
   committing. Do NOT run any eval or training — this is a docs-only task.
   One-variable discipline, stop-and-report.

# ================================================================================ SEED CONTENT

## Corrections to project memory (top of file)

C1. SUPERSEDED: "raw v9 PEFT adapter has 0 MoE expert keys; only 116 of 6,004
modules trained." CORRECT: experts WERE trained — stored as packed 3D
ParamWrapper tensors invisible to a naive key count. runs/train/lora_v9 has 92
expert keys; lora_v9_warm_start_300step has 184; the converter unpacks these to
11,776 per-expert tensors. EVIDENCE: safetensors key counts (this session,
2025-06-09).

C2. SUPERSEDED: "the local→Kaggle gap is a Kaggle runtime / vLLM-version /
moe_backend difference (Q2/Q3 framing)." CORRECT: the converter had a real
expert-B reshape bug (packed (out,E,r) read as (out,r,E)), AND fixing it
REGRESSED Kaggle 0.58→0.55. The bug was load-bearing. The gap is NOT a runtime
difference; the expert training itself is weak and the scramble was accidentally
masking it. EVIDENCE: round-trip check (this session) + Kaggle B_FIXED
submission = 0.55.

C3. RETAINED from prior memory (still true): eval is GREEDY (temp=0.0),
confirmed by Kaggle Overview + organizer. Canonical local baseline:
dev_frozen-raw-runs_train_lora_v9-v1.1-1780297968.json (acc=0.686).

C4. RETAINED: submitted artifacts DO contain MoE experts (12,008 tensors /
11,868 expert keys). Confirmed again this session.

C5. RETAINED: vLLM greedy is non-deterministic at ±0.5pp; sub-1pp deltas are
noise. (Relevant: B_FIXED local 0.686 vs buggy 0.694 is within noise.)

## Open

### Q3: Why do the trained MoE experts contribute weakly on Kaggle?

- **Status:** investigating
- **Opened:** 2025-06-09 (<SHA>)
- **Why it matters:** Gates whether expert-recipe / solver work can lift the
  Kaggle score, and whether local eval predicts Kaggle expert behavior at all.
- **Hypotheses:**
  - Expert training signal is weak (traces under-determine the routed-expert
    deltas) → predict: a clean expert assignment scores no better, possibly
    worse, than the scrambled one. CONFIRMED by Q3c below.
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
- Cross-links to RESOLVED Q3c below.

### Q5: Recipe gap to leaders (86–87%).

- **Status:** open
- **Opened:** (carry prior date) (<SHA>)
- **Why it matters:** Defines the path to a competitive score by June 15.
- **Hypotheses:** min-logprob/focal-CE loss unbuilt; example count ~4,468 vs
  winner ~7,830; no lm_head; 1 epoch; LR 3e-4.
- **Test:** ablations holding one variable at a time.
- **Evidence:** —
- **Resolution:** (open) NOTE: previously GATED on Q2/Q3 runtime repro; that
  gate is REMOVED (C2). Recipe/solver work is now the primary lever.

### Q6: Solver/rule-coverage on the weak three categories.

- **Status:** open → ELEVATED to primary priority
- **Opened:** (carry prior date) (<SHA>)
- **Why it matters:** Now the most likely lever for Kaggle gains (per C2, the
  gap is in expert/trace quality, not runtime). bit_manip base 9.5% (89%
  truncation unaided); eq_trans ~29% solver ceiling; text_enc Form 3 refuted in
  v10 (revert to v9 format).
- **Test:** per-category solver expansion, single-variable, eval on dev_frozen.
- **Evidence:** per-category v9 accuracy in handoff.
- **Resolution:** (open)

### Q3b: ~4pp non-expert transfer loss (experts-zeroed 0.590 local → 0.55 Kaggle).

- **Status:** open
- **Opened:** (carry prior date) (<SHA>)
- **Why it matters:** A separate, smaller gap on the base/non-expert path.
- **Hypotheses:** distribution shift dev_frozen vs Kaggle test; minor numerics.
- **Test:** —
- **Evidence:** Kaggle ws300_noexpert = 0.55; local zeroed = 0.590.
- **Resolution:** (open)

### Q4: Does Kaggle accept a raw 232-tensor PEFT adapter (pre-conversion)?

- **Status:** open
- **Opened:** (carry prior date) (<SHA>)
- **Why it matters:** Tests whether conversion is net-positive vs raw PEFT.
- **Test:** submit raw adapter to Kaggle.
- **Evidence:** —
- **Resolution:** (open)

### Q7: Recover a valid Huikang submitted adapter for key-layout comparison.

- **Status:** open
- **Opened:** (carry prior date) (<SHA>)
- **Why it matters:** Direct comparison of expert-key layout against the
  winner's working submission.
- **Evidence:** references/huikang/notebook_output/submission.zip is BadZipFile.
- **Resolution:** (open)

## Resolved

### Q1: Does local vLLM apply MoE expert LoRA keys?

- **Status:** resolved
- **Resolution:** YES — +10.4pp (0.694 live → 0.590 zeroed). Evidence:
  dev_frozen-raw-submissions_extracted_lora_v9_ws300_noexpert-1780949611.json
  (0.5900) vs ..._lora_v9_warm_start_300step-1780923962.json (0.6940).

### Q3a: Is the gap a key-prefix mismatch (model. vs backbone.)?

- **Status:** resolved
- **Resolution:** NO. Backbone-prefix A/B: local 0.6900 vs original 0.6940
  (within noise). Evidence:
  dev_frozen-raw-submissions_extracted_lora_v9_warm_start_300step_backbone_prefix-1780958643.json

### Q3c: Is the converter's expert-B reshape correct? (NEW — resolved this session)

- **Status:** resolved
- **Opened:** 2025-06-09 (<SHA>)
- **Why it matters:** A transposed reshape would scramble per-expert deltas and
  could explain the local→Kaggle expert-fidelity gap.
- **Hypotheses:**
  - Converter reshapes B as (out, r, E) per its docstring → predict the packed
    tensor's first-8 sample for expert 0 matches reshape(out,r,E)[:,:,0].
  - PEFT actually packs B as (out, E, r) → predict the sample matches
    reshape(out,E,r)[:,0,:] instead.
- **Test:** round-trip check_expert_unpack_roundtrip.py + two-reshape sample
  print on layer 1, expert 0.
- **Evidence:** reshape(out,r,E)[:,:,0] != reshape(out,E,r)[:,0,:] for expert 0
  (samples diverged after element 0). up_proj.A matched expert-major (correct);
  up_proj.B matched rank-major (transposed). Converter docstring claim of
  (out,r,E) is WRONG; actual PEFT packing is (out,E,r).
- **Resolution:** The reshape WAS transposed (real bug). BUT correcting it
  REGRESSED Kaggle 0.58→0.55 (B_FIXED submission) and was noise-neutral locally
  (0.686 vs 0.694). CONCLUSION: the bug is real but load-bearing — the scrambled
  B was accidentally masking weak expert training. Reverted to the buggy
  artifact as the working baseline. This SUPERSEDES the prior "Kaggle runtime
  difference" framing of Q3 (see C2); the reframed Q3 (weak expert training)
  remains open. 2025-06-09 (<SHA>).
