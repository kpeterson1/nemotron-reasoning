# Handoff — writeup reconcile (Phases 0–2), 2026-07-06

> *2026-07-17: commit-SHA citations replaced with descriptions — they pointed
> into the private development history, which the public snapshot does not carry.*

Committed in three user-approved commits on 2026-07-06 (writeup+handoff+log /
RESULTS erratum+provenance / README link) after verifying the submission-record commit. Not pushed.

Submission-record-commit verification (user-requested, pre-commit): it adds only
V9_R32PADDED_BACKBONE_SUBMISSION.md (2026-05-25, "public leaderboard score:
0.58", note "v9 r32padded backbone adapter reference-matched keys shapes
dtypes"); its sibling commit adds the matching KAGGLE_SUBMISSION_RESULTS row. The
submitted package `runs/train/lora_v9_warm_start_300step_vllm_r32padded_backbone/`
(built May 24 03:20-03:25, submission.zip alongside) has target_modules=LIST
(8 entries), 12,008 tensors all backbone-prefixed, experts r32-padded,
rank_pattern={} — matches RESULTS.md Adapter A exactly. Caveat: the similarly
named `submissions/lora_v9_warm_start_300step_r32padded.zip` is the PRE-REKEY
intermediate (regex + model.model) — do not mistake it for A. The cited commit is the
score-record commit, not the artifact build commit (adapters were never
tracked; gitignored early on).

---

## 1. Phase 0 merge summary

Created `writeup.md` (now at `docs/writeup.md`) from
`writeup_nemotron_retrospective.md` **verbatim** (canonical base: C7
runtime-divergence reversal, text_encryption phantom retraction, C11
packaging-confound correction all preserved), plus three detail chunks salvaged
from the older `writeup_best_finetuning_method.md`, reframed to not contradict
the corrections:

1. **text_encryption forensics -> §6b** (older draft §3b): 26/47 one-word-off
   misses, 0/47 mechanical-map-match on misses vs 81% (29/36) on correct
   predictions, 100% coverage / <=536-token traces. Framed as pre-C7
   observations of the artifact eval run — NOT a live learnability wall. The
   retracted "fix" stays retracted.
2. **12,008-tensor conversion count + PEFT 3D packing shapes -> §6a** as
   round-trip validation detail.
3. **Per-category gap table -> §2**, relabeled local-eval (Kaggle emits no
   per-category output).

Deliberately NOT ported: the older draft's "Pivotal submissions" table (its
version labels conflict: it calls the B-fixed submission "v13" @0.55, while the
canonical docs use v13 = the bit_manip-v5 trace swap @0.56). Retracted framings
(Open Contribution Award entry, "retired then reinstated" hedge, live
text_encryption learnability claim) confirmed absent.

## 2. Branch + final git status

Branch: `phase2/solver-expansion-v11` (all three phases ran here).
Submission commit cited in the writeup (2026-05-25, on `main`,
records the 0.58 Adapter A submission).

```
 M README.md              (added docs/writeup.md link)
 M RESULTS.md             (B_FIXED row erratum + correction note)
 M docs/SESSION_LOG.md    (session log entries, phases 0 and 2)
?? .claude/settings.json  (pre-existing, user's)
?? docs/writeup.md        (the canonical writeup — moved from repo root)
?? phase0_prompt.txt      (user's phase prompts)
?? phase1_prompt.txt
?? phase2_prompt.txt
?? references/README.md   (new provenance note)
```

Deleted this session (approved): `writeup_nemotron_retrospective.md`,
`writeup_best_finetuning_method.md` (both untracked; archived to session
scratchpad first), `docs/reference_solvers/tonghuikang/` (untracked, was
gitignored at `.gitignore:42`; provenance preserved in `references/README.md`).
No `.pdf` draft existed despite the phase-2 prompt mentioning one.

## 3. Phase 1 audit table (verbatim)

| # | Claim | writeup.md value | Repo value | Status | Source |
|---|---|---|---|---|---|
| 1 | v9 bit_manip local baseline | 33% | 0.333 (both v9 runs) | **Confirmed** | `runs/eval/...v9-v1.1-1780297968.json` + `...1780923962.json` `per_task_type` |
| 1 | v13 bit_manip local | ~54%, "roughly +21pp" | v13 = **0.536** (+20.3pp); the +21.4pp figure belongs to v12 (0.548) | **Confirmed (rounding nuance)** | `runs/eval/...v13_ws300_vllm-1781288020.json`; RESULTS.md:52-55; OPEN_QUESTIONS.md:400 |
| 2 | target_modules list-format Kaggle gain | ~2pp | 0.55->0.57 clean A/B, "+2pp, real (~4x noise floor)" | **Confirmed** | OPEN_QUESTIONS.md:192 (C10); RESULTS.md:88 |
| 3 | `src/training/convert_peft_to_vllm_moe.py` | cited | exists | **Confirmed** | — |
| 3 | `src/solvers/` | cited as solver location | directory exists but contains only `__pycache__`; solvers/trace generators live in **`src/data/`** | **Stale** | `src/data/bit_manip_trace_v5.py` etc. |
| 3 | `src/eval/run_eval.py` | cited | actual path **`src/evaluation/run_eval.py`** | **Stale** | — |
| 3 | probe scripts "under `docs/investigations/`" | cited | actual: **`scripts/bitmanip_logprob_probe.py`**, `scripts/text_enc_logprob_probe_vllm.py` | **Stale** | — |
| 3 | `docs/reference_solvers/tonghuikang/`, repo URL | cited | exist; remote = `github.com/kpeterson1/nemotron-reasoning.git` | **Confirmed** | `git remote -v` |
| 3 | "`SESSION_LOG.md`" | root-relative | actual `docs/SESSION_LOG.md` | **Stale (minor)** | — |
| 3 | "main at the tagged submission commit" | implied tag exists | only tag is `phase0-complete`; **no submission tag** | **Stale/aspirational** | `git tag` |
| 3 | phase-prompt extras: `extract_answer.py`, `generate.py`, `pad_lora_to_uniform_rank.py`, `rekey_to_backbone_reference.py` | (not cited in writeup) | all exist (`src/evaluation/`, `src/inference/`, `src/training/`, `scripts/`) | **Confirmed** | — |
| 4 | lora_B bug (out,r,E) vs fix (out,E,r) | as stated | converter uses `(out, E, r)`; buggy `(out, r, E)` preserved in comment | **Confirmed** | `src/training/convert_peft_to_vllm_moe.py:186-201` |
| 5 | text_enc 68.7% vs 43.4%, 26 flips, 0% truncation | as stated | 0.687 / 0.434 exactly; "26 text_enc rows flip"; truncation 0.0 both | **Confirmed** | the two JSONs; OPEN_QUESTIONS.md:116-133 (C7) |
| 5 | Spark 1 = vLLM 0.20.1 | as stated | EngineCore `v0.20.1` in local diag log | **Confirmed** | `runs/eval/_b_fixed_diag_1781016473.log` |
| 5 | 43.4% run was "a Spark 2 run"; delta = vLLM version | stated as fact | C7 marks this **"INFERRED (not in the artifact)"** — no engine version recorded for run 1779355462 | **Inferred** | OPEN_QUESTIONS.md:125-127 |
| 5 | Kaggle = vLLM 0.17.1 | stated as fact (table + prose) | C7: **"(c) user-reported, NOT independently verifiable"** (huikang notebook outputs cleared) | **Inferred/user-reported** | OPEN_QUESTIONS.md:134-141 |
| 6 | RECONCILE: char-by-char design attribution | flagged | huikang's reference `cipher.py` emits per-char `cc->plain` lookup steps before each decoded word — the design **is his**; older draft's credit was correct | **Resolved -> credit huikang** | `docs/reference_solvers/tonghuikang/reasoners/cipher.py:175-197` (local copy since removed; see references/README.md) |
| 7 | §6a: B-buggy vs B-fixed also differed regex-vs-list | as stated | **Artifact-confirmed**: B_FIXED zip = regex `target_modules` + `model.model` prefix + expert r8 rank_pattern; Adapter A = list + backbone + r32-pad | **Confirmed** — and it exposes **RESULTS.md:82 as wrong** (said "32 \| list \| backbone") | `submissions/extracted/lora_v9_ws300_B_FIXED.zip` adapter_config + tensor header |
| 7 | §6b: "intervention regressed 68.7% -> 53%" | attributed drop to char-by-char | chain: 68.7 (v9) -> **56.6 (v11, no char-by-char — bit_manip-trace interference)** -> 53.0 (v12 = Cut-1 + char-by-char, deliberate two-variable) -> 54.2 (v13 revert, didn't recover). Isolated char-by-char cost <=3.6pp | **Imprecise/Stale** | RESULTS.md:52-55; OPEN_QUESTIONS.md Q9:367-397; SESSION_LOG:522-526, 590-592 |
| 7 | v13 Kaggle 0.56 vs v9 0.58 | as stated | matches RESULTS ladder (note: an earlier naive-packaged v13 scored 0.55 — C8) | **Confirmed** | RESULTS.md:79,83; OPEN_QUESTIONS.md:148 |
| — | §2 ported table rounded rows | grav ~85%, numeral ~80%, unit ~75%, eq ~25% | canonical v9: grav/numeral/unit = **1.000** in every adapter from v6 on; eq_trans acc **0.155** | **Contradicted** | eval JSONs; RESULTS.md:52 |
| — | §2 coverage figures | bit 88%, eq ~29%, text_enc 100% | bit_manip **88.1%** (74/84) ok; eq_trans **28.6%** (24/84) ok; text_enc 100%: **no committed number** | Confirmed / Confirmed / **Unverifiable** | OPEN_QUESTIONS.md:253-258 (Q6) |
| — | §3 probe threshold "-0.69 ~= p 0.5" | as stated | probe uses **first argmax divergence** (rank>1, `prompt_logprobs=1`), no threshold; observed gold logprobs at divergence -0.7…-2.3 | **Unverifiable/contradicted** | `scripts/bitmanip_logprob_probe.py:1-25,182`; OPEN_QUESTIONS.md:322-324 |
| — | §4 44/47 diverge at RULE_STATEMENT | as stated | "44 (94%) first diverge at the RULE_STATEMENT line" | **Confirmed** | OPEN_QUESTIONS.md:319-321; SESSION_LOG:406 |
| — | §4 v3->v4 "+6.8pp" prior gain | as stated | **no repo source found** (searched docs, reports, session log) | **Unverifiable** | — |
| — | §6a 12,008 tensors, PEFT 3D shapes | as stated | 12,008 in B_FIXED header + "12,008/12,008" SHA match; shapes match converter code | **Confirmed** | artifact header; BACKBONE_DIAGNOSTIC.md:132; converter:187,201 |
| — | §6b forensics 26/47, 0/47, 29/36, <=536 tok | ported | **no committed artifact**; arithmetic is self-consistent (43.4% = 36/83 => 47 misses; 29/36 correct) but the decomposition exists only in the older draft | **Unverifiable** | — |
| — | §6c prefix noise (0.58 vs 0.57, opposite-sign r8 A/B) | as stated | matches C10 exactly | **Confirmed** | OPEN_QUESTIONS.md:192 |
| — | final 0.58 / eval params / ±0.5pp noise floor | as stated | match RESULTS.md headline, Kaggle Overview params, C5 | **Confirmed** | RESULTS.md:95; CLAUDE.md; diag log engine args |
| — | rank ~3452/4100, leaders 0.86–0.87, closed June 15 | as stated | **nowhere in repo** | **Unverifiable** (user-provided context) | — |

## 4. writeup.md edits applied (Phase 2)

All in `docs/writeup.md`. Diff vs pre-edit copy archived at session scratchpad
(`phase2_writeup.diff`, `writeup_pre_phase2.md`).

1. **Header + §1 UNVERIFIED markers** — close date, rank ~3452/4100, leaders
   0.86–0.87 marked `[UNVERIFIED: user-provided]`. Source: not in repo
   (audit rows "Unverifiable").
2. **§1 bullet + §4 title/body + §5 lead: +21pp -> +20pp** with exact numbers
   33.3% -> 53.6% (v13, submitted mix; v12 hit 54.8%). Source:
   `runs/eval/dev_frozen-raw-runs_train_lora_v13_ws300_vllm-1781288020.json`
   per_task_type; RESULTS.md:52-55.
3. **§2 table replaced** with canonical v9 per-category values (grav/numeral/
   unit 100%, text_enc 68.7%, bit_manip 33.3%, eq_trans 15.5%; coverage
   88.1% = 74/84 and 28.6% = 24/84; text_enc 100% coverage marked UNVERIFIED).
   Source: `runs/eval/...lora_v9_warm_start_300step-1780923962.json`;
   OPEN_QUESTIONS.md:253-258 (Q6, `scripts/audit_solver_coverage.py`).
4. **§3 probe method corrected** — "-0.69 threshold" replaced with
   first-argmax-divergence (rank > 1); observed gold logprobs -0.7…-2.3 cited.
   Source: `scripts/bitmanip_logprob_probe.py:1-25,182`;
   OPEN_QUESTIONS.md:322-324.
5. **§4 "+6.8pp v3->v4" CUT entirely** (per phase-2 resolution — no repo
   source, doesn't reconcile). Replaced with the derivation-grid description
   (Q8: v4 docstring "Drops the 8-row per-bit verification grid") and v4 max
   ~377 tokens (OPEN_QUESTIONS.md:255).
6. **§5 parity qualifiers** — Spark 2 attribution of the 43.4% run marked
   INFERRED (run artifact records no engine version); Kaggle 0.17.1 marked
   user-reported / not independently verifiable; Spark 1 0.20.1 marked
   confirmed-from-engine-logs. Source: OPEN_QUESTIONS.md:125-141 (C7);
   `runs/eval/_b_fixed_diag_1781016473.log` (EngineCore v0.20.1).
7. **§6b attribution** — "We designed" replaced with credit to tonghuikang's
   char-by-char cipher design; RECONCILE flag removed; §9 acknowledgments
   restore "the character-by-character cipher decode" as his contribution.
   Source: his reference `cipher.py:175-197` (per-char lookup steps before
   each decoded word; local copy since removed — see references/README.md).
8. **§6b regression chain corrected** — 68.7 (v9) -> 56.6 (v11, bit_manip-trace
   interference only) -> 53.0 (v12, deliberate two-variable) -> 54.2 (v13
   revert); isolated char-by-char cost <=3.6pp. Source: RESULTS.md:52-55;
   OPEN_QUESTIONS.md Q9:367-397; docs/SESSION_LOG.md:522-526,590-592.
9. **§6b forensics caveat** — counts labeled "uncommitted, contemporaneous CPU
   analysis"; arithmetic reconciliation (43.4% = 36/83 -> 47 misses) stated;
   "<=536 tokens" marked UNVERIFIED.
10. **§8 takeaway** — "purely from vLLM version" softened to "attributable to
    vLLM version under the inference documented in §5".
11. **§9 repro section** — paths fixed (`src/data/`, `src/evaluation/run_eval.py`,
    `scripts/*_logprob_probe*.py`, `docs/SESSION_LOG.md`); "tagged submission
    commit" -> "the submission-record commit" (2026-05-25, on main, records the
    0.58 Adapter A submission — `git merge-base --is-ancestor` verified; no
    today-dated tag created); tonghuikang reference pointer -> `references/README.md`.

## 5. Repo-reorg checklist

Approved + applied this session (each individually user-approved):
- [x] RESULTS.md:82 B_FIXED row erratum + dated correction note (artifact:
      regex + model.model + r8 rank_pattern; strengthens C11).
- [x] Delete local gitignored copy `docs/reference_solvers/tonghuikang/`;
      provenance recorded in new `references/README.md`.
- [x] Move writeup.md -> `docs/writeup.md`; README link added (README.md:5).
      RESULTS.md stays at root; OPEN_QUESTIONS.md stays in docs/investigations/.
- [x] Source drafts archived to session scratchpad, deleted from repo root.

Already done in earlier sessions (no action needed):
- [x] LICENSE (Apache-2.0)
- [x] README corrected (architecture + eval params + vLLM caveat)
- [x] RESULTS.md / OPEN_QUESTIONS.md reader-accessible (glossaries)

Pending / propose-only (higher-risk, PR-style diff required before applying):
- [ ] `src/` restructuring (e.g. remove empty `src/solvers/`, or rename
      `src/data/` -> `src/solvers/` — breaks imports in build scripts; needs a
      sweep of `scripts/` and `src/orchestration/`).
- [ ] Branch curation (13 remote branches) — pre-publish.
- [ ] History rewrite / sanitization — the real going-public gate per
      `docs/PUBLICATION_CHECKLIST.md`.
- [ ] Optional annotated tag pointing at the submission-record commit (user said writeup cites the
      SHA regardless; tag not created).

Rejected: none.

## 6. Open items / pre-publish gates

UNVERIFIED numbers still marked inline in docs/writeup.md:
- Rank ~3452/4100, leaders 0.86–0.87, close date June 15 2026 — user-provided;
  confirm against the Kaggle UI before publish.
- text_encryption solver coverage 100% and "<=536-token traces" — from the
  uncommitted CPU analysis; commit the analysis or rerun
  `scripts/audit_solver_coverage.py` for text_enc.
- §6b forensic counts (26/47, 0/47, 29/36) — kept per user resolution, labeled
  uncommitted; a committed artifact would upgrade them.

Other gates (pre-existing, unchanged): PUBLICATION_CHECKLIST Batch 2 — Huikang
licensing, competition-data redistribution (datasets/splits tracked), Spark
sync scripts, branch curation, history sanitization. Repo-side loose ends
noted: `reports/leaderboard/submission_log.csv` is header-only;
`docs/investigations/kaggle_lora_mismatch/KAGGLE_SUBMISSION_RESULTS.md` has a
single orphan row (row "4" with no rows 1-3 or header).

## 7. Next actions

1. User reviews docs/writeup.md diff + this handoff; on confirmation, commit
   (writeup + RESULTS erratum + references/README.md + README link — suggest
   separate commits for writeup vs errata).
2. Resolve the remaining UNVERIFIED markers (Kaggle UI check; text_enc
   coverage rerun) — they must not survive into the published writeup.
3. Publish gate: run PUBLICATION_CHECKLIST Batch 2 (licensing, data
   redistribution, history sanitization) before making the repo public.
4. Web explainer prototype: **not started** — no trace of it in the repo
   (grep "explainer/prototype" over docs/ returns nothing). Scope it after the
   writeup is frozen.
