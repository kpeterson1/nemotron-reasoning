# Publication checklist — MUST resolve before any public push

This repo is currently **private**. The items below are deferred privacy/legal
and curation tasks that **must be resolved before making it public**. They were
intentionally not done during the Batch-1 cleanup because they require an
external license/TOS check and/or are destructive (removing files, rewriting
history) while the material is still useful for the writeup.

Status legend: ☐ not done.

## ⚠️ Key principle — history, not branch tidiness, is the gate

**Deleting or merging branches does NOT sanitize git history.** Making the repo
public exposes *all of `main`'s commit history*. The Batch-1 scrubs fixed the
*current* tree, but earlier commits still contain the username paths, the
competition data, and the Huikang material. The **squash / orphan step (Step 3
below) is the only thing that actually removes the committed sensitive files
from history.**

## Going-public procedure

1. **Resolve Batch 2** (P0/P1 below): remove competition data + Huikang material
   (replace with download scripts); confirm **no username paths anywhere in the
   tree** (`git grep` clean). See the per-item checklist below.
2. **Harvest unique commits** worth keeping from the branches that aren't already
   in the canonical line, before they're abandoned. As of 2026-06-16, commits
   NOT in `phase2/solver-expansion-v11` (would be lost on deletion):
   - `phase2/trace-format-audit` — **16** (most; review first)
   - `phase2/text-enc-trace-audit` — 7
   - `solvers-v2` — 3 ; `phase2/text-enc-audit` — 3 (more than the v5 generator
     already cherry-picked into the trunk)
   - `phase3/huikang-warmstart` — 1 (Spark 2's reformatter + train_lora
     device_map) ; `adapter-linter-mvp` — 1
   - (0 unique, fully contained, nothing to harvest: `main`,
     `phase3/local-kaggle-gap-diagnosis`, `kaggle-runtime-parity-v9`,
     `huikang-adapter-metadata`, `expert-lora-kaggle-debug`)
3. **Create a fresh curated/squashed history** from the cleaned tree (squashed
   `main` or a new `git checkout --orphan` branch). This drops the sensitive
   content from history entirely. Nothing important is lost: the investigation
   narrative lives in the **docs** (`OPEN_QUESTIONS.md`, `RESULTS.md`,
   `SESSION_LOG.md`), not in commit granularity.
4. **Make that public**; keep the full-history private repo as the working
   archive.

## P0 — privacy / legal (blockers)

- ☑ **Huikang reference material licensing.** *RESOLVED 2026-07-17:* no
  license file or notice could be located in the vendored copies (absent an
  explicit license, redistribution defaults to all-rights-reserved), so his 6
  verbatim files (notebook_tinker.py ×2, the .ipynb + .md notebook copies,
  kernel-metadata.json, reference_adapter/adapter_config.json) were removed
  from tracking. Our 10 provenance/extraction docs remain;
  `references/README.md` (Cited sources) records the removal and the pinned
  public-commit recovery pointer.
  - (Already excluded — good: large `.safetensors` / `.zip` per its README.)

- ☑ **Competition data redistribution.** *RESOLVED 2026-07-17:* the six
  competition-verbatim JSONLs (`datasets/dev_frozen.jsonl` +
  `datasets/splits/{dev_frozen, dev_frozen_shuffled, hard200, speed_bench,
  train_remaining}.jsonl` — 10,750 rows, verified verbatim vs `train.csv`)
  were untracked; `.gitignore` now blocks `datasets/*.jsonl` and
  `datasets/splits/*.jsonl`. Regeneration is documented in
  `datasets/splits/README.md` (download `train.csv` from Kaggle, run
  `python -m src.data.split --csv datasets/raw/train.csv`; seed-42 defaults
  reproduce the splits byte-for-byte) and linked from the repo README.

- ☐ **Spark sync scripts (P0-5, deferred).** `scripts/pull_from_spark2.sh` and
  `scripts/sync_to_spark2.sh` hardcode `/home/nvidia/kaggle/...` remote paths +
  host refs — machine-specific, useless to the community. They are already
  tracked, so a `.gitignore` line alone won't help. Before public: `git rm`
  them (or move to an excluded `dev/`) and/or scrub the host paths to env vars.

## P1 — correctness / history

- ☐ **Branch curation / squash (P1-4, deferred).** 13 remote branches, many
  stale (`solvers-v2`, `trace-format-audit`, `expert-lora-kaggle-debug`,
  `huikang-adapter-metadata`, `kaggle-runtime-parity-v9`, the `phase2/*` audit
  branches, …). For a public release, consolidate to a curated `main` with a
  clean history (the "curated history" step — do at publication, not while the
  branches are live working artifacts).

- ☐ **`run_eval` / `make eval` defaults don't match Kaggle.** Defaults are
  `temperature=1.0, max_tokens=3584, max_model_len=4096`; the confirmed Kaggle
  harness is `0.0 / 7680 / 8192 / max_num_seqs=64` (now documented in README).
  Consider updating the defaults (or a `--kaggle` preset) so a default local run
  reproduces Kaggle conditions. (README already warns to pass them explicitly.)

## Done in Batch 1 (for reference — not deferred)

- ✅ Scrubbed hardcoded `~`/username absolute paths from source
  (`harvest_correct_traces.py`, `text_encryption_solver.py`) → repo-relative.
- ✅ Scrubbed username paths from `PHASE1_STEPA`, `PHASE1_STEPC`, and
  `SESSION_LOG.md`.
- ✅ Added `LICENSE` (Apache-2.0) — **our code only**; does NOT cover the
  Huikang material or competition data (those are P0 above).
- ✅ Fixed README eval params + architecture; added the vLLM-version caveat (C7)
  and the convert→pad→rekey packaging how-to.
- ✅ Pruned duplicate/empty `prompts/phase3/` files; added `RESULTS.md`.
- ✅ Gitignored `*.py.backup`.

## Verified clean (no action)

- No `.env` / keys / tokens / `.pem` / IPs tracked. `.env.example` is empty
  placeholders only.
