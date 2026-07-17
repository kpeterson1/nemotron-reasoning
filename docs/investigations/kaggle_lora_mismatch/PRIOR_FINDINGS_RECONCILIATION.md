# Prior findings reconciliation

**Date:** 2026-05-26
**Trigger:** Extractor fix landed (the brace-walking extractor commit); re-score
(`EXTRACTOR_RESCORE.md`) shows the fix moves the local v9 dev_frozen score
by exactly 0.0pp. We now know the 11.4pp local-vs-Kaggle gap is **not**
extractor-driven. This doc reconciles what's still load-bearing in the
prior investigation against that new information.

## What is already known about the Kaggle 0.58 score

(Synthesized from `KAGGLE_SUBMISSION_RESULTS.md`,
`V9_R32PADDED_BACKBONE_SUBMISSION.md`,
`PHASE2_PRE_SUBMIT_REPORT.md`, `HUIKANG_COMPARISON.md`,
`BACKBONE_DIAGNOSTIC.md`, `RANK_PATTERN_REPORT.md`.)

### Kaggle submission ladder

| Submission | Kaggle | Local n=500 (greedy) | Notes |
|---|---:|---:|---|
| `lora_v9_warm_start_300step.zip` (original, `rank_pattern={}`) | **0.56** | ~0.69 | mixed-rank adapter packaged without declaring rank pattern |
| `lora_v9_warm_start_300step_rankpattern.zip` | **0.57** | 0.692 (n=500) | safetensors byte-identical to original; added `rank_pattern`/`alpha_pattern` regex blocks |
| `lora_v9_warm_start_300step_rankpattern_backbone.zip` | **0.56** | (loads cleanly; not re-evaluated full split) | same weights as rankpattern, keys renamed `model.model` → `backbone` |
| `lora_v9_warm_start_300step_r32padded_backbone` (reference-matched keys/shapes/dtypes) | **0.58** | (not run on full split; r32-padded shows FP drift in vLLM kernel) | current best Kaggle score |

The full structural ladder explored (rank-pattern → backbone-prefix
→ r32-padding → reference-matched keys/shapes/dtypes) bought
**+2pp total**: 0.56 → 0.58.

### Local re-scoring on full v9 dev_frozen (this commit)

`EXTRACTOR_RESCORE.md`: brace-walking vs naive `[^}]*` on the saved
n=500 raw generations → **0.694 → 0.694** (delta 0.0pp). Per-task
delta also 0.0pp across all six categories.

## Prior conclusions: still load-bearing

These were established by direct safetensors/file/config inspection and
are unaffected by the extractor fix.

### Structural / on-disk

1. **rankpattern safetensors are bit-identical to the original
   unpadded warm-start 300.** Verified by SHA256 and matching inode
   (hardlinked). The only file delta is `adapter_config.json`
   gaining `rank_pattern`/`alpha_pattern` regex blocks. **Stands.**
   (`PHASE2_PRE_SUBMIT_REPORT.md` §Task 1; `RANK_PATTERN_REPORT.md` §1.)

2. **Per-tensor SHA256-by-value parity for the backbone-prefix rename.**
   All 12,008 expert/non-expert tensors are value-identical between
   rankpattern and rankpattern_backbone; only the key strings differ
   (+3 ASCII chars per key, +36,024 bytes raw, +631 bytes zipped).
   **Stands.** (`BACKBONE_DIAGNOSTIC.md` §2.)

3. **rank_pattern coverage is structurally correct.** The four regex
   patterns match exactly 5,888 / 5,888 expert modules and 0 / 116
   non-expert modules; protected `shared_experts.{up,down,gate}_proj`,
   Mamba `in_proj`/`out_proj`, and attention `q/k/v/o_proj` are
   correctly unmatched. Histogram is uniform (11,776 expert tensors
   at rank 8, 232 non-expert at rank 32). **Stands.**
   (`RANK_PATTERN_REPORT.md` §1; `PHASE2_PRE_SUBMIT_REPORT.md` §Task 1.)

### Loader behavior

4. **Stock vLLM 0.20.1 accepts both `.model.model.layers.…` and
   `.backbone.layers.…` prefixes locally.** Boot messages identical
   (`MoE model detected. Using fused MoE LoRA implementation.`); no
   `LoRA module … will be ignored` warnings; outputs on the test prompt
   are identical between the two forms. This means vLLM's loader does
   additional resolution beyond strict prefix match — most likely the
   FusedMoE attach path identifies expert modules by MoE-mapping suffix.
   **Stands.** (`BACKBONE_DIAGNOSTIC.md` §4.)

5. **Our `convert_peft_to_vllm_moe.py` already emits Kaggle-format
   expert keys.** Per-expert `experts.<E>.up_proj` / `down_proj`
   form, not packed `w1`/`w2`. No Tinker-style unfusing needed.
   **Stands.** (`HUIKANG_COMPARISON.md` T2; `PHASE2_PRE_SUBMIT_REPORT.md`
   §Task 3.)

6. **Mamba `in_proj` is trained directly; no `gate_proj + x_proj` SVD
   merge needed.** HF NemotronH exposes the fused `in_proj` natively.
   **Stands.** (`HUIKANG_COMPARISON.md` T3.)

### Negative results that still constrain

7. **r32 zero-padding introduces FP drift in vLLM's kernel** and
   diverges in inference from the unpadded baseline at the same input.
   (`RANK_PATTERN_REPORT.md` §intro reference to
   `PROMPT_PARITY_R32PADDED_REPORT.md`.) **Stands** as a warning, but
   the r32padded_backbone variant is what scored 0.58 on Kaggle, so
   either the FP drift is small enough to be net-positive on Kaggle, or
   the reference-matched structural changes recover whatever the
   padding loses. **Open whether the +2pp came from padding or from the
   other structural matches in that bundle.**

## Prior conclusions that depended on local eval numbers

These referenced specific local accuracies (which were computed under
the buggy extractor). Per `EXTRACTOR_RESCORE.md`, the extractor fix
moves those numbers by 0.0pp — **so the conclusions themselves stand,
but the underlying numbers are now confirmed valid under either
extractor**.

8. **rankpattern local n=500 Kaggle-exact greedy = 69.20% (346/500),
   +3 examples over v9 = 68.60%.** (`RANK_PATTERN_REPORT.md` §3.) The
   re-score shows brace-walking would give 0.694 (347/500) for
   `dev_frozen-raw-runs_train_lora_v9-*.json` — within the same
   ±1-example noise the prior doc already flagged. **Stands; no
   re-evaluation needed.**

9. **Same-boot rank_pattern vs unpadded delta +4 net (5 unpadded-wrong
   → rank_pattern-right, 1 reverse), out of 50.** (`RANK_PATTERN_REPORT.md`
   §2.) Both arms in the same boot use the same extractor, so the delta
   is extractor-independent. **Stands.**

10. **All structural-fix combinations together only buy +2pp on Kaggle
    (0.56 → 0.58).** This was already the stated conclusion of
    `V9_R32PADDED_BACKBONE_SUBMISSION.md` — that "gross structural
    /key/config mismatch is probably not the dominant remaining issue."
    The new extractor result strengthens this: extractor is also not
    the dominant issue. **Stands and strengthens.**

## Open questions NOT covered by prior docs

The prior docs concentrated on adapter packaging (config, keys, shapes,
ranks). Several runtime/data dimensions were not investigated.

### Q1 — Generation parameters

`docs/kaggle_metric_source.py:357-368` (the Kaggle metric's `score`
defaults):

```
max_lora_rank=32, max_tokens=3584, top_p=1.0, temperature=1.0,
max_num_seqs=128, gpu_memory_utilization=0.85, max_model_len=4096
```

vs typical local Kaggle-exact harness runs (per
`RANK_PATTERN_REPORT.md` §3): `max_tokens=7680, max_model_len=8192,
temperature=0.0, top_p=1.0`.

**`temperature=1.0` on Kaggle vs `temperature=0.0` locally is a major
divergence not flagged in any prior doc.** Greedy vs T=1.0 with
top_p=1.0 (pure sampling) will produce different rollouts. The Kaggle
n=1 score-per-input is thus a sample from a distribution, not a
deterministic point. Local greedy at 0.69-0.70 may sample down to 0.58
at T=1.0 simply from sampling variance + the head being suboptimal
under the sampled trajectory.

Local greedy → Kaggle T=1.0 alone could explain a meaningful chunk of
11.4pp.

### Q2 — max_tokens=3584 vs max_tokens=7680

Local rankpattern eval showed 0.8% truncation at max_tokens=7680. At
max_tokens=3584, truncation will increase, and truncated bit_manipulation
/ equation_transformation responses are likely to produce wrong (or
NOT_FOUND) extractions. Prior doc does not measure this.

### Q3 — `enable_chunked_prefill=True` and `enable_prefix_caching=True`

Kaggle's metric sets both flags True. Local boots in
`BACKBONE_DIAGNOSTIC.md` §4 and `RANK_PATTERN_REPORT.md` §2/§3 do not
list those flags. If local was running with these disabled (default off
in some vLLM versions) and Kaggle has them on, the generated tokens can
differ deterministically for the same model+input.

### Q4 — Backbone (base-model) weight identity

The Kaggle metric loads the base model from
`kagglehub.model_download('metric/nemotron-3-nano-30b-a3b-bf16/transformers/default')`.
Local presumably loads `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` from
HF or a local mirror. **No prior doc verifies these are bit-identical
snapshots.** Even a 1-revision delta (different conversion pass, BF16
rounding, or quantization mismatch) could produce single-digit-pp drift.

### Q5 — Test set identity

`dev_frozen` and the Kaggle public test set are assumed to be drawn from
the same distribution (or even overlap), but this is not verified
anywhere in the prior docs. If Kaggle's test items are systematically
harder in 1-2 categories — say, longer/harder equation_transformation
items — that would manifest as a local-vs-Kaggle gap independent of any
code.

### Q6 — Chat-template rendering at inference time

Prior docs verified `chat_template.jinja` is byte-equal across submissions,
but did not verify that the **rendered prompt** going into vLLM is
byte-equal between local and Kaggle. The Kaggle metric appends
`'\nPlease put your final answer inside `\boxed{}`...'` to the user
content (`docs/kaggle_metric_source.py:299-303`); local harnesses may
or may not append the same suffix. A prompt-byte diff was claimed in
`PROMPT_PARITY_REPORT.md` (referenced in Phase 2 Task 4) but that report
is not in the docs directory we just read.

## Recommended Phase 1 (vLLM parity) scope, narrowed

Given the extractor explains 0pp of the gap and structural fixes have
already saturated at +2pp, Phase 1 should now be a **real runtime
parity investigation, not a confirmation pass.** Narrowed scope:

1. **First, verify Q6 (prompt-byte parity).** Pull the actual rendered
   prompt that the Kaggle metric feeds to vLLM for a known test item;
   diff against what the local harness sends to vLLM for the same item.
   Cheap (no GPU needed). If different, fix and re-score.

2. **Then run ONE local eval that exactly mirrors `docs/kaggle_metric_source.py`'s
   `generate_predictions` call site.** Same vLLM kwargs
   (chunked_prefill, prefix_caching, max_num_seqs=128, gpu_mem=0.85,
   max_lora_rank=32), same sampling params (T=1.0, top_p=1.0,
   max_tokens=3584, max_model_len=4096), same prompt suffix
   ("Please put your final answer..."). Run on the same 500-item
   dev_frozen split. Expected outcome: score drops from 0.69 toward
   0.58. If it lands close to Kaggle, **the gap is runtime config**,
   not adapter packaging.

3. **Verify Q4 (base-model weight identity).** SHA256 the local
   NemotronH-3-Nano-30B BF16 weights against the Kaggle dataset snapshot
   if at all accessible (kagglehub local cache); compare a few sample
   tensors. If they diverge, that explains residual gap.

4. **Only if 1-3 fail to close the gap**, investigate Q5 (test-set
   identity) — would require obtaining a portion of the actual Kaggle
   test items, which is non-trivial since the competition does not
   expose them.

### What Phase 1 should NOT bundle

- Further adapter-packaging variants (key prefixes, rank patterns).
  Already saturated at +2pp Kaggle gain; diminishing returns.
- A new training run. Per `V9_R32PADDED_BACKBONE_SUBMISSION.md` and
  the user's standing constraint, do not start a new training run
  based on submission scores.
- vLLM upgrade/downgrade. Local is 0.20.1; Kaggle metric environment
  isn't pinned in the visible source. Out of scope for Phase 1.

### Which existing eval script is closest to what Phase 1 needs

(To be checked in Step 4.) `src/training/eval_kaggle_exact.py` is the
candidate — name suggests it's the Kaggle-exact harness. Need to
verify it uses the actual Kaggle metric's vLLM kwargs (chunked_prefill,
prefix_caching, T=1.0, max_tokens=3584, max_model_len=4096).
