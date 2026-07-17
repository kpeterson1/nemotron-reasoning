# Extractor re-score: brace-walking vs naive `[^}]*`

**Date:** 2026-05-26
**Context:** `src/evaluation/extract_answer.py` was fixed (the brace-walking extractor commit) to
use the brace-walking logic that the deployed Kaggle metric uses (re-verified
against the Kaggle website on 2026-05-26 — see
`docs/kaggle_metric_source.py`). The prior implementation used a naive
`r"\\boxed\{([^}]*)(?:\}|$)"` regex that mishandles nested LaTeX braces.

This report re-scores every saved raw-generation eval output in `runs/eval/`
under both extractors to bound how much of the 11.4pp local-vs-Kaggle gap
(v9 dev_frozen 0.694 → Kaggle public 0.58) is due to the extractor.

## Method

For each `runs/eval/*-raw-*.json` file (each contains the full `raw` model
text plus the `answer` ground-truth per prediction):
- **OLD score:** apply the naive `[^}]*` extractor (snapshot inline in
  `scripts/rescore_with_fixed_extractor.py`) + Kaggle's `verify()`.
- **NEW score:** apply the current `src/evaluation/extract_answer.py`
  (brace-walking) + the same `verify()`.

`verify()` is unchanged by the fix; only the boxed-extraction strategy
differs.

Re-scoring script: `scripts/rescore_with_fixed_extractor.py` (uncommitted,
depends on `runs/eval/` raw outputs which are gitignored).

## Results

| file | n | stored | old | new | delta | disagree |
|---|---|---|---|---|---|---|
| dev_frozen-raw-base-1778107065.json | 1 | 0.000 | 0.000 | 0.000 | +0.0000 | 0 |
| dev_frozen-raw-base-1778108772.json | 50 | 0.080 | 0.080 | 0.080 | +0.0000 | 0 |
| dev_frozen-raw-runs_train_lora_v10-1779147722.json | 500 | 0.662 | 0.662 | 0.662 | +0.0000 | 3 |
| dev_frozen-raw-runs_train_lora_v11-1779175851.json | 500 | 0.670 | 0.670 | 0.670 | +0.0000 | 5 |
| dev_frozen-raw-runs_train_lora_v6-1778628463.json | 500 | 0.612 | 0.612 | 0.612 | +0.0000 | 4 |
| dev_frozen-raw-runs_train_lora_v7-1778652991.json | 500 | 0.624 | 0.624 | 0.624 | +0.0000 | 3 |
| **dev_frozen-raw-runs_train_lora_v9-1779095404.json** | **500** | **0.694** | **0.694** | **0.694** | **+0.0000** | **3** |
| dev_frozen-raw-runs_train_lora_v9_warm_start_300step_vllm-1779355462.json | 500 | 0.634 | 0.634 | 0.634 | +0.0000 | 5 |
| dev_frozen_shuffled-raw-base-1778485851.json | 500 | 0.520 | 0.520 | 0.520 | +0.0000 | 37 |
| dev_frozen_shuffled-raw-base-baseline_n50-1778129737.json | 50 | 0.360 | 0.360 | 0.360 | +0.0000 | 4 |
| dev_frozen_shuffled-raw-base-brief_n50-1778130651.json | 50 | 0.480 | 0.480 | 0.480 | +0.0000 | 3 |
| dev_frozen_shuffled-raw-runs_train_lora_baseline-1778265114.json | 500 | 0.368 | 0.368 | 0.362 | **-0.0060** | 20 |
| dev_frozen_shuffled-raw-runs_train_lora_v2-1778450771.json | 500 | 0.584 | 0.584 | 0.584 | +0.0000 | 7 |
| dev_frozen_shuffled-raw-runs_train_lora_v4-1778545698.json | 500 | 0.616 | 0.616 | 0.616 | +0.0000 | 2 |
| dev_frozen_shuffled-raw-runs_train_lora_v5-1778559173.json | 500 | 0.626 | 0.626 | 0.626 | +0.0000 | 3 |

`disagree` = number of predictions where old/new produce different extracted
strings OR different verify() outcomes.

`stored` matches `old` exactly across every file: confirms the stored
accuracy values were computed under the pre-fix extractor (no other variable
moved).

## Per-task breakdown — v9 dev_frozen (n=500)

| task | n | old | new | delta |
|---|---|---|---|---|
| bit_manipulation | 84 | 0.333 | 0.333 | +0.0000 |
| equation_transformation | 84 | 0.250 | 0.250 | +0.0000 |
| gravitational_constant | 83 | 1.000 | 1.000 | +0.0000 |
| numeral_conversion | 83 | 1.000 | 1.000 | +0.0000 |
| text_encryption | 83 | 0.590 | 0.590 | +0.0000 |
| unit_conversion | 83 | 1.000 | 1.000 | +0.0000 |

**Every task category moves by exactly 0.0pp on v9 dev_frozen.** The
extractor accounts for zero of the 11.4pp gap.

## Representative disagreements where old/new extracted strings differ

### v9 dev_frozen (3 disagreements, all equation_transformation, all
`old_correct == new_correct == False`)

| id | gt | old_extracted | new_extracted | who's right |
|---|---|---|---|---|
| d911e8e1 | `':\\:'` | `'` | `'}})<'` | new (full literal) — model's answer was wrong anyway |
| 4bb8c6cd | `']}\\!'` | `!` | `!}*/\\` | new — model's answer was wrong anyway |
| e3bf0c2c | `'"`' | `` `' `` | `` `'}' `` | new — model's answer was wrong anyway |

Raw text for d911e8e1 ends with `\boxed{'}})<'}` — the answer literally
contains `}`. Brace-walking captures `'}})<'` (correct full literal); the
old regex stops at the first `}` and captures just `'`. Verify against the
ground truth fails in both cases because the model's reasoning produced the
wrong answer.

### baseline LoRA, dev_frozen_shuffled (20 disagreements, delta -0.6pp)

Sole driver of the -0.6pp delta: 3 `gravitational_constant` rows where the
raw output contains a single `\boxed{N}` followed by extensive trailing
text (post-`</think>` reasoning that includes unbalanced `}` characters
from LaTeX `\begin{aligned}...\end{aligned}` or `\[...\]` blocks).

Example (id=c2ebca34, gt='32.81'):
- Raw structure: `... \boxed{32.81}.\n\nWait, but let me check once more ...`
  (then ~1000 chars more reasoning with LaTeX `\]` `\)` etc., eventually
  ending in `}`).
- `old_extracted`: `'32.81'` → verifies True ✓
- `new_extracted`: `'32.81}.\n\nWait, but let me check once more with exact calculation...'`
  (everything up to the LAST `}` in the segment) → fails to parse as float,
  falls back to string compare, fails ✗.

This is the **expected trade-off of brace-walking**: it correctly captures
literal `}` inside the answer, but over-captures when the model writes
unbalanced LaTeX after the box. Net effect on baseline LoRA: -0.6pp.

This same trade-off applies to the Kaggle metric, since it uses the same
algorithm. So the trade-off does not contribute to the local-vs-Kaggle
gap — both sides experience it equally.

### Other disagreements (gravitational_constant pattern)

Across the dev_frozen_shuffled files, most disagreements look like:
- gt = `70.34`, raw ends with `\boxed{70.32\text{ m}}`
- old_ext: `70.32\text{ m` (drops the closing `}`)
- new_ext: `70.32\text{ m}` (keeps it)
- Both fail to verify against `70.34` (model got the wrong number); neither
  fail is because of the extractor.

## Conclusion

**The extractor fix moves v9 dev_frozen by exactly 0.0pp.** It moves the
overall gap (11.4pp) by 0.0pp. The 17 files re-scored show deltas of
[-0.6pp, +0.0pp], all driven by `gravitational_constant` over-capture on
the baseline LoRA, not anything that affects trained models.

### Implications

1. The Kaggle 0.58 vs local 0.694 gap is **not explained by the extractor**.
   Brace-walking and the naive regex give the same `verify()` outcome on
   every prediction in the v9 dev_frozen file, even when extracted strings
   differ — the model either got the answer right enough for `verify()` to
   accept, or got it wrong in a way no extractor can rescue.

2. The pre-fix extractor was nevertheless buggy in cases like
   `\boxed{}52}` and `\boxed{\frac{1}{2}}`. The fix is still correct and
   the parity test now enforces it. Future regressions in this code path
   are blocked.

3. **Real runtime divergence remains.** The 11.4pp gap must come from
   somewhere else. Candidates (to be evaluated in Step 3 and Step 4):
   - vLLM runtime flags (`enable_chunked_prefill`, `enable_prefix_caching`,
     `max_lora_rank`, `max_num_seqs`)
   - Sampling parameters (`temperature=1.0`, `top_p=1.0`, `max_tokens=3584`)
   - Prompt template (chat template, instruction wording)
   - Backbone weights (Kaggle's `nemotron-3-nano-30b-a3b-bf16` snapshot
     vs whatever local uses)
   - LoRA adapter format (see HUIKANG_COMPARISON.md, RANK_PATTERN_REPORT.md)
   - dev_frozen ≠ Kaggle test set (no overlap guarantee)

4. **TODO (flagged, not fixed in this step):** `src/evaluation/run_eval.py`
   already saves raw_generations in `runs/eval/*-raw-*.json`. Several
   other eval scripts in `src/training/eval_*.py` save only `raw_head`
   (160-char truncation) — those files cannot be re-scored. Consider
   standardizing on full `raw` everywhere.

### Phase 1 verdict signal

Per the original spec:
- ≥9pp closure → confirmation-only Phase 1.
- partial closure → narrow investigation.
- 0pp closure (this result) → **full Phase 1 investigation** of runtime
  divergence is required.
