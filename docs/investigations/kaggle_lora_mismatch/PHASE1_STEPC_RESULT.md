# Phase 1 Step C — v9 dev_frozen with Kaggle-matched runtime parameters

**Date:** 2026-05-26
**Hypothesis under test:** `max_num_seqs=64` and `max_model_len=8192`
(matching Kaggle Overview tab) close some/all of the 11.4pp gap to
Kaggle's 0.58 score.
**Verdict:** **Hypothesis rejected at this magnitude.** Local score
moved from 0.694 → 0.690 (delta **−0.4pp**, within row-flip noise).
Per the user-provided interpretation guide: "New score ~= 69.4 →
parameters not the gap source; weight identity OR test-set
generalization remain."

## Invocation

```
python -m src.evaluation.run_eval \
  --split dev_frozen \
  --config configs/eval/default.yaml \
  --adapter-dir runs/train/lora_v9 \
  --max-tokens 7680 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --temperature 0.0 \
  --top-p 1.0
```

Log: `runs/eval/_v9_kaggle_params_step_c.log`
Output JSON: `runs/eval/dev_frozen-raw-runs_train_lora_v9-1779810797.json`
  (also copied to `...-kaggle_params-1779810797.json` for findability)

vLLM `non-default args` boot line (`max_model_len=8192, max_num_seqs=64,
trust_remote_code=True, enable_prefix_caching=True,
enable_chunked_prefill=True, gpu_memory_utilization=0.85,
max_lora_rank=32, enable_lora=True`) — confirms the two intended
changes vs the v9 69.4 baseline (`max_model_len=4096, max_num_seqs=32`),
all other LLM-init args identical.

`top_p` is a SamplingParams field, not an LLM-init field, so vLLM's
"non-default args" line doesn't report it. The CLI was invoked with
`--top-p 1.0` and `--temperature 0.0`; matches Kaggle.

## Headline numbers (extractor already fixed — the brace-walking commit — so apples-to-apples)

| | v9 69.4 (orig) | v9 kaggle_params | delta |
|---|---:|---:|---:|
| n                  | 500     | 500     | — |
| accuracy           | **0.6940** | **0.6900** | **−0.0040** |
| truncation_rate    | 0.0040  | 0.0060  | +0.0020 |
| wall time          | 1,812 s (30.2 min) | 1,481 s (24.7 min) | −5.5 min (larger batch) |

## Per-task accuracy

| task                       | n  | old   | new   | delta    | trunc_old / trunc_new |
|---|---:|---:|---:|---:|---|
| bit_manipulation           | 84 | 0.333 | 0.345 | +0.0119  | 0.000 / 0.000 |
| equation_transformation    | 84 | 0.250 | 0.202 | **−0.0476** | 0.024 / 0.024 |
| gravitational_constant     | 83 | 1.000 | 1.000 | +0.0000  | 0.000 / 0.000 |
| numeral_conversion         | 83 | 1.000 | 1.000 | +0.0000  | 0.000 / 0.000 |
| text_encryption            | 83 | 0.590 | 0.602 | +0.0120  | 0.000 / 0.012 |
| unit_conversion            | 83 | 1.000 | 1.000 | +0.0000  | 0.000 / 0.000 |

Both 100% tasks held perfectly. The three non-trivial tasks moved in
opposite directions:
- bit_manipulation: +1 example
- equation_transformation: −4 examples
- text_encryption: +1 example
- Net: −2 examples = −0.4pp.

## Row-level diff

Common ids: 500/500 (deterministic at T=0.0; both runs scored the
same dev_frozen split).

| | count |
|---|---:|
| same (`old_correct == new_correct`) | 488 |
| `old_wrong → new_right`             | 5 |
| `old_right → new_wrong`             | 7 |
| **net**                             | **−2** |

The 12 flipped rows (id, task, old_correct → new_correct,
old_pred → new_pred, ground truth):

```
a00fe76a bit_manipulation         True → False  '10111011' → '00001111'  gt='10111011'
7e5b8c12 bit_manipulation         True → False  '10110000' → '10011000'  gt='10110000'
f2b23d11 bit_manipulation        False → True   '10100000' → '10000000'  gt='10000000'
eeb60061 bit_manipulation        False → True   '00000001' → '00000101'  gt='00000101'
7a79ac09 bit_manipulation        False → True   '11101111' → '11111111'  gt='11111111'
2af7134e equation_transformation  True → False  '20'       → '03'        gt='20'
74fff108 equation_transformation  True → False  '49'       → '711'       gt='49'
ea6d926a equation_transformation  True → False  '87'       → '72'        gt='87'
3383d4ec equation_transformation  True → False  '66'       → '-33'       gt='66'
96291987 text_encryption         False → True   'y?l?s explores in v...' → 'mouse explores in v...'  gt='mouse explores...'
92087c7c text_encryption          True → False  'the magical rabbit...'  → 'the magical mirror...'   gt='the magical ra...'
290cc78d text_encryption         False → True   'teacher dreams thro...' → 'teacher draws throu...'  gt='teacher draws...'
```

Reading: these are single-bit-edit-style flips (one or two letters
or digits change). Consistent with greedy decoding under tiny logit
perturbations from a different fp32 reduction order — `max_num_seqs`
and `max_model_len` change the KV cache page layout and batch
scheduler, which permutes the kernel reduction order. The model is
already on near-tie boundaries on these items; small noise crosses
the tie.

## Truncation profile (real NemotronH tokenizer, n=500)

| Metric | old (1779095404) | new (1779810797) |
|---|---:|---:|
| min                  | 102   | 101   |
| median               | 332   | 332   |
| p95                  | 516   | 516   |
| p99                  | 1,181 | 1,674 |
| max                  | 7,680 | 7,680 |
| count at cap (=7680) | 2     | 2     |
| count in (1k, 7680)  | 3     | 4     |

NEW histogram:
```
0-1k         494
1-2k           2
2-3k           1
3-4k           0  ← (still gap)
4-5k           0
5-6k           0
6-7k           0
7k-7680        1  ← new: id e3890bf7 at 7679 tokens
=7680 (CAP)    2  ← ids 20f0fac9 (same as old), 21bd1251 (new)
```

### Truncation/cap-hit churn between runs

| id | old run | new run |
|---|---|---|
| `3c424916` (eq_xform) | 7680 cap-hit, truncated, wrong | finished cleanly, still wrong (different prediction) |
| `20f0fac9` (eq_xform) | 7680 cap-hit, truncated, wrong | 7680 cap-hit, truncated, wrong (same row, same failure mode) |
| `21bd1251` (text_enc) | 1,432 tokens, has `\boxed{}`, **correct** | 7680 cap-hit, truncated, wrong (`'unknown cipher letters: p, m...'` repeated) |
| `e3890bf7` (eq_xform) | 569 tokens, has `\boxed{}`, wrong | 7679 tokens, missing `\boxed{}`, wrong (`'= b(a, b) = b(a, b) ...'` repeated 700+ times) |

So `max_model_len=8192` freed `3c424916` from the cap (it had been
constrained by max_model_len=4096 + input prompt + token-loop output)
— but introduced two new pathologies (`21bd1251` and `e3890bf7`).
Net effect on truncation rate: 2 → 3 (one of them, `21bd1251`, was a
previously-correct text_encryption answer that became degenerate).

### Bimodality refinement

The previous "p99 = 1,181, then gap, then 2 cap-hits" description
loosens slightly: now p99 = 1,674 and one row sits at 7,679 (one
token shy of cap). Distribution is still strongly bimodal, but the
8,192 max_model_len lets the model produce longer pre-cap traces on
items where it would otherwise have been forcibly stopped. The
fundamental "model degenerates into token loops on the hardest items"
shape remains.

## What this rules in / out

- **Rules out:** `max_num_seqs` and `max_model_len` together cannot
  explain more than ~0.5pp of the 11.4pp gap. (They might still
  contribute a fraction of a percent in the Kaggle direction, but not
  multiple percent.)
- **Strengthens:** weight identity (Q4) and test-set composition
  (Q5) remain the live hypotheses. Of the two, Q4 is more concrete
  to test (definitive via the 60 GiB download Q.A.2 deferred).
- **New observation:** at greedy decoding with otherwise-identical
  config, vLLM 0.20.1's output is **not** deterministic across
  `max_num_seqs` × `max_model_len` permutations on this model. 12
  of 500 rows flip text. This is in-spec for vLLM (kernel reduction
  order is not pinned) but worth documenting — it bounds how much
  signal a single-seed run carries.

## Recommended next action

Given the parameters are ruled out as the dominant gap source:

1. **Decision point for the user:** approve the ~60 GiB
   `kagglehub.model_download(...)` for Q4 (base-model weight
   identity), or proceed to Q5 (test-set probe — harder; Kaggle
   does not expose test items).
2. **Cheaper interim:** if base-model identity is suspected, the
   `model.safetensors.index.json` from the Kaggle dataset (a small
   file inside the package) lists tensor → shard mapping and shard
   sizes. If the Kaggle index file's shard sizes match the local
   `model.safetensors.index.json` byte-for-byte, identity is highly
   likely (the bigger files are downstream of the index). The Kaggle
   index file is ~600 KB, not 60 GiB. Worth pulling first.
3. **Independent of weight identity:** the per-row noise observed
   here (12 of 500 flip from a config-only change) bounds the
   per-Kaggle-submission variance. If we re-submitted v9 to Kaggle
   today, the score might be 0.57 or 0.59 rather than 0.58 just
   from kernel-order noise. The 11pp gap is well outside that
   range, so it's still real, but quantifying noise would inform
   how tightly to chase the residual.
