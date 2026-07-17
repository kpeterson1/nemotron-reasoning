# Phase 1 (vLLM parity) — scope

**Date:** 2026-05-26 (revised after parity recount)
**Verdict:** **Narrow investigation** of the remaining runtime divergences.

This revision retracts the earlier (wrong) "14 of 16 parity dimensions match"
claim and the "T=1.0 hypothesis." Both errors traced back to using the
defaults in `docs/kaggle_metric_source.py` as a runtime source of truth,
which CLAUDE.md now explicitly forbids ("Defaults are not values"). The
authoritative runtime values come from the Kaggle competition Overview tab.

## Recount: corrected parity table with explicit source of truth

For each dimension, the **Source** column flags how the Kaggle value was
established and how the local value was established.

| Parameter | v9 69.4 run actual | Kaggle (Overview tab, per CLAUDE.md) | Match? | Source-of-truth (Kaggle) | Source-of-truth (local) |
|---|---|---|---|---|---|
| `max_lora_rank`         | **32**           | **32**       | match    | (a) confirmed Overview | (a) confirmed `_lora_eval_500.log` |
| `max_tokens`            | **7680**         | **7680**     | match    | (a) confirmed Overview | (a) confirmed v9-run-report commit msg + `--max-tokens 7680` would have to have been passed (run_eval default is 3584) |
| `top_p`                 | **1.0** (assumed default) | **1.0** | match (b)| (a) confirmed Overview | (b) inferred from run_eval default — not logged |
| `temperature`           | **0.0**          | **0.0**      | match    | (a) confirmed Overview | (a) confirmed v9-run-report commit msg |
| `max_num_seqs`          | **32**           | **64**       | **MISMATCH** | (a) confirmed Overview | (a) confirmed `_lora_eval_500.log` `max_num_seqs=32` |
| `gpu_memory_utilization`| **0.85**         | **0.85**     | match    | (a) confirmed Overview | (a) confirmed `_lora_eval_500.log` |
| `max_model_len`         | **4096**         | **8192**     | **MISMATCH** | (a) confirmed Overview | (a) confirmed `_lora_eval_500.log` `max_model_len=4096` |
| `trust_remote_code`     | **True**         | **True**     | match    | (a) confirmed Overview | (a) confirmed `_lora_eval_500.log` non-default args |
| `enable_prefix_caching` | **True**         | **True**     | match    | (a) confirmed Overview | (a) confirmed `_lora_eval_500.log` |
| `enable_chunked_prefill`| **True**         | **True**     | match    | (a) confirmed Overview | (a) confirmed `_lora_eval_500.log` |
| `dtype`                 | **bfloat16**     | **`auto`** (→ bf16 for this checkpoint) | match (b) | (a) confirmed Overview | (a) confirmed `_lora_eval_500.log` `dtype=torch.bfloat16` |

**Substantive divergences vs the v9 69.4 actual run: 2** (`max_num_seqs`, `max_model_len`).

The earlier draft also listed `max_tokens` (3584 vs 7680) and
`trust_remote_code` (unset vs True) as divergences. Those were against
`eval_kaggle_exact.py` **defaults**, which were not actually used to
produce v9 69.4. v9 69.4 was produced by `src/evaluation/run_eval.py`
+ `src/inference/generate.py`, run from the CLI with `--max-tokens 7680`,
which set `max_num_seqs=32` (CLI flag — run_eval.py's default is 16),
trust_remote_code=True (hardcoded in `src/inference/generate.py:52`).

### Where the v9 69.4 evidence comes from on disk

| Claim | Evidence |
|---|---|
| max_tokens=7680, temp=0.0 | the v9-run-report commit message: "Eval @ dev_frozen (n=500, temp=0.0, max_tokens=7680)" |
| max_num_seqs=32, max_model_len=4096, trust_remote_code=True, enable_chunked_prefill=True, enable_prefix_caching=True, gpu_memory_utilization=0.85, max_lora_rank=32, dtype=bfloat16, model=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | `runs/eval/_lora_eval_500.log` lines 1-3 (vLLM boot "non-default args" dump) |
| run script = `src/evaluation/run_eval.py` (NOT eval_kaggle_exact.py) | `runs/eval/dev_frozen-raw-runs_train_lora_v9-1779095404.json` has `prompt_family`, `system_message`, `user_instruction`, `arm_label` keys which only `run_eval.py` produces. `eval_kaggle_exact.py` produces `results_by_n_temp` nesting and `has_boxed_rate` etc. |
| Prompt template = HARNESS_SUFFIX + apply_chat_template (matches Kaggle) | `src/inference/generate.py:19-22, 98-108` |

## Truncation profile: max_tokens hypothesis falsified

The earlier draft (and the user's stated hypothesis to test) was that
`max_tokens` halving might cause heavy truncation. The v9 69.4 raw
data falsifies this:

| Metric | Value |
|---|---|
| `truncation_rate` (stored, = fraction with no `\boxed{}` in raw) | **0.40%** (2/500) |
| `has_boxed` | 99.60% |
| Raw char-length median | 652 |
| Raw char-length p75 | 753 |
| Raw char-length p95 | 1,085 |
| Raw char-length max | 23,190 |

Approximation note: an earlier draft of this section reported
character/4 as a token proxy. The actual NemotronH tokenizer
(`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, vocab 131,072) was run
on **all 500 raws** to confirm. Mean tokens/(char÷4) ratio on the
longest 10 outputs is **1.76**, ranging up to 2.01 — so char/4 was a
~44% under-estimate per token. The corrected real-token numbers below
supersede the approximation and **strengthen** the falsification (the
distribution is bimodal: everything ≤1,181 tokens or exactly 7,680).

| Metric | Real tokens (NemotronH tokenizer, n=500) |
|---|---|
| Token-count min      | 102 |
| Token-count median   | 332 |
| Token-count p75      | 415 |
| Token-count p95      | 516 |
| Token-count p99      | **1,181** |
| Token-count max      | **7,680** (exactly the cap) |
| Predictions ≥ 7,680 tokens (hit cap) | **2 / 500** (0.4%) |
| Predictions in [1,182, 7,679] tokens | **0 / 500** |

Real-token histogram (1k buckets):
```
0-1k         495  #################################################
1-2k           3
2-3k           0
3-4k           0
4-5k           0
5-6k           0
6-7k           0
7-7680         0
=7680 (CAP)    2
```

Per-task `\boxed{}`-absence rate (same on real or approx tokens):
- bit_manipulation:        0/84   (0.00%)
- equation_transformation: 2/84   (2.38%)
- gravitational_constant:  0/83   (0.00%)
- numeral_conversion:      0/83   (0.00%)
- text_encryption:         0/83   (0.00%)
- unit_conversion:         0/83   (0.00%)

The 2 cap-hit predictions and the 2 stored-`truncated=True` predictions
are **the same two rows** (ids `3c424916` and `20f0fac9`, both
equation_transformation). Both raws end in a token loop — the model
got stuck and ran the budget out. Importantly, both were scored
**`correct=False`** in the stored report:

```
id        task                      tokens  truncated  correct  predicted  answer
3c424916  equation_transformation   7680    True       False    '3234'     '9351'
20f0fac9  equation_transformation   7680    True       False    '6'        '2976'
```

So there is **no over-scoring** of truncated outputs via fallback
extraction in v9 dev_frozen — the user's stated concern ("local was
over-scoring on truncated outputs via fallback extraction") is also
falsified by the actual data. Max effect on score: 0 / 500 = 0pp.

Additional finding: at max_tokens=3584 (the old `eval_kaggle_exact.py`
default, fixed in 5a618b1), v9 dev_frozen would have truncated the
**same 2 outputs**, no more — because no other outputs fall in the
[1182, 7679] token range. So the trap fix in 5a618b1 wouldn't have
changed the v9 number either; it's still worth fixing because Kaggle's
public test items may have a different token-length distribution.

So `max_tokens` is **not** the gap source. Falsification confirmed
with real tokenizer counts.

## Remaining substantive divergences

After the recount, only two parameters differ between the v9 69.4 run
and Kaggle:

### D1. `max_num_seqs`: 32 (local) vs 64 (Kaggle)
At T=0.0 (greedy), the per-sequence sampling is deterministic given the
logits. Different batch sizes change the kernel reduction order in the
fused MoE and possibly attention, producing tiny logit-noise (typically
within 1e-5 magnitude). This can flip a token under near-tie
distributions but does not generally rewrite reasoning trajectories.
Plausible contribution: ±1pp, not ±11pp.

### D2. `max_model_len`: 4096 (local) vs 8192 (Kaggle)
For v9 dev_frozen, output lengths are <300 tokens at p95. Input
prompts (problem + HARNESS_SUFFIX, chat-templated) are short. Even at
1k input + 6k output, total stays within 7k tokens — under both 4096
and 8192. But: **the v9 max output was 5,797 approx-tokens.** If the
input is 500 tokens, that's 6,297 total — over 4096. So `max_model_len=4096`
would have **rejected** that prediction. Need to check whether the
stored "raw" for those long outputs was actually returned in full or
was silently truncated.

Also: `max_model_len` affects vLLM's KV cache page layout and may
interact with `enable_prefix_caching` differently across values. Plausible
contribution: depends on whether long outputs exist on Kaggle's harder
items (which we cannot directly observe).

## What other dimensions could explain the gap (and weren't recounted)

Now-flagged candidates that go beyond the vLLM parity table:

1. **Base-model weight identity (Q4 from PRIOR_FINDINGS_RECONCILIATION).**
   Kaggle loads from `kagglehub.model_download('metric/nemotron-3-nano-30b-a3b-bf16/transformers/default')`;
   local uses HF `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. SHA256
   parity of these weight shards is **unverified**.

2. **Test-set identity (Q5).** `dev_frozen` is locally constructed;
   Kaggle's public test set composition is not exposed. The 11.4pp
   could be a difficulty distribution difference.

3. **Tokenizer/chat-template state at runtime.** Both sides call
   `apply_chat_template` with the same flags; the rendered prompt
   parity has been asserted but not byte-diffed on a real item.

4. **`enable_prefix_caching=True` interactions on Mamba state.** The
   v9 boot log mentions `Mamba cache mode is set to 'all'` and "Its
   support for Mamba layers is experimental." Different `max_model_len`
   and `max_num_seqs` change the cache layout. The deterministic
   effects are bounded but not zero.

## Revised next-action hypothesis

Given the corrected evidence:

1. **`max_tokens` is not the constraint.** Drop that hypothesis.
2. **Temperature is already matched** (both T=0.0). Drop the T=1.0
   hypothesis.
3. **The two real runtime divergences (`max_num_seqs`, `max_model_len`)
   plausibly account for ~1-3pp combined**, not the full 11.4pp.
4. **The most likely big-ticket remaining source is base-model
   weight identity (Q4).** It is concretely testable on-disk
   without a GPU run.

## Phase 1 — revised work plan

### Step A (priority 1, cheap, no GPU): base-model weight identity

Check whether the local HF cache for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
matches the Kaggle-distributed `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`.
Approach:
- Inspect `~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/`
  for the 13 safetensors shards (already on disk per the load log:
  "Loading safetensors checkpoint shards: 13/13").
- Compute SHA256 of each shard.
- Compare against any kagglehub snapshot we can pull (this requires
  network + Kaggle credentials; flag and defer if not feasible
  on Spark 1 without user action).

Effort: ~30 min if kagglehub access is available; otherwise document
the SHA256 of local shards as a reference point for future comparison.

### Step B (priority 2, cheap, no GPU): prompt-byte parity sanity

Take one dev_frozen item. Render through:
- `src/inference/generate.py` (used by run_eval.py)
- The exact `docs/kaggle_metric_source.py:298-315` lines (using the
  same local tokenizer).

Byte-diff. Expected: identical (HARNESS_SUFFIX and apply_chat_template
flags are the same). Verifies the assumption.

Effort: ~15 min.

### Step C (priority 3, requires GPU): one minimal-delta local run

Re-run v9 dev_frozen with `max_num_seqs=64, max_model_len=8192`
(matching Kaggle exactly), everything else held equal. If the local
score moves from 0.694 toward 0.58, these two flags are load-bearing.
If it stays ~0.694, the gap is elsewhere (Q4 base weights, Q5 test
set, or kernel-level differences).

Estimated GPU time: ~30 min on Spark 1. Worth running, but only after
Steps A and B.

**Done 2026-05-26 — see `PHASE1_STEPC_RESULT.md`.** Headline: local
moved 0.694 → 0.690 (delta −0.4pp; within noise). Hypothesis rejected
at this magnitude. 12 of 500 rows flipped, consistent with kernel-order
perturbations from changed batch/cache layout. Live remaining
hypotheses: Q4 (base-model weight identity) and Q5 (test-set
composition).

### Step D (deferred): full test-set identity probe

Out of scope for Phase 1 unless A-C all fail to close. Would require
either obtaining a portion of the actual Kaggle test set (unlikely)
or constructing a held-out distribution check that's a different
project.

## Observations not pursued in Phase 1

### Bimodal generation length on equation_transformation

When all 500 v9 dev_frozen raws were tokenized with the real NemotronH
tokenizer, the distribution turned out **bimodal**: 498/500 below
1,200 tokens (p99 = 1,181), then a 5,500-token gap, then exactly 2
outputs at the 7,680-token cap. Both cap-hits are on
`equation_transformation` (ids `3c424916` and `20f0fac9`). The model
isn't producing wrong-but-terminated answers on these — it's running
the clock out in token loops. Both are scored `correct=False` against
ground truths that bear no resemblance to the predicted strings
(`'3234'` vs `'9351'`, `'6'` vs `'2976'`).

This is a **Phase 2 lead, not a Phase 1 gap source.** The cap-hit
rate is 0.4% (2/500), too small to move the score even if perfectly
recovered. But the qualitative behavior — degenerating into a token
loop only on the hardest task category — suggests the training
trace's terminal-token / `\boxed{}` boundary on equation_transformation
items may be malformed in a way the model has learned to imitate
poorly. Worth investigating after Phase 1 resolves the runtime gap.

### top_p=1.0 confirmation

`references/huikang/README.md` explicitly states Kaggle uses
`top_p=1.0`, removing what was an inferred-from-defaults claim about
the local v9 run. Both sides confirmed at top_p=1.0; this dimension
is closed.

### Loop-on-hard-items pathology now spans two task categories

`PHASE1_STEPC_RESULT.md` records two new degenerate outputs that
appeared when `max_model_len` went from 4096 → 8192:
- `21bd1251` (text_encryption): previously correct at 1,432 tokens
  with `\boxed{}`; now stuck at 7,680 tokens repeating "unknown
  cipher letters: p, m (both known: p" indefinitely.
- `e3890bf7` (equation_transformation): now produces 7,679 tokens
  of `'= b(a, b) = b(a, b) = ...'` repetition.

Combined with the prior `equation_transformation` cap-hits, the
"model degenerates into a token loop instead of finishing the
problem" pathology now appears in **two** task categories. The
previous interpretation (training-trace terminal-token boundary
malformed only on `equation_transformation`) understates the
problem: the failure mode is general, not category-specific.

This is a **Phase 2 training-data lead**, not a Phase 1 gap source.
Combined cap-hit rate remains 2/500 = 0.4% (the eq_xform
3c424916 freed up by the larger context cancels the new text_enc
hit), so it can't account for the 11pp gap. But the qualitative
trend (the model has learned to imitate "keep generating" rather
than "stop and box" on items it can't solve cleanly) suggests
SFT trace surgery in Phase 2.

## Out of scope and trap-fix

Independent of this scope doc, the next commit fixes the
`eval_kaggle_exact.py` defaults so the same trap doesn't bite the
next session:
- `--max-tokens` default: 3584 → 7680
- `--max-model-len` default: 4096 → 8192
- `--max-num-seqs` default: hardcoded 16 → CLI flag with default 64
- Add `trust_remote_code=True` to the `LLM(...)` init

This is a single-purpose commit, NOT bundled with Phase 1 work, NOT a
new run.

## Single recommended next action

**Step A.** Compute SHA256 of local HF NemotronH-3-Nano-30B BF16
safetensors shards as a baseline. If kagglehub credentials are
available, fetch the Kaggle snapshot and compare. This is the
cheapest way to test the highest-probability remaining hypothesis
given the recount: base-model weight identity.

If Step A reveals identity, move to Step C (one GPU run with
max_num_seqs=64 and max_model_len=8192). Skip Step B unless A and C
both fail — prompt rendering parity is already strongly suggested by
the code path.

---

# Phase 1 conclusion (2026-05-26)

Phase 1 began with the working hypothesis that the 11.4pp local
(0.694) vs Kaggle (0.58) gap had a small number of identifiable
runtime causes — extractor, sampling temperature, token budget,
batching, weight identity, or test-set composition. After
investigation, **every concrete vLLM-runtime / extractor / packaging
hypothesis is falsified or shows movement below the measurement
noise floor**. The remaining unidentified factor is most plausibly
real generalization on Kaggle's test distribution.

## Verdict on each hypothesis

| Hypothesis | Status | Evidence |
|---|---|---|
| **H1: Extractor mismatch** is the gap source | **Falsified.** Re-score delta 0.0pp on v9 dev_frozen. | `EXTRACTOR_RESCORE.md`. After fixing `src/evaluation/extract_answer.py` to use brace-walking (the brace-walking extractor commit), re-scoring all 17 saved raw-output files shows 0.0pp delta on v9; range across all files [−0.6pp, +0.0pp]. |
| **H2: Kaggle uses T=1.0 sampling**, local was greedy | **Never true.** Kaggle runs at T=0.0 (greedy). | Kaggle Overview tab (CLAUDE.md "Confirmed Kaggle leaderboard parameters") and `references/huikang/README.md` both say T=0.0. Local v9 69.4 also ran at T=0.0 per the v9-run-report commit message. |
| **H3: max_tokens halving** (3584 vs 7680) causes truncation-driven over-scoring | **Falsified by tokenizer counts.** | Real NemotronH tokenizer on all 500 v9 raws: p99 = 1,181 tokens, then 2 outliers at exactly 7,680. Both cap-hits scored correct=False. At max_tokens=3584, only the same 2 outputs would have truncated. `PHASE1_VLLM_PARITY_SCOPE.md` §"Truncation profile". |
| **H4: max_num_seqs (32 vs 64) + max_model_len (4096 vs 8192)** explain the gap | **−0.4pp, inside noise.** Falsified at the multi-pp scale. | `PHASE1_STEPC_RESULT.md`. Local re-ran v9 dev_frozen with `max_num_seqs=64, max_model_len=8192`: score 0.694 → 0.690. 488/500 rows unchanged; 12 single-bit-edit-style flips consistent with kernel-order perturbations (now documented as the noise floor in CLAUDE.md). |
| **H5: Base-model weight identity (Q4)** differs between local HF and Kaggle | **Highly likely identical** (effectively closed without 60 GiB download). | `PHASE1_STEPA_BASE_MODEL_SHA.md` § "Resolution". All 8 small companion files (`model.safetensors.index.json`, `config.json`, `chat_template.jinja`, `tokenizer_config.json`, etc.) are **byte-identical** between Kaggle's `metric/.../1` dataset and local HF revision `cbd3fa9f...`. Identical index implies identical tensor-to-shard map and identical total_size. Exotic same-name-different-bytes case not ruled out without one full shard download (~5 GiB), deferred. |
| **H6: Test-set composition (Q5)** — dev_frozen ≠ Kaggle public test distribution | **Live, untestable without external action.** | Kaggle does not expose the test items. Cheapest signal would be the Kaggle submission ladder itself showing different per-category accuracy than local; but per-category Kaggle results are not visible. Phase 2 starting point, not Phase 1. |

## What's left after Phase 1

- **±0.5pp noise floor** (CLAUDE.md "Measurement noise"). At least
  half a percentage point of the 11pp gap is just kernel-order
  jitter between the two environments.
- **Up to a few percent** could be hardware/driver/CUDA stack
  differences between Spark 1 (NVIDIA GB10) and Kaggle's runtime
  (typically T4 or L4 per the competition setup). Same vLLM kernel
  + different fp32 precision behavior + different cudagraph
  capture sizes could compound. Unverifiable without access to
  Kaggle's runtime environment.
- **The remainder (~7-10pp) is most plausibly real generalization
  difference between local dev_frozen and Kaggle public test set.**
  This is the residual after everything testable has been ruled
  out.

## Recommended pivot

**Stop debugging the gap. Start improving the model.**

The 11pp gap is most likely a combination of:
1. Genuine generalization deficit (the model overfits to dev_frozen-like
   items, underperforms on whatever Kaggle's distribution looks like).
2. Hardware/kernel-level noise we cannot reproduce.

Both are addressed by training a better model, not by chasing more
parity. Phase 2 should focus on training-data improvements where
local signal is actionable. **The Phase 2 leads visible from Phase 1
work** (do NOT scope here — saved for the Phase 2 kickoff session):

- **Loop-on-hard-items pathology** spanning equation_transformation
  AND text_encryption. Model generates token loops to the cap rather
  than producing a wrong-but-terminated answer. Suggests
  training-trace surgery on the `</think>` → `\boxed{}` boundary.
- **equation_transformation at 25%** is the weakest non-degenerate
  category. Three of the 7 right→wrong flips in Step C came from
  this category — the model's correct answers here are luck-on-near-tie,
  not robust reasoning.
- **bit_manipulation at 33%** despite v9 training specifically on
  bit_manip traces. Solver-coverage gap (per `runs/eval/bit_manip_solver_coverage.json`)
  + wrong-rule-trained behavior (per the v9-run-report commit's post-eval note)
  suggest the solver mix needs revisiting before more SFT cycles.

Phase 1 is closed.
