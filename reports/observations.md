# Observations Log

Append-only log of non-obvious findings, data quirks, harness behavior, and decisions. Newest entries at the bottom of each day's section.

---

## 2026-05-06 — Repo scaffold complete

- All 21 unit tests pass after one regex fix: the "final answer is" fallback in `extract_boxed_answer` originally used `[^\n\.]+`, which truncated decimals like `3.14` to `3`. Changed to `[^\n]+` and trim trailing periods.
- Initial repo commit created.

## 2026-05-06 — Kaggle credentials incident

- The user's `~/.kaggle/kaggle.json` had a literal newline + the word `Today` appended inside the `key` value, breaking JSON parsing.
- A diagnostic command I ran printed the partial key value to the transcript before the regex mask could catch it (the `\n` inside the string broke the regex). Suggested the user expire the token.
- After cleaning the file (regex-trimmed everything after the first non-control run of key chars), the API returned 401. The user reissued credentials; the second attempt succeeded.
- **Lesson for future runs:** when inspecting credential files, mask byte-by-byte rather than relying on a regex over whole strings — control characters can split the match boundary.

## 2026-05-06 — train.csv / test.csv shape

- `train.csv` is 9,500 rows with columns `[id, prompt, answer]`.
- `test.csv` is **only 3 rows** with columns `[id, prompt]`. The three IDs are identical to the first three rows of `train.csv`, so the public test file is a format/sample stub — the real test set is held by the Kaggle harness at scoring time.
- Implication: we have **no held-out test set locally**. All local eval has to come from carving splits out of `train.csv`. The 500-row stratified `dev_frozen` becomes our scoreboard proxy; treat it as fragile (only ~83 examples per task type).

## 2026-05-06 — Task type distribution is uniform

`classify_task` matched 100% of 9,500 train rows (zero `unknown`). Counts:

| task_type                | count | %    |
|--------------------------|------:|-----:|
| bit_manipulation         | 1,602 | 16.9 |
| gravitational_constant   | 1,597 | 16.8 |
| unit_conversion          | 1,594 | 16.8 |
| text_encryption          | 1,576 | 16.6 |
| numeral_conversion       | 1,576 | 16.6 |
| equation_transformation  | 1,555 | 16.4 |

Distribution is essentially uniform (range 16.4–16.9%), so stratified sampling won't materially distort the marginal. The classifier's prompt-prefix matching is sufficient — no need for a smarter classifier.

## 2026-05-06 — Splits carved; hard200/speed_bench are length-skewed by task type

Carved splits with seed 42:

| split            | n    | per-type counts                                     | prompt-len min/max/mean |
|------------------|-----:|-----------------------------------------------------|-------------------------|
| dev_frozen       |  500 | bit=84, equation=84, others=83 each                  | 181 / 510 / 300         |
| train_remaining  | 9000 | bit=1518, equation=1471, grav=1514, num=1493, txt=1493, unit=1511 | 177 / 510 / 302         |
| hard200          |  200 | bit=84, gravitational=52, text_encryption=64        | 318 / 510 / 410         |
| speed_bench      |   50 | equation_transformation=50                          | 181 / 197 / 190         |

**Key insight: prompt length correlates strongly with task type.** Selecting "longest 200" yields only 3 task types (the ones with verbose problem statements); "shortest 50" yields 100% `equation_transformation`. This is spec-faithful (the prompt asked for length-based selection without re-stratification), but means:

- `hard200` is not a balanced "hard subset" — it's "the long-prompt task types subset". Use it for stress testing on long-context behavior, not for general difficulty.
- `speed_bench` measures inference latency on `equation_transformation` only; it isn't representative of overall throughput.
- If we later want a balanced hard split, we should stratify within each task type by some intrinsic difficulty proxy (e.g., number of examples in the prompt, answer length).

Bug fix during this step: the original `split.py` used `dev_size // n_types` per bucket, yielding 498 (not 500) for `dev_size=500, n_types=6`. Fixed to allocate the 2-row remainder to the first two buckets in sorted order — deterministic across seeds.

## 2026-05-06 — Aligned extract_answer to actual Kaggle metric source

Pulled the metric notebook via `kaggle kernels pull metric/nvidia-nemotron-metric` and exec'd just the `extract_final_answer` and `verify` function ASTs (without the heavy module-level setup) to run side-by-side parity. Found and fixed 4 small divergences in our `src/evaluation/extract_answer.py`:

1. **Empty-only boxed:** Kaggle returns `""` if `\boxed{...}` matches exist but all are empty. We were falling through to the next strategy. Now matches.
2. **Final-answer fallback patterns:** Kaggle uses 4 patterns:
   - `The final answer is:\s*([^\n]+)`
   - `Final answer is:\s*([^\n]+)`
   - `Final answer\s*[:：]\s*([^\n]+)` (note the **full-width colon `：`**)
   - `final answer\s*[:：]\s*([^\n]+)`

   We had only one. The full-width colon variant matters if the model emits Chinese-style punctuation; we now match it.
3. **Trailing period trim:** I had added `rstrip(".")` to the fallback. Kaggle does not. Removed.
4. **Binary string detection:** Kaggle runs `re.fullmatch(r'[01]+', stored_answer)` on the **ground truth only**. We were also checking the prediction. Behavior diverges only when GT is non-binary numeric and prediction happens to look binary (rare). Aligned to GT-only.

Parity verified: **20/20 extract cases, 14/14 verify cases** across a fuzz set covering empty boxed, multi-boxed, full-width colon, unclosed boxed, negative numbers, near-zero abs_tol, binary edge cases. Test suite expanded from 21 to 29 cases to lock in the new behavior.

**Watch-outs not covered by parity:**
- Kaggle's `verify` does NOT preserve sign-aware float comparison; both sides go through `float(...)` so `"  3.14  "` parses fine but anything with units or symbols falls back to string compare. The string fallback is case-insensitive, which means `"XLVII"` vs `"xlvii"` matches but `"24.6 m/s"` vs `"24.6"` does NOT (string compare).
- The metric expects the model output to come from vLLM with `enable_thinking=True`, but `extract_final_answer` is called on the **whole text including `<think>...</think>`**. So if the thinking trace contains `\boxed{...}`, that match gets included in the search and could win over the post-think final answer. Mitigation: training data should not put `\boxed{}` inside `<think>` blocks.
- The Kaggle `score` function has a sneaky line: `submission['prediction'].iloc[0]` is treated as the LoRA path, not the per-row prediction. Predictions are regenerated server-side — we don't pre-compute predictions, just submit the adapter.

These are now codified in `tests/unit/test_extract_answer.py` and `test_eval.py`.

## 2026-05-06 — Nemotron-H is hybrid Mamba/attention + sparse MoE (not pure MoE)

Pulled the model `config.json` while waiting for weights to download. The architecture is more nuanced than "MoE" alone:

- 52 layers total, decomposed via `hybrid_override_pattern = MEMEM*EMEMEM*…`:
  - 23 Mamba (SSM) layers
  - 23 attention layers (with heavy GQA: 32 query heads, 2 KV heads → tiny KV cache)
  - 6 MoE feed-forward layers (`num_experts_per_tok=6` plus a shared expert via `moe_shared_expert_intermediate_size=3712` — DeepSeek-style)
- "30B total / ~3B active per token" comes from MoE expert routing, not from Mamba.
- Implication: KV cache per token is unusually small, so we can run large batches at 4096 ctx on the GB10's 119 GB unified memory. Mamba layers add constant per-token cost regardless of context length, so long contexts are cheap.

Full daily entry at `reports/daily/2026-05-06.md`. Action item for Phase 5: after `get_peft_model`, dump the named LoRA-touched modules to confirm we cover MoE expert weights, not just mamba/attention projections.

## 2026-05-06 — format_training verified with real tokenizer

Loaded `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` tokenizer (vocab 131,072, ChatML-style `<|im_start|>/<|im_end|>` markers) and ran `format_training_example` on the first 3 train rows with synthetic reasoning traces. All 8 markers passed on all 3 rows: chat tokens, `<think>...</think>`, `\boxed{answer}`, harness suffix on user message, trace inside text, answer inside text. The chat template detects inlined `<think>` blocks and does NOT double-wrap, so we don't need to migrate to the `reasoning_content` field — current implementation works directly.

Token counts on row 0: prompt=253, full=295, completion=42. Well under the 4096 max_model_len. Worth noting for Phase 4 — even a long synthetic trace of ~500 tokens leaves plenty of room.

vLLM cold-load measured during smoke test: ~25 sec per safetensors shard × 13 shards ≈ 5.5 min just for weight load (sequential, since checkpoint 58.8 GiB > 90% of available RAM 53.4 GiB → auto-prefetch disabled). Plus CUDA graph capture. So expect ~7-10 min cold-start before first inference. Not a per-call cost — the LLM is cached in process via `_LLM` global in `src/inference/generate.py`.

## 2026-05-06 — First baseline smoke result (n=50, direct prompt) — TWO CRITICAL FINDINGS

Ran 50 rows of `dev_frozen.jsonl` through the base model (no LoRA, no template, just raw prompt + harness suffix). `acc=0.080`, walltime 27.5 min. Per-task: bit_manipulation 4/50 = 8%. Other tasks: 0/0 (n=0 — see finding #1).

**Finding 1 — `dev_frozen.jsonl` is task-bucketed, not interleaved.** `split.py` writes per-bucket samples in alphabetical task-type order, so a sequential prefix of dev_frozen is dominated by one task: rows[0:50] is 100% bit_manipulation, rows[0:100] is 84 bit_manipulation + 16 equation_transformation, etc. **Implication**: any `--limit N` smoke test with N<500 gives a skewed signal. **Fix**: added `datasets/splits/dev_frozen_shuffled.jsonl` (seed=42 shuffle of the same rows). Use the shuffled view for diagnostic sampling; keep `dev_frozen.jsonl` as the canonical frozen file (immutable). For full-split eval, ordering doesn't matter so either file is fine.

**Finding 2 — At max_tokens=3584 (harness setting), 94% of thinking traces are truncated.** Numbers from the 50-row smoke:

|                         | count | % |
|-------------------------|------:|--:|
| Output contains `</think>` (trace closed) | 3 | 6% |
| Output contains `\boxed{` (final answer present) | 5 | 10% |
| Correct answer | 4 | 8% |
| Raw output length: median 8,890 chars / max 11,444 chars | – | – |

The chat template emits `<|im_start|>assistant\n<think>\n` as the prompt suffix, so the model's generated tokens start *inside* the think block. To produce a valid answer it must close `</think>` and emit `\boxed{...}` — but at max_tokens=3584 it usually runs out of tokens mid-reasoning. The 4 correct answers all happen to fall in the small tail that did finish. extract_answer's "last numeric in text" fallback then captures noise from the truncated trace ("0", "1", "2") for the rest.

**This is the dominant signal.** Prompt selection is secondary to thinking-budget management. Possible mitigations to explore:
- Add an explicit "Be brief — under 1500 thinking tokens" instruction to the user message.
- Use a prompt that pushes the model toward direct answer writing rather than verbose self-reasoning (e.g., a minimal `direct` template).
- After SFT, training data should reward concise traces (synthetic data should have ~500-1500-token traces, not 3000+).
- Consider whether the harness's `max_tokens=3584` is actually a soft cap — if the leaderboard truly applies it, this is a hard architectural constraint and SFT must teach concision.

**Throughput numbers worth recording**:
- Aggregate output throughput at saturation: 140 tok/s (from vLLM's tqdm)
- Effective throughput: 50 prompts × ~3000 generated tokens / 1260 s of inference ≈ 119 tok/s
- Single-prompt throughput (n=1 test earlier): 13 tok/s
- Speedup from batching: ~10×. Limited by GB10 LPDDR5X memory bandwidth (CPU+GPU shared) and per-token MoE expert routing.
- Cold start: weight load 5:24 + engine init 0:45 = ~6 min (init cached after first run thanks to torch.compile cache).
- At 140 tok/s aggregate, full dev_frozen sweep (500 prompts × 3584 max) ≈ 3.6 hours per prompt family (assuming most go to max_tokens). A 5-family sweep ≈ 18 hours.

## 2026-05-07 — Two-arm n=50 cross-task baseline + 'be brief' (dev_frozen_shuffled[0:50])

Ran both arms back-to-back in one process so weight load was paid only once. Same 50 rows, mixed across 6 task types (4–13 per type).

| metric | baseline | brief | delta |
|---|---:|---:|---:|
| overall accuracy | 36.0% | **48.0%** | +12.0pp |
| overall truncation rate | 54.0% | 50.0% | −4.0pp |
| mean raw output (chars) | 7,606 | 6,661 | −12% |
| inference walltime | 1513s | 914s | −40% |

Per task:

| task | n | baseline acc | brief acc | Δacc | baseline trunc | brief trunc |
|---|---:|---:|---:|---:|---:|---:|
| gravitational_constant | 13 | 46.2% | **92.3%** | **+46.1** | 38.5% | 7.7% |
| text_encryption | 7 | 0.0% | 14.3% | +14.3 | 100% | 85.7% |
| bit_manipulation | 9 | 0.0% | 11.1% | +11.1 | 100% | 88.9% |
| numeral_conversion | 7 | 100% | 100% | 0 | 0% | 0% |
| equation_transformation | 10 | 10.0% | 10.0% | 0 | 60% | **80%** ↑ |
| unit_conversion | 4 | 100% | **50.0%** | **−50.0** | 0% | 50% ↑ |

**Brief arm definition** (from this experiment, not yet in `prompts/`):
- system: `"Solve the puzzle concisely. Show your reasoning in under 1000 tokens, then give your final answer."`
- user prefix: `"Be concise. Do not enumerate all examples — identify the pattern quickly and apply it."`

**Takeaways**:

1. **The earlier bit_manipulation-only smoke (8% acc, 94% trunc) was the worst-case slice.** Across all tasks the base model is 36%, not 8%.
2. **`gravitational_constant` is the dominant beneficiary of brief mode** — went 46% → 92% (n=13 is small but the effect size is large). Hypothesis: this task has a clean formula (`d = 0.5*g*t²` → `g = 2d/t²`); without the concision instruction the model wanders into physics derivations it doesn't need.
3. **`unit_conversion` regressed 100% → 50%** (n=4 noisy, but trunc went 0% → 50%). Hypothesis: the brief instruction either confused the model or cut off a verification step. Easy tasks should be exempted from brief.
4. **`equation_transformation` truncation got WORSE under brief** (60% → 80%) with no accuracy gain. Hypothesis: this task genuinely requires per-symbol enumeration that brief discourages.
5. **`numeral_conversion` is 100% on both arms** (n=7) — appears to be a "free" task at default settings, modulo small-n noise.
6. **Brief mode runs 40% faster** because shorter completions mean fewer decode steps. This dominates the per-prompt-length comparison.

**Next-move implications**:

- Brief is **net positive but heterogeneous**. Naively applying it costs us on `unit_conversion` and doesn't help `equation_transformation`. The right next prompt is **task-conditional brief** — apply brief instructions ONLY to tasks where it helps (probably gravitational_constant + bit_manipulation + text_encryption).
- The `task_typed` prompt template was scaffolded for exactly this — with task-specific hints. Augmenting that with task-conditional concision instructions is the natural next experiment.
- Sample sizes are too small (n=4 for unit_conversion in particular) to confidently call regressions. A bigger run (n≥150 shuffled, ~25/task) would tighten the per-task confidence intervals.
- The walltime gain matters for cost: at 40% faster, a full 500-row sweep drops from ~2 hours to ~1.2 hours per prompt family.

**Cost recorded**: each arm = ~25 min (baseline) / ~15 min (brief) of inference + shared 6 min cold start = 46 min total walltime for both arms.

## 2026-05-07 — Qualitative inspection: bit_manipulation vs text_encryption failure modes (brief arm, n=5 each)

Pulled raw outputs for all 5 visible bit_manipulation rows and 5 of 7 text_encryption rows from the brief arm to understand WHY they fail. The two task types have **completely different failure modes** even though both show 11–14% accuracy and 86–89% truncation rates.

### bit_manipulation — wrong-strategy failure (more tokens won't help)

All 5 outputs do **hypothesis-test-discard cycles**: propose a transformation (e.g., "rotate left 3", "XOR with 0b10101010", "permute bits"), bit-by-bit verify against the example pairs, find mismatch, abandon, propose next. Each hypothesis-and-check pass burns ~1,500–2,000 chars. At ~8,500 chars before truncation that's 4–5 attempts. **None converge.**

Example endings (mid-thought, all truncated):
- `"Rotated had 1 at pos1 and pos7,8? Actually rotated..."`
- `"Perhaps the transformation is: output = (~input) >> 1"`
- `"Rotating left by 3 positions yields..."`

Implication: even with unlimited tokens the model would keep generating hypotheses at random. What's needed is a **structured search procedure** (try simple ops first: inversion, rotation, swap; then composition; verify each on ALL examples before moving on). This must come from SFT or a much more directive prompt — the brief instruction does not produce it.

### text_encryption — token-budget failure (more tokens would fix most)

All 5 outputs follow the **correct procedure**:
1. Align cipher words to plaintext words (length-based) from examples
2. Build per-letter substitution map (cipher_letter → plain_letter)
3. Decode the test phrase letter by letter

They run out of tokens mid-decode. Example #2 actually arrived at the right answer (`"dragon imagines the colourful crystal"`) in the trace before truncation — but with British "colourful" vs ground-truth American "colorful". That answer would also have failed `verify` on string compare even if it had reached `\boxed{}`.

So text_encryption has TWO failure modes worth tracking:
- (i) Correct trace, truncated before `\boxed{}` (~most cases). Fix: shorter procedure, or longer budget.
- (ii) Spelling locale (British vs American) on full English-word answers. Smaller in absolute terms but worth a normalizer if it shows up systematically. Need to check answer distribution for British/American variant pairs (`colorful`/`colourful`, `gray`/`grey`, etc.).

### Critical cross-task finding: brief is effectively ignored on hard tasks

`</think>` is **absent from ALL 10 inspected outputs** — every one ran to `max_tokens=3584`. The brief system message ("under 1000 tokens") had ZERO observable effect on bit_manipulation and text_encryption trace length. The +12pp overall accuracy gain from brief mode came entirely from tasks where the model could already converge (gravitational_constant especially). **Concision instructions are a soft signal the model can ignore — and does, when stuck.** This argues against treating brief mode as a general fix and reinforces that SFT (or task-conditional structured prompts) is the necessary intervention.

### Action items now codified

1. **bit_manipulation prompt experiment**: write a `prompts/bit_manipulation_structured/v1.yaml` that lists the search procedure explicitly: "First try (a) bit inversion, (b) bit rotation, (c) bit permutation, (d) XOR with a constant. For each candidate, verify against ALL examples. Move to the next candidate only after eliminating the prior. Once verified, apply." Test on n≥30 bit_manipulation rows.
2. **text_encryption prompt experiment**: ask for the substitution map in a tabular single-line form (e.g., `a→x, b→y, ...`) before the decode, to compress the per-letter mapping section.
3. **Spelling normalizer**: scan train.csv answers for common British/American pairs and decide if we need a pre-normalize step on either side. If <0.5% of answers, ignore.
4. **Capture text_encryption budget shortfall**: re-run the same 7 text_encryption rows with `max_tokens=4096` (above harness — purely diagnostic) to confirm the token-budget hypothesis. If accuracy jumps from 14% → 60%+ at higher budget, that confirms the diagnosis and gives an SFT target (teach the procedure to fit in 3584).

## 2026-05-07 — Two negative results: prompt engineering won't fix the hard tasks

Ran two cheap diagnostics to test the hypotheses from the qualitative inspection. Both hypotheses were wrong; both tasks need SFT.

### Test 1 — Structured search prompt regressed bit_manipulation

Wrote `prompts/bit_manipulation_structured/v1.yaml`:

> "For each example pair, compute: XOR, AND, OR, NOT, left shift, right shift, and rotation of the input. Compare each result against the output. Once you find the operation that matches ALL examples, apply it to the test input."

Ran on all 84 bit_manipulation rows in dev_frozen.jsonl at the harness `max_tokens=3584`.

| variant | n | acc | trunc | mean raw_len |
|---|---:|---:|---:|---:|
| brief (system+user prefix) | 9 | 11.1% | 88.9% | 6,661 |
| **structured search prompt** | **84** | **2.4%** | **93%** | **9,269** |

**The structured prompt regressed accuracy** (11% → 2%) and made outputs longer. Likely failure modes:
- The instruction adds ~60 tokens of prefix to every prompt, eating into the thinking budget.
- Forcing brute-force enumeration of XOR/AND/OR/NOT/shifts on every example burns even more tokens than the model's already-bad random hypothesis approach.
- The actual transformations in the puzzles aren't simple ops — they're permutations or compositions the listed operations can't reach.

**Conclusion**: bit_manipulation cannot be rescued by prompt structure. Confirms SFT is necessary.

### Test 2 — Doubling the budget did NOT fix text_encryption

Re-ran the same 7 text_encryption rows from `dev_frozen_shuffled[0:50]` at `max_tokens=8192` (2.3× the harness limit, diagnostic only) with no system message and no user prefix.

| variant | n | acc | trunc | mean raw_len |
|---|---:|---:|---:|---:|
| baseline 3584 | 7 | 0.0% | 100% | ~7,600 |
| brief 3584 | 7 | 14.3% | 85.7% | ~6,700 |
| **baseline 8192** | **7** | **14.3%** | **71%** | **24,412** |

**At 2.3× budget, output length grew 3.7× — but accuracy stayed at 1/7.** Five of seven traces hit `max_tokens=8192` without closing `</think>`. The model uses every additional token it's given without converging.

This kills the qualitative-inspection hypothesis ("right strategy, just out of tokens"). The model isn't running out of space; it's running in circles. Same pathology as bit_manipulation, different surface presentation.

**Conclusion**: text_encryption is not budget-bound. SFT is required for this task too.

### What we know about each task type now

| task | base acc | brief acc | bottleneck | path forward |
|---|---:|---:|---|---|
| numeral_conversion | 100% | 100% | none | leave as-is |
| unit_conversion | 100% (n=4) | 50% (n=4, noisy) | brief regresses; data is small | re-test on bigger n |
| gravitational_constant | 46% | 92% | base model over-reasons | brief-style prompt |
| equation_transformation | 10% | 10% | unclear | needs separate inspection |
| text_encryption | 0% | 14% | model can't converge | **SFT needed** |
| bit_manipulation | 0% | 11% | wrong-strategy + truncation | **SFT needed** |

Three of six tasks (numeral, unit, gravitational) are largely solved by base + concision. Two (bit_manip, text_enc) are blocked until SFT. equation_transformation is unclear and warrants its own inspection (similar to what we did for bit_manip and text_enc).

### Implications for Phase 4 (synthetic data + SFT)

- **Trace length budget for training data**: target ~1500 thinking tokens per example, hard cap ~2500. The base model needs to learn brevity it doesn't have.
- **bit_manipulation traces** must teach a structured search that converges (not just "try ops"). Hand-craft a procedure that handles permutations and compositions, then have a teacher model emit traces that follow it.
- **text_encryption traces** must teach the substitution-map → decode procedure in a compact form (single-line tabular map, then linear decode). Avoid letting the model re-derive the map mid-decode.
- **Use synthetic generation with verification**: only keep traces whose final `\boxed{}` answer is correct. Throw away the rest. We need a teacher model that's better at these tasks — possibly Claude itself, possibly a larger Nemotron, possibly the same model with different decoding settings.

## 2026-05-07 — equation_transformation has a different failure profile (and exposes a harness-suffix bug)

Pulled all 10 equation_transformation rows from the brief n=50 run. The 10% accuracy turns out to be entirely an `answers_match` artifact: the only "correct" row (#4) had `pred='-1'`, `gt='-01'`, both parse as float `-1.0`, so `verify` returned True. **True semantic accuracy on this task is ~0%.**

Two distinct failure modes, both novel:

### Failure A — Multi-operator puzzles waste tokens on irrelevant operators

These puzzles introduce ≥3 secret symbol operators in the example block (e.g., `*`, `` ` ``, `#`) but the test query only uses ONE. The model decodes ALL operators in its thinking trace — operators it doesn't need — burning the budget. By the time it gets to applying the relevant operator it's truncated.

**Implication for SFT**: training traces should *first* identify the operator used in the test query, *then* decode only that operator's examples. The teacher model must demonstrate this triage step.

### Failure B — Harness-suffix poisoning on symbolic-answer tasks

The harness ALWAYS appends `For example: \`\boxed{your answer}\`` to every prompt. On rows whose ground-truth answer is a non-alphanumeric string (e.g., `}@`, `{<<<`, `|&`, `]}\!`), the model sometimes copies the example *literally*:

- Row 2 (gt=`}@`): wrote `\boxed{}` → pred=`''`
- Row 7 (gt=`{<<<`): wrote `\boxed{your answer}` → pred=`'your answer'`

So the harness's own example string becomes a confounder for tasks with unusual answer formats. The model has learned that "your answer" is a legitimate answer template, and emits it when stuck on symbolic outputs.

**This affects more than equation_transformation**: any task that produces non-alphanumeric or non-English answers risks the same failure. Worth checking whether bit_manipulation/text_encryption/etc. ever emit `\boxed{your answer}` too — quick post-hoc grep across all run JSONs.

**Mitigation paths**:
1. SFT training data on equation_transformation must show clean symbolic answers in `\boxed{...}` so the model learns "answer is a literal sequence of characters, not English prose".
2. Consider adding a post-extract filter: if `pred == 'your answer'`, treat as `NOT_FOUND` and re-run. **DO NOT do this** without verifying it matches Kaggle metric behavior — the metric does NOT have such a filter, so this would diverge from leaderboard scoring. Instead, fix at the model-output layer via training.
3. Phase 4 trace generation should use the actual harness suffix in prompts (already handled in `format_training.py`) so training distribution matches inference distribution.

### Updated failure mode taxonomy after all three inspections

| task | brief acc | failure mode | fix |
|---|---:|---|---|
| numeral_conversion | 100% | none | leave alone |
| unit_conversion | 50% (n=4) | possible brief regression on easy task; small n | retest with bigger sample |
| gravitational_constant | 92% | base over-reasons; brief sufficient | use brief/concise prompt |
| **equation_transformation** | ~0% (10% is float-tolerance artifact) | (A) decodes irrelevant operators; (B) "your answer" suffix poisoning | **SFT — operator triage + symbolic-answer training** |
| **text_encryption** | 14% | wrong strategy, can't converge at 8192 budget either | **SFT — compact substitution-map procedure** |
| **bit_manipulation** | 11% | wrong strategy, no convergence; structured prompt regressed | **SFT — structured search procedure** |

### Phase 4 teacher-model decision now sharpened

Three of six tasks need SFT, each with different procedural requirements:
1. **bit_manipulation**: structured search over bit operations including permutations and compositions. The teacher must produce traces that converge — the base model can't.
2. **text_encryption**: substitution-map-then-decode in compact form, fitting in <2500 thinking tokens.
3. **equation_transformation**: triage which operator is used in the test query, then decode only that one. Handle symbolic answers without copying the harness suffix's "your answer" placeholder.

These are quite specific behaviors to teach. Hand-crafted procedures + algorithmic trace generation (filling in puzzle-specific values) might actually work better than a strong teacher LM because we have full control over the trace structure. Worth piloting on bit_manipulation first — it's the most procedural of the three.

## 2026-05-07 — bit_manipulation procedural solver + trace generator

Built a shift-invariant rule solver (`src/data/bit_manip_solver.py`) and a trace generator (`src/data/bit_manip_trace.py`). Result on full train_remaining (1,518 rows):

| metric | value |
|---|---:|
| Rule matched (fits all examples) | 1,095 (72.1%) |
| Solver answer matches ground truth | 1,076 (70.9%) |
| Generalization (matched → correct on test) | 98.3% |
| Walltime | 13 min |
| Trace token p50 / p90 / max | 735 / 933 / 946 |

**Solver design**: searches for rules of the form `output[i] = f(input[i+d_1], ..., input[i+d_K])` for K∈{1,2,3}, offsets in [-7,+7], with edge handling either zero-pad or wrap. Truth tables enumerated from non-degenerate values (filtering out fns that ignore an input). NumPy-vectorized at the bit level (`a if minterm[m] else ~a`) → 13 min on 1,518 rows.

**Rule shape distribution** in solved set:

| K | wrap | count | example pattern |
|---:|---|---:|---|
| 2 | zero | 466 | `output[i] = input[i+a] XOR input[i+b]` |
| 3 | zero | 460 | `output[i] = MAJ(input[i+a], i+b, i+c)` |
| 1 | wrap | 82 | rotation: `output[i] = input[(i+k) mod 8]` |
| 1 | zero | 54 | shift: `output[i] = input[i+k]` zero-pad |
| 2 | wrap | 32 | XOR with rotated copy |
| 3 | wrap | 1 | rare |

**Inspection of 5 unmatched puzzles** revealed three classes the current solver doesn't handle:
1. **Position-uniform with edge exceptions** (rule applies to most positions but edges differ).
2. **Fully position-dependent** (each output bit has its own logic, no shift pattern).
3. **Complex K≥2 with no uniform shift** (every output bit depends on multiple non-adjacent inputs).

Coverage could be pushed higher with K=4 invariants, per-bit fallback with position-uniformity preference, or 2-op compositions. Deferred — 1,076 verified examples is enough for an SFT pilot. The pilot will tell us whether the model can learn the procedure at all; if so, we expand solver coverage later.

**Trace structure**:
1. Frame: "I need to find an operation that maps each 8-bit input to its 8-bit output."
2. Two rejected hypotheses (NOT, ROL_1) on the first example.
3. Winning rule with per-position verification on the first example.
4. Spot check on a second example.
5. Apply to test input position by position.
6. Emit `\boxed{<answer>}`.

K=3 with arbitrary truth tables originally inflated traces to 2k+ tokens because the truth table got respelled at every position; refactored to introduce `f` once at the top with the full table, then reference `f` per position. Final p90 = 933 tokens.

## 2026-05-08 — SFT pilot FAILED: hand-crafted procedural traces don't transfer

Trained `runs/train/lora_baseline` on 1,076 bit_manipulation procedural traces. 2 epochs, r=32, alpha=16, LR=5e-5, target regex `.*\.(in_proj|out_proj|up_proj|down_proj)$` (Mamba + shared-expert FFN — does NOT touch routed experts). 8 hours walltime on the GB10.

**Training loss dropped 55× (0.82 → 0.015) — model memorized the training set perfectly.**

**Held-out evaluation on full `dev_frozen` (n=500) was a regression:**

| task | n | LoRA acc / trunc | brief acc / trunc | base acc / trunc | LoRA−brief | LoRA−base |
|---|---:|---:|---:|---:|---:|---:|
| bit_manipulation | 84 | 3.6% / 90% | 11.1% / 89% | 0% / 100% | **−7.5pp** | +3.6 |
| equation_transformation | 84 | 6.0% / 83% | 10.0% / 80% | 10.0% / 60% | −4.0 | −4.0 |
| gravitational_constant | 83 | 63.9% / 14% | 92.3% / 8% | 46.2% / 38% | **−28.4** | +17.7 |
| numeral_conversion | 83 | 98.8% / 0% | 100% / 0% | 100% / 0% | −1.2 | −1.2 |
| text_encryption | 83 | 4.8% / 94% | 14.3% / 86% | 0% / 100% | −9.5 | +4.8 |
| unit_conversion | 83 | 44.6% / 55% | 50% / 50% | **100%** / 0% | −5.4 | **−55.4** |
| **OVERALL** | 500 | **36.8%** | **48.0%** | **36.0%** | **−11.2** | **+0.8** |

The LoRA matches the pure baseline overall (36.8% vs 36.0%) but the brief-prompt-no-LoRA arm beats it by 11.2pp (48.0% vs 36.8%). On bit_manipulation specifically — the only task we trained on — the LoRA gets 3.6%, **worse than just adding a "be brief" instruction (11.1%)**.

### Three diagnoses (all support each other)

1. **Memorization, not procedure-learning.** 1,076 traces with only ~6 rule-shape variants gave the LoRA enough surface variety to fit the trace *form* (loss → 0.015) but not the underlying reasoning steps. At inference, faced with a novel puzzle, the model produces something that *looks* like our trace template but doesn't actually parse, hypothesize, verify, or apply. bit_manipulation truncation rate: 90.5% — same as base. Only 8/84 outputs reach `\boxed{`.

2. **Catastrophic forgetting on shared paths.** The LoRA target regex hits Mamba `in_proj/out_proj` and the shared-expert FFN — both of which see every token regardless of task. Training only on bit_manipulation shifted these layers in ways that hurt unrelated tasks: `unit_conversion` (which the base model nailed at 100%) crashed to 44.6%. `gravitational_constant` regressed from 92.3% (brief) to 63.9% (LoRA).

3. **Inference-time distribution shift.** At training, the model sees `<think>FULL_TRACE</think>\boxed{ANSWER}` end-to-end as labels. At inference, it must generate token-by-token from `<think>\n`. Once the first generated token diverges from our hand-crafted style, the trajectory unravels and reverts to the model's natural verbose pattern.

### What this rules out, and what to try next

**Ruled out**:
- Hand-crafted procedural traces alone, even at 1k+ verified examples.
- Target-region pattern from the demo notebook (Mamba + shared expert) on a single task (causes forgetting).

**Worth trying** (in cost-order):
1. **Generate traces from the model's own correct outputs.** When the base model gets a bit_manip puzzle right, capture its trace and use that as SFT data. Stays in-distribution; no forgetting expected. Coverage will be very low (~0–5%) but the few examples we get may transfer better than synthetic.
2. **Use a strong teacher LM** (Claude / DeepSeek-R1 / a larger Nemotron) to produce traces. The teacher's traces will be more naturalistic than ours and span a wider rule space. Verifier-filter to keep only correct ones.
3. **Mix tasks in training** to prevent the unit_conversion-style forgetting. Even if we only have good bit_manip data, dilute with general data so shared-path updates don't collapse easy tasks.
4. **Narrow the LoRA target to attention-only or routed-experts-only** so single-task training can't break shared paths.

### Important meta-finding for the project

Loss → 0.015 with regression on the training-target task tells us **the SFT objective itself is suspect for this model on these tasks**. Token-level next-token prediction over a fixed trace template doesn't teach a procedure — it teaches the surface form of the trace. A teaching signal that more directly rewards correct final answers (RL with verifier reward, GRPO-style) might be needed eventually, but it's expensive. First we should rule out the cheaper path: better trace data from a stronger teacher.

## 2026-05-09 — Harvest pass complete: 3,975 self-distilled correct traces

Ran the base model with brief-arm prompt (system + user prefix) on all 9,000 train_remaining puzzles at temperature=1.0, max_tokens=3584. Walltime: 27.1 hours at chunk-saturated throughput (max_num_seqs=64, ~220 tok/s aggregate). Resumable via id-skip; written incrementally to `datasets/processed/harvest_brief.jsonl`.

**Per-task yield (final)**:

| task | correct/total | yield | trace p50 chars |
|---|---:|---:|---:|
| gravitational_constant | 1,386/1,514 | 91.5% | 3,118 |
| numeral_conversion | 1,384/1,493 | 92.7% | 718 |
| unit_conversion | 839/1,511 | 55.5% | 5,394 |
| text_encryption | 162/1,493 | 10.9% | 9,105 |
| equation_transformation | 139/1,471 | 9.4% | 5,358 |
| bit_manipulation | 65/1,518 | 4.3% | 5,398 |
| **OVERALL** | **3,975/9,000** | **44.2%** | — |

The hard tasks remain hard: combined bit_manip + text_enc + equation_transformation = 366 self-distilled correct (9.1% of total corpus). The 4.3% bit_manipulation yield gives only 65 examples — far thinner than the 1,076 procedural traces from the failed pilot.

## 2026-05-10 — Competitive intelligence corrections + v2 training mix

User shared the actual harness eval parameters (verified vs earlier function defaults):
- temperature 0.0 (was 1.0)
- max_tokens 7680 (was 3584)
- max_model_len 8192 (was 4096)

`configs/inference/baseline.yaml` updated. The earlier 3,584 cap explains some of why bit_manipulation/text_encryption traces got cut off in our diagnostics — the live harness allows 2.1× that.

LoRA hyperparameter corrections applied to `configs/train/lora_baseline.yaml`:
- lora_alpha: 16 → 32 (effective scale alpha/r = 1.0 instead of 0.5)
- lora_dropout: 0.05 → 0.0
- learning_rate: 5e-5 → 3e-4 (standard LoRA LR)
- target_modules widened: now includes `q_proj|k_proj|v_proj|o_proj|gate_proj` (catches the 6 standard-attention layers and any MoE gate projections)
- max_seq_length: 1300 → 8192

### v2 training mix (sqrt-rebalanced + 200 procedural bit_manip)

Final composition `datasets/processed/train_formatted_v2.jsonl` (2,338 records):

| task | self-distilled | procedural | total | % | p50 tokens |
|---|---:|---:|---:|---:|---:|
| numeral_conversion | 658 | — | 658 | 28.1% | 291 |
| gravitational_constant | 653 | — | 653 | 27.9% | 1,581 |
| unit_conversion | 480 | — | 480 | 20.5% | 2,292 |
| bit_manipulation | 61 | 200 | 261 | 11.2% | 1,185 |
| text_encryption | 160 | — | 160 | 6.8% | 3,162 |
| equation_transformation | 126 | — | 126 | 5.4% | 1,676 |

Filters applied: correct=True AND `</think>` in raw_output AND `\boxed{` in raw_output (drops 146 false-positive "correct" rows where extract_answer's last-numeric fallback happened to match GT). Token cap 7,000 dropped 0 records. sqrt rebalancing with budget B=2500: easy tasks (grav, num, unit) capped at 480-658 vs available 737-1383; hard tasks kept in full.

Hard tasks now make up 23.4% of the mix (vs 9% in raw harvest). Token distribution: min 172 / p50 1,253 / p90 3,174 / max 3,851. All within max_seq_length=8192.

### mamba_ssm + causal_conv1d installed

Got both fused kernels building from source on aarch64 with `--no-build-isolation`:
- mamba_ssm 2.3.2.post1 (350 MB wheel)
- causal_conv1d 1.6.2.post1 (206 MB wheel)

Side effect: tilelang got downgraded from 0.1.9 → 0.1.8 (vLLM's pin), but vLLM 0.20.1 still imports cleanly so the downgrade is benign. `train_lora.py` no longer force-disables `use_mamba_kernels`; native transformers' lazy loader will pick up the fused path. Expected speedup: training step time should drop substantially from the naive ~13.5 sec/micro-batch we measured in the pilot.

## 2026-05-10 — LoRA v2 training + eval: 58.4% overall accuracy

**Training**: 1 epoch on `train_formatted_v2.jsonl` (2,338 records). With fused mamba kernels enabled, training finished in **83 minutes** — vs the pilot's 8 hours for 2 epochs of 1,076 records. Per-step time dropped roughly 8-10× from naive Mamba.

**Eval at corrected harness params** (temperature=0.0, max_tokens=7680, max_model_len=8192, no system message, no user prefix — matches the harness exactly) on `dev_frozen_shuffled` (n=500):

| task | n | **LoRA v2** | LoRA v1 | brief (n=50) | base (n=50) |
|---|---:|---:|---:|---:|---:|
| numeral_conversion | 83 | **100%** | 98.8% | 100% | 100% |
| gravitational_constant | 83 | **100%** | 63.9% | 92.3% | 46.2% |
| unit_conversion | 83 | **89.2%** | 44.6% | 50% | 100% |
| text_encryption | 83 | **44.6%** | 4.8% | 14.3% | 0% |
| equation_transformation | 84 | **13.1%** | 6.0% | 10.0% | 10.0% |
| bit_manipulation | 84 | **4.8%** | 3.6% | 11.1% | 0% |
| **OVERALL** | 500 | **58.4%** | 36.8% | 48.0% | 36.0% |

Truncation rate dropped from 56% (v1) to 32% (v2) — primarily from the larger max_tokens budget plus the model learning to close `</think>` more reliably.

### Confounds (the comparison is not fully apples-to-apples)

LoRA v2 ran at temp=0.0/max_tokens=7680. All baselines (LoRA v1, brief, base) ran at temp=1.0/max_tokens=3584. Some fraction of the +22pp gain comes from better sampling and a 2.1× larger token budget, not the LoRA itself. A clean comparison needs a base-model run at corrected params (~3.5 h walltime).

### Per-task analysis

**Recovered from v1's catastrophic forgetting**:
- gravitational_constant 64% → 100% (and exceeds brief's 92%)
- unit_conversion 45% → 89% (close to base's 100%)
- numeral_conversion 99% → 100%

**Real new capability** (LoRA v2 beats EVERY prior arm including brief):
- text_encryption 0/14% → 44.6%. This is the single most important gain. The model could not solve these at all before; now it solves ~half of them. The 162 self-distilled text_encryption traces in the training mix did real work.

**Marginal**:
- equation_transformation 10% → 13%. Small lift, still mostly broken. Traces are massive (p50 27k chars) — model still wandering.

**Regression vs brief**:
- bit_manipulation 11% → 5%. Despite mixing 200 procedural traces with 61 self-distilled, accuracy *decreased* relative to just adding "be brief" instruction. The brief instruction was actually working better than what we trained.

### Trace length distribution shows the bit_manip + equation_transformation problem

| task | p50 chars | max chars |
|---|---:|---:|
| numeral_conversion | 395 | 1,387 |
| gravitational_constant | 2,501 | 6,310 |
| unit_conversion | 6,998 | 24,274 |
| bit_manipulation | 15,716 | 26,989 |
| text_encryption | 20,148 | 27,325 |
| equation_transformation | 27,055 | 32,297 |

Approximate char→token ratio is ~4:1, so equation_transformation p50 is ~6,800 tokens, just at the budget. bit_manipulation p50 ~3,900 tokens — fits, but the model isn't producing useful reasoning at that length.

### What this tells us for next decisions

1. **Self-distillation works**: even with only 162 text_encryption examples we tripled accuracy. The brief-arm self-distilled traces ARE high-quality teaching signal when the model can solve the task at all.
2. **bit_manipulation needs a different intervention**. Self-distillation has only 61 correct examples — too few to teach. Procedural traces (200 mixed) didn't help and may have hurt. Options: (a) much stronger teacher LM for bit_manip specifically, (b) RL with verifier reward, (c) accept ~5% and focus elsewhere.
3. **Epoch 2 might help equation_transformation** (still 13%, lots of room). Cheap to try now that we know 1 epoch = 83 min.
4. **Need a base-at-corrected-params baseline** to nail down how much of +22pp is the LoRA vs the corrected sampling.

## 2026-05-10 — bit_manipulation v2 inspection: procedural traces transferred at FORMAT level but not REASONING level

Pulled 5 bit_manipulation outputs from the LoRA v2 eval (1 correct, 4 incorrect). **Every output adopted the hand-crafted procedural-trace structure from the 200 mixed-in examples.**

Sample opening pattern across all 5 outputs:
```
I need to find an operation that maps each 8-bit input to its 8-bit output...
Try NOT(input): NOT(00000111) = 11111000. Expected output: 00001000. Doesn't match.
Try rotate-left-by-1: ROL_1(00000111) = 00001110. Expected: 00001000. Doesn't match.
Try a rule of the form output[i] = f(input[i-7], input[i+1], input[i+2]), zero-padding edge bits...
Verify on the first example: i=0: ... ✓
```

This is the literal template from `src/data/bit_manip_trace.py`. The procedural data taught the surface format successfully.

### Why it's not helping

The model adopted the form (hypothesis → verify → reject → next) but lacks the heuristics needed to *guess* good hypotheses. It picks arbitrary offsets like `input[i-3] XOR input[i+5]` with no semantic basis. Failure modes seen:

1. **Random rule guessing**: model proposes K=3 shift-invariant rules with random offsets and arbitrary truth tables, none fit.
2. **Hypothesis loops without convergence**: outputs #3 and #5 spent 16-17k chars cycling through guesses then truncated.
3. **Wrong final answer despite "verification"**: outputs #2 and #4 emitted answers, but verification was self-consistent with a wrong rule (verifying that the rule produces what the rule produces, not against external truth).
4. **One "correct" answer that's actually trivial**: the single correct case (#1, gt=`11111111`) found the rule "output = input OR 0b11111111" — which produces all 1s regardless of input. The model noticed all example outputs were the same and predicted that constant. Got lucky on a degenerate puzzle, not real reasoning.

### Implications

- **Procedural traces taught a counter-productive search pattern**. The bit_manip space is enormous (8-bit permutations × Boolean fns × offset combos); brute-forcing with random guesses can't converge in 7,680 tokens. Real solvers need heuristic priors ("if output bit-count matches input, try permutation; if output is all-1s, try OR with mask"; etc.) — those aren't in our procedural traces.
- **Dropping the 200 procedural traces should not hurt and may help**: the base model + brief prompt got 11% on bit_manip. v2 (with procedural mix) got 5%. v3 (self-distilled only, 61 bit_manip examples) should land closer to 11% — recovering bit_manip without losing the wins on text_encryption.
- **What WOULD help bit_manipulation**: traces from a stronger teacher (Claude, DeepSeek-R1) that actually solve these puzzles with semantic insight, OR a separate few-shot bit_manip prompt library, OR an entirely different objective (RL-with-verifier).

### v3 training mix prepped (no training yet)

`datasets/processed/train_formatted_v3.jsonl` (2,251 records):

| task | v2 | v3 | delta |
|---|---:|---:|---:|
| bit_manipulation | 261 | 61 | **−200** |
| numeral_conversion | 658 | 699 | +41 |
| gravitational_constant | 653 | 694 | +41 |
| unit_conversion | 480 | 511 | +31 |
| text_encryption | 160 | 160 | 0 |
| equation_transformation | 126 | 126 | 0 |

Self-distilled only. The 200 procedural slots are not redistributed to other tasks (those tasks were already at sqrt-rebalance cap or below); the freed budget just shrinks the total.

## 2026-05-11 — Base baseline at corrected params: 52%. LoRA contribution isolated.

Ran the base model at the same harness params used for the LoRA v2 eval (temperature=0.0, max_tokens=7680, max_model_len=8192, no system message, no user prefix). n=500 on dev_frozen_shuffled, walltime 4h15m. Result: **acc=52.0%, trunc=36%**.

### Isolated decomposition

| task | base@old (t=1/3584) | base@corr (t=0/7680) | LoRAv2 (t=0/7680) | Δ sampling | Δ LoRA | Δ total |
|---|---:|---:|---:|---:|---:|---:|
| numeral_conversion | 100% | 100% | 100% | 0 | 0 | 0 |
| gravitational_constant | 46.2% | 74.7% | 100% | +28.5 | **+25.3** | +53.8 |
| unit_conversion | 100% (n=4) | 72.3% | 89.2% | −27.7 | **+16.9** | −10.8 |
| equation_transformation | 10.0% | 11.9% | 13.1% | +1.9 | +1.2 | +3.1 |
| text_encryption | 0% | 41.0% | 44.6% | **+41.0** | +3.6 | +44.6 |
| bit_manipulation | 0% | 13.1% | 4.8% | +13.1 | **−8.3** | +4.8 |
| **OVERALL** | **36.0%** | **52.0%** | **58.4%** | **+16.0** | **+6.4** | **+22.4** |

### Key insights

1. **Most of the "v2 win" was the sampling fix, not the LoRA.** Corrected sampling (temp=0.0 + max_tokens=7680) alone accounts for +16.0pp of the +22.4pp overall gain. The LoRA contributes +6.4pp on top.

2. **text_encryption was unlocked by sampling, not LoRA.** Base@old=0% → base@corr=41% → LoRAv2=45%. The 162 self-distilled text_encryption traces barely moved the needle (+3.6pp). We were misattributing this win.

3. **The LoRA's two genuine wins are gravitational_constant (+25pp) and unit_conversion (+17pp).** These are tasks where the LoRA learned something base+corrected-sampling couldn't do alone.

4. **The LoRA actively HURTS bit_manipulation by 8.3pp** (base@corr 13% → LoRAv2 5%). Confirms the inspection finding: the 200 procedural traces taught a counter-productive search template. The base model at corrected sampling does better on bit_manip than our trained model.

5. **unit_conversion gets WORSE with corrected sampling** (base@old=100% n=4 → base@corr=72%). Caveat: base@old was n=4, possibly noise — but base@corr=72% on n=83 is real. Hypothesis: temp=0 makes the model commit to wrong unit-conversion formulas early without sampling variance to escape. The LoRA recovers it.

6. **Truncation rate is barely improved by the LoRA** (36% → 32%). Most of the truncation reduction is also from the sampling fix.

### Implications for next training run

The LoRA strategy is making +6pp marginal contribution overall, concentrated on grav + unit_conversion. The hard tasks (bit_manip, equation_transformation) aren't getting LoRA benefit. The text_encryption "win" was largely sampling-driven.

This justifies the v3 plan (drop procedural bit_manip):
- v3 prediction: bit_manipulation recovers toward 11-13% (matches brief / base@corr).
- v3 total prediction: ~60% (similar to v2; lose nothing else because the same self-distilled corpus drives the other tasks).

**Bigger strategic question raised**: if the LoRA is only adding +6pp and is hurting the hardest task, is more SFT iteration the right next move? Possible alternative bets:
- Train a SMALLER, more targeted LoRA on just grav + unit (where the LoRA actually helps)
- Use a stronger teacher (Claude / DeepSeek-R1) for bit_manip + text_enc + equation specifically
- Skip LoRA entirely on bit_manip and run inference with the LoRA disabled or masked on bit_manip prompts (router approach)
- RL-with-verifier on the hard tasks (expensive)

## 2026-05-11 — Phase 6: deterministic solvers + solver-backed trace training (LoRA v4)

Pivoted from self-distillation to deterministic Python solvers, one per task type. Built 6 solvers + trace generators. Coverage on full train.csv (verified-correct):

| category | coverage | verified |
|---|---:|---:|
| unit_conversion | 100% | 1,594 |
| gravitational_constant | 100% | 1,597 |
| numeral_conversion | 100% | 1,576 |
| text_encryption | 100% | 1,576 |
| bit_manipulation | 75% (shift-invariant search, K=1-3) | 1,144 |
| equation_transformation | 12% (heterogeneous operators) | 191 |
| **TOTAL** | **81%** | **7,678** |

v4 training mix (`datasets/processed/train_formatted_v4.jsonl`, 2,991 records, sqrt-rebalanced B=3000):

| task | count | % | p50 tokens |
|---|---:|---:|---:|
| grav | 579 | 19.4% | 577 |
| unit | 579 | 19.4% | 387 |
| numeral | 576 | 19.3% | 324 |
| text_enc | 576 | 19.3% | 641 |
| bit_manip | 490 | 16.4% | 1,043 |
| equation | 191 | 6.4% | 290 |

Training: 1 epoch, same corrected LoRA config (r=32, alpha=32, dropout=0.0, LR=3e-4, widened targets), 64 min walltime (faster than v2's 83 min on smaller dataset because traces are shorter). Final train_loss=0.082 — HIGHER than v2's 0.015. This is a GOOD sign: solver traces are diverse per-puzzle so the model can't memorize a single template.

### v4 eval @ corrected harness (n=500, dev_frozen_shuffled)

acc=61.6%, truncation=16%. Walltime 87 min (much faster than v2's 222 min — model writes shorter completions).

| task | base@corr | v2 | v4 | Δ v4-v2 | Δ v4-base |
|---|---:|---:|---:|---:|---:|
| numeral_conversion | 100% | 100% | 100% | 0 | 0 |
| gravitational_constant | 74.7% | 100% | 100% | 0 | +25.3 |
| unit_conversion | 72.3% | 89.2% | **100%** | **+10.8** | +27.7 |
| text_encryption | 41.0% | 44.6% | **51.8%** | **+7.2** | +10.8 |
| equation_transformation | 11.9% | 13.1% | 15.5% | +2.4 | +3.6 |
| bit_manipulation | 13.1% | 4.8% | **3.6%** | **−1.2** | **−9.5** |
| OVERALL | 52.0% | 58.4% | **61.6%** | **+3.2** | **+9.6** |

### Per-task trace length reduction (v2 → v4 raw output, p50 chars)

The dominant qualitative change is that v4 outputs are dramatically shorter:

| task | v2 p50 | v4 p50 | reduction |
|---|---:|---:|---:|
| equation_transformation | 27,055 | 583 | −97.8% |
| text_encryption | 20,148 | 1,020 | −94.9% |
| unit_conversion | 6,998 | 526 | −92.5% |
| gravitational_constant | 2,501 | 662 | −73.5% |
| bit_manipulation | 15,716 | 15,083 | −4% (still rambling) |

text_encryption truncation went 52% → 1%. The model learned that this task has a SHORT correct procedure and now executes it cleanly.

### Two findings worth recording

**1. Trace style mattered more than trace quantity.** v2 had 2,338 self-distilled records with long exploratory traces. v4 has 2,991 solver-derived records with short deterministic traces. Same SFT setup. v4 is +3pp better with FAR less compute at inference time. The bottleneck wasn't data volume; it was teaching the model to commit to a procedure.

**2. bit_manipulation is poisoned by SFT exposure.** Three consecutive data points:
   - base@corr (no bit_manip training): 13.1%
   - v2 (261 bit_manip in training): 4.8%
   - v4 (490 bit_manip in training): 3.6%

Both v2's procedural traces and v4's procedural traces teach the model to do random-hypothesis-test that doesn't converge. More training on this task makes the model WORSE. The next obvious experiment is to **drop bit_manipulation from the training mix entirely** and measure whether (a) the OTHER tasks retain v4's gains and (b) bit_manipulation recovers toward base@corr's 13.1%.

### Where this leaves us

- 61.6% with 81% train coverage and only a partial bit_manip / equation_transformation solver. Ceiling for the current solver set without bit_manip improvement is probably 65-68% (push equation higher, drop the bit_manip damage).
- Top teams reportedly at 86-87% — they presumably have stronger solvers (better bit_manip + equation_transformation rule enumeration).
- Next experiments worth running: v5 mix WITHOUT bit_manipulation, expand bit_manip solver to K=4 / per-bit / 2-op compositions, expand equation_transformation operator library.

## 2026-05-12 — v5 (no bit_manip training) + bit_manip solver v2

### bit_manip solver v2: per-output-bit decomposition

Added a per-output-bit fallback to the existing shift-invariant solver. For each of the 8 output positions, enumerate ~580 candidate rules (2 constants + 16 single-input + 280 K=2 natural fns + 280 K=3 natural fns) and pick the simplest fitting rule. Reference design from the competition winner *[correction 2026-07-09: tonghuikang won the midpoint Open Prize, not the overall competition]* (docs/reference_solvers/tonghuikang, read-only) — same idea but our enumeration is broader (10 K=2 fns vs their 6).

Coverage on full train.csv:
  v1 (invariant only):              1076 / 1602  (70.9%)
  v2 (invariant + per-bit fallback): 1254 / 1602  (78.3%)
  -- precision: 82.6% (downstream verification filters overfits)

The reference reportedly hits 85% — gap is their more sophisticated rule selection (left/right run analysis preferring position-correlated rules). Worth porting if bit_manip ever moves the needle in SFT.

### v5 training: bit_manipulation DROPPED from training mix

Built `datasets/processed/train_formatted_v5.jsonl` (2,952 records) — same as v4 but no bit_manip examples. sqrt rebalancing redistributes budget to the other 5 tasks; final composition:

| task | count | % |
|---|---:|---:|
| gravitational_constant | 693 | 23.5% |
| unit_conversion | 692 | 23.4% |
| numeral_conversion | 688 | 23.3% |
| text_encryption | 688 | 23.3% |
| equation_transformation | 191 | 6.5% |

Training: 1 epoch, 59 min, train_loss=0.086.

### v5 eval @ corrected harness (n=500, dev_frozen_shuffled)

acc=62.6%, truncation=5% (down from v4's 16%), elapsed 50 min (down from v4's 87 min).

| task | base@corr | v4 | v5 | Δ v5-v4 |
|---|---:|---:|---:|---:|
| numeral_conversion | 100% | 100% | 100% | 0 |
| gravitational_constant | 74.7% | 100% | 100% | 0 |
| unit_conversion | 72.3% | 100% | 100% | 0 |
| **text_encryption** | 41.0% | 51.8% | **60.2%** | **+8.4** |
| equation_transformation | 11.9% | 15.5% | 14.3% | −1.2 |
| **bit_manipulation** | 13.1% | 3.6% | **2.4%** | −1.2 |
| OVERALL | 52.0% | 61.6% | **62.6%** | +1.0 |

### Three findings

**1. Cross-task interference is real.** text_encryption jumped +8.4pp with no change to its own training data — just removing bit_manip from the mix. The bit_manip traces (which teach a counter-productive search pattern) were leaking into text_encryption inference. The LoRA's wider target modules (Mamba + shared expert + attention) all see every token, so the bit_manip-style updates affect everything.

**2. "Compact answer" habit transfers across tasks.** bit_manipulation truncation went from 74% (v4) to 6% (v5) even though we removed bit_manip training. The model learned from the OTHER tasks' compact solver-traces to commit to an answer and emit `\boxed{}` cleanly, and that habit applies to bit_manip too.

**3. bit_manipulation did NOT recover to base@corr.** Predicted: dropping bit_manip training would lift accuracy from 3.6% (v4) toward 13.1% (base). Actual: 2.4%. The model now gives up cleanly on bit_manip but doesn't fall back to base-model behavior, because the SFT changed its overall reasoning style. The cleaner, more confident style helps on tasks it knows but hurts the one task where verbose-exploratory was actually finding occasional correct answers.

### Ceiling analysis

Weighted across dev_frozen task distribution (~16.6% per task):

| task | acc upper bound | weighted |
|---|---:|---:|
| numeral_conversion | 100% | 16.6 |
| gravitational_constant | 100% | 16.8 |
| unit_conversion | 100% | 16.8 |
| text_encryption | 60% | 10.0 |
| equation_transformation | 15% | 2.5 |
| bit_manipulation | 13% (base ceiling) | 2.2 |
| **ceiling** |  | **~64.9%** |

We're at 62.6%; ~2pp of headroom under current assumptions. To break the 65% ceiling we need to actually improve text_enc/equation/bit_manip — either via deeper solvers OR by accepting that SFT can't teach this model bit_manip and focusing on data we CAN teach.

**Top teams at 86% must have moved bit_manip from 13% → 80%+** via solver-traces. Our experiments suggest this base model resists SFT teaching on bit_manip; their approach must work either because (a) their solver traces are different in some subtle way, or (b) they use a different training objective, or (c) their model is different.
