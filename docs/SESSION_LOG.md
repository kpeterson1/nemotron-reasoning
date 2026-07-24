# Session log

Append-only record of substantive sessions. Per CLAUDE.md: claim what was
asserted, verify what was actually checked, distinguish what was assumed.

## 2026-05-26 03:39 — Extractor fix + Phase 1 scope (with mid-session retraction)
**Branch:** kaggle-runtime-parity-v9
**Claimed:**
- (initial, wrong) docs/kaggle_metric_source.py was the source of truth for
  Kaggle's deployed boxed-extractor; local extract_answer.py diverged from it.
- (initial, wrong) "14 of 16 vLLM parity dimensions match" between
  eval_kaggle_exact.py and Kaggle.
- (initial, wrong) Kaggle runs at temperature=1.0 (sampled); local greedy
  at T=0.0 could explain a multi-pp chunk of the gap.
- (initial, wrong) max_tokens halving (3584 vs Kaggle 7680) could be a
  primary truncation source.
- (final, surviving) The local naive `[^}]*` boxed extractor and the
  brace-walking deployed Kaggle extractor are not equivalent. The local
  extractor was fixed. Test parity is enforced.
- (final, surviving) Re-scoring 17 *-raw-*.json eval files shows the
  extractor fix moves v9 dev_frozen by 0.0pp; deltas elsewhere in
  [-0.6pp, +0.0pp]. The extractor is not the gap source.
- (final, surviving) After re-counting against the Kaggle Overview tab
  values, the v9 69.4 run vs Kaggle has only two substantive vLLM
  divergences: max_num_seqs (32 vs 64), max_model_len (4096 vs 8192).
- (final, surviving) max_tokens hypothesis is falsified by the v9 raw
  outputs: 0/500 predictions within 1800 tokens of the 7680 cap;
  stored truncation_rate of 0.40% reflects missing \boxed{}, not
  token-cap hits.

**Verified:**
- Local extract_final_answer was the same regex as docs/kaggle_metric_source.py
  prior to today (`python -c` regex equivalence check + identical findall
  output on 5 cases).
- Re-vendored docs/kaggle_metric_source.py with current brace-walking
  logic (`grep boxed_starts` confirms presence after Write).
- Fixed src/evaluation/extract_answer.py and added
  tests/test_extract_answer_kaggle_parity.py (55 tests, 0 fails;
  full suite 84 tests pass). The parity test loads
  docs/kaggle_metric_source.py via importlib with kagglehub/pandas/tqdm
  stubbed and subprocess.run no-op'd, asserts byte-identical
  extract_final_answer output across a 41-string corpus.
- Re-scored 17 eval files with scripts/rescore_with_fixed_extractor.py
  (uncommitted, depends on gitignored runs/eval/). All old_score values
  match stored accuracy exactly (validates that the inline pre-fix
  snapshot reproduces what produced the stored predictions).
- The v9 dev_frozen 69.4 invocation: per the v9-run-report commit's message
  ("Eval @ dev_frozen (n=500, temp=0.0, max_tokens=7680)") and
  runs/eval/_lora_eval_500.log's vLLM "non-default args" boot line
  (`max_num_seqs=32, max_model_len=4096, trust_remote_code=True,
  enable_prefix_caching=True, gpu_memory_utilization=0.85,
  max_lora_rank=32, enable_chunked_prefill=True,
  enable_lora=True, dtype=torch.bfloat16`). Output JSON schema
  (with system_message/user_instruction/arm_label fields) identifies
  src/evaluation/run_eval.py as the producer, not eval_kaggle_exact.py.
- Truncation profile: 498/500 predictions are <1000 approx-tokens
  (char/4); max is 5797 approx-tokens; 0 predictions ≥ 7000.
  Per-task `\boxed{}` absence: only equation_transformation (2/84).

**Assumed:**
- "char/4" is a reasonable approximation for tokens. NOT verified
  against the actual NemotronH tokenizer. For prose this is roughly
  right; for code/LaTeX it can over-estimate token count by 1.5-2x.
  Even at 2x, the v9 max would be ~11.6k tokens — over the 7680 cap.
  So this assumption is load-bearing for the truncation conclusion.
  TODO if challenged: run actual tokenizer on the longest 5 raws.
- top_p=1.0 in the v9 run (run_eval.py default). NOT recorded in the
  boot log or commit message; inferred from defaults.
- Kaggle Overview tab values per CLAUDE.md are current (not stale).
  Treated as source-of-truth per the CLAUDE.md rule. If the Overview
  tab has changed since CLAUDE.md was authored, all parity claims
  re-open.
- `dtype='auto'` on Kaggle resolves to bfloat16 (matching local
  log's `dtype=torch.bfloat16`). Inferred from NemotronH-3-Nano-30B
  being shipped in BF16.

**Next:**
Step A from PHASE1_VLLM_PARITY_SCOPE.md — compute SHA256 of local HF
NemotronH-3-Nano-30B BF16 safetensors shards. If kagglehub access is
available on Spark 1, pull the Kaggle dataset snapshot and diff.
Base-model weight identity is the highest-EV remaining hypothesis
given the recount.

**Commits this session:**
- Fix extract_final_answer: brace-walking to match Kaggle metric
- Re-score historical eval outputs with brace-walking extractor
- Reconcile prior findings against extractor-fix result
- Scope Phase 1: narrow runtime-parity investigation
- Revise Phase 1 scope: parity recount + max_tokens hypothesis falsified
- Fix eval_kaggle_exact.py defaults to match Kaggle Overview tab

**Skipped commits flagged in conversation:**
- "Re-vendor docs/kaggle_metric_source.py from current Kaggle website" —
  the file is in .gitignore (Local reference, do not redistribute); only
  the on-disk content was updated.
- eval_kaggle_exact_multi.py has the same defaults trap as
  eval_kaggle_exact.py; not fixed this session per user's narrow
  instruction. Flagged for follow-up.

## 2026-05-26 06:08 — Tokenizer falsification, multi fix, Step A baseline
**Branch:** kaggle-runtime-parity-v9
**Claimed:**
- The "char/4 token approximation" used in the prior session entry was
  a 1.76× under-estimate (mean) for NemotronH reasoning traces; max
  observed 2.01×. Real tokens are 44% denser than char/4 suggested.
- Despite the under-estimate, the max_tokens=7680 falsification stands
  and is **strengthened**. Real-token distribution is bimodal: p99 =
  1,181 tokens, then exactly 2 outputs at 7,680 (the cap). Nothing in
  between. Both cap-hits were scored correct=False — no over-scoring
  via fallback extraction.
- At max_tokens=3584 (the old `eval_kaggle_exact.py` default) v9 dev_frozen
  would have truncated the same 2 outputs, no more — because no outputs
  fall in [1182, 7679].
- `eval_kaggle_exact_multi.py` shared the `max_num_seqs=16` hardcoded
  trap (its max_tokens and max_model_len defaults were already correct);
  fixed in this session.
- HF blob naming for `tokenizer.json` and the 13 safetensors shards
  uses SHA256 (git-LFS style); for small files like `config.json` it
  uses git's SHA1 blob hash. The earlier "HF blob name ≠ SHA256" check
  was right for the small file I picked but wrong as a general claim.

**Verified:**
- Ran `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` tokenizer via
  `transformers.AutoTokenizer` (`hf_env` venv) on the longest 10 raws
  from `dev_frozen-raw-runs_train_lora_v9-1779095404.json`. Top two
  (ids 20f0fac9 and 3c424916) tokenize to **exactly 7,680** — the
  cap. Mean tokens/(char÷4) ratio = 1.76, min 1.32, max 2.01.
- Tokenized **all 500** raws to get the true distribution: min 102,
  median 332, p95 516, p99 1,181, max 7,680. Histogram confirmed
  bimodal — 495 below 1k, 3 in [1k, 2k], then a 5,500-token gap to
  the 2 cap-hits.
- Both cap-hit predictions are equation_transformation, both
  `truncated=True` and `correct=False` in the stored report. Predicted
  values `'3234'` and `'6'` vs ground truths `'9351'` and `'2976'`.
- Fixed `src/training/eval_kaggle_exact_multi.py` defaults to mirror
  the eval_kaggle_exact-fix commit's pattern (max_num_seqs CLI flag default 64, trust_remote_code=True
  in LLM init). max_tokens and max_model_len defaults were already
  7680 and 8192 — only the two fields needed change.
- Added `**Skipped / deferred:**` field to the SESSION_LOG.md template
  in CLAUDE.md.
- Computed SHA256 of all 13 safetensors shards + config.json,
  tokenizer.json, chat_template.jinja, generation_config.json,
  model.safetensors.index.json for HF revision
  `cbd3fa9f933d55ef16a84236559f4ee2a0526848`. Recorded in
  `docs/investigations/kaggle_lora_mismatch/PHASE1_STEPA_BASE_MODEL_SHA.md`.
  Total shard bytes: 63,155,694,272 — matches vLLM boot log's
  "Checkpoint size: 58.82 GiB".

**Assumed:**
- The HF snapshot at `cbd3fa9f...` is the one vLLM loaded for the v9
  69.4 run. Inferred from it being the only snapshot in the cache
  matching that model id. Not directly cross-referenced against the
  vLLM run's load log timestamp.
- Top-10 by char-length captures all cap-hit candidates. Verified by
  the full 500-row tokenization — no surprises in the long tail.
- The Kaggle Overview tab `dtype='auto'` resolves to bfloat16. Still
  assumed (no Kaggle-side runtime log accessible). High confidence
  because the model ships in BF16.

**Next:**
- User decision: do we (a) ask the Kaggle dataset card for the source
  revision pin (cheap, no download); (b) install kagglehub + pull
  ~60 GiB (definitive); or (c) skip Step A and go to Step C — a
  local v9 dev_frozen re-run with max_num_seqs=64 and max_model_len=8192.
  Step C is a stronger signal regardless of Step A's outcome.

**Skipped / deferred:**
- Kaggle-side SHA256 comparison: blocked on kagglehub install + 60 GiB
  download; documented as pending in
  `PHASE1_STEPA_BASE_MODEL_SHA.md` § "Pending: Kaggle-side hashes".
  No agent action taken — significant resource cost, user approval
  required.
- The SESSION_LOG-template commit's message understates its scope: the commit also
  contains the user's earlier in-context CLAUDE.md additions (Parity
  claims, contradiction rule, Confirmed Kaggle params, Reading prior
  investigation docs, top-level Skipped/deferred section, Session
  logging rule) that hadn't been committed yet. Per project rule
  ("prefer to create a new commit rather than amending"), not amended.
  If audit clarity matters, a follow-up empty-content commit can
  document the bundling.
- top_p=1.0 in the v9 run is still inferred from run_eval.py default,
  not confirmed from any log. Cheap to verify if the next run records
  SamplingParams.

**Commits this session:**
- Confirm max_tokens falsification with real NemotronH tokenizer counts
- Fix eval_kaggle_exact_multi.py defaults to match Kaggle Overview tab
- Add Skipped/deferred field to SESSION_LOG.md template
- Step A baseline: local HF NemotronH SHA256s recorded

## 2026-05-26 11:56 — Step A.1 dead end + Step C result (-0.4pp / noise)
**Branch:** kaggle-runtime-parity-v9
**Claimed:**
- Step A.1: no HF revision pin for the Kaggle `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`
  dataset is findable on disk. The Kaggle dataset's `/1` is an internal
  version number, not an HF SHA. Skip A.2 (60 GiB download) for now.
- `references/huikang/README.md` independently confirms Kaggle uses
  `max_lora_rank=32, temperature=0.0, top_p=1.0, max_tokens=7680,
  max_model_len=8192`. The last inferred-from-defaults runtime claim
  (top_p=1.0 for v9 run) is now confirmed.
- Step C run completed: v9 dev_frozen with `max_num_seqs=64,
  max_model_len=8192` scored **0.690** vs the original **0.694**
  (delta **−0.4pp**). Per the user's interpretation guide, this
  rejects the "two runtime params explain the 11.4pp gap" hypothesis.
- vLLM 0.20.1 greedy decoding (T=0.0) is NOT deterministic across
  `(max_num_seqs, max_model_len)` permutations on this model. 12/500
  rows produced different text from the same prompts. Per-row flips
  are single-bit-edit style — consistent with kernel-reduction-order
  perturbations from changed KV cache layout.
- `max_model_len=8192` freed prior cap-hit `3c424916` from truncation
  but introduced two new degenerate outputs (`21bd1251` text_enc
  loop at 7,680 tokens, `e3890bf7` eq_xform loop at 7,679 tokens).
  Net at-cap count unchanged (2 → 2); identity of the cap-hits churned.

**Verified:**
- 5-min on-disk grep for HF revision SHA, `cbd3fa9f...`,
  `nemotron-3-nano-30b-a3b-bf16` + revision pin, `snapshot_download`,
  `from_pretrained.*revision`, `hf_hub` in `docs/`, `references/`,
  `scripts/`. Only Kaggle-path references (`metric/.../1`) appear;
  no HF SHA anywhere.
- Read `references/huikang/README.md` — confirms all five Kaggle
  runtime parameters explicitly. Independent of the CLAUDE.md
  Overview-tab claim.
- Step C invocation:
  `~/kaggle/hf_env/bin/python -m src.evaluation.run_eval
   --split dev_frozen --config configs/eval/default.yaml
   --adapter-dir runs/train/lora_v9 --max-tokens 7680
   --max-model-len 8192 --max-num-seqs 64 --temperature 0.0 --top-p 1.0`.
- vLLM `non-default args` boot line (in `runs/eval/_v9_kaggle_params_step_c.log`):
  `{'trust_remote_code': True, 'max_model_len': 8192, ...,
  'max_num_seqs': 64, ..., 'max_lora_rank': 32, ...,
  'enable_chunked_prefill': True, 'enable_prefix_caching': True,
  'gpu_memory_utilization': 0.85, 'model': 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'}`.
  Confirms intended runtime config.
- Output JSON `dev_frozen-raw-runs_train_lora_v9-1779810797.json`
  reports `n=500, accuracy=0.6900, truncation_rate=0.006,
  elapsed_sec=1481.6`. Copied to `...-kaggle_params-1779810797.json`
  for findability.
- Per-task accuracy + 12-row flip table + real-tokenizer histogram
  recorded in `docs/investigations/kaggle_lora_mismatch/PHASE1_STEPC_RESULT.md`.

**Assumed:**
- The Kaggle Overview tab values in CLAUDE.md are current. Now also
  triangulated against Huikang README; consistent.
- vLLM 0.20.1 in `hf_env` venv is the same version that produced the
  original v9 69.4 result (per `_lora_eval_500.log`: `v0.20.1`).
  Verified by `import vllm; print(vllm.__version__)`.
- The new run targeted the same `runs/train/lora_v9` adapter dir as
  the original. Verified by `--adapter-dir runs/train/lora_v9`
  matching the original boot log's adapter reference.

**Next:**
- User decision on Q4 (base-model weight identity): cheapest signal
  is the Kaggle dataset's `model.safetensors.index.json` (~600 KB,
  one small download) — if its shard sizes match the local index
  byte-for-byte, weight identity is highly likely. Definitive
  signal is the full 60 GiB `kagglehub.model_download(...)`.
- If user approves the small index download, proceed there. Otherwise
  Phase 1 is largely exhausted; Q5 (test-set composition) requires
  external Kaggle action and is the appropriate Phase 2 starting
  point.

**Skipped / deferred:**
- Step A.2 (60 GiB kagglehub download) NOT initiated — significant
  resource cost without prior approval. Cheap interim: pull only
  the Kaggle dataset's `model.safetensors.index.json` (~600 KB) to
  cross-check shard sizes against the local file recorded in
  `PHASE1_STEPA_BASE_MODEL_SHA.md`. Not done in this session.
- The SESSION_LOG-template commit's message understates its scope (it also bundled
  user-side CLAUDE.md edits made earlier in-context). User decision
  this session: leave as-is, don't rewrite history; this entry is
  the lasting record. Per project rule ("prefer new commit over
  amend"), no follow-up needed.
- vLLM kernel-order non-determinism at greedy (12/500 rows flip
  text from a config-only change) is bigger than I'd assumed. This
  bounds per-submission Kaggle noise to ≈ ±0.5pp from kernel order
  alone. The 11pp gap is well outside that noise, but every future
  comparison <1pp should treat that as the noise floor. Worth a
  dedicated investigation if any future result claims < 1pp
  significance.
- The new text_encryption cap-hit (`21bd1251`) is interesting because
  text_encryption was previously 0% cap-hit. The model's degeneration
  mode now extends to a second task category. Phase 2 lead, not
  Phase 1 gap source.

**Commits this session:**
- A.1 dead end + bimodal-trace observation + top_p confirmed
- Step C result: max_num_seqs=64, max_model_len=8192 → −0.4pp (noise)

## 2026-05-26 16:15 — Q4 closed; Phase 1 conclusion + pivot recommendation
**Branch:** kaggle-runtime-parity-v9
**Claimed:**
- Q4 (base-model weight identity) is **effectively closed without
  the 60 GiB download.** kagglehub.model_download supports per-file
  fetch (~890 KB total for 8 small companion files); all 8 are
  byte-identical between Kaggle's `metric/.../1` dataset and local
  HF revision `cbd3fa9f...`. Byte-identical index file = identical
  weight_map and total_size; byte-identical modeling code = same
  forward path. Tiny residual risk (same names, different shard
  contents) requires one shard download to definitively rule out.
- CLAUDE.md now has a "Measurement noise" section codifying the
  ±0.5pp floor observed in Step C: 12/500 (2.4%) of greedy outputs
  flip text under a config-only `(max_num_seqs, max_model_len)`
  change due to fp32 kernel reduction order.
- The loop-on-hard-items pathology spans two task categories now
  (text_encryption gained a cap-hit in Step C). Recorded as a
  Phase 2 lead, not acted on.
- **Phase 1 is closed.** All six investigated hypotheses (H1
  extractor, H2 T=1.0 sampling, H3 max_tokens halving, H4
  max_num_seqs+max_model_len, H5 weight identity, H6 test-set
  composition) are now falsified or shown to be inside the noise
  floor. The residual ~7-10pp gap is most plausibly real
  generalization on Kaggle's distribution + ≤few-pp kernel/hardware
  noise we cannot reproduce.
- Recommended pivot: **stop debugging the gap, start improving the
  model.** Phase 2 work targets visible from Phase 1: loop pathology,
  equation_transformation weakness, bit_manipulation solver coverage.

**Verified:**
- `pip install kagglehub` in `hf_env` venv → kagglehub 1.0.1.
- `~/.kaggle/access_token` is sufficient for
  authenticated kagglehub calls. `kaggle.json` not required.
- `kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
   path="<file>")` works for individual files. Downloaded files land in
  `~/.cache/kagglehub/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1/`.
- 8 files SHA256-compared with `cmp -s` against
  `~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/
  snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848/`:
  all 8 byte-identical.
- The byte-identical `model.safetensors.index.json` carries
  `total_parameters=31,577,937,344`, `total_size=63,155,886,464`, and a
  6,243-entry weight_map mapping every tensor to a shard
  (model-NNNNN-of-00013.safetensors). Identical on both sides.
- Phase 1 conclusion section in `PHASE1_VLLM_PARITY_SCOPE.md`
  tabulates verdicts for all six hypotheses with evidence links.

**Assumed:**
- Same-tensor-name-but-different-bytes case in shards is unlikely.
  Not formally ruled out without one full shard download.
  Argument: identical `modeling_nemotron_h.py` would refuse a model
  whose shard dtypes diverged; identical `config.json` fixes the
  architecture; the BF16 in the model name is explicit (no
  re-quantization to a different precision is plausible).
- Kaggle's compute hardware is in the T4/L4 family (per typical
  Kaggle competition setup). Not confirmed against the specific
  competition's runtime. The kernel-order-noise estimate of ±0.5pp
  could be larger on a substantially different GPU/CUDA combo.

**Next:**
- Phase 1 is closed. **Phase 2 scoping is explicitly NOT part of
  this session** per user instruction. The Phase 2 kickoff should
  start from the leads recorded in
  `PHASE1_VLLM_PARITY_SCOPE.md` § "Observations not pursued in
  Phase 1" and the recommended pivot in
  `PHASE1_VLLM_PARITY_SCOPE.md` § "Recommended pivot".

**Skipped / deferred:**
- One-shard byte-comparison (~5 GiB, ~5 min) NOT done. Closes the
  tiny same-name-different-bytes risk on Q4 if it ever becomes
  load-bearing in Phase 2.
- vLLM 0.20.1 greedy kernel-order non-determinism: the 12-row Step C
  flip set is one data point. Larger surveys (e.g., 5 seeds × 500
  rows) would tighten the noise floor estimate from "±0.5pp inferred"
  to "p95 X.XXpp measured." Not done; the order-of-magnitude is what
  matters for closing Phase 1.
- The SESSION_LOG-template commit's scope conflation persists in (private) git history
  per earlier user decision ("leave the commit as-is").

**Commits this session:**
- Close Phase 1: Q4 weights byte-identical via single-file fetch

## 2026-06-10 02:59 — Seed OPEN_QUESTIONS tracker + /question slash command
**Branch:** phase3/local-kaggle-gap-diagnosis
**Commits this session:**
- docs: seed OPEN_QUESTIONS tracker + /question slash command
**Claimed:** Docs-only deliverable; no investigative claims made. The tracker
restates prior findings (C1–C5, Q1/Q3a/Q3c resolved, Q3/Q3b/Q4/Q5/Q6/Q7 open)
as seeded by the user — these were not re-verified this session.
**Verified:** Current HEAD short-SHA captured (filled into all `<SHA>` placeholders);
investigations dir had only `kaggle_lora_mismatch/` before; existing slash
command was only `log.md`. Confirmed the seeded file + command committed cleanly
as a single commit (2 files, 205 insertions) via `git show --stat`.
**Assumed:** Accuracy of the seeded evidence (Kaggle 0.55/0.58, local 0.686/
0.694, artifact filenames) taken on faith from the user's seed content — not
independently re-confirmed. Five entries retain the literal `(carry prior date)`
marker because the seed supplied no real opening date; left unfabricated.
**Next:** Backfill `(carry prior date)` fields on Q3b/Q4/Q5/Q6/Q7 with real
opening dates if recoverable. Substantive work (per C2) now points at Q6
(per-category solver expansion) as the primary Kaggle lever; use `/question` to
log evidence as it accrues.

## 2026-06-10 03:50 — Solver-coverage audit + bit_manip logprob diagnostic (option 1 + 3)
**Branch:** phase2/solver-expansion-v11
**Commits this session:**
- phase2: solver-coverage audit + bit_manip logprob diagnostic (2 scripts)
- docs(open-questions): C6 correction + Q6 reframe + new Q8
**Claimed:** (1) eq_trans has no clean numeric solver bugs to fix; (2) bit_manip's
gap is learning, not solver coverage or truncation; (3) the gap is localized to
the trace's rule-statement line.
**Verified:** dev_frozen solver coverage via scripts/audit_solver_coverage.py
(bit_manip 88.1% / max 377 tok / 0 trunc; eq_trans 28.6%). Premise figures traced
to base-model eval (dev_frozen-raw-base-1780830752.json: bm 0.095/0.893, eq
0.119/0.548) vs trained ws300 (bm 0.333/0.0, eq 0.155/0.06). All 11 eq numeric
misses classified: 5 operator-absent, 4 single-example-ambiguous, 2 inconsistent.
Teacher-forced logprob probe (scripts/bitmanip_logprob_probe.py, vLLM converted
adapter, prompt_logprobs) over 74 covered traces: 47 model-wrong, 44/47 (94%)
first diverge at RULE_STATEMENT; divergence tokens = bit-index digits/operators/
atom-shape words; invariant_k1 69% vs invariant_k2 12% vs per_bit(7-8) 35%.
**Assumed:** Probe used the vLLM-converted adapter (the deployed 33.3% model) —
faithful to the eval, but NOT the raw-PEFT trained weights (HF PeftModel.from_
pretrained is blocked by a transformers↔peft version bug: WeightConverter got
unexpected kwarg 'distributed_operation'). Logprob magnitudes assumed stable
under vLLM greedy non-determinism (±0.5pp regime; conclusions rest on rank/region
patterns, not exact logprobs).
**Next:** Decide bit_manip trace redesign (Q8 hypothesis-2): build
bit_manip_trace_v5 with a compact per-output-bit derivation, single-variable
rebuild + train + re-eval, predicting RULE_STATEMENT divergence↓ and acc↑ from
33%. Or pursue recipe lever (Q5). Both require training — not started.
**Skipped / deferred:**
- eq_trans numeric solver: deliberately NOT changed (no clean bugs; max gain
  ~0.8pp, near/below noise floor). Revisit only if a derivation-aware reframe
  of the symbolic puzzles is attempted.
- bit_manip_trace_v5 + dataset rebuild + training: deferred pending approval
  (user asked to stop before dataset rebuild).
- runs/eval/bitmanip_logprob_probe.json is LOCAL ONLY (runs/eval/ gitignored);
  regenerate via scripts/bitmanip_logprob_probe.py.
- HF raw-PEFT teacher forcing blocked by transformers↔peft version bug; if raw-
  weight logprobs are ever needed, monkeypatch WeightConverter or pin versions.
- Pre-existing uncommitted changes left untouched: src/training/convert_peft_to_
  vllm_moe.py (M), scripts/check_expert_unpack_roundtrip.py (??),
  convert_peft_to_vllm_moe.py.backup (??) — not part of this session.

## 2026-06-10 06:05 — bit_manip_trace_v5 (derivation-first) + v11 build (Q8 hyp-2)
**Branch:** phase2/solver-expansion-v11
**Commits this session:**
- phase2: bit_manip_trace_v5 (derivation-first) + v11 builder
**Claimed:** A derivation-first trace where every operator/index is forced by a
prior self-verifying line is buildable under 2000 tokens with high coverage
retention; v11 can swap only bit_manip while keeping other categories identical.
**Verified:** v5 generator on dev_frozen (74/74 covered retained, 0 errors, 0
internally-inconsistent traces, max 1692 tok) and on train_remaining (1190/1226
= 97.1% retained, 36 dropped, 0 inconsistent, max 1935 tok). Worked example
5d77eff6 matches the approved spec (two stride runs; columns consistent;
answer==GT). v11 built: 4432 records; all 5 non-bit_manip categories
byte-identical to v9 (per-category sha256 + line-order check); bit_manip 1190.
Two bugs found+fixed during validation: (1) double-NOT in displayed *-NOT
columns (showed NOT T, not T), (2) _runs strided against run-start instead of
current operands (split valid runs).
**Assumed:** The 36 dropped bit_manip use families v5 intentionally does not
derive (NAND/NOR/MAJ/XOR3/Popcount/invariant-k3) — confirmed by enumerating
their solver rules; accepted as principled (derivability over coverage).
PerBit traces narrate the solver's exact atoms (answer guaranteed); Invariant
traces use structural conversion with answer cross-check + drop-on-mismatch.
**Next:** TRAIN on v11 (single-variable vs v9: only bit_manip traces differ),
re-eval dev_frozen, predict bit_manip RULE_STATEMENT divergence↓ and acc↑ from
33.3%. Then optionally re-run the logprob probe to confirm the divergence moved.
**Skipped / deferred:**
- TRAINING NOT STARTED (user gate). v11 ready at
  datasets/processed/train_formatted_v11.jsonl.
- v11 dataset is gitignored/local-only (datasets/processed/ ignored; v9 also
  untracked). Regenerate via scripts/build_v11_bitmanip.py.
- v5 does NOT derive NAND/NOR/MAJ/XOR3/Popcount/invariant-k3 (36 train bm
  dropped). If those matter later, extend the elimination engine + _TT2_MAP /
  add triple/global derivations.
- eq_trans solver still unchanged (prior session: no clean bugs).

## 2026-06-10 19:40 — Trained v11bm; v5 derivation-first traces confirmed (Q8) + text_enc regression (Q9)
**Branch:** phase2/solver-expansion-v11
**Commits this session:** pipeline save-fix, + docs below
**Claimed:** The derivation-first v5 bit_manip trace raises bit_manip accuracy
and reduces rule-region teacher-forced divergence (Q8 hyp-2).
**Verified:** Two-stage v11bm (single-variable vs v9: only bit_manip traces
differ; buggy/load-bearing converter; eff batch 8). Canonical dev_frozen eval:
overall 0.694→0.704 (+1.0pp); **bit_manip 33.3%→50.0% (+16.7pp, 28→42/84)**,
22/47 previously-failing now correct. Probe (v5, 47 set): teacher-forced clean
1/47→17/47, rule-region first-divergence 44/47→30/47. Q8 hypothesis CONFIRMED.
**Assumed:** The eval/probe used the converted vLLM adapter (deployed-equivalent).
v5 trace length is the suspected interference driver (untested).
**Next:** Decide Q9 (ship vs mitigate). Mitigation candidates: shorten v5
derivation, raise LoRA rank, or isolate capacity; re-eval text_enc + bit_manip.
**Skipped / deferred:**
- **text_encryption regressed −12pp (0.687→0.566)** despite byte-identical
  training data — cross-task interference (NOT contamination/truncation; flips
  are small letter errors). Net overall only +1.0pp. Tracked as Q9. DO NOT
  submit v11 to Kaggle until Q9 is resolved (user gate).
- First training run lost ~5h: stage-2 ran 300 steps without --smoke and the
  save guard (`if smoke or max_steps is None`) skipped saving. Fixed
  + made resumable; re-ran stage-2 only.
- bit_manip truncation rose 0→4.8% (v5 traces ~3× longer); a length cap would
  help both truncation and Q9 interference.
- Artifacts gitignored/local-only: runs/train/lora_v11bm*, the v11 dataset, all
  runs/eval/*.json (eval + both probe JSONs).

## 2026-06-10 23:10 — text_enc interference probe (Q9): regression is in map-construction, not final_decode
**Branch:** phase2/solver-expansion-v11
**Claimed:** The v11bm text_enc regression would hit final_decode (Spark 2's
prediction: model writes correct map then ignores it / fluent-English leap).
**Verified:** Teacher-forced the unchanged text_enc gold trace under v9 vs v11bm
(all 83 + 16 flipped). Interference concentrated in **assembled_map** (cipher->
plain map construction): flipped-16 first-divergence assembled_map 5->10, clean
6->1, final_decode 0->0; assembled_map div-rate 0.0066->0.0138 (~2.1x).
CONTRADICTS the final_decode prediction.
**Assumed:** Ran a vLLM-equivalent probe (scripts/text_enc_logprob_probe_vllm.py)
because the pulled HF script (scripts/text_enc_logprob_probe.py, from
origin/phase2/text-enc-audit) can't load PEFT in this env (WeightConverter /
distributed_operation bug). Region markers + gold trace copied verbatim.
**Next:** Decide v12. Probe-grounded: shorten v5 bit_manip derivations targets
the measured regression (capacity -> map construction) AND kills 4.8% bit_manip
truncation. Char-by-char text_enc decode (Spark 2 Q9) targets a DIFFERENT
failure (free-run map usage) that teacher-forcing can't observe — defensible but
orthogonal to this regression.
**Skipped / deferred:**
- KEY CAVEAT: teacher-forcing pins final_decode to the correct gold decode, so
  it CANNOT observe the Spark 2 free-run "ignore-the-map" behavior. final_decode
  div=0 is partly an artifact, NOT proof free-run decode is fine.
- Pulled HF probe scripts/text_enc_logprob_probe.py is unrunnable here; kept for
  parity with Spark 2 / other envs.
- If v12 bundles shorten-v5 + char-by-char text_enc, it is a DELIBERATE
  two-variable run (flag in log); attribution will need a follow-up ablation.
- No Kaggle submission pending review (v11/v12).

## 2026-06-11 — Built v12 (shortened bit_manip v5 + char-by-char text_enc v5); stop before training
**Branch:** phase2/solver-expansion-v11
**Commits:** v12 build code
**Claimed:** v12 = v9 with bit_manip (shortened v5 / Cut 1) + text_enc (Spark 2
char-by-char v5) regenerated, all else byte-identical. DELIBERATE TWO-VARIABLE.
**Verified:** Guardrails all pass — bit_manip dev coverage 74/74 (==full-v5, 0
answer deltas: Cut 1 only drops the redundant example list, +16.7pp protected;
spot-checked 5d77eff6: elimination/conclusions/run-verify all present); 4
non-changed categories byte-identical to v9 (SHA); text_enc 83/83 dev
answer-preserving. v12: 4432 records (bit_manip 1190 [36 dropped, unsupported
families], text_enc 688/688, others v9-identical). Tokens: bit_manip
completion med 1072->899 / max 1935->1739 (Cut 1); text_enc med 581 max 771.
**Assumed:** Built via record-replacement (scripts/build_v12.py), NOT
build_v9_mix.py — the shared sampler rng-samples text_enc + final-shuffles,
which would perturb unchanged categories' selection AND order (byte-identity
fail). Swapped build_v9_mix bit_manip import anyway so the shared file is
correct for future full rebuilds.
**Next:** AWAIT user review of the build, then train v12 (same two-stage v9
pipeline, eff batch 8, buggy converter) -> eval -> probes. NOT started.
**Skipped / deferred:**
- TRAINING NOT STARTED (user gate: review build first).
- Cut 2 (tighten apply block) HELD — would be a 3rd variable; it's the next
  single-variable test only if v12 text_enc still shows map-construction
  interference.
- +rank / capacity-isolation HELD (3rd variable; next single-variable test if
  shorten-v5 doesn't recover map construction).
- Two-variable run: needs a follow-up shorten-only-vs-bundle ablation for clean
  attribution if GPU permits before deadline.
- v12 dataset gitignored/local-only (datasets/processed/train_formatted_v12.jsonl).
- Pulled HF text_enc probe still unrunnable here (PEFT loader bug); vLLM
  equivalent used.

## 2026-06-12 — Resolved 43%/68.7% text_enc parity (C7); built+launched v13 (revert char-by-char)
**Branch:** phase2/solver-expansion-v11
**Claimed:** The 43% text_enc figure (Spark 2, motivating char-by-char) is an
eval artifact, not a real ceiling; v13 (revert char-by-char, keep shortened
bit_manip) is the clean shippable delta.
**Verified:** 43% source = ...vllm-1779355462.json; canonical 68.7% =
...submissions_extracted...-1780923962.json. The two adapters are BYTE-IDENTICAL
(cmp), same model/prompt, text_enc truncation 0.0 in BOTH (refutes truncation
hypothesis). 26 text_enc rows flip; raw outputs begin identically then diverge
-> eval-ENVIRONMENT parity failure (inferred vLLM 0.22.1 vs 0.20.1; not in
artifact). v13 built (record-replacement): all non-bit_manip categories incl
text_encryption byte-identical to v9; bit_manip = v12 cut1 set (1190). v12
result recap: bit_manip 54.8% (Cut 1 win, trunc 0%), text_enc 53.0% (char-by-char
regressed; behavior changed [leap gone] but propagates wrong maps).
**Assumed:** vLLM version is the parity root (user-stated 0.22.1 vs host 0.20.1);
which env matches Kaggle is UNKNOWN -> Kaggle-true text_enc in [43%,69%].
**Next:** v13 train -> eval; confirm guardrails bit_manip ~54.8% AND text_enc
~68.7%. Then (separate, single-variable) +rank for text_enc map construction.
**Skipped / deferred:**
- v13 TRAINING IN PROGRESS (tmux v13); ~5-6h. Guardrails to confirm post-eval:
  bit_manip 54.8%, text_enc 68.7%.
- +rank / capacity-isolation HELD until after v13 (separate single-variable test
  on the clean baseline; map construction is the real text_enc bottleneck).
- KAGGLE-PARITY RISK (C7): text_enc 43-69% depending on vLLM env; Kaggle env
  unconfirmed. Decision-relevant before any submission.
- char-by-char text_enc v5 generator + build_v9_mix text_enc wiring remain in
  the tree (Spark 2's) but are NOT used by v13; revisit only if map-construction
  capacity fix lands first.
- No Kaggle submission pending review.

## 2026-06-12 (cont) — v13 eval: revert FAILED text_enc guardrail (cross-task interference); packaged v13 + launched +rank
**Branch:** phase2/solver-expansion-v11
**Commits:** v13+C7 build, C7-version note, +rank (expert r32)
**Claimed:** v13 (revert char-by-char, keep shortened bit_manip) would restore
text_enc ~68.7% while keeping bit_manip ~54.8%.
**Verified:** v13 eval overall 0.704; bit_manip 53.6% (45/84, HELD ~54.8%) *[note 2026-07-17: 53.6 vs expected 54.8 = −1.2pp, outside the ±0.5pp noise floor]*;
text_enc 54.2% (45/83, FAILED — did NOT return to 68.7%). Mechanism CONFIRMED
across 3 independent runs sharing byte-identical v9 text_enc data: v9(v4-bm)
68.7% -> v11(full-v5-bm) 56.6% -> v13(cut1-v5-bm) 54.2%. The text_enc regression
is cross-task interference FROM the bit_manip v5 traces, NOT the text_enc trace
format; reverting text_enc and shortening bit_manip both failed to recover it.
No trace-format route to bit_manip-win-without-text_enc-cost; net ~+1pp (0.704).
**Assumed:** expert-rank test mechanism (specialization relieves shared path).
**Next:** (1) v13 SUBMITTED to Kaggle by user (submissions/packaged/v13_submission.zip,
3 files, rank32, sha256 5bf7fa8f) — first post-0.58 submission, resolves text_enc
[43,69]. (2) +rank test running (tmux v13r): expert rank 8->32 off v13, predict
text_enc recovers. Select both as final if +rank beats v13 on Kaggle.
**Skipped / deferred:**
- CONSTRAINT: the interference site (stage-1 reasoning LoRA r=32) is at Kaggle's
  max_lora_rank=32 ceiling -> cannot raise it Kaggle-valid. +rank test uses
  EXPERT rank (8->32, Kaggle-valid) instead; if it fails, the direct reasoning
  r=64 test is LOCAL-ONLY (not submittable) — informative but needs a rank<=32
  realization for Kaggle.
- v13 Kaggle score pending manual upload; only way to resolve text_enc [43,69]
  (Kaggle runs vLLM the local hosts can't reproduce; user-reported 0.17.1, C7).
- +rank pipeline reuses lora_v13 stage-1 (stage-2 only, ~5h).

## 2026-06-12 (cont) — v13 Kaggle 0.55 = packaging confound (C8); reconstructed pad->backbone->refmatch; repackaged v13
**Branch:** phase2/solver-expansion-v11
**Claimed:** v13's 0.55 (vs A's 0.58) is a conversion/packaging confound, not a
trace regression; the rekey+reference-match step is reproducible.
**Verified:** Side-by-side build paths reconstructed from submissions/*.md +
docs/investigations/kaggle_lora_mismatch/*. v13 shipped via NAIVE converter
output (experts r8 unpadded, rank_pattern={}, model.model prefix) = original
0.56 packaging; A (0.58) = +pad(r32)+backbone-rekey+reference-match. Training
single-variable (only dataset differs). REVERSE-ENGINEERED the rekey step by
diffing _r32padded vs _r32padded_backbone: EXACTLY (1) key prefix
model.model->backbone, (2) adapter_config target_modules regex->reference list;
tensor values byte-identical (max-abs-diff 0.0). Wrote
scripts/rekey_to_backbone_reference.py. Applied full path to v13
(pad->rekey). STRUCTURAL VERIFICATION vs A: key count 12008==12008, key sets
identical, 0 shape/dtype mismatches, 12008/12008 backbone, expert rank (32,1856),
adapter_config IDENTICAL; tensor values differ (0.056, the v5 weights). Zip
908MB (A=905MB), under 1.5GB cap.
**Assumed:** repackaged v13 loads on Kaggle like A (identical structure).
**Next:** USER submits submissions/packaged/v13_r32padded_backbone_submission.zip
— THAT is the clean apples-to-apples test of v5 traces vs the 0.58 (same
packaging, only traces differ). +rank (expert r32) still training in parallel.
**Skipped / deferred:**
- DID NOT submit (user gate). Repackaged zip ready + structurally verified.
- OLD submissions/packaged/v13_submission.zip (naive packaging) is the WRONG
  one — do NOT submit it; use the _r32padded_backbone zip.
- Reproducibility note: pad + rekey now both scripted
  (pad_lora_to_uniform_rank.py, rekey_to_backbone_reference.py). The full
  Kaggle-packaging path is: convert(HEAD/buggy) -> pad --target-rank 32 ->
  rekey_to_backbone_reference.
- +rank expert-r32 run (tmux v13r) still in flight; report when waiter fires.

## 2026-06-12 23:49 [phase2/packaging/adapterA-modelmodel-squeeze/v01] — v13 Kaggle 0.56 (traces closed); built A pad+model.model squeeze
**Branch:** phase2/solver-expansion-v11
**Result:** v13 (correctly repackaged r32padded_backbone) scored **0.56** on Kaggle < Adapter A 0.58 — clean read, correct packaging: the v5 bit_manip traces do NOT translate to the 0.17.1 Kaggle runtime. Trace iteration CLOSED; Adapter A (0.58) stays best.
**Skipped:** +rank packaging — 3.0 GiB zip, 2x over the 1.5 GB cap (genuine r32 experts dont compress, unlike padded-r8 zeros). Confirmed Kaggle dead-end.
**Built (NOT submitted):** pad+model.model squeeze on Adapter A. Source = runs/train/lora_v9_warm_start_300step_vllm_r32padded (pre-rekey intermediate, kept from the May-21 A build); rekey NOT run. Sanity vs A submitted package: 12008 tensors, uniform r32 experts, model.model prefix (0 backbone), rank_pattern={}; WEIGHTS byte-identical (max-abs-diff 0.0 over all 12008 after key-rename); ONLY 2 metadata diffs — key prefix (model.model vs backbone) + target_modules (regex-str vs list). Zip submissions/packaged/adapterA_r32padded_modelmodel_submission.zip = 0.84 GiB (under cap), safetensors sha256 9fda697a. Hypothesis: backbone was neutral-to-negative (Kaggle rankpattern 0.57->0.56; local Q3a null), so model.model may score >= A 0.58.
**Constraints:** no train/GPU (CPU pack only); not submitted (user submits manually); committed, not pushed.
**Next:** user submits the squeeze; isolates the backbone-prefix variable (only diff vs A: prefix + target_modules format).

## 2026-06-13 16:19 [phase2/analysis/backbone-vs-targetmodules-confound/v01] — squeeze 0.55 reconciled: load-bearing factor is target_modules format, not prefix
**Branch:** phase2/solver-expansion-v11
**Claimed (user hyp):** backbone prefix is load-bearing; dropping it caused the 3pp drop (A 0.58 -> squeeze 0.55).
**Verified:** squeeze differs from A in TWO fields (prefix model.model<-backbone AND target_modules regex<-list) -> NOT a clean prefix test. The clean prefix-only A/B (rankpattern 0.57 vs rankpattern_backbone 0.56, byte-identical weights, only prefix renamed, target_modules=regex held; SHA-verified BACKBONE_DIAGNOSTIC §2) shows prefix NEUTRAL-to-negative. So the 3pp most likely = target_modules regex->list (mechanism: Kaggle vLLM 0.17.1 may not parse regex target_modules -> partial LoRA load). Huikang ref adapter = 12008/12008 backbone + LIST (uses both, doesnt isolate). Reverses earlier "reference-match neutral": the LIST component is load-bearing, the backbone-prefix component is neutral. Recorded C9.
**Assumed:** target_modules-format mechanism (inferred, not run on 0.17.1); rank x prefix interaction not fully excluded.
**Next:** DISAMBIGUATOR (1 submission) = model.model + LIST at r32: ~0.58 => list is lever; ~0.55 => prefix is. Can package CPU-only on request.
**Constraints:** analysis only; no submission; no GPU; committed not pushed.

## 2026-06-13 16:26 [phase2/packaging/modelmodel-list-disambiguator/v01] — packaged model.model+LIST r32 (resolves prefix vs target_modules confound)
**Branch:** phase2/solver-expansion-v11
**Built (NOT submitted):** model.model + LIST target_modules at r32. safetensors = the squeeze (runs/train/lora_v9_warm_start_300step_vllm_r32padded, model.model keys, r32), config = squeeze config with target_modules flipped regex->list. Sanity: 12008 tensors, model.model prefix (0 backbone), r32 experts; WEIGHTS byte-identical to A (max-abs-diff 0.0 over all 12008). config diff vs squeeze = ONLY target_modules (regex->list); config diff vs A = NONE (only the safetensors key prefix differs, model.model vs backbone). Zip submissions/packaged/adapterA_r32padded_modelmodel_LIST_submission.zip = 0.84 GiB (under cap).
**Disambiguation:** this one cell gives TWO clean A/Bs — vs squeeze isolates target_modules format (prefix+weights held); vs A isolates prefix (config+weights identical). Read: ~0.58 => target_modules LIST is the lever (regex silently fails to apply LoRA on Kaggle 0.17.1; prefix irrelevant); ~0.55 => prefix load-bearing after all.
**Constraints:** CPU pack only; not submitted (user submits); committed not pushed.

## 2026-06-13 21:12 [phase2/analysis/targetmodules-list-confirmed/v01] — disambiguator 0.57 resolves confound: target_modules LIST is the lever, prefix is noise
**Branch:** phase2/solver-expansion-v11
**Result:** model.model+LIST r32 = Kaggle **0.57**. Matrix (byte-identical weights): A(backbone+list) 0.58, this(model.model+list) 0.57, squeeze(model.model+regex) 0.55.
**Verified:** target_modules regex->list (prefix held) = 0.55->0.57 = +2pp REAL (~4x floor). prefix model.model->backbone (target_modules held) = +1pp, but r8 prefix-only A/B was -1pp -> opposite signs -> prefix is NOISE. CONCLUSION: packaging lever = target_modules must be a LIST (regex under-applies LoRA on Kaggle vLLM 0.17.1); prefix NOT load-bearing. Confirms C9; refutes backbone-load-bearing. Recorded C10.
**Assumed:** target_modules-format mechanism (0.17.1 loader; inferred). Residual: fully-applied Kaggle 0.58 vs local 0.69 ~=11pp = runtime-version parity (C7), separate.
**Next:** A (backbone+list) 0.58 stays best submission. Packaging solved; remaining gap is runtime version (irreducible locally w/o 0.17.1 env).
**Constraints:** analysis only; no submission; no GPU; committed not pushed.

## 2026-06-14 04:01 [phase2/cleanup/converter-commit-split/v01] — split converter into fix+refactor commits; repo-hygiene commits staged (pre-push)
**Branch:** phase2/solver-expansion-v11
**Task 1 (split):** done as clean 2-commit reconstruction (not the fallback): a fix commit (surgical expert-B reshape + docstring math; msg notes "mathematically correct per round-trip; Kaggle-clean-test pending") and a refactor commit (guardrails RESTORED: 23-layer MoE coverage check + shape assertions; config nulling restored; or-treats-0 bug fixed; docstring cleanup kept).
**Task 2 (config verify):** converter (old AND refactored) emits REGEX target_modules; LIST came ONLY from rekey. Fix folded into the refactor commit: converter now emits target_modules=LIST (REFERENCE_TARGET_MODULES, single source of truth; rekey imports it), so converter output is Kaggle-valid standalone (C9/C10). Verified cross-module import resolves; py_compile clean.
**Task 3 (commits):** session tooling (CLAUDE.md auto-save + /save-session + prompts/phase3); round-trip diagnostic + gitignore *.py.backup (backup NOT committed). C11 opened (reconciles C2 vs C8/C10).
**Verified:** py_compile both edited files; cross-module REFERENCE_TARGET_MODULES is same object; secret-scan of untracked = clean (only ML-sense "token"); working tree clean.
**Next:** SHOW push plan (29 commits, branch has no upstream) -> on approval push origin phase2/solver-expansion-v11 -> git fetch -> repo-readiness review (recs only).
**Constraints:** NOT pushed (awaiting approval); no GPU/train/submission; single-line commands.

## 2026-06-14 12:53 [phase3/repo-readiness/review/v01] — pushed phase2 + phase3 branches; delivered repo-readiness review (recs only)
**Branch:** phase2/solver-expansion-v11
**Pushed (approved):** origin/phase2/solver-expansion-v11 (29 commits, new branch) + origin/phase3/local-kaggle-gap-diagnosis (new). Both now backed up; upstreams set. git fetch -> Spark 2 origin/phase3/huikang-warmstart visible.
**Review delivered (RECOMMENDATIONS ONLY, nothing moved/renamed/deleted):** P0 privacy/legal = competition data (datasets/splits/*.jsonl) + Huikang material (references/huikang*, his notebook/code/metadata) are tracked -> redistribution risk for public; no LICENSE; 2 SOURCE files hardcode ~ paths (harvest_correct_traces.py, text_encryption_solver.py) = privacy + breaks reuse; Spark sync scripts leak absolute host paths. P1 = README eval params stale (temp=1.0/3584 vs confirmed 0.0/7680/8192) + missing vLLM-version caveat (C7); scrub username paths in 2 PHASE1 docs; branch curation (13 remote branches). P2 = prune dup/empty prompts/phase3; commit a consolidated RESULTS.md (eval JSONs gitignored; Kaggle scores ARE preserved in KAGGLE_SUBMISSION_RESULTS.md + submission_log.csv).
**Verified:** no .env/keys/tokens/IPs tracked (.env.example is placeholders); 5 tracked files with ~; README+pyproject+Makefile present, no LICENSE.
**Next:** user approves which P0/P1/P2 items to action (scrub/gitignore/download-script/LICENSE). No action taken.
**Constraints:** read-only review; nothing moved/renamed/deleted; this log committed locally (not pushed).

## 2026-06-15 18:46 [phase3/repo-readiness/batch1/v01] — Batch 1 repo cleanup committed (not pushed)
**Branch:** phase2/solver-expansion-v11
**Committed (4 groups, no squash):** scrub abs-paths/usernames (2 sources -> Path(__file__).parents[2]; PHASE1_STEPA/STEPC/SESSION_LOG redacted; git grep username = 0); Apache-2.0 LICENSE (our code only); README fixes + RESULTS.md; PUBLICATION_CHECKLIST + prune phase3 _v0/_v2.
**Verified:** README architecture corrected AGAINST config (hybrid_override_pattern len 52, M=23/E=23/*=6, E-positions == converter MoE layers -> 23 Mamba + 23 MoE + 6 attn, top-6 of 128 + 1 shared); eval params -> Kaggle 0.0/7680/8192/64; SESSION_LOG sed diff eyeballed (3 substitutions, text intact); both edited sources py_compile clean.
**Deferred (Batch 2, PUBLICATION_CHECKLIST):** Huikang licensing, competition-data redistribution, Spark sync scripts, branch curation, run_eval defaults — resolve before any public push.
**Next:** user decides whether to push; then writeup-only.
**Constraints:** not pushed; no GPU/train/submission.

## 2026-06-23 23:46 — Reader-readability pass on RESULTS.md + OPEN_QUESTIONS.md (additive glossaries + outdated-claim markers)
**Branch:** phase2/solver-expansion-v11
**Claimed:** the two writeup docs were full of project-internal shorthand (C1–C11, v9–v13/+rank, Adapter A / squeeze / B_FIXED, dev_frozen, rankpattern, backbone-rekey, target_modules list-vs-regex, noise floor) that an outside reader cannot parse; making them readable does not require altering any technical claim.
**Verified:** every glossary definition sourced from the docs themselves + SESSION_LOG + kaggle_lora_mismatch/ (recommend-first diff reviewed and approved before applying). "Form 3"/"v10" confirmed grep-clean across docs/, submissions/, reports/ (genuinely undefined -> TODO flag, not guessed). Post-edit grep: 0 leftover [[wiki-links]], 0 leftover 2025-dates, 0 leftover (carry prior date), 0 stray chars; 4 <a id> heading anchors + 9 [QN](#qN) links present. Diff is additive except approved typo/format cleanups.
**Assumed:** nothing material taken on faith — the one inference ("Form 3"/"v10" = an earlier text_enc trace-format experiment) is explicitly marked unverified in the in-doc TODO.
**Changes:** RESULTS.md +Legend/glossary section. OPEN_QUESTIONS.md +Glossary/legend section; C2 prefixed *[SUPERSEDED — see C11 and C7]* (names both overturned conclusions: "load-bearing bug/fix regressed" and "gap is NOT runtime"); Q3->C11 + Q3c-stub->C11 + ParamWrapper inline pointers; Form 3/v10 TODO; cleanups: [[#qN]]->GitHub anchors, 2025-06-09->2026-06-09 (incl. one inside the Q3c Resolution — user-approved: a year typo changes no claim), (carry prior date)->2026-06-09.
**Next:** writeup-only. Open offer still standing: push origin phase2/solver-expansion-v11 + main to make origin canonical (user's call; not done).
**Skipped / deferred:** no push (docs committed locally only). Batch 2 (PUBLICATION_CHECKLIST: Huikang licensing, competition-data redistribution, Spark sync scripts, branch curation, squash/orphan history sanitization) remains the real pre-public gate. [[#qN]]->anchor + date-typo fixes applied ONLY to OPEN_QUESTIONS.md; if other docs use the same wiki-link/date conventions they were not swept this session.

## 2026-07-06 15:11 — Phase 0 writeup merge: created writeup.md (retrospective base + 3 salvaged detail chunks)
**Branch:** phase2/solver-expansion-v11
**Commits this session:** none (phase0_prompt.txt: no commit, no push).
**Claimed:** writeup.md = writeup_nemotron_retrospective.md verbatim + 3 ports from writeup_best_finetuning_method.md, all reframed to not contradict C7/C11 corrections: (1) text_enc forensics (26/47 one-word-off; 0/47 mechanical-map-match on misses vs 81% = 29/36 on correct; 100% coverage, ≤536-token traces) into §6b, framed as pre-C7 observations of the Spark 2 vLLM-0.22.1 artifact run, NOT a live learnability wall; (2) 12,008-tensor conversion count + PEFT 3D packing shapes into §6a as round-trip validation detail; (3) per-category gap table into §2, relabeled as local eval (not "best submission" — Kaggle has no per-category output), text_enc row starred as pre-C7 Spark 2 number with canonical 68.7% noted.
**Verified:** both source drafts read in full; every ported number traced to older draft §3b/§4/§5; retracted claims (live text_enc learnability wall, "fix" as valid intervention, Open Contribution Award framing, retired-then-reinstated hedge) confirmed NOT carried over.
**Assumed:** rounded table rows (~85/~80/~75/~25) are trustworthy local numbers — older draft doesn't record which machine; caveat noted inline in writeup.md rather than dropped. One RECONCILE flag left inline in §6b: interleaved lookup-and-emit format attribution conflict (older draft credits tonghuikang; retrospective says "we designed").
**Next:** Phase 1 (repo audit) per phase1_prompt.txt; resolve the §6b attribution RECONCILE before publication.
**Skipped / deferred:** older draft's "Pivotal submissions" table NOT ported — its version labels conflict with the retrospective (older: v13 = B-fixed @0.55; retrospective: v13 = bit_manip v5 swap @0.56); flagged for reconciliation rather than merged. Source drafts left in place per prompt.

## 2026-07-06 16:05 — Phases 1+2: writeup audited against repo, corrected, moved to docs/; RESULTS.md B_FIXED erratum
**Branch:** phase2/solver-expansion-v11
**Commits this session:** three, user-approved post-verification — docs(writeup) writeup+handoff+log; fix(results) B_FIXED erratum+provenance README; docs(readme) writeup link. Pre-commit check (user-requested): the submission score-record commit confirmed as the 0.58 Adapter A score-record commit; submitted package verified at runs/train/lora_v9_warm_start_300step_vllm_r32padded_backbone/ (target_modules LIST, 12,008 backbone-prefixed tensors, r32-padded experts, built May 24). Trap noted: submissions/lora_v9_warm_start_300step_r32padded.zip is the pre-rekey intermediate (regex+model.model), NOT Adapter A.
**Claimed (Phase 1 audit):** writeup numbers verified per-claim vs runs/eval JSONs, OPEN_QUESTIONS C7/C8/C10/C11, RESULTS.md, converter code, and the B_FIXED submission artifact. Confirmed: 33.3%→53.6% (v13 bit_manip), 68.7/43.4/26-flips (C7), +2pp list-format lever, (out,E,r) reshape, 44/47 rule-region divergence, 12,008 tensors. Newly discovered: RESULTS.md:82 B_FIXED row contradicted the actual zip (regex+model.model+r8 rank_pattern, NOT list+backbone+r32) — artifact check strengthens C11.
**Verified:** B_FIXED adapter_config + safetensors header read from submissions/extracted/lora_v9_ws300_B_FIXED.zip; per_task_type extracted from 6 eval JSONs; probe method read from scripts/bitmanip_logprob_probe.py (argmax-rank, no −0.69 threshold); huikang char-by-char design confirmed in his reference cipher.py (per-char lookup steps) before local copy deletion; the submission score-record commit is on main.
**Assumed:** rank ~3452/4100, leaders 0.86–0.87, close date, text_enc 100% coverage, ≤536-tok traces, 26/47 / 0/47 / 29/36 forensics — all marked UNVERIFIED/uncommitted inline in docs/writeup.md, none guessed.
**Changes (each user-approved):** docs/writeup.md = writeup.md edited per audit (11 edits: +20pp exact numbers, canonical §2 table, argmax-divergence method, +6.8pp cut, C7 parity qualifiers, §6b huikang attribution + 68.7→56.6→53.0→54.2 chain + uncommitted-forensics caveat, §9 paths + submission-record-commit citation) then moved to docs/; RESULTS.md B_FIXED row fixed + dated note; references/README.md provenance added; docs/reference_solvers/tonghuikang/ deleted (was gitignored); source drafts archived to scratchpad + deleted; README links writeup. Handoff: handoffs/2026-07-06-writeup-reconcile.md.
**Next:** user reviews diff + handoff → commit on confirmation; resolve UNVERIFIED markers (Kaggle UI, text_enc coverage rerun); PUBLICATION_CHECKLIST Batch 2 before public.
**Skipped / deferred:** src/ restructuring (empty src/solvers/ vs src/data/ rename — breaks imports; propose-only), branch curation, history sanitization, optional annotated tag at the submission-record commit, submission_log.csv still header-only + KAGGLE_SUBMISSION_RESULTS.md single orphan row (noted, not fixed). Full writeup diff archived at session scratchpad phase2_writeup.diff.

## 2026-07-10 — LB split (public 0.58 / private 0.604), Open Prize attribution fix, text_enc coverage audited 100%
**Branch:** phase2/solver-expansion-v11
**Commits this session:** one semantic unit (user-approved plan, Co-Authored-By dropped per user): writeup+RESULTS public/private LB labeling; winner-attribution correction (tonghuikang = midpoint Open Prize, NOT overall winner — writeup ack, references/README two-entry split [1. tonghuikang @GitHub 82bd1880, 2. Team NullSira first-place writeup URL], observations.md bracketed correction, OPEN_QUESTIONS glossary note; phase3 prompt archive left verbatim by design); markers resolved (close date 2026-06-15; winners ~0.92 private, 0.86–0.87 = public pack; rank 3488/4182).
**Verified:** text_enc solver coverage RE-AUDITED on dev_frozen via extended scripts/audit_solver_coverage.py --tasks text_encryption (CPU-only, hf_env python, v9 trace format): **100% (83/83), trace tokens min 384 / med 470 / max 536** — the old draft's "≤536 tokens" was exact. JSON: runs/eval/text_enc_solver_coverage.json (gitignored; script committed = reproducible). references/README entry 2 confirmed NullSira-only (no huikang repo link).
**Assumed:** private LB 0.604 / rank 3488/4182 / NullSira handle+URL / ~0.92 winners / close date — user-provided from Kaggle UI, taken as authoritative (repo has no Kaggle-UI artifacts).
**Next:** commit is the publish-ready docs state; remaining pre-public gates unchanged (PUBLICATION_CHECKLIST Batch 2: licensing, competition-data redistribution, history sanitization — history still carries absolute-path blobs, an internal hostname on an unmerged branch, and author-email metadata). No push.
**Skipped / deferred:** OPEN_QUESTIONS Q6 evidence not updated with the new text_enc coverage line (writeup + JSON carry it); audit script's bit_manip/eq default behavior unchanged.

## 2026-07-17 — P0 removals at source: competition JSONLs untracked, huikang verbatim files removed, registry stub dropped
**Branch:** phase2/solver-expansion-v11
**Commits this session:** one (user-approved): P0 removal unit. Also folded: RESULTS.md bit_manip +20.3pp fix, Q9 + SESSION_LOG:589 bracketed correction notes (convention: original words untouched).
**Removed:** 6 competition-verbatim JSONLs (git rm --cached — DISK COPIES KEPT for local eval, now gitignored: datasets/*.jsonl + datasets/splits/*.jsonl, dev_frozen negation dropped); 6 huikang verbatim files (full rm — no license located in vendored copies => all-rights-reserved default; our 10 provenance/extraction docs kept; references/README.md huikang section rewritten as removal record); runs/registry.csv (empty header-only stub; run_report.md KEPT — force-added curated artifact, PII-clean).
**Verified:** all 6 JSONLs id-joined vs datasets/raw/train.csv = verbatim competition rows + derived task_type only; split.py seed-42 defaults reproduce 4 splits (dev_frozen_shuffled = ad-hoc permutation, documented); runs/ tracked files PII-clean ("base@corr" = arm label, email-pattern false positive); +6.8pp mystery SOURCED: runs/train/lora_v9/run_report.md headline = v9 overall 69.4% vs v5-mix 62.6% — an adapter-mix delta the old draft misattributed to the v3→v4 trace format (writeup cut stands, now explained).
**Docs:** datasets/splits/README.md rewritten (regeneration procedure); repo README data line; writeup §9 + split.py bullet.
**Next:** re-run fresh-history build + gate. Known dangling pointers to removed huikang files: CLAUDE.md read-first list item 4, HUIKANG_COMPARISON.md:3 reference line — flagged, not edited (await scrub-vs-note decision).
**Skipped / deferred:** OPEN_QUESTIONS:194 SHA scrub proposed but not applied (user reviewing same-class sweep scope); datasets/raw/train.csv remains on disk untracked (needed to regenerate splits).

## 2026-07-17 (2) — SHA scrub across shipping docs; P0 boxes ticked; rebuild pending
**Branch:** phase2/solver-expansion-v11
**Commits this session:** pointer-fix commit (huikang dangling refs) + this scrub/checklist commit — described by subject only per the scrub itself.
**Changes:** ~60 private-history git-SHA citations replaced with dates/descriptions across OPEN_QUESTIONS.md, SESSION_LOG.md, handoffs/2026-07-06-writeup-reconcile.md, observations.md (whitelist-verified per-site; problem IDs, HF revision pins, and sha256 content hashes untouched; legend notes added to OPEN_QUESTIONS + handoff). PUBLICATION_CHECKLIST P0 boxes ticked with dated resolution notes (data JSONLs removed + split.py regeneration; huikang 6 files removed + provenance stub).
**Verified:** post-scrub whitelist grep = 0 residual SHAs in the four files.
**Next:** rebuild ~/kaggle/nemotron-public from new HEAD, re-gate, STOP; first public push is the user's (`git push -u origin main` to the empty public repo); future public changes via branch → PR.

## 2026-07-18 14:16 — web/explainer reconciled to post-publish canon; rebased onto main; 3-commit split
**Branch:** web/explainer
**Commits this session:**
- (rebase) web: add retrospective explainer with log-verified numbers (T1-T4) — replayed onto main head, zero conflicts
- web: correct explainer claims to post-publish canon (probe mechanism, text_enc chain, C7 hedge, +20.3pp, public/private LB)
- web: un-stub repo links + huikang attribution; absolute GitHub Pages-safe hrefs
- web: rename --indigo token to --blue (value #1d4ed8 unchanged)
**Claimed:** web/index.html carried pre-correction claims: a fabricated −0.69 logprob threshold as the probe mechanism (writeup defines divergence as gold token rank > 1); the text_enc 68.7→54.2 drop collapsed to one variable (corrected chain: 68.7 v9 → 56.6 v11 interference-only → 53.0 v12 two-variable → 54.2 v13 revert, ≤3.6pp isolable to char-by-char); "dominant cause" overclaim on C7; +20.2pp vs canonical +20.3pp; all scores unlabeled public-LB; private 0.604 / rank 3488/4182 absent; huikang attribution stubbed. Also claimed zero dead file paths in the HTML.
**Verified:** rebase safety via `git merge-tree --write-tree main web/explainer` (clean tree, main never touches web/); every audited number against docs/writeup.md + RESULTS.md + OPEN_QUESTIONS.md on main (0.694 local, +1.0pp net, 25.3pp swing, 0.55/0.56/0.57/0.58 ladder, mixed-rank 0.56→0.57–0.58); the 2026-06-13 matrix date against SESSION_LOG lines 650/658/664 and OPEN_QUESTIONS C10 (kept); cited paths exist on main via `git cat-file -e` / ls-tree; commit partition purity by grep over the three staged diffs (no hrefs in c1, no --blue outside c3); final tree byte-identical to the user-reviewed diff via `cmp`.
**Assumed:** GitHub Pages will serve from this repo's public mirror so absolute blob/main URLs resolve (repo is public per writeup §9); "C8 v13's 0.55 = naive-packaged submission" read from OPEN_QUESTIONS C8 text, not re-derived from submission artifacts.
**Next:** push web/explainer (needs --force-with-lease; origin diverged at the rebase), then user-run GitHub Pages deploy and a browser pass over the hero tri-stat responsive layout (new repeat(3) grid, untested visually).
**Skipped / deferred:** no visual/browser verification of the edited page (text-only edits reviewed as diffs); footer "prototype" tag removed per approval — revisit if the page should still be marked pre-final; citation block has no URL field (add the Pages URL once it exists); explainer_prompt.txt and other untracked session files left untracked.

## 2026-07-22 08:16 — writeup restructured around three-ceilings thesis; README rewritten as explainer
**Branch:** web/explainer
**Commits this session:**
- docs: restructure writeup around three-ceilings thesis; README becomes explainer
**Claimed:** README.md (not web/index.html) is the right explainer/report target — the repo's accessible entry point for the recruiter/new-engineer audience; the old README project tree was stale (configs/openclaw listed but absent); the writeup's ~3.5B and README's ~3B active-param figures are both convention-dependent derivations the model config cannot arbitrate; the §5 trace excerpts initially pulled from train_formatted_v9/v13.jsonl were solver-generated text wrapping competition-derived binary strings (row id present in raw train.csv).
**Verified:** every path cited in both docs exists (20 checked); all 19 relative links resolve and all 5 README→writeup anchor fragments match GitHub-slugged headings (scripted check); hedge-survival grep confirmed every required qualifier post-restructure (Spark 2 attribution inferred ×3, 0.17.1 user-reported, "a dominant contributor not the sole cause" ×2+README, C2 untested ×3, "uncommitted, contemporaneous", ≈11pp inferred, public/private caveat, ±0.5pp ×3); model config read from HF cache (no explicit active-param field; ≈2.8B excl. / ≈3.5B incl. embeddings by derivation); excerpt provenance checked via source field + train.csv id join; replacement excerpts generated by running bit_manip_trace_v4/v5 + bit_manip_solver_v3 on synthetic pairs in a scratchpad venv (numpy; PEP 668 blocks user install); grep confirms no original competition-row bit strings remain; layout block checked against actual tree (src/solvers is an empty stub — solvers live in src/data).
**Assumed:** "9,500 rule-induction puzzles" carried forward on user's confirmation against the competition row count (not independently re-counted); GitHub's Mermaid rendering for the new pipeline diagram (not renderable locally); public-repo README divergence (user-reported: it has its own drifted tree) — this session edited the dev README only.
**Next:** follow-up pass on web/index.html to harmonize "~3.5B active" with the A3B phrasing now used in both markdown docs; user to sync the public repo's README in a separate PR; consider trimming the writeup's ~840-word growth if reviewers find it long.
**Skipped / deferred:** markdownlint not run (not installed; npx would download — structural checks scripted instead); web/index.html untouched this session per approval (its writeup links are anchor-free so the heading restructure cannot break them, but its "~3.5B" figure now diverges from both markdown docs); internal-only dirs (handoffs/, logs/, web/, submissions/) intentionally omitted from the README layout block; scratchpad venv is session-temporary and was not added to the repo.
