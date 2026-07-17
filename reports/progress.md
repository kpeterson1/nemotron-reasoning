# Progress Checklist — Nemotron Reasoning Challenge

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

Deadline: **2026-06-15**.

---

## Phase 0 — Repo scaffold

- [x] Create directory tree at `~/kaggle/nemotron-reasoning/`
- [x] Root config files (README, pyproject, .env.example, .gitignore, Makefile)
- [x] `configs/{inference,train,eval}/*.yaml` with actual harness params
- [x] `prompts/{direct,rule_induction,verify_then_answer,few_shot,task_typed}/v1.yaml`
- [x] `src/` Python modules — evaluation, inference, data, training, packaging, verifiers
- [x] Unit tests (`tests/unit/test_eval.py`, `test_extract_answer.py`, `test_format_training.py`) — 21/21 passing
- [x] `git init` + Phase 0 commit

## Phase 1 — Data acquisition & splits

- [x] Download competition data (train.csv: 9,500 rows; test.csv: 3-row sample)
- [x] Verify task type classifier covers 100% of train (no `unknown` rows)
- [x] Carve frozen splits via `src/data/split.py`:
    - [x] `dev_frozen.jsonl` — 500 examples, stratified by task_type, seed 42
    - [x] `train_remaining.jsonl` — 9,000 examples
    - [x] `hard200.jsonl` — 200 longest prompts from dev_frozen
    - [x] `speed_bench.jsonl` — 50 shortest prompts from dev_frozen
- [x] Verify task distribution per split, commit splits

## Phase 2 — Verify extract_answer matches Kaggle metric

- [x] Diff our `src/evaluation/extract_answer.py` against the Kaggle metric source
- [x] Run the Kaggle docstring test cases end-to-end (extract + verify)
- [x] Document any divergences in `reports/observations.md`

## Phase 3 — Diagnostic baseline inference

- [x] Set up inference (vLLM 0.20.1 aarch64; `NemotronHForCausalLM` natively supported)
- [x] Download `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` weights (59 GB / 13 shards)
- [x] Smoke test: 1-row dry run (495s, vLLM works end-to-end)
- [x] 50-example smoke run (27.5 min, base model, no template, **acc=0.080 — but bit_manipulation only**)
- [x] Identified **truncation crisis**: 94% of traces hit max_tokens=3584 before closing `</think>` / writing `\boxed{}` (only 5/50 emit `\boxed`)
- [x] Identified split-ordering bug in `dev_frozen.jsonl` (task-bucketed, not interleaved). Added `dev_frozen_shuffled.jsonl`.
- [x] Re-run 50-example smoke on `dev_frozen_shuffled.jsonl` for cross-task signal — **acc=36%** (vs 8% on bit_manip-only)
- [x] Two-arm comparison (baseline vs 'be brief'): **brief is +12pp overall (36 → 48%)** but heterogeneous per task — see observations.md
- [ ] Validate brief on larger n (≥150 shuffled, ~25/task) to tighten per-task error bars
- [x] **Diagnostic — bit_manipulation structured prompt (n=84)**: 2.4% acc, regressed vs brief — prompt engineering can't fix it
- [x] **Diagnostic — text_encryption @ max_tokens=8192 (n=7)**: 14.3% acc unchanged vs 3584 — NOT a budget problem
- [x] Conclusion: both hard tasks (bit_manip, text_enc) require SFT
- [x] **Inspect equation_transformation failures** — true accuracy is ~0% (10% is a float-tolerance artifact on `'-1'` vs `'-01'`). Two new failure modes: (A) multi-operator puzzles waste tokens decoding irrelevant operators, (B) **the harness's own `\boxed{your answer}` example poisons symbolic-answer tasks** — the model copies "your answer" verbatim when stuck.
- [ ] (Deferred) Full sweep on bigger n to validate per-task brief signal (esp. unit_conversion regression)

## Phase 4 — Training data preparation

- [x] **Verify `format_training.py` works with real Nemotron tokenizer** — all 8 markers (im_start, im_end, think tags, \boxed, harness suffix, trace, answer) pass on 3 train rows. Tokens: 253 prompt + 42 completion = 295 (well under 4096 ctx).
- [x] **bit_manipulation solver** (`src/data/bit_manip_solver.py`) — shift-invariant rule search (K=1, 2, 3 over offsets [-7,+7] × {wrap, zero}, NumPy-vectorized). On 1,518 train_remaining bit_manip rows: 72.1% match, **70.9% test-correct, 98.3% generalization**. ~13 min walltime.
- [x] **Inspected 5 unmatched puzzles** — three pattern classes: position-uniform with edge exceptions, fully position-dependent, and complex K≥2 rules. Could push coverage higher with K=4 / per-bit fallback / 2-op compositions, but 70% is enough for an SFT pilot.
- [x] **bit_manipulation trace generator** (`src/data/bit_manip_trace.py`) — produces 5-section traces: frame, two rejected hypotheses, winning rule + verification on first example, spot check, apply to test. K=3 truth tables introduced once to keep length bounded.
- [x] **Generated 1,076 verified bit_manipulation traces** at `datasets/processed/bit_manip_traces.jsonl`. Token length p50=735, p90=933, max=946 — every trace fits under 1,500 thinking tokens.
- [ ] **Decide teacher model for trace generation** — base model isn't strong enough on bit_manip/text_enc to self-distill. Candidates: stronger Nemotron variant, Claude (via API), or hand-crafted procedure templates.
- [ ] **Trace-length budget**: target ≤1500 thinking tokens, hard cap 2500. Anything past that loses to truncation at inference time.
- [ ] **Task-specific procedure traces**:
    - bit_manipulation: structured search that handles permutations and compositions (not just basic ops)
    - text_encryption: tabular cipher→plain map first, then linear decode
    - equation_transformation: (A) triage operator from test query first; (B) handle symbolic answers without copying "your answer" from harness suffix
- [ ] Generate reasoning traces for `train_remaining` examples
- [ ] Filter generated traces by verifier correctness (use `answers_match`)
- [ ] Build `datasets/processed/train_formatted.jsonl` via `format_training.py`

## Phase 4 — Training data preparation

- [ ] Pick top-performing prompt family from Phase 3 for trace generation
- [ ] Generate reasoning traces for `train_remaining` examples
- [ ] Filter generated traces by verifier correctness
- [ ] Build `datasets/processed/train_formatted.jsonl` via `format_training.py`

## Phase 5 — LoRA fine-tuning

- [x] First training run with `configs/train/lora_baseline.yaml` — 2 epochs, 8h walltime. Loss 0.82 → 0.015 (55× drop, memorized training set).
- [x] Eval adapter on `dev_frozen` (n=500) — **acc 36.8% vs brief-arm 48.0% (n=50). Regression.** Per-task: bit_manip went 0%→3.6% (training target), but unit_conversion crashed 100%→44.6% (catastrophic forgetting on shared paths). All other tasks regressed too.
- [x] Diagnosed pilot failure: memorization-not-generalization + catastrophic forgetting + inference-time distribution shift. Hand-crafted procedural traces ruled out as a sufficient SFT signal.
- [~] **Phase 5b — combined approach: self-distillation + multi-task mix.**
    - [x] Harvest pass complete: 3,975/9,000 correct (44.2%). 27.1 h walltime.
    - [x] Per-task yields: grav 91.5%, num 92.7%, unit 55.5%, text_enc 10.9%, eq 9.4%, bit 4.3%.
    - [x] **v2 training mix built** (`datasets/processed/train_formatted_v2.jsonl`, 2,338 records). sqrt-rebalanced + 200 procedural bit_manip. Hard tasks 23.4% of mix (vs 9% raw).
    - [x] **Competitive intel corrections applied**: harness uses temperature=0.0, max_tokens=7680. LoRA config updated (alpha 32, LR 3e-4, dropout 0.0, widened targets, max_seq 8192).
    - [x] **mamba_ssm + causal_conv1d installed** (built from source, aarch64). Should give significant training speedup vs naive Mamba fallback.
    - [x] **Training run** (1 epoch, 83 min walltime with fused mamba kernels — 6× faster than pilot).
    - [x] **Eval at corrected harness params** (temp=0.0, max_tokens=7680, n=500): **overall 58.4% accuracy** (vs LoRA v1 36.8%, brief baseline 48%, pure base 36%). Truncation 32% (vs 56%).
    - [x] Per-task wins: gravitational_constant 64→100%, unit_conversion 45→89%, **text_encryption 5→45%** (the headline win), grav/numeral/unit recovered from v1's catastrophic forgetting.
    - [x] Per-task losses: **bit_manipulation regressed 11→5% vs brief** (procedural mix did not help). equation_transformation only 13% (still mostly broken).
- [x] **Base at corrected params baseline complete (n=500, 4h15m)**: acc=52.0%, trunc=36%. Confirms most of the "LoRA v2 win" was the sampling fix.
- [x] **LoRA contribution isolated**: corrected sampling alone gives +16pp (36→52%); LoRA adds only +6.4pp (52→58%). LoRA wins concentrated on grav (+25) and unit (+17); HURTS bit_manip (−8); only +4 on text_enc.
- [x] **v3 training mix prepared** (`datasets/processed/train_formatted_v3.jsonl`, 2,251 records, no procedural bit_manip).
- [x] **bit_manip inspection**: procedural traces transferred at format level but reasoning is random-guessing. Confirms the −8pp regression is from harmful template.
- [ ] **Decision pending**: train v3 (predict ~60% overall, bit_manip recovers to ~11%) vs pivot to a stronger teacher for the hard tasks.

## Phase 6 — Deterministic solvers + solver-backed CoT traces

- [x] Built deterministic Python solvers for all 6 categories
- [x] Coverage on train.csv: 7,678 / 9,500 verified-correct (80.8%)
- [x] Trace generators per category that mirror the solver's reasoning steps
- [x] v4 training mix built (2,991 sqrt-rebalanced records)
- [x] LoRA v4 trained (1 epoch, 64 min, train_loss=0.082)
- [x] LoRA v4 eval @ corrected harness (n=500): **61.6% acc / 16% trunc**. Per-task: grav 100%, unit 100% (+11pp), num 100%, text_enc 51.8% (+7pp), eq 15.5%, bit_manip **3.6% (−1pp, REGRESSED)**.
- [x] Headline qualitative finding: model now writes 90%+ shorter outputs on text_enc / unit / equation. Truncation 36% → 16% overall.
- [x] **v5 trained** (no bit_manipulation in training mix). Eval n=500: **62.6% acc / 5% trunc**. New best. text_encryption jumped +8.4pp to 60.2% — cross-task interference from bit_manip training was poisoning text_enc.
- [x] **bit_manip solver v2** added per-output-bit fallback: 71% → 78% coverage on train.csv.
- [ ] bit_manipulation still stuck at 2-4% across all our trained LoRAs — SFT cannot teach this task to this model (3 independent training runs confirm). Need different intervention (RL-w/-verifier, stronger teacher, or accept the ceiling).
- [ ] Expand equation_transformation solver past 12%

## Phase 6 — Submission

- [ ] Package adapter via `make package`
- [ ] Validate submission.zip locally (size, adapter_config.json, weight files)
- [ ] Upload to Kaggle, log score in `reports/leaderboard/submission_log.csv`

---

## Recent updates

- **2026-05-06** — Phase 0 complete (initial scaffold commit). Phase 1 download done; classifier coverage verified at 100%.
- **2026-05-06** — Phase 1 splits carved: `dev_frozen` (500, 83–84/type), `train_remaining` (9000), `hard200` (length-skewed, 3 types only), `speed_bench` (50, all equation_transformation).
- **2026-05-06** — Phase 2 done: pulled actual Kaggle metric notebook, fixed 4 small divergences in `extract_answer.py`, achieved 20/20 + 14/14 parity on a fuzz set. 29 unit tests pass.
- **2026-05-06** — Phase 3: vLLM 0.20.1 working on aarch64 GB10. Baseline smoke n=50 acc=0.08 (bit_manipulation only) — DOMINATED by trace-truncation (94% hit max_tokens). format_training verified with real tokenizer.
- **2026-05-07** — Phase 3: cross-task n=50 baseline acc=36% / trunc=54%. 'Be brief' arm: acc=48% / trunc=50% / 40% faster. Big gain on gravitational_constant (46→92%). Regression on unit_conversion (100→50% on n=4). Per-task heterogeneity argues for task-conditional brief, not global.
- **2026-05-07** — Phase 3: structured prompt regressed bit_manip (11→2%); 8192-token budget did NOT help text_enc (14% unchanged, output 3.7× longer). Both hard tasks need SFT, not prompt tweaks.
- **2026-05-08** — Phase 5 pilot SFT'd LoRA on 1,076 hand-crafted bit_manip traces. Train loss 0.82→0.015. Eval n=500: acc 36.8% (vs brief-arm 48%, base 36%). Per-task regression on every task. Hand-crafted procedural traces ruled out — need base-model self-distillation or stronger teacher LM.
- **2026-05-09** — Harvest complete (27 h walltime). 3,975/9,000 self-distilled correct. Hard tasks remain hard: bit 4.3%, text_enc 10.9%, eq 9.4%.
- **2026-05-10** — v2 mix built (2,338 records, sqrt-rebalanced, hard tasks 23%). Configs corrected per competitive intel (temperature=0.0, max_tokens=7680, LR 3e-4, alpha 32, widened LoRA targets). mamba_ssm + causal_conv1d installed for fused kernels. Ready for 1-epoch training run.
- **2026-05-10** — LoRA v2 trained in 83 min (1 epoch). Eval n=500 @ corrected harness: **58.4% acc (+22pp vs base, +10pp vs brief)**. text_encryption 0→45% breakthrough. bit_manipulation still broken at 5%.
- **2026-05-11** — Base @ corrected params baseline (n=500): **52.0% acc**. Most of the v2 "win" was sampling (+16pp), not LoRA (+6pp). LoRA HURTS bit_manip by 8pp. v3 mix ready (no procedural).
- **2026-05-11** — Pivoted to deterministic Python solvers (4 of 6 at 100% coverage on train). v4 LoRA trained on 2,991 solver-backed CoT traces. Eval: **61.6% (+9.6pp vs base, +3.2pp vs v2)**. Truncation 36% → 16%. Output length dropped 90%+ on text_enc / unit / eq. bit_manipulation still broken (3.6%) — needs different intervention.
- **2026-05-12** — v5 (no bit_manip in training): **62.6% acc, 5% trunc**. text_enc +8.4pp from cross-task interference removal. bit_manip solver v2 hits 78% coverage. Ceiling under current approach ~65%.
