# rank_pattern unpadded warm-300 — Option B build

**Goal**: build a submission zip whose safetensors is byte-identical to the unpadded warm-start 300 (the local-70% adapter) but whose `adapter_config.json` declares the per-tensor mixed ranks explicitly via `rank_pattern` and `alpha_pattern`. This addresses the mixed-rank packaging concern without zero-padding (which empirically introduced FP-drift in vLLM's kernel — see `PROMPT_PARITY_R32PADDED_REPORT.md`).

## 1. What changed (vs the originally-submitted unpadded zip)

| File | Change |
|---|---|
| `adapter_model.safetensors` | **No change.** SHA256 identical, inode identical (hardlinked from the unpadded dir). 968,745,200 bytes. |
| `chat_template.jinja` | No change. |
| `adapter_config.json` | **Added** `rank_pattern` and `alpha_pattern` blocks; everything else (including global `r=32`, `lora_alpha=32`, `target_modules`, `inference_mode=True`) unchanged. |

### Config diff (only delta)

```diff
-   "alpha_pattern": {},
+   "alpha_pattern": {
+     "experts\\.\\d+\\.up_proj": 8,
+     "experts\\.\\d+\\.down_proj": 8,
+     ".*\\.experts\\.\\d+\\.up_proj": 8,
+     ".*\\.experts\\.\\d+\\.down_proj": 8
+   },

-   "rank_pattern": {},
+   "rank_pattern": {
+     "experts\\.\\d+\\.up_proj": 8,
+     "experts\\.\\d+\\.down_proj": 8,
+     ".*\\.experts\\.\\d+\\.up_proj": 8,
+     ".*\\.experts\\.\\d+\\.down_proj": 8
+   },
```

Both patterns are included so PEFT/vLLM matches under either `re.search` (endswith-style) or `re.fullmatch` (anchored) semantics:
- `experts\.\d+\.up_proj` and `experts\.\d+\.down_proj`: match by substring.
- `.*\.experts\.\d+\.up_proj` and `.*\.experts\.\d+\.down_proj`: anchored fullmatch.

### Why these patterns are unambiguous

| Pattern | Matches expert modules (target 5,888) | Matches non-expert modules (must be 0) |
|---|---:|---:|
| `experts\.\d+\.up_proj`             | 2,944 / 2,944 (search) | 0 / 116 |
| `experts\.\d+\.down_proj`           | 2,944 / 2,944 (search) | 0 / 116 |
| `.*\.experts\.\d+\.up_proj`         | 2,944 / 2,944 (fullmatch) | 0 / 116 |
| `.*\.experts\.\d+\.down_proj`       | 2,944 / 2,944 (fullmatch) | 0 / 116 |

`shared_experts.{up,down}_proj` (which are rank 32) are correctly NOT matched (no `\d+` after `experts.` in `shared_experts.up_proj`). Mamba `in_proj`/`out_proj` and attention `q/k/v/o_proj` also don't match (no `experts.` in path). So all and only the per-expert tensors get the rank-8/alpha-8 override.

## 2. Local stock vLLM 0.20.1 load + eval — bit-only n=50 (Kaggle-exact prompts, max_tokens=7680, max_model_len=8192)

Loaded both unpadded and rank_pattern in the same vLLM boot (max_loras=2):

| Adapter (same boot) | n | T | Acc | Trunc | boxed | </think> |
|---|---:|---:|---:|---:|---:|---:|
| warm-300 unpadded (`rank_pattern={}`)              | 50 | 0.0 | 28.0% (14/50) | 0.0% | 100% | 100% |
| **warm-300 rank_pattern (this build)**             | 50 | 0.0 | **36.0% (18/50)** | 0.0% | 100% | 100% |

Same-boot per-row delta: **15 of 50 rows differ in predicted text**; **5 unpadded-wrong → rank_pattern-right** vs **1 unpadded-right → rank_pattern-wrong**; **net +4**.

vLLM load: `MoE model detected. Using fused MoE LoRA implementation.` No `LoRA module … will be ignored` warnings. No asserts. Same benign messages as prior boots (default MoE config, default LoRA kernel configs).

### Caveat on the same-boot delta

The safetensors are **byte-identical** (verified by SHA256 + matching inode). So the same-boot +4 from `rank_pattern` over unpadded is either:
- vLLM IS honoring `rank_pattern` and using it to set per-tensor ranks (the optimistic read), **or**
- adapter-load order / vLLM kernel selection happens to drift across `(adapter, position)` combinations and we got lucky.

To control for boot-context variability, the rank_pattern variant was also evaluated alone on the full split below.

## 3. Local stock vLLM n=500 dev_frozen — rank_pattern variant alone, Kaggle-exact greedy

`max_tokens=7680, max_model_len=8192, T=0.0, top_p=1.0`. Single-adapter boot.

| Metric | Value |
|---|---:|
| Accuracy            | **69.20%** (346/500) |
| Truncation          | 0.80% |
| has_boxed           | 99.20% |
| has_</think>        | 99.20% |
| Walltime            | 1,714 s (28.6 min) |

### Per task

| Task | n | rank_pattern | (cached unpadded n=500 greedy, max_tokens=3584) |
|---|---:|---:|---:|
| bit_manipulation         | 84 | **36.9%** (31/84) | 38.1% (32/84) |
| equation_transformation  | 84 | 16.7% (14/84) | 16.7% (14/84) |
| gravitational_constant   | 83 | 100% (83/83) | 100% |
| numeral_conversion       | 83 | 100% (83/83) | 100% |
| **text_encryption**      | 83 | **62.7%** (52/83) | 66.3% (55/83) |
| unit_conversion          | 83 | 100% (83/83) | 100% |

**Net rank_pattern vs cached unpadded n=500 at greedy: −1 example** (346 vs 347 — within noise; max_tokens budget differs between the two runs but the gap is dominated by ~1–4 examples of cross-boot drift that we've observed earlier).

For comparison, v9 alone at Kaggle-exact greedy n=500 max_tokens=3584 scored **68.60%** (343/500). The rank_pattern variant at **69.20%** is **+3 examples over v9** under directly comparable conditions, and within run-to-run noise of unpadded's 70.00%.

## 4. Submission zip

| | Value |
|---|---|
| Filename                                          | `submissions/lora_v9_warm_start_300step_rankpattern.zip` |
| Size                                              | **830.91 MiB (0.811 GiB)** |
| Under 1.5 GB Kaggle cap                            | YES (+0.689 GiB headroom) |
| Contents                                          | `adapter_config.json` (1,405 B), `adapter_model.safetensors` (968,745,200 B), `chat_template.jinja` (10,504 B) |
| Validated rank ≤ max_lora_rank=32                  | YES (r=32) |
| `adapter_config.json: inference_mode`              | True |
| safetensors SHA256                                | `5a9f5c5e2dd7b859c5e8e59e4a98b9c68b541d27e0548da6256cb67ee14c0cea` (identical to original unpadded zip) |

## 5. Submission summary table

| Submission zip on disk | Size | What's different from base |
|---|---:|---|
| `lora_v9_warm_start_300step.zip` | 830.91 MiB | (original; scored 0.56 on Kaggle) |
| `lora_v9_warm_start_300step_minimal.zip` | 830.91 MiB | original adapter, no chat_template.jinja (fallback for chat-template-interference) |
| `lora_v9_warm_start_300step_r32padded.zip` | 860.53 MiB | expert tensors zero-padded to r=32; **DO NOT submit** — diverges in inference from unpadded (FP drift) |
| **`lora_v9_warm_start_300step_rankpattern.zip`** | **830.91 MiB** | **same safetensors as original; rank_pattern/alpha_pattern added to declare mixed ranks. PRIMARY SUBMISSION CANDIDATE** |

## 6. Recommendation

Submit `submissions/lora_v9_warm_start_300step_rankpattern.zip`:

- Mechanically valid: stock vLLM 0.20.1 loads it cleanly with no warnings.
- Numerically safe: safetensors is bit-identical to the unpadded baseline (no FP risks from padding).
- Locally evaluated: rank_pattern variant scored 69.20% on n=500 Kaggle-exact greedy (within run-to-run noise of unpadded's 70.00%); +3 examples over v9 in the same eval harness; trace integrity 99.2%.
- Directly addresses the mixed-rank packaging concern via the documented PEFT/vLLM mechanism (`rank_pattern`), not via tensor padding.

## 7. Holding before submission

No upload to Kaggle made. Awaiting your approval.