# Backbone-renamed diagnostic zip

**Context**: `lora_v9_warm_start_300step_rankpattern.zip` scored **0.57 on Kaggle** (essentially unchanged from the original 0.56). So `rank_pattern` alone did not close the local/Kaggle gap. This zip tests the **next hypothesis** from the Huikang comparison: that the Kaggle metric environment requires LoRA keys to use the `backbone` prefix rather than `model.model`.

**Zip**: `submissions/lora_v9_warm_start_300step_rankpattern_backbone.zip`
**Size**: 830.91 MiB (+0.689 GiB headroom under 1.5 GB Kaggle cap)

## 1. Construction

| Step | Detail |
|---|---|
| Source                                | `runs/train/lora_v9_warm_start_300step_vllm_rankpattern/` (the rankpattern variant that scored 0.57) |
| `adapter_config.json`                 | **Copied byte-for-byte** from rankpattern (`rank_pattern` and `alpha_pattern` retained verbatim) |
| `chat_template.jinja`                 | **Copied byte-for-byte** from rankpattern |
| `adapter_model.safetensors` keys      | All 12,008 keys rewritten: `base_model.model.model.layers.…` → `base_model.model.backbone.layers.…` |
| `adapter_model.safetensors` tensors    | Values, shapes, dtypes preserved verbatim |

## 2. Key rename audit

**Rename rule**: `key.startswith("base_model.model.model.") → "base_model.model.backbone." + key[len("base_model.model.model."):]`

| Check | Result |
|---|---:|
| Source tensors loaded                                        | 12,008 |
| Keys renamed                                                  | **12,008** |
| Keys unchanged                                                | 0 |
| Final keys starting with `base_model.model.model.` (must be 0) | **0** |
| Final keys starting with `base_model.model.backbone.` (target 12,008) | **12,008** |
| Final key-prefix histogram (depth 3)                          | `12,008  base_model.model.backbone` (single bucket) |

### Per-tensor identity verification (SHA256-by-value)

For every renamed key in the destination, the SHA256 of the raw tensor bytes (`tensor.detach().contiguous().cpu().numpy().tobytes()`) was compared to the SHA256 of the corresponding source tensor under the old key. Additionally shape and dtype were compared.

**Result: 12,008 / 12,008 tensors are value-identical to source under the old → new key mapping.** 0 shape mismatches, 0 dtype mismatches, 0 value hash mismatches.

This guarantees the new safetensors holds the **exact same weights as `lora_v9_warm_start_300step_rankpattern.zip`**, only the key strings differ.

### First 20 old → new key mappings

```
base_model.model.model.layers.0.mixer.in_proj.lora_A.weight
 -> base_model.model.backbone.layers.0.mixer.in_proj.lora_A.weight
base_model.model.model.layers.0.mixer.in_proj.lora_B.weight
 -> base_model.model.backbone.layers.0.mixer.in_proj.lora_B.weight
base_model.model.model.layers.0.mixer.out_proj.lora_A.weight
 -> base_model.model.backbone.layers.0.mixer.out_proj.lora_A.weight
base_model.model.model.layers.0.mixer.out_proj.lora_B.weight
 -> base_model.model.backbone.layers.0.mixer.out_proj.lora_B.weight
base_model.model.model.layers.1.mixer.experts.0.down_proj.lora_A.weight
 -> base_model.model.backbone.layers.1.mixer.experts.0.down_proj.lora_A.weight
base_model.model.model.layers.1.mixer.experts.0.down_proj.lora_B.weight
 -> base_model.model.backbone.layers.1.mixer.experts.0.down_proj.lora_B.weight
base_model.model.model.layers.1.mixer.experts.0.up_proj.lora_A.weight
 -> base_model.model.backbone.layers.1.mixer.experts.0.up_proj.lora_A.weight
base_model.model.model.layers.1.mixer.experts.0.up_proj.lora_B.weight
 -> base_model.model.backbone.layers.1.mixer.experts.0.up_proj.lora_B.weight
base_model.model.model.layers.1.mixer.experts.1.down_proj.lora_A.weight
 -> base_model.model.backbone.layers.1.mixer.experts.1.down_proj.lora_A.weight
base_model.model.model.layers.1.mixer.experts.1.down_proj.lora_B.weight
 -> base_model.model.backbone.layers.1.mixer.experts.1.down_proj.lora_B.weight
base_model.model.model.layers.1.mixer.experts.1.up_proj.lora_A.weight
 -> base_model.model.backbone.layers.1.mixer.experts.1.up_proj.lora_A.weight
base_model.model.model.layers.1.mixer.experts.1.up_proj.lora_B.weight
 -> base_model.model.backbone.layers.1.mixer.experts.1.up_proj.lora_B.weight
base_model.model.model.layers.1.mixer.experts.10.down_proj.lora_A.weight
 -> base_model.model.backbone.layers.1.mixer.experts.10.down_proj.lora_A.weight
base_model.model.model.layers.1.mixer.experts.10.down_proj.lora_B.weight
 -> base_model.model.backbone.layers.1.mixer.experts.10.down_proj.lora_B.weight
base_model.model.model.layers.1.mixer.experts.10.up_proj.lora_A.weight
 -> base_model.model.backbone.layers.1.mixer.experts.10.up_proj.lora_A.weight
base_model.model.model.layers.1.mixer.experts.10.up_proj.lora_B.weight
 -> base_model.model.backbone.layers.1.mixer.experts.10.up_proj.lora_B.weight
base_model.model.model.layers.1.mixer.experts.100.down_proj.lora_A.weight
 -> base_model.model.backbone.layers.1.mixer.experts.100.down_proj.lora_A.weight
base_model.model.model.layers.1.mixer.experts.100.down_proj.lora_B.weight
 -> base_model.model.backbone.layers.1.mixer.experts.100.down_proj.lora_B.weight
base_model.model.model.layers.1.mixer.experts.100.up_proj.lora_A.weight
 -> base_model.model.backbone.layers.1.mixer.experts.100.up_proj.lora_A.weight
base_model.model.model.layers.1.mixer.experts.100.up_proj.lora_B.weight
 -> base_model.model.backbone.layers.1.mixer.experts.100.up_proj.lora_B.weight
```

## 3. Backbone zip vs rankpattern zip comparison

| Check | Backbone zip | Rankpattern zip | Equal? |
|---|---|---|---|
| File list                  | `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja` | same | YES |
| Zip total size              | 871,271,665 B (830.91 MiB) | 871,271,034 B (830.91 MiB) | nearly (Δ = +631 B in zip; +36,024 B raw) |
| `adapter_config.json` bytes | (rankpattern verbatim)     | (rankpattern verbatim)    | **YES** — byte-equal |
| `chat_template.jinja` bytes | (verbatim)                 | (verbatim)                | **YES** — byte-equal |
| `adapter_model.safetensors` raw size | 968,781,224 B | 968,745,200 B | +36,024 B (= 3 extra chars per key × 12,008 keys; "backbone" is 3 chars longer than "model") |
| `adapter_model.safetensors` SHA256 | `185c92f2b2fce506c0781ee30fa0ab8b02aec21921b1720085be207e5044e94b` | `5a9f5c5e2dd7b859c5e8e59e4a98b9c68b541d27e0548da6256cb67ee14c0cea` | **differs as expected** (header key strings changed); tensor *values* are bit-identical per the 12,008-tensor SHA256-by-value check |

The +36,024-byte raw-safetensors delta is exactly accounted for by the longer key strings (`backbone` adds 3 ASCII chars per key × 12,008 keys = 36,024 bytes). Zip-DEFLATE recovers most of that overhead, leaving only +631 bytes in the compressed zip.

## 4. Local stock vLLM 0.20.1 load probe

`src/training/load_adapter_vllm_test.py --adapter-dir … --compare-base --max-tokens 32` on the backbone-renamed adapter:

| Probe | Outcome |
|---|---|
| Engine boot                                 | Succeeded (`vLLM v0.20.1`, `MoE model detected. Using fused MoE LoRA implementation.`) |
| `assert isinstance(lora_a, list)` failure   | **None** |
| `LoRA module ... will be ignored` warning  | **None** |
| Rank-mismatch warning                       | **None** |
| Any LoRA-related error                      | **None** (only benign: `Using default MoE config`, `Using default LoRA kernel configs`, `Enforce eager set`, `Add 1 padding layers`) |
| LoRA attach                                 | Succeeded (`[vllm-test] LLM ready. Attaching LoRA adapter from ...rankpattern_backbone`) |
| Output for `"What is 2+3?"` (32 tokens, T=0) | adapter: `'The sum of 2 and 3 is 5.'`   ·   base: `'5'`   ·   outputs differ: **True** |
| Comparison to rankpattern (non-renamed) on same prompt | adapter text **identical**: `'The sum of 2 and 3 is 5.'` |

**Reading**: stock vLLM 0.20.1 accepts the `base_model.model.backbone.layers.…` key form and applies the LoRA. The locally-observed behavior is **indistinguishable** from the `base_model.model.model.layers.…` form on this short test. This means vLLM's loader does additional resolution beyond a strict-prefix-match against `parse_fine_tuned_lora_name`'s output — most likely the FusedMoE LoRA attach path identifies expert modules by their MoE-mapping suffix (`experts.<E>.up_proj` / `down_proj`) rather than by full path equality.

This is **informational, not a fix**: the fact that both prefixes load locally means we cannot tell from local evidence whether Kaggle prefers one over the other. The submission is the only way to test the hypothesis. The submission is built and ready.

## 5. Submission queue (updated)

| Zip | Status | Kaggle score | Notes |
|---|---|---|---|
| `lora_v9_warm_start_300step.zip`              | submitted | **0.56** | original mixed-rank + `rank_pattern={}` |
| `lora_v9_warm_start_300step_minimal.zip`      | not submitted | — | unpadded + no chat_template; lower priority |
| `lora_v9_warm_start_300step_r32padded.zip`    | not submitted | — | DO NOT submit — FP drift from padding |
| `lora_v9_warm_start_300step_rankpattern.zip`  | submitted | **0.57** | unpadded + rank_pattern + `model.model` prefix |
| **`lora_v9_warm_start_300step_rankpattern_backbone.zip`** | **not submitted** | **—** | **same weights as rankpattern, keys renamed to `backbone` prefix** |

## 6. Recommended next submission

# **SUBMIT `lora_v9_warm_start_300step_rankpattern_backbone.zip`**

Rationale:

1. **Cleanest A/B test against the 0.57 score**: tensor values byte-identical to rankpattern (verified per-tensor SHA256-by-value, 12,008 / 12,008 match). Only the key prefix changed. Any Kaggle-score difference is unambiguously attributable to the key-prefix hypothesis.

2. **Matches the format used by Huikang's known-working conversion** (`notebook_tinker.py:92-94` renames `model.model` → `backbone`, and Huikang's reference adapter `nvidia-nemotron-all-linear` is in `backbone` form).

3. **Local stock vLLM 0.20.1 accepts both prefixes equivalently**, so this variant is unlikely to *break* anything that worked in the rankpattern submission. The downside risk is bounded.

4. **Holds rank_pattern / alpha_pattern in place** (unchanged from rankpattern), so the previously-built fix is preserved.

5. **Single hypothesis change** — does not bundle other speculative edits (e.g., target_modules-as-list, lm_head LoRA addition). If this also fails to recover, the next hypotheses are testable in isolation.

### What would constitute strong evidence

- **If Kaggle score rises to ~0.70 (matching local)** → the key-prefix hypothesis is confirmed; Kaggle's vLLM is strict about the live module path.
- **If Kaggle score is still ~0.57** → key prefix is not the root cause; remaining candidates are (a) Kaggle-specific kernel difference, (b) some other format aspect we have not identified, (c) target_modules-format sensitivity, (d) Tinker's `lm_head` LoRA being load-bearing.

## 7. Holding before submission

No training. No submission. Awaiting your approval to upload `submissions/lora_v9_warm_start_300step_rankpattern_backbone.zip`.
