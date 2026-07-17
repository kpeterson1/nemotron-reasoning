# Phase 2 pre-submit comparison

**Candidate**: `submissions/lora_v9_warm_start_300step_rankpattern.zip`

## Task 1 — rankpattern vs original unpadded warm-start 300

| Check | Result |
|---|---|
| safetensors SHA256 (both zips) | `5a9f5c5e2dd7b859c5e8e59e4a98b9c68b541d27e0548da6256cb67ee14c0cea` — **identical** |
| Same file list                  | `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja` |
| `chat_template.jinja` bytes      | **identical** |
| `adapter_model.safetensors` bytes | **identical** |
| Differing files                  | only `adapter_config.json` |

### Full adapter_config.json diff (rankpattern vs original)

```diff
  "rank_pattern":
-   {}
+   {
+     "experts\\.\\d+\\.up_proj": 8,
+     "experts\\.\\d+\\.down_proj": 8,
+     ".*\\.experts\\.\\d+\\.up_proj": 8,
+     ".*\\.experts\\.\\d+\\.down_proj": 8
+   }
  "alpha_pattern":
-   {}
+   {
+     "experts\\.\\d+\\.up_proj": 8,
+     "experts\\.\\d+\\.down_proj": 8,
+     ".*\\.experts\\.\\d+\\.up_proj": 8,
+     ".*\\.experts\\.\\d+\\.down_proj": 8
+   }
```

No other key in the config differs.

### Per-tensor rank audit (inside the rankpattern safetensors)

| | value |
|---|---:|
| Expert tensor rank histogram (lora_A first-dim / lora_B last-dim) | **{8: 11,776}** (uniform) |
| Non-expert tensor rank histogram                                   | **{32: 232}** (uniform) |
| All 11,776 expert tensors rank 8?                                   | YES |
| All 232 non-expert tensors rank 32?                                 | YES |

### rank_pattern / alpha_pattern coverage

- `rank_pattern` matches **5,888 / 5,888 expert modules** and **0 / 116 non-expert modules**.
- `alpha_pattern` matches **5,888 / 5,888 expert modules** and **0 / 116 non-expert modules**.
- Protected non-expert kinds in the safetensors that **must not match** the patterns: `shared_experts.up_proj`/`down_proj` (23 each), `mixer.in_proj`/`out_proj` (23 each), attention `q/k/v/o_proj` (6 each). **Wrongly matched: 0** of any kind.

## Task 2 — rankpattern vs v9 known-working Kaggle baseline

> *Huikang's converted submission zip and `notebook_tinker.py` are not present locally (only `docs/reference_solvers/tonghuikang/reasoners/` is here, which contains solver algorithms — no PEFT/LoRA conversion artifacts).* The most relevant **known-working** Kaggle-compatible reference we do have is `submission/submission.zip` — our own v9 adapter as previously submitted in working form. Comparison is against that.

### File list comparison

Both zips contain the same 3 files: `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja`. No extras in either direction.

### Classified differences

| Class | Key | Rankpattern | v9 | Note |
|---|---|---|---|---|
| **A — required** | `rank_pattern`  | 4 regex entries → expert r=8 | `{}` | Our adapter has 11,776 rank-8 expert tensors + 232 rank-32 non-expert; v9 has only rank-32 tensors and so does not need this. **Required for our adapter; this is the entire point of the rebuild.** |
| **A — required** | `alpha_pattern` | 4 regex entries → expert α=8 | `{}` | Same reasoning; ensures scaling = α/r = 8/8 = 1.0 per-expert, matching the train-time per-module scaling. **Required.** |

**No other config keys differ between rankpattern and v9.** That includes:

| Key | rankpattern | v9 | Class |
|---|---|---|---|
| `base_model_name_or_path` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | same | B (harmless metadata — matches the harness's base model) |
| `inference_mode`           | `True` | `True` | (good) |
| `r` (global)               | `32`  | `32`  | (good) |
| `lora_alpha`               | `32`  | `32`  | (good) |
| `target_modules` regex     | `.*\.(in_proj|out_proj|up_proj|down_proj|gate_proj|q_proj|k_proj|v_proj|o_proj)$` | same | (good — same regex governs which keys vLLM expects) |
| `peft_version`             | `0.19.1` | same | (good) |
| `target_parameters`        | `None`  | `None` | (good — both nulled; our adapter has already been split to per-expert keys by the converter) |
| `task_type`                | `CAUSAL_LM` | same | (good) |
| `use_dora`, `use_qalora`, etc. | matching nulls/false | matching | (good) |

### File-structure audit

| Audit | rankpattern | v9 (known-working) | Verdict |
|---|---|---|---|
| **safetensors dtype** | 12,008 tensors, all `torch.float32` | 232 tensors, all `torch.float32` | match (both fp32) |
| **key-prefix convention** | 12,008 keys under `base_model.model.model.layers.…` | 232 keys under `base_model.model.model.layers.…` | **match — no `backbone` rename** |
| **`.backbone.` in keys**  | 0 | 0 | match |
| **`.model.model.` in keys** | 12,008 | 232 | match |
| **lm_head LoRA present**  | No | No | match — both adapters intentionally do not LoRA `lm_head` |
| **chat_template.jinja**   | present (10,504 B) | present (10,504 B) | matching — both packaged with the base model's chat template (vLLM uses base tokenizer regardless; this is harmless) |

## Task 3 — audit against notebook_tinker.py-style conversion

We do not have `notebook_tinker.py` on disk; comparing against the **transformations the task description lists** for that conversion pipeline:

| Tinker transformation | Required for our adapter? | Why |
|---|---|---|
| **`model.layers` → `backbone.layers`** rename | **NO** | Our adapter already uses `base_model.model.model.layers.…` and so does v9 (which works on Kaggle). Zero keys in either zip contain `.backbone.`. The Tinker rename is for a different base-model code path; our `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` exposes `.model.layers`, not `.backbone.layers`. |
| **`experts.w1` / `experts.w2` unfusing** | **NO** | The native NemotronH MoE uses `up_proj` and `down_proj` 3-D Parameters (not `w1`/`w2`). Our `src/training/convert_peft_to_vllm_moe.py` already splits the trained 3-D ParamWrapper deltas into per-expert `experts.<E>.up_proj.lora_{A,B}` and `experts.<E>.down_proj.lora_{A,B}` keys — which is exactly what stock vLLM 0.20.1's `FusedMoE.make_expert_params_mapping` consumes (per `PATH2_REPORT.md` §3). This already matches the expected Kaggle-format keys. |
| **`gate_proj + x_proj → in_proj` SVD** | **NO** | NemotronH Mamba uses `in_proj` and `out_proj` natively; the model never exposed `gate_proj`/`x_proj` separately for SVD merging. v9 was trained directly against `in_proj`/`out_proj` and works on Kaggle. Our adapter does the same. |
| **`lm_head` retention** | **N/A** | Neither v9 nor our adapter trains `lm_head` LoRA. There is nothing to retain or discard. |

**Verdict: our `convert_peft_to_vllm_moe.py` already emits Kaggle-format keys.** No Tinker-style conversion needs to be applied on top.

## Task 4 — known-good adapter load check

| | Outcome |
|---|---|
| Huikang's actual converted submission.zip | **Not accessible locally.** Only solver algorithm files exist under `docs/reference_solvers/tonghuikang/reasoners/`; no PEFT/safetensors artifacts. |
| Public Huikang notebook (`notebook_tinker.py`) | **Not present on disk.** Inferring expected transformations from the task description (above). |
| **Our v9 (known to work on Kaggle previously)** | Loads cleanly in stock vLLM 0.20.1 (`MoE model detected. Using fused MoE LoRA implementation.`, no `will be ignored` warnings, no asserts). |
| **Our rankpattern zip** | Loads cleanly in stock vLLM 0.20.1 in two separate boots (single-adapter and 2-adapter). No `will be ignored` warnings, no rank-mismatch errors, no asserts. Same `MoE model detected. Using fused MoE LoRA implementation.` message as v9. |
| Local vLLM == Kaggle metric env? | vLLM version (0.20.1) is known; PEFT 0.19.1; tokenizer / chat template come from the base model. The metric prompt builder is byte-identical to local (`PROMPT_PARITY_REPORT.md`). Differences if any are below what we can observe without direct Kaggle access. |

## Task 5 — final recommendation

# SUBMIT rankpattern as-is

**Why this is safe:**

1. **safetensors are byte-identical to the original unpadded warm-start 300** (SHA256 match) — we are not introducing any new weight values, no quantization, no padding-induced FP drift (which the `_r32padded.zip` showed empirically). The model that scored ~70% locally is the model in this zip.

2. **The only config delta vs the v9 known-working Kaggle baseline is `rank_pattern` + `alpha_pattern`** — and this delta is both **required** (our adapter has mixed-rank tensors, v9 does not) and **structurally correct** (matches all 5,888 / 5,888 expert modules and 0 / 116 non-expert modules; protected `shared_experts` / Mamba / attention modules are not wrongly matched).

3. **No Tinker-style key/structure rewrites are needed.** Our key prefix (`base_model.model.model.layers.…`), tensor dtype (fp32), `target_modules` regex, `inference_mode`, `r`, `lora_alpha`, `peft_version`, `base_model_name_or_path`, `lm_head` policy, and `chat_template.jinja` are all identical to v9's working format.

4. **Local stock vLLM 0.20.1 loads it cleanly** with no `LoRA module … will be ignored` warnings and no rank-mismatch asserts; same engine message as v9 (`MoE model detected. Using fused MoE LoRA implementation.`).

5. **Local eval evidence**: rankpattern at Kaggle-exact greedy n=500 max_tokens=7680 scored **69.20% (346/500)** — within 1 example of the unpadded n=500 cached result (350/500), **+3 examples over v9 (343/500)** in the same harness; trace integrity 99.2% (boxed) / 99.2% (`</think>`).

**No safetensors changes**; no further config-only rebuilds suggested at this time.

**No submission made.** Awaiting your approval to upload `submissions/lora_v9_warm_start_300step_rankpattern.zip` to Kaggle.
