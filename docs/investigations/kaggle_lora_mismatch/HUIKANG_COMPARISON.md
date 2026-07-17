# Step-by-step comparison: our pipeline vs Huikang `notebook_tinker.py`

**Reference file**: tonghuikang's `notebook_tinker.py` at pinned source commit https://github.com/tonghuikang/nemotron/blob/82bd1880/notebook_tinker.py — see `references/README.md` (Cited sources) for provenance. The local vendored copy this comparison was made against was removed 2026-07-17 (no license located; all-rights-reserved default).

**Goal**: classify every transformation Huikang performs as **required**, **harmless**, **suspicious**, or **must change** for our adapter, given that our training-source format and our target-runtime model are different from Huikang's.

## Critical context: different training-source formats

Huikang's pipeline starts from a **Tinker-trained** adapter (Thinking Machines's Tinker fine-tuning framework). Tinker's serialization conventions differ from PEFT 0.19.1 (what we use) in three relevant ways:

1. Tinker uses a `.backbone.` model attribute internally; Huikang's `model.model` → `backbone` rename brings the adapter into Tinker's expected layout (which the Kaggle public reference `nvidia-nemotron-all-linear` also uses).
2. Tinker emits **packed** MoE experts as 3-D `experts.w1` and `experts.w2` tensors (one tensor per layer, indexed by expert id along dim 0); Huikang's unfusing splits these into per-expert keys.
3. Tinker exposes the Mamba projection as **two separate** modules `gate_proj` and `x_proj`; Huikang's SVD merge folds them back into the canonical `in_proj` single-module form the inference model expects.

By contrast, our pipeline starts from **PEFT 0.19.1** training against `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` directly:
- Uses the HF `NemotronHForCausalLM` with `.model.layers.…` (no `.backbone.`).
- Uses 3-D `nn.Parameter` `experts.up_proj` / `experts.down_proj` via `target_parameters` ParamWrapper (not packed `w1`/`w2`).
- LoRAs Mamba `in_proj` directly (it's a single fused module in HF NemotronH).

So **most of Huikang's transformations don't have an analogue in our pipeline because the source format is already different**. The right question to ask isn't "do we do each of Huikang's steps?" but "does the **output** Huikang produces match the **output** we produce?"

## Step-by-step classification

| # | Huikang transformation | Our adapter state | Need to change ours? | Class |
|---|---|---|---|---|
| **T1** | `base_model.model.model` → `base_model.model.backbone` key rename | 12,008 / 12,008 keys use `base_model.model.model.` prefix; 0 use `backbone.` | **UNCERTAIN — see §"Open question" below**. Local stock vLLM 0.20.1 loads our adapter cleanly and applies it (output diverges from base; n=500 greedy = 69.2%), so locally this is a non-issue. Kaggle behaviour unverified. | **C — suspicious / could affect vLLM** |
| **T2** | Unfuse packed `experts.w1`/`w2` into per-expert `experts.<i>.up_proj`/`down_proj`. Broadcast singleton dim. Skip empty `w3`. | 0 packed w1/w2/w3 keys; **11,776 already-per-expert keys** in final `experts.<i>.(up|down)_proj.lora_(A|B).weight` form. Our `src/training/convert_peft_to_vllm_moe.py` did this split offline (from PEFT's 3-D `ParamWrapper`). | NO — already done. | **A — likely required, already satisfied** |
| **T3** | Mamba `gate_proj` + `x_proj` → `in_proj` via SVD (rank-64 → rank-32). Skip `gate_proj`/`x_proj` keys from the direct path. | 0 Mamba-layer `.gate_proj` LoRA keys; 0 `.x_proj` LoRA keys; **46 Mamba-layer `.in_proj` LoRA keys** already trained directly. HF NemotronH exposes the fused `in_proj` natively (it concatenates `[z, x, B, C, dt]` projections internally; PEFT applies LoRA to the whole fused module). | NO — base model uses `in_proj` natively; nothing to merge. | **A — likely required, already satisfied** |
| **T4** | `lm_head` retention (LoRA on the LM head, kept in submission via the `"lm_head"` entry in `target_modules`) | **0 `lm_head` LoRA keys.** Our training intentionally did NOT LoRA `lm_head`. Our `target_modules` regex does not match `lm_head` (verified: `re.search(regex, "model.lm_head")` returns `None`). | NO — this is a training-time choice, not a packaging issue. v9 (our previously-built working adapter) also has no `lm_head` LoRA. Adding fake `lm_head` LoRA weights now would not be a "packaging" change. | **B — harmless metadata difference** |
| **T5** | `target_modules` set to a Python **list of suffix names**: `["k_proj","o_proj","in_proj","q_proj","up_proj","v_proj","down_proj","out_proj","lm_head"]` | Our `target_modules` is a **regex string**: `.*\.(in_proj|out_proj|up_proj|down_proj|gate_proj|q_proj|k_proj|v_proj|o_proj)$`. PEFT 0.19.1 accepts both forms via `LoraConfig`; vLLM's `PEFTHelper.from_dict` types it as `list[str] | str`. Both work. | NO — equivalent semantically; PEFT and vLLM accept both. (Note: ours also includes `gate_proj`, but no `gate_proj` modules are LoRA'd in the saved safetensors, so the extra regex alternative is inert.) | **B — harmless metadata difference** |
| **T6** | `rank_pattern` / `alpha_pattern` left as `{}` (Huikang's final adapter is uniform-rank after his merge+unfuse steps) | Our `rank_pattern` and `alpha_pattern` declare rank/alpha=8 for per-expert `up_proj`/`down_proj`; non-expert tensors are rank 32 governed by global `r=32`/`lora_alpha=32`. | NO — our adapter is *genuinely* mixed-rank (11,776 expert tensors at rank 8 + 232 non-expert at rank 32), so `rank_pattern` is required to declare that to vLLM. This is the whole point of the `_rankpattern.zip` rebuild. | **A — required for OUR adapter, doesn't apply to Huikang** |
| **T7** | Final submission zip: flat archive of `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja` (plus whatever PEFT/Tinker emitted in the working dir, with `reference/` removed). | Same: 3 files (no `runs/`, no checkpoints, no junk). | NO. | **B — harmless** |
| **T8** | Submission file is `submission.zip` in the working dir | Ours is also `submissions/lora_v9_warm_start_300step_rankpattern.zip` | NO — Kaggle just needs *a* `submission.zip` upload. | **B — harmless** |

### Tensor-level extras Huikang does that we don't need
- **Float32 cast before SVD** (`gate_A.float()`): only relevant inside the merge math, irrelevant to us.
- **`.expand(...).contiguous()` for singleton-dim broadcasts** on packed `w1`/`w2`: only relevant when source has packed expert tensors with a degenerate dim; we never have packed tensors.

### Tensor dtype
Both Huikang's submission and our submission save tensors as **fp32** (verified for ours: 12,008 tensors, 100% `torch.float32`). No dtype change is implied by Huikang's pipeline.

## Open question — T1: the `backbone` rename

This is the only Huikang transformation that has a real "could affect Kaggle" question, so it deserves a dedicated look.

### Why Huikang renames
Huikang's input (Tinker-trained) adapter has keys like `base_model.model.model.layers.X.mixer.in_proj.lora_A.weight`. He renames the **second** `.model.` segment to `.backbone.`, producing `base_model.model.backbone.layers.X.mixer.in_proj.lora_A.weight`. This matches the format of the **`nvidia-nemotron-all-linear` reference adapter** he loads from `/kaggle/input/notebooks/huikang/nvidia-nemotron-all-linear`, which is the Kaggle-distributed public reference.

The implication: when Huikang ran this notebook, the reference adapter on Kaggle expected `backbone` keys.

### What vLLM 0.20.1's loader actually does
From `vllm/lora/utils.py:155 parse_fine_tuned_lora_name`:

```python
if name.startswith("base_model.model."):
    name = name.replace("base_model.model.", "")    # strip first 2 dot-parts
    ...
    name = "base_model.model." + name                # recover prefix
start_index = 2 if name.startswith("base_model.model.") else 0
parts = name.split(".")
if parts[-1] == "weight" and (parts[-2] == "lora_A" or parts[-2] == "lora_B"):
    new_name = ".".join(parts[start_index:-2])       # skip base_model + model
    return new_name, parts[-2] == "lora_A"
```

So vLLM strips exactly `base_model.model.` (the first two dot-parts). What remains is the path used to find the live module.

- **Ours, after strip**: `model.layers.X.mixer.in_proj` — this is exactly `<NemotronHForCausalLM>.model.layers.X.mixer.in_proj`, which exists in the live HF model.
- **Huikang's, after strip**: `backbone.layers.X.mixer.in_proj` — would need `<…>.backbone.layers…`, which does NOT exist in the live HF `NemotronHForCausalLM` (the attribute is `.model`, not `.backbone`).

For Huikang's `backbone` keys to load at all on Kaggle, **one of these must be true** (we cannot pick between them from local evidence alone):

1. The Kaggle metric environment loads the model from a code path where the inner attribute is `.backbone` (e.g., a Tinker-compatible model class, or a transformers version that uses `.backbone`).
2. Or vLLM does additional fuzzy/suffix matching after `parse_fine_tuned_lora_name` returns the stripped name, so both `model.layers.…in_proj` and `backbone.layers.…in_proj` ultimately resolve to the right live module.

We don't have Huikang's actual converted `submission.zip` to load locally and probe (only the `notebook_tinker.py` source). Local experiments cannot resolve this.

### Why this probably is NOT our root cause anyway

- **Our local vLLM 0.20.1 loads our adapter cleanly and applies it.** No `LoRA module … will be ignored` warning. Output diverges from base on test prompts. n=500 Kaggle-exact greedy = **69.2%** (consistent with full LoRA application; if our keys were silently dropped, n=500 would collapse to base-model ~38%).
- **Our `v9` known-baseline adapter also uses `.model.model.` keys** (verified, 232/232). If `.model.model.` is incompatible with Kaggle's vLLM, v9 would also fail; the user has not reported that.
- The observed Kaggle gap (~0.56 vs local ~0.70) is the size we'd expect from **partial LoRA application** — non-expert applies but mixed-rank experts get mis-loaded — which is exactly the `rank_pattern={}` failure mode the `_rankpattern.zip` is designed to fix. If our keys were globally rejected on prefix, we'd expect ~base (lower).

So T1 stays at **C — suspicious** but **not the primary suspect**.

## Final recommendation

# **SUBMIT rankpattern as-is**

`submissions/lora_v9_warm_start_300step_rankpattern.zip`:

1. **Addresses the most-likely root cause**: declares the mixed-rank LoRA state (per-expert r=8 / non-expert r=32) via `rank_pattern` + `alpha_pattern`. This is exactly what was missing in the original 0.56 submission.

2. **Does not alter the safetensors** — bit-identical to the original unpadded warm-start 300 (SHA256 match), same as the user's "config-only rebuild" constraint requires.

3. **Implements every Huikang transformation that's relevant to our adapter**:
   - Per-expert key form (T2): ✓ already done by our converter.
   - Mamba `in_proj` (T3): ✓ trained directly; no merge needed.
   - target_modules: ✓ equivalent regex form; PEFT/vLLM accept both.
   - Junk-free flat zip (T7/T8): ✓.

4. **Does NOT bundle the backbone rename (T1) with this submission.** Reasons:
   - The user instruction is "config-only rebuild, do not alter adapter_model.safetensors". Renaming keys IS altering the safetensors (key names are part of the file structure).
   - Local evidence shows our `.model.model.` keys load and apply correctly in stock vLLM 0.20.1; v9 with the same prefix has been used as our working baseline.
   - We should change one thing at a time. If `_rankpattern.zip` still scores low on Kaggle, the backbone rename is the next thing to try.

5. **If `_rankpattern.zip` does not recover to ~70%, the follow-up is a backbone-renamed variant** — a separate experiment that does mutate the safetensors keys but is a strict superset of the rankpattern config. We have not built that variant yet because (a) it requires safetensors edits which the user instructed us not to make in this pass, and (b) without Huikang's reference submission locally to load-test, we can't fully validate it.

### Evidence summary

| Verification | Result |
|---|---|
| rankpattern safetensors == original unpadded | SHA256 identical |
| rankpattern adapter_config.json delta | **only** `rank_pattern` + `alpha_pattern` added |
| rank_pattern coverage | 5,888 / 5,888 expert modules, 0 / 116 non-expert (including 0 of `shared_experts.*`, Mamba, attention) |
| target_modules suffix matching against `lm_head` | False (we do not target `lm_head`) |
| Local vLLM 0.20.1 boot | `MoE model detected. Using fused MoE LoRA implementation.`; no `will be ignored` warnings, no asserts, no rank-mismatch |
| Local n=500 Kaggle-exact greedy (max_tokens=7680, max_model_len=8192) | **69.20% (346/500)** — within 1 example of unpadded local 70.0%, +3 over v9 (343/500) |
| Trace integrity | 99.2% boxed + 99.2% `</think>` |

No training. No submission. Awaiting your approval to upload `submissions/lora_v9_warm_start_300step_rankpattern.zip`.