 
● Phase 2 sanity check complete

  Task 1 — rankpattern vs original unpadded

  - safetensors SHA256: identical (5a9f5c5e…)
  - Only adapter_config.json differs; chat_template.jinja and adapter_model.safetensors byte-equal
  - Full diff: only rank_pattern: {} → 4 regex entries (expert r=8), same for alpha_pattern. No other config key
  changed.
  - Per-tensor: 11,776 expert tensors all rank 8 ✓ and 232 non-expert tensors all rank 32 ✓
  - Pattern coverage: 5,888 / 5,888 expert modules matched, 0 / 116 non-expert;
  shared_experts.{up,down,gate}_proj, Mamba in_proj/out_proj, attn q/k/v/o_proj all correctly unmatched

  Task 2 — rankpattern vs v9 known-working baseline (submission/submission.zip)

  Only two differences in adapter_config.json, both classified A (required):

  ┌───────┬───────────────┬───────────────────────────────────────────────────────────────────────────────────┐
  │ Class │      Key      │                                      Reason                                       │
  ├───────┼───────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ A     │ rank_pattern  │ Required — our adapter has mixed rank-8 expert + rank-32 non-expert; v9 has only  │
  │       │               │ rank-32 and doesn't need it.                                                      │
  ├───────┼───────────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ A     │ alpha_pattern │ Required — matches expert α=8 so scaling stays at 1.0 per-expert.                 │
  └───────┴───────────────┴───────────────────────────────────────────────────────────────────────────────────┘

  Everything else matches v9 exactly: same target_modules regex, same base_model_name_or_path, same r=32, same
  lora_alpha=32, same peft_version (0.19.1), same inference_mode=True, same dtype (fp32), same key prefix
  (base_model.model.model.layers.…), no lm_head LoRA, same chat_template.jinja. 0 backbone keys in either.

  Task 3 — Tinker-style conversion audit (against task description)

  ┌──────────
  ┌───────────────────────┬──────────────┬──────────────────────────────────────────────────────────────────┐
  │    Transformation     │ Needed for   │                               Why                                │
  │                       │ our adapter? │                                                                  │
  ├───────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────┤
  │ model.layers →        │              │ Our keys (and v9's) use .model.model.layers.; 0 .backbone. keys. │
  │ backbone.layers       │ NO           │  v9 works on Kaggle in this form.                                │
  │ rename                │              │                                                                  │
  ├───────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────┤
  │ experts.w1/w2         │              │ Native NemotronH uses up_proj/down_proj; our converter already   │
  │ unfusing              │ NO           │ split the 3-D ParamWrapper into per-expert experts.<E>.up_proj / │
  │                       │              │  down_proj keys (stock vLLM format).                             │
  ├───────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────┤
  │ gate_proj + x_proj →  │ NO           │ NemotronH Mamba exposes in_proj/out_proj natively; v9 was        │
  │ in_proj SVD           │              │ trained directly against these names.                            │
  ├───────────────────────┼──────────────┼──────────────────────────────────────────────────────────────────┤
  │ lm_head retention     │ N/A          │ Neither v9 nor our adapter LoRAs lm_head.                        │
  └───────────────────────┴──────────────┴──────────────────────────────────────────────────────────────────┘

  Our convert_peft_to_vllm_moe.py already emits Kaggle-format keys. No Tinker conversion to apply.

  Task 4 — known-good load check

  - Huikang's converted submission.zip is NOT accessible locally (only solver algorithms exist under
  docs/reference_solvers/tonghuikang/reasoners/). notebook_tinker.py is not on disk.
  - v9 (our own known-working baseline) loads cleanly in stock vLLM 0.20.1.
  - rankpattern zip loads cleanly in two separate stock-vLLM-0.20.1 boots: MoE model detected. Using fused MoE
  LoRA implementation., no will be ignored warnings, no rank-mismatch errors, no asserts.

  Task 5 — Recommendation

  SUBMIT rankpattern as-is

 - safetensors byte-identical to the original unpadded warm-300 (no new weight values, no FP-drift risk like
  _r32padded).
  - Sole config delta vs the v9 known-working baseline is rank_pattern + alpha_pattern — both required (our
  adapter is mixed-rank, v9 isn't) and structurally correct (5,888/5,888 expert match, 0/116 non-expert match).
  - No Tinker-style rewrites needed; key prefix / dtype / config fields all already match v9's working format.
  - Local stock vLLM loads cleanly without warnings; n=500 Kaggle-exact greedy ≈ 69.20% (+3 examples over v9,
  within 1 of unpadded local).

  Full write-up at submissions/PHASE2_PRE_SUBMIT_REPORT.md. No submission made. No safetensors changes; no
  further config rebuilds suggested.
