 Backbone diagnostic zip built and tested

  Filename

  submissions/lora_v9_warm_start_300step_rankpattern_backbone.zip (830.91 MiB, +0.689 GiB Kaggle headroom)

  Construction summary

  - adapter_config.json: byte-equal to rankpattern (rank_pattern + alpha_pattern preserved verbatim)
  - chat_template.jinja: byte-equal to rankpattern
  - adapter_model.safetensors: all 12,008 / 12,008 keys renamed base_model.model.model.layers.… →
  base_model.model.backbone.layers.…. No tensor values, shapes, dtypes, ranks, or counts changed.

  Key-rename audit

  - 12,008 renamed, 0 unchanged, 0 added, 0 dropped
  - 0 keys still start with base_model.model.model.
  - 12,008 keys start with base_model.model.backbone.
  - Per-tensor SHA256-by-value verification: 12,008 / 12,008 match under old→new mapping (0 shape, 0 dtype, 0 value-hash
  mismatches)
  - First 20 sample mappings printed in the report

  Backbone zip vs rankpattern zip

  ┌──────────────────────┬────────────────────────────────────────┬─────────────┬───────────────────────────────────────┐
  │                      │                Backbone                │ Rankpattern │                Equal?                 │
  ├──────────────────────┼────────────────────────────────────────┼─────────────┼───────────────────────────────────────┤
  │                      │ adapter_config.json,                   │             │                                       │
  │ File list            │ adapter_model.safetensors,             │ same        │ YES                                   │
  │                      │ chat_template.jinja                    │             │                                       │
  ├──────────────────────┼────────────────────────────────────────┼─────────────┼───────────────────────────────────────┤
  │ adapter_config.json  │ (verbatim)                             │ (verbatim)  │ YES                                   │
  │ bytes                │                                        │             │                                       │
  ├──────────────────────┼────────────────────────────────────────┼─────────────┼───────────────────────────────────────┤
  │ chat_template.jinja  │ (verbatim)                             │ (verbatim)  │ YES                                   │
  │ bytes                │                                        │             │                                       │
  ├──────────────────────┼────────────────────────────────────────┼─────────────┼───────────────────────────────────────┤
  │                      │                                        │ 968,745,200 │ +36,024 B (= 3 × 12,008, exactly the  │
  │ safetensors raw size │ 968,781,224 B                          │  B          │ cost of "backbone" vs "model" in key  │
  │                      │                                        │             │ strings)                              │
  ├──────────────────────┼────────────────────────────────────────┼─────────────┼───────────────────────────────────────┤
  │                      │                                        │             │ differs as expected; tensor values    │
  │ safetensors SHA256   │ 185c92f2…                              │ 5a9f5c5e…   │ are bit-identical per the             │
  │                      │                                        │             │ 12,008-tensor value-hash check        │
  └──────────────────────┴────────────────────────────────────────┴─────────────┴───────────────────────────────────────┘

  Local stock vLLM 0.20.1 load probe

  - MoE model detected. Using fused MoE LoRA implementation. ✓
  - No LoRA module ... will be ignored warnings ✓
  - No asserts, no rank mismatches ✓
  - Adapter output: 'The sum of 2 and 3 is 5.' vs base '5' → outputs differ ✓
  - Output identical to rankpattern non-renamed on same prompt — vLLM 0.20.1 accepts both model.model and backbone prefixes
  equivalently (most likely the FusedMoE LoRA path matches experts by suffix experts.<E>.up_proj/down_proj)

  Verdict

  Local evidence cannot distinguish whether Kaggle's vLLM has the same flexibility. The submission is the only way to test the
   hypothesis. The backbone zip is the cleanest A/B test against the 0.57 rankpattern submission because tensor values are
  bit-identical — any score change is unambiguously attributable to the prefix.

  Recommendation

  SUBMIT lora_v9_warm_start_300step_rankpattern_backbone.zip next (single-variable change vs the 0.57 submission, matches
  Huikang's reference format).

  If Kaggle still scores ~0.57, the next hypotheses to test are target_modules-format (list vs regex), lm_head LoRA inclusion,
   or a Kaggle-specific kernel quirk.

  Full write-up at submissions/BACKBONE_DIAGNOSTIC.md. No training. No submission. Awaiting your approval.
