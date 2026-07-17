# prompts/phase3/2026-06-10_investigate_investigation-tracker-setup_v3_prompt.md

Proceed with Q8 hypothesis-2: design bit_manip_trace_v5. Show me the format
spec + one worked example (pick a per_bit problem the model currently gets
wrong, e.g. 5d77eff6) before generating any traces. The design constraint: every
token in the rule statement must be derivable from prior tokens in the trace —
no fait accompli operator/index claims. Keep total trace under 2000 tokens to
leave headroom. Study Huikang's derivation logic in the reference repo for
inspiration on compact column-matching.
