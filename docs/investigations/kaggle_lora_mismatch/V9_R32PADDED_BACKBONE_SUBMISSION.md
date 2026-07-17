# v9 r32-padded backbone reference-matched submission

## Summary

Submitted a v9 r32-padded backbone adapter variant intended to match Huikang/reference-style structure more closely.

Submission note:
`v9 r32padded backbone adapter reference-matched keys shapes dtypes`

## Actual Kaggle result

Public leaderboard score: **0.58**

## Interpretation

This result remains in the same failure band as the previous adapter-format diagnostics:

- original warm-start 300: ~0.56
- rankpattern: ~0.57
- rankpattern + backbone prefix: ~0.56
- v9 r32-padded backbone reference-matched: **0.58**

The r32-padded backbone adapter was accepted by Kaggle and was intended to match the reference structure on key prefix, shapes, dtypes, expert layout, and Mamba `in_proj` layout. The score improved only slightly, so gross structural/key/config mismatch is probably not the dominant remaining issue.

This does **not** prove training quality is the issue. It means the tested packaging fixes did not recover the local-vs-Kaggle gap.

## Next debugging direction

Next debugging should focus on reproducing Kaggle inference locally from the exact submitted zip and inspecting raw outputs:

1. exact chat template
2. exact prompt wrapper
3. exact generation parameters
4. exact `\boxed{}` answer formatting
5. exact metric extraction behavior
6. raw output inspection by category
7. whether local vLLM/PEFT loading applies the adapter identically to Kaggle despite structural parity

Do not start a new training run based only on this score.
