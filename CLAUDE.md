For Kaggle LoRA mismatch debugging, read these first:

1. docs/investigations/kaggle_lora_mismatch/PHASE2_PRE_SUBMIT_REPORT.md
2. docs/investigations/kaggle_lora_mismatch/HUIKANG_COMPARISON.md
3. docs/investigations/kaggle_lora_mismatch/BACKBONE_DIAGNOSTIC.md
4. references/README.md (huikang provenance — his notebook_tinker.py was
   removed 2026-07-17; recover it from the pinned public commit cited there)

Do not assume external URLs are available. Use local references.
Do not train or submit unless explicitly asked.
Do not mutate safetensors values for format diagnostics; key/config-only changes must be documented as single-variable tests.

# Parity claims
When evaluating parity to an external reference, list every parameter in
a table with a source-of-truth column. Defaults are not values.
Distinguish: (a) confirmed from primary source (Kaggle Overview tab,
deployed code), (b) inferred from naming or docstrings, (c) assumed.
Never collapse (b) or (c) into (a).

# When a hypothesis is contradicted by evidence
Stop. Surface the contradiction explicitly before proposing a new
hypothesis. Do not silently retcon. Past examples: extractor "mirrors
Kaggle" claim contradicted by diff; eval_kaggle_exact.py "14/16 parity"
claim contradicted by re-count; T=1.0 hypothesis contradicted by Kaggle
Overview tab.

# Confirmed Kaggle leaderboard parameters (Overview tab)
max_lora_rank=32, max_tokens=7680, top_p=1.0, temperature=0.0,
max_num_seqs=64, gpu_memory_utilization=0.85, max_model_len=8192,
trust_remote_code=True, enable_prefix_caching=True,
enable_chunked_prefill=True, dtype='auto'

# Reading prior investigation docs
Before scoping new work in docs/investigations/, list existing files in
that directory and read the TLDRs first. New scope must reconcile against
prior findings.

# Measurement noise
vLLM 0.20.1 greedy (`temperature=0.0`) is NOT strictly deterministic
across `(max_num_seqs, max_model_len)` permutations on NemotronH.
Observed 2026-05-26: 12 of 500 rows (≈2.4%) produced different text
on the same prompts/seeds/adapter under a config-only delta
(`max_num_seqs=32→64, max_model_len=4096→8192`). Cause is fp32
reduction-order changes in fused MoE / attention kernels as KV-cache
page layout and batch scheduling shift. Per-row flips are
single-bit-edit style — model lands on different tokens at near-tie
logit boundaries, then cascades.

**Expected Kaggle-submission noise floor: ±0.5pp.** Treat any
single-submission delta below 1pp as statistically meaningless. If a
finding hinges on <1pp, repeat with a different seed/config first,
or expand to a larger split.

# Skipped / deferred
Items intentionally not done this session that
future sessions should know about (gitignore-blocked actions, follow-up
files, hypotheses untested). Without this, follow-ups silently vanish.

# Session logging
At the end of every substantive session (after committing work, before
ending), append one entry to docs/SESSION_LOG.md with this format:

## YYYY-MM-DD HH:MM — <one-line summary>
**Branch:** <branch name>
**Claimed:** <hypotheses or assertions made this session>
**Verified:** <what was actually checked, and how>
**Assumed:** <what was taken on faith, and why>
**Next:** <recommended next action>
**Skipped / deferred:** <items intentionally not done this session that
future sessions should know about — gitignore-blocked actions, follow-up
files, hypotheses untested, secondary scripts that share a fixed trap,
etc. Without this, follow-ups silently vanish.>

Do this without being asked. Skip only if the session was purely exploratory
read-only with no commits.

# Prompt & output auto-save
If the user's prompt begins with a session tag in brackets, automatically
save session artifacts when the task completes.

Tag format: [{phase}/{category}/{topic}/{version}]
Example: [phase3/investigate/bit-manip-learning-gap/v01]

Categories: eval, train, run, investigate, test

At task completion, after writing the summary (after ---):
1. Get today's date as YYYY-MM-DD.
2. Save the user's original prompt (everything after the tag) to:
   prompts/{phase}/{date}_{category}_{topic}_{version}_prompt.md
3. Save the summary (everything after the final --- delimiter) to:
   logs/interactive/{date}_{category}_{topic}_{version}_cli.log
4. Create directories as needed. Preserve text verbatim.
5. Briefly confirm both saves at the end.

Example for tag [phase3/investigate/bit-manip-learning-gap/v01] on 2026-06-09:
  prompts/phase3/2026-06-09_investigate_bit-manip-learning-gap_v01_prompt.md
  logs/interactive/2026-06-09_investigate_bit-manip-learning-gap_v01_cli.log

If no session tag is present, do not auto-save. The user can invoke
/save-session {phase}/{category}/{topic}/{version} manually instead.
