# prompts/phase3/2026-06-10_investigate_investigation-tracker-setup_v05_prompt.md

Train v11. Single variable: only bit_manip traces differ from v9. Same
hyperparams as v9 (LR 3e-4, rank 32, 1 epoch, batch 32). Run in tmux on Spark 1.
After training completes, run eval on dev_frozen with the canonical invocation
(temp=0.0, max_tokens=7680, max_model_len=8192, max_num_seqs=64, gpu_mem=0.85).
Then run the teacher-forcing logprob probe (bitmanip_logprob_probe.py) against
the new adapter to compare RULE_STATEMENT divergence rates vs v9. Report: (a)
overall dev_frozen accuracy, (b) bit_manip accuracy, (c) RULE_STATEMENT
divergence count for the 47 previously-failing problems. Stop after eval — do
not submit to Kaggle until we review.


