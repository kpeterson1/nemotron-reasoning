# Evaluation splits — regenerate locally (not distributed)

The split JSONLs that used to live here are verbatim rows from the
competition's `train.csv` (plus a derived `task_type` label). Redistribution
of competition data isn't ours to grant, so the files are not in this
repository. They are exactly reproducible:

1. Download `train.csv` from the competition page
   (`kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge`) to
   `datasets/raw/train.csv`.
2. Run:

   ```bash
   python -m src.data.split --csv datasets/raw/train.csv
   ```

   Defaults (`--output-dir datasets/splits --dev-size 500 --seed 42`,
   stratified across the 6 task types) reproduce byte-for-byte:
   - `dev_frozen.jsonl` (n=500) — the canonical local eval split
   - `train_remaining.jsonl` (n=9,000)
   - `hard200.jsonl` (n=200) — longest-prompt subset of dev
   - `speed_bench.jsonl` (n=50) — shortest-prompt subset of dev

   (`dev_frozen_shuffled.jsonl`, referenced by some eval scripts, was an
   ad-hoc row-order permutation of `dev_frozen.jsonl` used for
   order-sensitivity checks — any shuffle of dev_frozen serves the purpose.)

Once created, treat the splits as frozen: they are version-pinned by the
seed, not by version control, and must not be modified after creation.
