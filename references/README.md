# References — provenance

Third-party reference material used during the competition. Nothing in this
directory is our work; see per-entry provenance and licensing notes below.
**Licensing review is a pre-publication gate** (see `docs/PUBLICATION_CHECKLIST.md`)
— do not make this repo public before resolving redistribution rights for
tracked third-party content.

## Cited sources

Two distinct sources — do not conflate them. tonghuikang won the **midpoint
Open Prize** and is a source the eventual winners built on; he is **not** the
overall winner. The first-place team's writeup is a separate source with its
own entry below.

### 1. tonghuikang — midpoint Open Prize winner

GitHub `@tonghuikang` — https://github.com/tonghuikang/nemotron (this repo
pins commit `82bd1880`; see the provenance sections below). Attribution
source for the min-logprob pre-submission gate and the character-by-character
cipher decode credited in `docs/writeup.md`, and the structural benchmark for
our adapter packaging (key prefix, shapes, dtypes, `target_modules` format).

### 2. First-place team — NullSira

Team **NullSira** — first-place solution writeup:
https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution
Cited **only** to corroborate the evaluation parameters (vLLM harness,
`\boxed{}` answer extraction, `temperature=0.0` greedy decoding). This entry
is deliberately separate from tonghuikang's: NullSira built on his Open
Prize work, but their writeup is not his and does not point at his repo.

## references/huikang/ and references/huikang_nemotron_commit_82bd1880/

tonghuikang's (midpoint Open Prize winner — see Cited sources above) public
Kaggle submission material for the NVIDIA Nemotron Model Reasoning Challenge
(`kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge`) was
previously vendored here and was **removed 2026-07-17**: no license file or
notice could be located in the vendored copies, and absent an explicit
license the default is all-rights-reserved. Removed files: his Tinker
submission notebook (`notebook_tinker.py` ×2,
`kaggle_kernel/tinker-submission-notebook.ipynb`,
`tinker-submission-notebook.md`), `kaggle_kernel/kernel-metadata.json`, and
`reference_adapter/adapter_config.json`. Recover them from his public repo
(pinned commit `82bd1880` — see Cited sources entry 1).

What remains here is **our own** provenance and extraction material: the
per-directory READMEs, `reference_adapter/METADATA_EXTRACTION.md`, and our
tooling's outputs describing his adapter's layout (`conversion_stdout.txt`,
`file_list.txt`, `manifest_summary.md`, `safetensors_keys_first_200.txt`,
`tensor_manifest.tsv`). His material was used as the structural benchmark for
adapter packaging (key prefix, shapes, dtypes, `target_modules` format) and
as the source of the min-logprob diagnostic technique and the
character-by-character cipher decode credited in the writeup.

## docs/reference_solvers/tonghuikang/ (removed 2026-07-06)

A local, read-only copy of tonghuikang's per-category reasoners
(`bit_manipulation.py`, `cipher.py`, `cryptarithm.py`, `equation_numeric.py`,
`gravity.py`, `numeral.py`, `unit_conversion.py`, plus `dictionary.txt` /
`wonderland.txt`). It was always gitignored (`.gitignore:42`, never tracked)
and the local copy was deleted 2026-07-06 during repo cleanup. Notable citation
preserved from it: his `cipher.py` emits a per-character `cipher→plain` lookup
step immediately before each decoded word (formerly `cipher.py:175-197`) — the
design our v12 char-by-char text_encryption trace followed. Recover the
originals from his public competition material if needed.
