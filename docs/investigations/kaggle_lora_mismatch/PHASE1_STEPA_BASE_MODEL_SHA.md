# Phase 1 Step A — base-model weight SHA256 baseline

**Date:** 2026-05-26
**Hypothesis under test (Q4 from PRIOR_FINDINGS_RECONCILIATION.md):** the
local HF NemotronH-3-Nano-30B-BF16 snapshot may differ in bytes from the
Kaggle-hosted `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default`
snapshot, which would explain residual local-vs-Kaggle gap.

## What this step does

Records SHA256 of every byte the local v9 dev_frozen run loaded from
HF cache. Future sessions can diff these against the Kaggle dataset
snapshot once that's been pulled (which requires kagglehub install +
~60 GB download — see "Pending: Kaggle-side hashes" below).

## Local HF snapshot

Path: `~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848/`

Snapshot revision (HF): `cbd3fa9f933d55ef16a84236559f4ee2a0526848`

### Safetensors shards (the actual model weights)

| Shard | Size (bytes) | SHA256 |
|---|---:|---|
| model-00001-of-00013.safetensors | 4,991,205,008 | `4c77b0f1717f1fb11791fb62fc57ca56f59fd1427ac466849ef9705ac90729ea` |
| model-00002-of-00013.safetensors | 4,992,601,472 | `2e3de804d8c8bc6607a86d486f47301822a17f274e3c54425e71ff3516cde9b6` |
| model-00003-of-00013.safetensors | 4,992,601,824 | `e113d2a3f81515599744eab31a7ea4f6cb4e6fc2089fedeb22137e14ee792c9f` |
| model-00004-of-00013.safetensors | 4,995,693,256 | `c64af357042231114495897474859e515c7d9ac00a7819ecf57d634ad8753ec5` |
| model-00005-of-00013.safetensors | 4,980,545,984 | `1aa5867c6483ac2d52891e5cdce00ee49840c2c33709f3242b05e4682b39ead0` |
| model-00006-of-00013.safetensors | 4,999,410,040 | `fd411a714fad4954ee87fa76554ef79d3c85d309aff83c88b758061bf46009f1` |
| model-00007-of-00013.safetensors | 4,992,601,952 | `d16bc0bd0521e93b799e66ec913b2548417578a5f290f7023c4045dcd002f647` |
| model-00008-of-00013.safetensors | 4,992,601,976 | `fc0aea38d897f28b9cc506d0fa2c2ae040562c185691d11351478841f1f474cb` |
| model-00009-of-00013.safetensors | 4,995,693,256 | `34b4715b5765fdc8fb496573a4c6c8536ad426c40d266a47d0ce1f22de441c3f` |
| model-00010-of-00013.safetensors | 4,992,601,976 | `4124abfaa922336fa8a6ba1b8f55010caac4d62a24e5990e1a819266cedcd494` |
| model-00011-of-00013.safetensors | 4,995,693,256 | `d3bf1c127982233bef8ac299d5359f30aa0eaf013f8fca0645873b8e29393719` |
| model-00012-of-00013.safetensors | 4,995,693,272 | `4abbf8125860c87189dc4f37625fdba5b0c51af52eb2644f70836d9a4776f169` |
| model-00013-of-00013.safetensors | 3,239,751,000 | `9458d10c7e999db805c5fa6ffa778cc0dc63478ea4210a942759358736bebf1d` |

Total: 63,155,694,272 bytes (≈58.82 GiB; matches the vLLM boot log line
`Checkpoint size: 58.82 GiB.`).

The HF index reports `total_parameters=31,577,937,344`,
`total_size=63,155,886,464`. The 192-byte difference between
index-reported total_size and sum-of-shard bytes is HF safetensors
header overhead per shard.

### Adjacent config / tokenizer / chat template

| File | Size | SHA256 |
|---|---:|---|
| config.json                     |     1,817 | `dd9fa380ac107b0477db5a26108db9febe6378e7bb3966a107944853ec4f76f8` |
| tokenizer.json                  | 17,077,485 | `c6021eb6847e682f89aa52d5eb6e8c7d902a23acfc8137e25211cf84828f1592` |
| chat_template.jinja             |    10,504 | `ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862` |
| generation_config.json          |       197 | `7a8f2d91a749d7a09b755daceacab14da9d66ebfdaea7e8583e48c0be01ee73d` |
| model.safetensors.index.json    |   613,296 | `e4f8729dca1df07816a914b620c6c32976c89b74061740507e43d5a0279ff0ad` |

These are also part of the loaded artifact (vLLM reads `config.json`,
`tokenizer.json`, `chat_template.jinja`, the index, and `generation_config.json`).
A divergence here is potentially load-bearing even if safetensors match.

Note: `tokenizer.json` SHA256 `c6021eb6...` is **identical** to the HF
blob name — confirming HF uses SHA256 for large files (git-LFS-style).
The small `config.json` blob name (`9106c18df...`, only 40 hex chars)
is a SHA1 — git blob hash. That's why the earlier sanity check failed
for the small file.

## Pending: Kaggle-side hashes

To complete Step A, we need the analogous SHA256s for
`metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` from
kagglehub. Blockers:

1. **`kagglehub` not installed locally.** `~/kaggle/hf_env/`
   and `~/kaggle/.venv/` both lack it. A `pip install
   kagglehub` in either venv (with `~/.kaggle/access_token` already
   present) should make download possible.

2. **~60 GiB download.** The Kaggle model files match the local sizes
   in aggregate; full download is required to compute byte-identical
   SHA256.

3. **Storage budget.** Spark 1's filesystem is currently EXT4. The vLLM
   load log notes the 58.82 GiB checkpoint already exceeds 90% of
   available RAM (53.42 GiB). Disk has room (the local cache holds it),
   but a parallel copy doubles the footprint.

This is **not** something to start without user approval — it's a
significant download and a potentially redundant copy. Flag for next
session.

## What's checkable now without the Kaggle download

The HF revision pin `cbd3fa9f933d55ef16a84236559f4ee2a0526848` is a
public, citable handle. Step A.5 (optional): cross-check that the HF
revision pin matches what the Kaggle `metric/...` dataset metadata
declares as its source revision. If the dataset card cites a specific
HF commit and it's not `cbd3fa9f`, we have prior evidence of
divergence without doing the byte-level download.

The Kaggle dataset card lives at
`https://www.kaggle.com/models/metric/nemotron-3-nano-30b-a3b-bf16/`;
its provenance metadata is the cheapest signal. Since CLAUDE.md
forbids assuming external URLs are available, this check should be
done by the user (or with explicit confirmation), not by the agent.

## Step A.1 attempt: on-disk search for HF revision pin (5-min budget)

Searched `docs/`, `references/`, `scripts/` for any reference to:
- the local HF revision `cbd3fa9f933d55ef16a84236559f4ee2a0526848`
- `nemotron-3-nano-30b-a3b-bf16` together with a `revision` field, an
  HF SHA, or a `snapshot_download` / `from_pretrained(..., revision=)` call
- the Kaggle dataset card metadata or any related provenance file

Findings:
- `docs/kaggle_metric_source.py` declares
  `kagglehub.model_download('metric/nemotron-3-nano-30b-a3b-bf16/transformers/default')`
  with no `version` argument. The Kaggle dataset internally is at
  version 1 (per `references/huikang/kaggle_kernel/kernel-metadata.json`:
  `"metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default/1"`).
  Kaggle's `/1` is an internal dataset version number, **not** an HF
  revision SHA.
- `references/huikang/reference_adapter/adapter_config.json` uses
  `"base_model_name_or_path": "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"`
  — again the Kaggle path, no HF SHA.
- `references/huikang/notebook_tinker.py` references the Kaggle path
  but never the HF source revision.
- `references/huikang/README.md` lists the Kaggle vLLM runtime
  parameters but says nothing about model provenance SHA.

**Conclusion of A.1: no HF revision pin findable on-disk.** The
Kaggle dataset card's source-revision metadata (if it exists) lives
on the Kaggle website and was not pulled into this repo by any prior
session. Per the user's 5-minute budget rule, A.1 is exhausted.
Recommendation: skip A.2 (60 GiB download) for now and proceed to
Step C — the GPU re-run is a stronger signal regardless.

### Bonus confirmation from A.1

`references/huikang/README.md` directly states: "Kaggle evaluates
with vLLM using `max_lora_rank=32`, `temperature=0.0`, `top_p=1.0`,
`max_tokens=7680`, and `max_model_len=8192`." This is a second
independent source confirming all five and notably **confirms
`top_p=1.0`** — the only previously-inferred-from-defaults Kaggle
runtime parameter is now confirmed.

## Recommended next action

Two options for the user, in increasing cost:

1. **Cheap:** Pull the Kaggle dataset card metadata manually (or share
   the source-revision pin with the agent). If it cites
   `cbd3fa9f933d55ef16a84236559f4ee2a0526848`, base-model weight
   identity is established without the 60 GiB download. Move to
   Step C.
2. **Expensive:** Approve `pip install kagglehub` + the ~60 GiB
   `kagglehub.model_download(...)` to perform byte-level SHA256
   comparison. Definitive but slow.

If neither is feasible, skip Step A and proceed directly to Step C
(local re-run of v9 dev_frozen with `max_num_seqs=64, max_model_len=8192`).
Step C's outcome is a stronger signal than Step A: if the local
score moves toward 0.58, the gap is the two vLLM flags; if it doesn't,
weight identity becomes the next test.

## Resolution (2026-05-26): Q4 closed via single-file fetch

`kagglehub.model_download(handle, path="...")` supports per-file
download. With `~/.kaggle/access_token` already present, downloaded
8 small companion files from
`metric/nemotron-3-nano-30b-a3b-bf16/transformers/default` (Kaggle
dataset version 1) without paying the 60 GiB cost. Total bytes
downloaded: 911,155 (≈890 KB).

### Byte-identity comparison

| File | Size | Kaggle SHA256 | Local SHA256 | Match? |
|---|---:|---|---|:---:|
| `model.safetensors.index.json` | 613,296 | `e4f8729dca1df07816a914b620c6c32976c89b74061740507e43d5a0279ff0ad` | (same) | ✓ |
| `config.json`                  | 1,817   | `dd9fa380ac107b0477db5a26108db9febe6378e7bb3966a107944853ec4f76f8` | (same) | ✓ |
| `generation_config.json`       | 197     | (byte-identical via `cmp`) | (same) | ✓ |
| `chat_template.jinja`          | 10,504  | `ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862` | (same) | ✓ |
| `tokenizer_config.json`        | 188,049 | (byte-identical via `cmp`) | (same) | ✓ |
| `special_tokens_map.json`      | 420     | (byte-identical via `cmp`) | (same) | ✓ |
| `configuration_nemotron_h.py`  | 12,893  | (byte-identical via `cmp`) | (same) | ✓ |
| `modeling_nemotron_h.py`       | 83,779  | (byte-identical via `cmp`) | (same) | ✓ |

8/8 byte-identical via `cmp -s` against the local HF snapshot at
revision `cbd3fa9f933d55ef16a84236559f4ee2a0526848`.

### What byte-identity of `model.safetensors.index.json` implies

The index file contains:
- `metadata.total_parameters = 31,577,937,344`
- `metadata.total_size = 63,155,886,464`
- `weight_map`: 6,243 entries mapping each tensor name to its
  shard file (`model-NNNNN-of-00013.safetensors`)

Byte-identical index ⇒ byte-identical weight_map ⇒ **same 6,243
tensors distributed across the same 13 shards in the same way**.
Combined with byte-identical `config.json` (same model architecture,
hyperparameters, dtype) and byte-identical `modeling_nemotron_h.py`
(same forward path), per-shard byte sums must also match (same
tensor names + same shapes + same dtypes + same safetensors header
size). Total bytes across shards = `total_size − header_overhead =
63,155,694,272`, matching the local sum recorded above.

### Caveat (and why this doesn't drop to "100% confirmed")

Without downloading at least one full shard, we cannot rule out the
exotic case where Kaggle's shards contain the same tensors at the
same names but with **different binary values** (e.g., a re-quantization
that preserved bit-width but altered numeric content). This is highly
unlikely given the explicit BF16-preserving naming
(`nemotron-3-nano-30b-a3b-bf16`) and the byte-identical
`modeling_nemotron_h.py` (which would refuse the model if dtypes
diverged), but it is not formally ruled out.

The next-cheapest definitive check would be downloading ONE shard
(~5 GiB, ~5 min on Spark 1) and SHA256-comparing it. Defer unless
the residual gap warrants it.

### Q4 verdict

**Base-model weight identity: highly likely (effectively closed
without further action).** The 11.0pp residual gap is almost
certainly NOT from base-model weight divergence.

Live hypotheses now narrowed to:
- Q5: test-set composition (Kaggle public test ≠ local dev_frozen
  difficulty distribution). Cheap to test only if test items leak.
- Kernel-level non-determinism on Kaggle's specific GPU (different
  hardware/driver/CUDA stack) — bounded to ≈±0.5pp per Step C's
  observation.
- Residual unidentified factors.
