# Huikang Reference Adapter Metadata

This directory contains small, git-safe metadata extracted from Huikang’s reference submission-format LoRA adapter.

It does not contain the full adapter weights.

## Why this directory exists

This metadata is used as a practical reference for what a Kaggle/vLLM-compatible Nemotron LoRA adapter can look like.

It helps debug adapter conversion problems, especially when comparing:

- a raw Tinker adapter
- a converted adapter
- Huikang’s reference submission-format adapter

The goal is to catch structural mismatches before spending time on Kaggle submissions.

## Source

The source artifact was Huikang’s notebook input:

```text
/kaggle/input/notebooks/huikang/nvidia-nemotron-all-linear/submission.zip
