#!/usr/bin/env bash
set -euo pipefail
python -m src.evaluation.run_eval \
  --split dev_frozen \
  --config configs/eval/default.yaml \
  --adapter-dir submissions/extracted/lora_v9_warm_start_300step \
  --temperature 0.0 --top-p 1.0 \
  --max-tokens 7680 --max-model-len 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85
