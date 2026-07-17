#!/usr/bin/env bash
# +rank test (Kaggle-valid): reuse v13 stage-1 (runs/train/lora_v13), train
# stage-2 with MoE expert rank 32 -> convert (buggy/HEAD) -> canonical eval.
# Single variable vs v13: only the expert rank (8->32).
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/kaggle/hf_env/bin/activate

LOG=runs/train/v13_expr32_pipeline.log
SENTINEL=runs/train/v13_expr32_pipeline.DONE
rm -f "$SENTINEL"
exec > >(tee -a "$LOG") 2>&1
echo "================ v13_expr32 pipeline start $(date -u +%FT%TZ) ================"

if [ ! -f runs/train/lora_v13/adapter_model.safetensors ]; then
  echo "FATAL: stage-1 warm-start source runs/train/lora_v13 missing"; exit 1
fi
echo "[1/3] stage-2 expert LoRA rank 32, 300 steps (warm-start from lora_v13)"
if [ -f runs/train/lora_v13_expr32_ws300/adapter_model.safetensors ]; then
  echo "  already saved — skipping."
else
  python -m src.training.train_lora --config configs/train/lora_v13_expr32_ws300.yaml --max-steps 300 --smoke
fi

echo "[2/3] convert raw PEFT -> vLLM using git-HEAD (buggy/load-bearing) converter"
git show HEAD:src/training/convert_peft_to_vllm_moe.py > runs/train/_convert_buggy.py
python runs/train/_convert_buggy.py \
  --src runs/train/lora_v13_expr32_ws300 \
  --dst runs/train/lora_v13_expr32_ws300_vllm

echo "[3/3] canonical dev_frozen eval"
python -m src.evaluation.run_eval \
  --split dev_frozen --config configs/eval/default.yaml --prompt-family raw \
  --adapter-dir runs/train/lora_v13_expr32_ws300_vllm \
  --max-tokens 7680 --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --max-num-seqs 64 --temperature 0.0 --top-p 1.0

echo "================ v13_expr32 pipeline DONE $(date -u +%FT%TZ) ================"
echo OK > "$SENTINEL"
