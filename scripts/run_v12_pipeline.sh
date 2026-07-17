#!/usr/bin/env bash
# v12 full pipeline (Q8 single-variable test): stage1 -> stage2(warm-start
# expert, 300 steps) -> convert to vLLM (BUGGY/load-bearing converter from git
# HEAD, matching the 33.3% baseline) -> canonical dev_frozen eval.
#
# Run from repo root inside tmux. Hyperparams match v9 exactly; only the
# bit_manipulation traces (v11 dataset) differ.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/kaggle/hf_env/bin/activate

LOG=runs/train/v12_pipeline.log
SENTINEL=runs/train/v12_pipeline.DONE
rm -f "$SENTINEL"
exec > >(tee -a "$LOG") 2>&1
echo "================ v12 pipeline start $(date -u +%FT%TZ) ================"

echo "[1/4] stage-1 reasoning LoRA (lora_v12)"
if [ -f runs/train/lora_v12/adapter_model.safetensors ]; then
  echo "  stage-1 already saved — skipping."
else
  python -m src.training.train_lora --config configs/train/lora_v12.yaml
fi

echo "[2/4] stage-2 warm-start expert LoRA, 300 steps (lora_v12_ws300)"
# --smoke is required ALONGSIDE --max-steps so the adapter is actually saved
# (save guard is `if smoke or max_steps is None`). --smoke only adds a
# grad/loss audit; it does NOT reduce data or steps. Matches how the v9 ws300
# baseline was produced.
if [ -f runs/train/lora_v12_ws300/adapter_model.safetensors ]; then
  echo "  stage-2 already saved — skipping."
else
  python -m src.training.train_lora --config configs/train/lora_v12_ws300.yaml --max-steps 300 --smoke
fi

echo "[3/4] convert raw PEFT -> vLLM using git-HEAD (buggy/load-bearing) converter"
git show HEAD:src/training/convert_peft_to_vllm_moe.py > runs/train/_convert_buggy.py
python runs/train/_convert_buggy.py \
  --src runs/train/lora_v12_ws300 \
  --dst runs/train/lora_v12_ws300_vllm

echo "[4/4] canonical dev_frozen eval (temp=0, max_tokens=7680, max_model_len=8192, max_num_seqs=64, gpu=0.85)"
python -m src.evaluation.run_eval \
  --split dev_frozen --config configs/eval/default.yaml --prompt-family raw \
  --adapter-dir runs/train/lora_v12_ws300_vllm \
  --max-tokens 7680 --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --max-num-seqs 64 --temperature 0.0 --top-p 1.0

echo "================ v12 pipeline DONE $(date -u +%FT%TZ) ================"
echo OK > "$SENTINEL"
