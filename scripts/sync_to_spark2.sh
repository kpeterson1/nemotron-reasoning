#!/bin/bash

# --- CONFIGURATION ---
TARGET_USER="nvidia"
TARGET_HOST="spark-5c81-openclaw" 
TARGET_REPO_ROOT="/home/nvidia/kaggle/nemotron-reasoning/"
IDENTITY_KEY="$HOME/.ssh/spark2-setup"

# Dynamically find the absolute path of the repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use absolute paths so it never breaks based on where you run it
RUNS_DIR="$REPO_ROOT/runs/eval"
DATASETS_DIR="$REPO_ROOT/datasets/splits"
# ---------------------

echo "🚀 Launching direct Tailscale sync..."

# Ensure target directories exist on Spark 2 before running rsync
ssh -i "$IDENTITY_KEY" ${TARGET_USER}@${TARGET_HOST} "mkdir -p ${TARGET_REPO_ROOT}runs/ ${TARGET_REPO_ROOT}datasets/"

# 1. Sync all evaluation runs (including all V9 files)
echo "📁 Syncing evaluation runs folder..."
rsync -avzP -e "ssh -i $IDENTITY_KEY" "$RUNS_DIR" ${TARGET_USER}@${TARGET_HOST}:${TARGET_REPO_ROOT}runs/

# 2. Sync dataset splits (dev_frozen.jsonl)
echo "📄 Syncing dataset splits..."
rsync -avzP -e "ssh -i $IDENTITY_KEY" "$DATASETS_DIR" ${TARGET_USER}@${TARGET_HOST}:${TARGET_REPO_ROOT}datasets/

echo "✅ Transfer complete! Spark 2 now has the identical file structure."
