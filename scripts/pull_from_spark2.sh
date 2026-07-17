#!/bin/bash

# --- CONFIGURATION ---
REMOTE_USER="nvidia"
REMOTE_HOST="spark-5c81-openclaw" # Tailscale MagicDNS name or IP
REMOTE_REPO_ROOT="/home/nvidia/kaggle/nemotron-reasoning"
IDENTITY_KEY="$HOME/.ssh/spark2-setup"

# Dynamically determine local repository root absolute path
LOCAL_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# ---------------------

echo "🔄 Pulling all session changes from Spark 2..."

# 1. Sync the investigations docs folder (grabs any new .md reports)
echo "📁 Syncing docs/investigations/..."
rsync -avzP -e "ssh -i $IDENTITY_KEY" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_REPO_ROOT}/docs/investigations" \
  "$LOCAL_REPO_ROOT/docs"

# 2. Sync the scripts folder (grabs any newly generated python or bash tools)
echo "📁 Syncing scripts/..."
rsync -avzP -e "ssh -i $IDENTITY_KEY" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_REPO_ROOT}/scripts" \
  "$LOCAL_REPO_ROOT"

echo "📁 Syncing prompts/..."
rsync -avzP -e "ssh -i $IDENTITY_KEY" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_REPO_ROOT}/prompts" \
  "$LOCAL_REPO_ROOT"

echo "✅ General sync complete! Spark 1 is fully updated with all artifacts."
