#!/bin/bash
# Start Qwen3-VL-30B server via ms-swift (vllm backend) for MIRAGE QA.
# Uses all 4 H100 NVL GPUs (TP=4) for maximum throughput.
# max_model_len=131072 covers long-context checkpoints (d80k ≈ 81k tokens).
#
# Run from repo root: bash scripts/start_servers.sh
#
# Note: 30B occupies all 4 GPUs. The 8B cannot run concurrently.
# Use serve_qwen8b.sh separately if needed for compaction/judge.
#
# Logs:
#   logs/vllm_30b.log

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_30B="$HF_CACHE/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"

export HF_HOME="$HF_CACHE"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONNOUSERSITE=1

echo "[start_servers] Starting 30B on GPUs 0-3 (TP=4), port 8000 ..."
swift deploy \
  --model "$MODEL_30B" \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port 8000 \
  --served_model_name qwen3-vl-30b-instruct \
  --vllm_max_model_len 131072 \
  --vllm_tensor_parallel_size 4 \
  --vllm_enforce_eager true \
  > "$LOG_DIR/vllm_30b.log" 2>&1 &
PID_30B=$!
echo "[start_servers] 30B PID: $PID_30B"

echo "[start_servers] Waiting for server to become ready ..."
for i in $(seq 1 72); do
  if curl -sf "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then
    echo "[start_servers] 30B server on :8000 is READY"
    break
  fi
  [ "$i" -eq 72 ] && echo "[start_servers] WARNING: 30B server on :8000 not ready after 6 min"
  sleep 5
done

echo "[start_servers] Done. Log: $LOG_DIR/vllm_30b.log"
