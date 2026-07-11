#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mirage

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL_PATH="${MODEL_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/hub/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-configs/chat_templates/gemma3_permissive.jinja}"
PORT="${PORT:-8004}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
TP_SIZE="${TP_SIZE:-4}"

exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name gemma-3-27b-it \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --tensor-parallel-size "$TP_SIZE" \
  --disable-custom-all-reduce \
  --enforce-eager \
  --chat-template "$CHAT_TEMPLATE" \
  --chat-template-content-format openai \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
