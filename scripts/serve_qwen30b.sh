#!/bin/bash
# Serve Qwen3-VL-30B-A3B-Instruct via ms-swift (vllm backend) on port 8000.
# Uses all 4 H100 NVL GPUs (TP=4) for maximum throughput.
# max_model_len=131072 covers the 80k+ token contexts in d80k checkpoints.
#
# Usage: bash scripts/serve_qwen30b.sh

MODEL_PATH="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONNOUSERSITE=1

exec swift deploy \
  --model "$MODEL_PATH" \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port 8000 \
  --served_model_name qwen3-vl-30b-instruct \
  --vllm_max_model_len 131072 \
  --vllm_tensor_parallel_size 4 \
  --vllm_enforce_eager true
