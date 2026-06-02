#!/bin/bash
# Serve Qwen3-VL-8B-Instruct via ms-swift (vllm backend) on port 8001.
# Used as axis-3 judge / compaction model when 30B has all 4 GPUs.
# If running alongside 30B (TP=4 on GPUs 0-3), this cannot run concurrently.
# Run standalone (all GPUs) or defer to a separate machine.
#
# Usage: bash scripts/serve_qwen8b.sh

MODEL_PATH="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONNOUSERSITE=1

exec swift deploy \
  --model "$MODEL_PATH" \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port 8001 \
  --served_model_name qwen3-vl-8b-instruct \
  --vllm_max_model_len 131072 \
  --vllm_tensor_parallel_size 4 \
  --vllm_enforce_eager true
