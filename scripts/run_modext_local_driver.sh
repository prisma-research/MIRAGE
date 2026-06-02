#!/usr/bin/env bash
# Unattended driver: modality-extension sweep on local open VLMs (single H100, TP=1).
# Serves one model at a time (8B/30B/4B share the one GPU), runs the modext probe,
# frees the GPU, moves on. Outer loop = REP so all three models get run1 FIRST
# (guarantees "all 3 models have results" even if later reps don't finish).
# Resumable: fixed per-(model,rep) run-id => re-launch skips probes already done.
#
# Config strictly follows the prior Doubao modext run (probe_modext_doubao_v2.sbatch):
#   --bank combined_modality_bank.json --family modext_v2
#   --states d0,d50k,d80k,S2 --cf-conditions C0 --max-workers 8
# and the serve scripts: vllm_max_model_len=131072 (TP=1 here, single H100).

set -uo pipefail
PROJ="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJ"
export PATH="$HOME/.nvm/versions/node/v22.22.2/bin:$HOME/.local/bin:$PATH"
export OPENCLAW_TOKEN=none
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
unset PYTHONNOUSERSITE          # so ~/.local deps (safetensors, sentencepiece) are visible
BIN="$(dirname "$(command -v python)")"
HF_HUB="$HF_HOME/hub"
mkdir -p logs/serve
DLOG=logs/serve/driver.log

REPS="${REPS:-1 2 3}"
MODELS="${MODELS:-qwen8b qwen30b qwen4b}"
MAXLEN=131072
WORKERS=8

served(){ case $1 in qwen8b)echo qwen3-vl-8b-instruct;; qwen30b)echo qwen3-vl-30b-instruct;; qwen4b)echo qwen3-vl-4b-instruct;; esac; }
ocid(){   case $1 in qwen8b)echo local_qwen8b;; qwen30b)echo local_qwen30b;; qwen4b)echo local_qwen4b;; esac; }
port(){   case $1 in qwen8b)echo 8001;; qwen30b)echo 8000;; qwen4b)echo 8002;; esac; }
glob(){   case $1 in
  qwen8b)  echo "$HF_HUB/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"*;;
  qwen30b) echo "$HF_HUB/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"*;;
  qwen4b)  echo "$HF_HUB/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/"*;;
esac; }

log(){ echo "[$(date +%F_%T)] $*" | tee -a "$DLOG"; }

# kill serve by EXACT pid (never pattern-match 'vllm' — that matches our own shell).
# MUST also kill the VLLM::EngineCore child: it (not `swift deploy`) holds the GPU,
# so killing only swift deploy leaves ~84GB reserved and the next model OOMs.
kill_serve(){
  for p in $(ps -eo pid,args | awk '/[b]in\/swift deploy/{print $1}'); do kill -9 "$p" 2>/dev/null; done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  # also kill any API server still LISTENing on our ports (uvicorn survives EngineCore death
  # -> stale /v1/models = false READY); kill by listener pid (not pattern, avoids self-kill)
  for pt in 8000 8001 8002; do
    for p in $(ss -ltnp 2>/dev/null | grep ":$pt " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do kill -9 "$p" 2>/dev/null; done
  done
  # wait for GPU to actually drain (<5GB) so the next model has room
  for i in $(seq 1 45); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${u:-99999}" -lt 5000 ] 2>/dev/null && break
    # re-reap any lingering GPU procs each iteration
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
    sleep 4
  done
}

serve(){
  local m=$1 p mp; p=$(port "$m"); mp=$(ls -d $(glob "$m") 2>/dev/null | head -1)
  [ -d "$mp" ] || { log "  SKIP $m: snapshot missing"; return 1; }
  log "serve $m ($mp) on :$p  maxlen=$MAXLEN TP=1"
  setsid $BIN/swift deploy --model "$mp" --infer_backend vllm --host 127.0.0.1 --port "$p" \
    --served_model_name "$(served "$m")" --vllm_max_model_len "$MAXLEN" \
    --vllm_tensor_parallel_size 1 --vllm_enforce_eager true --vllm_gpu_memory_utilization 0.9 \
    > "logs/serve/${m}_serve.log" 2>&1 < /dev/null &
  for i in $(seq 1 120); do        # up to 20 min (30B is slow at TP=1)
    curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && { log "  $m READY (~$((i*10))s)"; return 0; }
    if grep -qE "ValueError|Engine core initialization failed|CUDA error|No module named|OutOfMemory|torch.OutOfMemoryError" "logs/serve/${m}_serve.log" 2>/dev/null; then
      log "  $m serve FATAL:"; grep -E "ValueError|Engine core init|CUDA error|No module|OutOfMemory" "logs/serve/${m}_serve.log" | tail -3 | tee -a "$DLOG"; return 1; fi
    sleep 10
  done
  log "  $m NOT READY after 20min"; return 1
}

probe(){
  local m=$1; local rep=$2; local rid="modext_${m}_c0_run${rep}"
  log "PROBE $rid START"
  $BIN/python -m harness.run_unified_pilot \
    --bank configs/study/combined_modality_bank.json --family modext_v2 \
    --states d0,d50k,d80k,S2 --skip-s3 --cf-conditions C0 \
    --max-workers "$WORKERS" \
    --model "$(ocid "$m")/$(served "$m")" \
    --run-id "$rid" >> "logs/serve/${rid}.log" 2>&1
  log "PROBE $rid rc=$?  ($(wc -l < logs/probes/unified_v1/$rid/results.jsonl 2>/dev/null || echo 0) probes)"
}

log "==== DRIVER START node=$(hostname) reps='$REPS' models='$MODELS' ===="
for REP in $REPS; do
  for M in $MODELS; do
    kill_serve
    if serve "$M"; then
      probe "$M" "$REP"
    else
      log "  $M rep$REP: serve failed, skipping"
    fi
    kill_serve
  done
done
log "==== DRIVER COMPLETE ===="
# final summary
for M in $MODELS; do for R in $REPS; do
  f="logs/probes/unified_v1/modext_${M}_c0_run${R}/summary.json"
  [ -f "$f" ] && $BIN/python - "$f" <<'PY' | tee -a "$DLOG"
import json,sys
d=json.load(open(sys.argv[1])); print("  ", sys.argv[1].split("/")[-2], "pf_rate=", d.get("parse_failure_rate"))
for st,v in d.get("by_state",{}).items():
    print(f"     {st}: SC={v.get('source_correct')} GC={v.get('grounded_correct')} WS={v.get('wrong_source')} n={v.get('n')}")
PY
done; done
