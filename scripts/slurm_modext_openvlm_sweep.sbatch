#!/usr/bin/env bash
#SBATCH --job-name=gb_modext_openvlm
# data-inst = the 4xH100 NVL partition the serve scripts assume (TP=4); 96 CPU / 1.5TB / 3-day.
#SBATCH --partition=data-inst
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/gb_modext_openvlm_%j.out
#SBATCH --error=logs/slurm/gb_modext_openvlm_%j.err
#
# Modality-extension on the OPEN-WEIGHT backbones (rebuttal T1 scope / L137 \todo:
# "additional paper-suite backbones (Qwen3-VL-4B/8B/30B) and repetitions").
# Re-uses the already-built 12-object modality trunk (family=modext_v2) + the modality
# bank, swapping only the answering model — same protocol/states/scoring as the Doubao
# modext run (logs/probes/unified_v1/modext_doubao_v2_c0).
#
# GPU constraint: 8B and 30B both serve at TP=4 (all 4 GPUs), so models are served
# SEQUENTIALLY (serve -> probe x REPS -> stop -> next model).
# RESUMABLE: fixed per-(model,rep) run-id => re-submitting skips probes already in
# results.jsonl. If a model's vLLM dies mid-run, re-submit; it resumes that run-id.

set -uo pipefail
PROJ="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJ"

# ---- environment ----
export PATH="$HOME/.nvm/versions/node/v22.22.2/bin:$HOME/.local/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || \
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate groundingbench 2>/dev/null || true
export OPENCLAW_TOKEN=none
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
PY=python
HF_HUB="$HF_HOME/hub"
mkdir -p "$PROJ/logs/slurm" "$PROJ/logs/serve"

# ---- sweep config (env-overridable: smoke = MODELS=qwen8b REPS=1 COUNT=2) ----
BANK=configs/study/combined_modality_bank.json
FAMILY=modext_v2
STATES="${STATES:-d0,d80k,S2}"        # signal states (matches rebuttal Tab.~modality); add d50k for full curve
WORKERS="${WORKERS:-8}"               # concurrent probe gateways hitting the one vLLM server
REPS="${REPS:-1 2 3}"                 # 3 reps -> per-cell CIs + matches paper's 3-run methodology
MODELS="${MODELS:-qwen8b qwen30b qwen4b}"   # served sequentially; drop 4b if not cached
COUNT="${COUNT:-}"                    # blank = all base questions; set (e.g. 2) for a smoke

# per-model: served name | openclaw provider id | vLLM port | HF snapshot glob
served_name(){ case $1 in qwen8b) echo qwen3-vl-8b-instruct;; qwen30b) echo qwen3-vl-30b-instruct;; qwen4b) echo qwen3-vl-4b-instruct;; esac; }
ocid(){       case $1 in qwen8b) echo local_qwen8b;; qwen30b) echo local_qwen30b;; qwen4b) echo local_qwen4b;; esac; }
port(){       case $1 in qwen8b) echo 8001;; qwen30b) echo 8000;; qwen4b) echo 8002;; esac; }
hfglob(){     case $1 in
                qwen8b)  echo "$HF_HUB/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"*;;
                qwen30b) echo "$HF_HUB/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/"*;;
                qwen4b)  echo "$HF_HUB/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/"*;;
              esac; }

log(){ echo "[$(date +%F_%T)] $*"; }

SERVE_PID=""
serve(){
  local m=$1 p; p=$(port "$m")
  local mp; mp=$(hfglob "$m"); mp=$(ls -d $mp 2>/dev/null | head -1)
  if [ -z "$mp" ] || [ ! -d "$mp" ]; then log "SKIP $m: HF snapshot not found ($(hfglob "$m"))"; return 1; fi
  log "serve $m: $mp on :$p (TP=4)"
  swift deploy --model "$mp" --infer_backend vllm --host 0.0.0.0 --port "$p" \
    --served_model_name "$(served_name "$m")" \
    --vllm_max_model_len 131072 --vllm_tensor_parallel_size 4 --vllm_enforce_eager true \
    > "$PROJ/logs/serve/${m}_${SLURM_JOB_ID:-NA}.log" 2>&1 &
  SERVE_PID=$!
  # wait for readiness (30B can take several minutes to load)
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1; then log "  $m READY on :$p"; return 0; fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then log "  $m serve died early — see logs/serve/${m}_*.log"; return 1; fi
    sleep 10
  done
  log "  $m NOT READY after 20min — abort this model"; return 1
}
stop_serve(){
  [ -n "$SERVE_PID" ] || return 0
  log "stopping serve pid=$SERVE_PID"
  kill "$SERVE_PID" 2>/dev/null || true
  for i in $(seq 1 30); do kill -0 "$SERVE_PID" 2>/dev/null || break; sleep 2; done
  kill -9 "$SERVE_PID" 2>/dev/null || true
  pkill -9 -f "vllm" 2>/dev/null || true   # reap stray vLLM workers so the next model gets the GPUs
  SERVE_PID=""
  sleep 10
}
trap stop_serve EXIT

log "JOB START node=$(hostname) job=${SLURM_JOB_ID:-NA} gpus=$CUDA_VISIBLE_DEVICES models='$MODELS' reps='$REPS' states=$STATES"

for M in $MODELS; do
  serve "$M" || { stop_serve; continue; }
  for R in $REPS; do
    RID="modext_${M}_c0_run${R}"; [ -n "$COUNT" ] && RID="modext_${M}_c0_smoke"
    log "PROBE $RID START (family=$FAMILY states=$STATES workers=$WORKERS count=${COUNT:-all})"
    $PY -m harness.run_unified_pilot \
      --bank "$BANK" --family "$FAMILY" \
      --states "$STATES" --skip-s3 --cf-conditions C0 \
      --max-workers "$WORKERS" \
      ${COUNT:+--count "$COUNT"} \
      --model "$(ocid "$M")/$(served_name "$M")" \
      --run-id "$RID"
    log "PROBE $RID rc=$?"
  done
  stop_serve
done

log "ALL DONE — summaries:"
for M in $MODELS; do for R in $REPS; do
  f="logs/probes/unified_v1/modext_${M}_c0_run${R}/summary.json"
  [ -f "$f" ] && $PY - "$f" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1])); print("  ", sys.argv[1].split("/")[-2])
for st,v in d.get("by_state",{}).items():
    print(f"     {st}: SC={v.get('source_correct')} GC={v.get('grounded_correct')} WS={v.get('wrong_source')} n={v.get('n')}")
PYEOF
done; done
log "JOB COMPLETE"
