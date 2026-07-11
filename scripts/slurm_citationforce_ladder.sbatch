#!/usr/bin/env bash
#SBATCH --job-name=gb_cf_ladder
#SBATCH --partition=work
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/gb_cf_ladder_%j.out
#SBATCH --error=logs/slurm/gb_cf_ladder_%j.err
#
# CitationForce mitigation ladder (bXAt W2: "diagnosis-heavy, solution-light").
# Single unified answering model = doubao-seed-2.0-pro, paper-matched checkpoint
# family, POST-COMPACTION state (S2) only — all 5 conditions share the SAME S2
# checkpoint, varying only the mitigation rung:
#   C0  no mitigation (control)
#   C1  prompt-only citation constraint (NO tool)
#   C2  C1 + artifact_recall retrieval tool available
#   C3  C2 + post-compaction-aware re-injection ("context was compacted, re-retrieve")
#   Cm  mandatory artifact_recall routing (memory_search prohibited)
# 5 conditions x S2 x 200 Q = 1000 probes, fully parallel + resumable.
# Story: even maximal forcing (Cm) does not recover SC at S2 => reinforces RQ3
# (post-compaction bottleneck is degraded stored evidence, not prompting).
#
# Resumable: re-submitting skips probes already in the run's results.jsonl
# (keyed by condition x state x base_question_id).

set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Compute-node environment (per doubao-runtime-setup memory):
export PATH="$HOME/.nvm/versions/node/v22.22.2/bin:$HOME/.local/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || \
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
conda activate mirage
export OPENCLAW_TOKEN=none

PY=python
QA=volcengine/doubao-seed-2-0-pro-260215          # unified answering model
EP=pilot_v3_100k_prewrite_generic                 # paper Table 1 checkpoint family
RID=doubao2pro_cf_ladder_s2                        # single run_id => one results.jsonl
W=24                                               # concurrent probes (=24 isolated gateways)

log(){ echo "[$(date +%F_%T)] $*"; }
log "JOB START node=$(hostname) job=${SLURM_JOB_ID:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA} mem=${SLURM_MEM_PER_NODE:-NA}MB"
log "node version: $(node --version 2>&1) | openclaw: $(command -v openclaw || echo MISSING)"
log "ark reachability:"; timeout 8 curl -s -o /dev/null -w "  ark HTTP=%{http_code}\n" \
  https://ark.cn-beijing.volces.com/api/v3 || { log "  ARK UNREACHABLE — abort"; exit 1; }

log "QA $RID START (model=$QA ep=$EP state=S2 conditions=C0,C1,C2,C3,Cm workers=$W)"
$PY -m harness.run_unified_pilot \
  --model "$QA" \
  --postcomp-episode "$EP" \
  --states S2 --skip-s3 \
  --cf-conditions C0,C1,C2,C3,Cm \
  --count 200 \
  --max-workers "$W" \
  --run-id "$RID"
rc=$?
log "QA $RID rc=$rc"

# ── Summary: full ladder + delta-vs-C0 (print_analysis already emits both) ──
SUMMARY="logs/probes/unified_v1/$RID/summary.json"
if [ -f "$SUMMARY" ]; then
  log "LADDER SUMMARY (S2):"
  $PY - "$SUMMARY" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1]))
bc=d.get("by_condition",{})
order=["C0","C1","C2","C3","Cm"]
print(f"  {'cond':4s} {'n':>4s} {'SC':>6s} {'GC':>6s} {'AC':>6s} {'WS':>6s}")
for c in order:
    v=(bc.get(c,{}) or {}).get("S2",{})
    if not v: continue
    print(f"  {c:4s} {v.get('n',0):>4d} {v.get('source_correct',0):>6.3f} "
          f"{v.get('grounded_correct',0):>6.3f} {v.get('answerable_correct',0):>6.3f} "
          f"{v.get('wrong_source',0):>6.3f}")
# R_tool share per condition at S2
rp=d.get("rpath",{})
print("  R_tool share @S2:")
for c in order:
    st=rp.get(c,{}).get("S2",{})
    tot=sum(st.values()) or 1
    print(f"    {c}: R_tool={st.get('R_tool',0)}/{tot} ({st.get('R_tool',0)/tot:.2%})")
PYEOF
else
  log "WARNING: no summary.json at $SUMMARY"
fi
log "JOB COMPLETE rc=$rc"
