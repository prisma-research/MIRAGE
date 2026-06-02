#!/usr/bin/env bash
#SBATCH --job-name=gb_compactor_abl
#SBATCH --partition=ondemand
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=09:00:00
#SBATCH --output=logs/slurm/gb_abl_%j.out
#SBATCH --error=logs/slurm/gb_abl_%j.err
#
# Compactor ablation (compactor/summarizer = doubao), fully parallel + resumable.
# Probes only S2 per arm (d0/d50k/d80k are pre-compaction + hardcoded => no signal).
#   strict arm = GPT-5 native compaction (reuse pilot_v3_100k_prewrite_generic S2)
#   custom/off arms = doubao summarizer, identifierPolicy custom/off
# Resumable: re-submitting skips probes already in each run's results.jsonl.

set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export OPENCLAW_TOKEN=none
PY=python
SUMM=volcengine/doubao-seed-2-0-pro-260215       # compactor (summarizer) = doubao-2.0-pro (.env default)
AGENT=volcengine/deepseek-v4-flash-260425        # build agent (reliable prewrite)
QA=volcengine/doubao-seed-2-0-pro-260215         # answering model (multimodal, ~0% parse-fail vs seed-1-6's 20-30%)
GEN_EP=pilot_v3_100k_prewrite_generic            # GPT-5-built checkpoint family (= paper Table 1)
W=16                                             # concurrent probes per QA job
CUSTOM_INSTR='Preserve all opaque identifiers exactly as written, especially every source FILE PATH (the data/... file each datum came from). In addition, for EVERY fact, datum, or number you include in the summary, you MUST attach its exact source artifact path as an explicit inline citation, e.g. [source: data/gdp/gdp_extracted_2000-2020.json]. Never drop, merge, or paraphrase a source path.'

log(){ echo "[$(date +%F_%T)] $*"; }
# Remove any branch state dirs + gateway sockets left by a prior crashed build,
# so restore_checkpoint's copytree into the SHARED build branch dir never collides.
clean_build_branches(){
  for d in $HOME/.openclaw-branch-build_postcomp_generic \
           $HOME/.openclaw-branch-build_near_comp_base; do
    [ -e "$d" ] && { rm -rf "$d"; log "  cleaned stale branch dir $d"; }
  done
}
log "JOB START node=$(hostname) job=${SLURM_JOB_ID:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA} mem=${SLURM_MEM_PER_NODE:-NA}MB"
log "ark reachability:"; timeout 8 curl -s -o /dev/null -w "  ark HTTP=%{http_code}\n" https://ark.cn-beijing.volces.com/api/v3 || log "  ARK UNREACHABLE — abort"

# ───────────────────────── BUILD arms (doubao summarizer) ─────────────────────────
if [ -f "logs/checkpoints/pilot_v3_idpol_custom_s2pro/postcomp_100k/manifest.json" ]; then
  log "BUILD custom: checkpoint already exists — skip"
else
  log "BUILD custom START (summarizer=doubao, identifierPolicy=custom)"
  clean_build_branches
  # allow-unverified: a transient usage-proxy drop can lose 1/6 artifact persistence
  # (seen as 5/6 verified); the custom policy still preserves provenance for the rest,
  # so probe the salvaged S2 rather than abort. Verified count is logged for honesty.
  OPENCLAW_ABLATION_IDENTIFIER_POLICY=custom \
  OPENCLAW_ABLATION_IDENTIFIER_INSTRUCTIONS="$CUSTOM_INSTR" \
  OPENCLAW_ABLATION_ALLOW_UNVERIFIED=1 \
  $PY -m harness.build_generic_prewrite_checkpoint --phase 2 \
    --target-episode pilot_v3_idpol_custom_s2pro \
    --compaction-model "$SUMM" --agent-model "$AGENT"
  log "BUILD custom rc=$?"
fi

if [ -f "logs/checkpoints/pilot_v3_idpol_off_s2pro/postcomp_100k/manifest.json" ]; then
  log "BUILD off: checkpoint already exists — skip"
else
  log "BUILD off START (summarizer=doubao, identifierPolicy=off, allow-unverified)"
  clean_build_branches
  OPENCLAW_ABLATION_IDENTIFIER_POLICY=off \
  OPENCLAW_ABLATION_ALLOW_UNVERIFIED=1 \
  $PY -m harness.build_generic_prewrite_checkpoint --phase 2 \
    --target-episode pilot_v3_idpol_off_s2pro \
    --compaction-model "$SUMM" --agent-model "$AGENT"
  log "BUILD off rc=$?"
fi

if [ -f "logs/checkpoints/pilot_v3_idpol_strict_s2pro/postcomp_100k/manifest.json" ]; then
  log "BUILD strict: checkpoint already exists — skip"
else
  log "BUILD strict START (summarizer=doubao, identifierPolicy=strict, verbatim preservation)"
  clean_build_branches
  # allow-unverified for parity with the other arms: a transient usage-proxy drop
  # can lose 1/6 persistence; strict should otherwise verify cleanly (6/6).
  OPENCLAW_ABLATION_IDENTIFIER_POLICY=strict \
  OPENCLAW_ABLATION_ALLOW_UNVERIFIED=1 \
  $PY -m harness.build_generic_prewrite_checkpoint --phase 2 \
    --target-episode pilot_v3_idpol_strict_s2pro \
    --compaction-model "$SUMM" --agent-model "$AGENT"
  log "BUILD strict rc=$?"
fi

# ───────────────────────── QA (parallel + resumable) ─────────────────────────
qa(){
  local ep=$1 rid=$2 states=$3
  log "QA $rid START (ep=$ep states=$states workers=$W)"
  $PY -m harness.run_unified_pilot --model "$QA" \
    --postcomp-episode "$ep" --states "$states" --cf-conditions C0 \
    --count 200 --max-workers "$W" --run-id "$rid"
  log "QA $rid rc=$?"
}

# strict arm (= GPT-5 native compaction S2) + shared d80k pre-comp reference, answered by seed-2-0-pro
qa "$GEN_EP" doubao2pro_papermatched_run1 d80k,S2
# doubao-compactor arms (S2 only)
qa pilot_v3_idpol_custom_s2pro doubao2pro_compactor_custom_n200 S2
qa pilot_v3_idpol_off_s2pro    doubao2pro_compactor_off_n200    S2
qa pilot_v3_idpol_strict_s2pro doubao2pro_compactor_strict_n200 S2

log "ALL DONE — summaries:"
for r in doubao2pro_papermatched_run1 doubao2pro_compactor_custom_n200 doubao2pro_compactor_off_n200 doubao2pro_compactor_strict_n200; do
  f="logs/probes/unified_v1/$r/summary.json"
  [ -f "$f" ] && $PY - "$f" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1]))
print("  ", sys.argv[1].split("/")[-2])
for st,v in d.get("by_state",{}).items():
    print(f"     {st}: SC={v['source_correct']:.3f} AC={v['answerable_correct']:.3f} GC={v['grounded_correct']:.3f} WS={v['wrong_source']:.3f} n={v['n']}")
PYEOF
done
log "JOB COMPLETE"
