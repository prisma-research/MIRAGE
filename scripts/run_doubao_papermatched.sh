#!/usr/bin/env bash
# Paper-matched doubao main eval: SAME checkpoint family as paper Table 1
# (pilot_v3_100k_prewrite_generic), SAME 4 states, C0, full 200-Q bank,
# 3 independent runs (paper averages 3). Only the answering model is swapped
# to doubao — exactly the run_unified_pilot model-swap the paper uses per row.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export OPENCLAW_TOKEN=none
MODEL=volcengine/doubao-seed-1-6-vision-250815
EP=pilot_v3_100k_prewrite_generic
for R in 1 2 3; do
  RID="doubao_papermatched_c0_run${R}"
  echo "==== RUN ${R} run_id=${RID} $(date) ===="
  python -m harness.run_unified_pilot \
    --model "$MODEL" \
    --postcomp-episode "$EP" \
    --states d0,d50k,d80k,S2 --cf-conditions C0 \
    --count 200 --skip-s3 \
    --run-id "$RID"
  echo "==== RUN ${R} rc=$? $(date) ===="
done
echo "==== PAPERMATCHED SWEEP COMPLETE $(date) ===="
