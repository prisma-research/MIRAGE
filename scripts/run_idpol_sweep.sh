#!/usr/bin/env bash
# Provenance-preservation sweep: vary compaction identifierPolicy (off / custom),
# holding summarizer=deepseek-v4-flash, answering backbone=doubao-seed-1-6-vision,
# safeguard mode + prewrite + memoryFlush fixed. Baseline strict = arm B (existing).
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export OPENCLAW_TOKEN=none
SUMM=volcengine/deepseek-v4-flash-260425
AGENT=volcengine/deepseek-v4-flash-260425
QAMODEL=volcengine/doubao-seed-1-6-vision-250815

CUSTOM_INSTR='Preserve all opaque identifiers exactly as written, especially every source FILE PATH (the data/... file each datum came from). In addition, for EVERY fact, datum, or number you include in the summary, you MUST attach its exact source artifact path as an explicit inline citation, e.g. [source: data/gdp/gdp_extracted_2000-2020.json]. Never drop, merge, or paraphrase a source path.'

run_arm () {
  local POL="$1" EP="$2" RID="$3"
  echo "==== BUILD arm=$POL episode=$EP $(date) ===="
  OPENCLAW_ABLATION_IDENTIFIER_POLICY="$POL" \
  OPENCLAW_ABLATION_IDENTIFIER_INSTRUCTIONS="$CUSTOM_INSTR" \
  python -m harness.build_generic_prewrite_checkpoint --phase 2 \
    --target-episode "$EP" \
    --compaction-model "$SUMM" \
    --agent-model "$AGENT"
  local BUILD_RC=$?
  echo "==== BUILD arm=$POL rc=$BUILD_RC $(date) ===="
  if [ $BUILD_RC -ne 0 ]; then echo "BUILD FAILED arm=$POL"; return 1; fi
  echo "==== QA arm=$POL run_id=$RID $(date) ===="
  python -m harness.run_unified_pilot \
    --model "$QAMODEL" \
    --count 10 --skip-s3 --cf-conditions C0 --states d80k,S2 \
    --postcomp-episode "$EP" \
    --run-id "$RID"
  echo "==== QA arm=$POL DONE $(date) ===="
}

run_arm off    pilot_v3_idpol_off    ablation_idpol_off
run_arm custom pilot_v3_idpol_custom ablation_idpol_custom
echo "==== SWEEP COMPLETE $(date) ===="
