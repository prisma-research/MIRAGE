#!/usr/bin/env bash
# Compactor ablation, compactor(summarizer)=doubao. Only S2 probed (d0/d50k/d80k
# are pre-compaction + hardcoded to pilot_v3_100k => identical across arms, no signal).
# strict arm reuses line B's finished S2 (GPT-5 native compaction) — handled separately.
# Here: off + custom arms, doubao summarizer, n=200 S2 probes.
set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export OPENCLAW_TOKEN=none
SUMM=volcengine/doubao-seed-1-6-vision-250815     # compactor = doubao (changed from deepseek)
AGENT=volcengine/deepseek-v4-flash-260425          # build agent (reliable prewrite persistence)
QA=volcengine/doubao-seed-1-6-vision-250815        # answering model
CUSTOM_INSTR='Preserve all opaque identifiers exactly as written, especially every source FILE PATH (the data/... file each datum came from). In addition, for EVERY fact, datum, or number you include in the summary, you MUST attach its exact source artifact path as an explicit inline citation, e.g. [source: data/gdp/gdp_extracted_2000-2020.json]. Never drop, merge, or paraphrase a source path.'

echo "==== BUILD custom (doubao summarizer) $(date) ===="
OPENCLAW_ABLATION_IDENTIFIER_POLICY=custom \
OPENCLAW_ABLATION_IDENTIFIER_INSTRUCTIONS="$CUSTOM_INSTR" \
python -m harness.build_generic_prewrite_checkpoint --phase 2 \
  --target-episode pilot_v3_idpol_custom_doubao \
  --compaction-model "$SUMM" --agent-model "$AGENT"
echo "==== BUILD custom rc=$? $(date) ===="

echo "==== BUILD off (doubao summarizer, allow-unverified) $(date) ===="
OPENCLAW_ABLATION_IDENTIFIER_POLICY=off \
OPENCLAW_ABLATION_ALLOW_UNVERIFIED=1 \
python -m harness.build_generic_prewrite_checkpoint --phase 2 \
  --target-episode pilot_v3_idpol_off_doubao \
  --compaction-model "$SUMM" --agent-model "$AGENT"
echo "==== BUILD off rc=$? $(date) ===="

# wait for line B run1 to finish (800 rows) before probing, to avoid 3-way QA contention
RES=logs/probes/unified_v1/doubao_papermatched_c0_run1/results.jsonl
echo "==== WAIT for line B run1 to finish $(date) ===="
while true; do
  n=$(wc -l < "$RES" 2>/dev/null || echo 0)
  if [ "$n" -ge 800 ] || ! kill -0 3833816 2>/dev/null; then echo "line B run1 done (n=$n)"; break; fi
  sleep 60
done

for arm in custom off; do
  echo "==== QA ${arm} S2 x200 $(date) ===="
  python -m harness.run_unified_pilot --model "$QA" \
    --postcomp-episode pilot_v3_idpol_${arm}_doubao \
    --states S2 --cf-conditions C0 --count 200 \
    --run-id doubao_compactor_${arm}_n200
  echo "==== QA ${arm} done $(date) ===="
done
echo "==== COMPACTOR ABLATION (doubao) COMPLETE $(date) ===="
