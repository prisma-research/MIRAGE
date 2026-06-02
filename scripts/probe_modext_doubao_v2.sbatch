#!/bin/bash
#SBATCH --job-name=modext_probe_v2
#SBATCH --partition=work
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/modext_probe_v2_%j.out
#SBATCH --error=logs/slurm/modext_probe_v2_%j.out
#
# Full Doubao parallel probe over the combined 12-object modality family.
# 309 questions x 4 states (d0/d50k/d80k/S2) x C0 = ~1236 probes, run with
# parallel branch gateways (--max-workers). RESUMABLE: a fixed --run-id means
# re-submitting skips probes already in results.jsonl (done_keys).

set -uo pipefail
PROJ="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJ"
export PATH="$HOME/.nvm/versions/node/v22.22.2/bin:$HOME/.local/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
export OPENCLAW_TOKEN=none
mkdir -p "$PROJ/logs/slurm"

echo "=== modext Doubao probe on $(hostname) ==="
python -m harness.run_unified_pilot \
  --bank configs/study/combined_modality_bank.json \
  --family modext_v2 \
  --states d0,d50k,d80k,S2 --skip-s3 \
  --max-workers 8 \
  --model volcengine/doubao-seed-2-0-pro-260215 \
  --run-id modext_doubao_v2_c0
echo "=== probe run finished ==="
