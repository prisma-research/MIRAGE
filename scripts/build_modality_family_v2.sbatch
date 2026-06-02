#!/bin/bash
#SBATCH --job-name=modext_build_v2
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/modext_build_v2_%j.out
#SBATCH --error=logs/slurm/modext_build_v2_%j.out
#
# RESUMABLE native checkpoint build for the combined 12-object modality family.
# Each checkpoint is built from the previous saved one; stages whose target
# checkpoint already exists are SKIPPED. Re-submitting after a timeout/crash
# resumes automatically. Checkpoints are saved incrementally (d0→d50k→d80k→S2).

set -uo pipefail
PROJ="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJ"

# ── env: node (openclaw v22) + anaconda base python + ark key ──
export PATH="$HOME/.nvm/versions/node/v22.22.2/bin:$HOME/.local/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
export OPENCLAW_TOKEN=none
export S1_D0_EIT_CEILING=45000   # 12-object trunk → higher (still-shallow) d0
mkdir -p "$PROJ/logs/slurm"

FAMILY=modext_v2
CK="$PROJ/logs/checkpoints/$FAMILY"
echo "=== modext build on $(hostname) — family=$FAMILY ==="
echo "node=$(which node) openclaw=$(which openclaw) python=$(which python)"

have() { [ -f "$CK/$1/manifest.json" ]; }
need_prev() { have "$1" || { echo "FATAL: prerequisite $1 missing — cannot build next stage"; exit 1; }; }

# ── Stage d0: plant 12 artifacts + work tasks → s1_prequery_d0 ──
if have s1_prequery_d0; then echo "[skip] d0 exists"; else
  echo "[build] d0 (plant 12 objects)"
  OPENCLAW_TOKEN=none python -m harness.episode_checkpoint_runner \
    --episode configs/episodes/modality_ext_100k_plant_v2.json \
    --model volcengine/doubao-seed-2-0-pro-260215 --episode-id "$FAMILY"
  have s1_prequery_d0 || { echo "FATAL: d0 build failed"; exit 1; }
fi

# ── Stage d50k: resume d0 → filler to 50k ──
if have s1_prequery_d50k; then echo "[skip] d50k exists"; else
  need_prev s1_prequery_d0
  echo "[build] d50k"
  OPENCLAW_TOKEN=none python -m harness.resume_native_stage \
    --source-episode "$FAMILY" --source-checkpoint s1_prequery_d0 \
    --target-episode "$FAMILY" --mode s1 --target-eit 50000 --depth-label 50000
  have s1_prequery_d50k || { echo "FATAL: d50k build failed"; exit 1; }
fi

# ── Stage d80k: resume d50k → filler to 80k ──
if have s1_prequery_d80k; then echo "[skip] d80k exists"; else
  need_prev s1_prequery_d50k
  echo "[build] d80k"
  OPENCLAW_TOKEN=none python -m harness.resume_native_stage \
    --source-episode "$FAMILY" --source-checkpoint s1_prequery_d50k \
    --target-episode "$FAMILY" --mode s1 --target-eit 80000 --depth-label 80000
  have s1_prequery_d80k || { echo "FATAL: d80k build failed"; exit 1; }
fi

# ── Stage S2: resume d80k → filler until native compaction @100k ──
if have postcomp_100k; then echo "[skip] postcomp_100k exists"; else
  need_prev s1_prequery_d80k
  echo "[build] postcomp_100k (native compaction)"
  OPENCLAW_TOKEN=none python -m harness.resume_native_stage \
    --source-episode "$FAMILY" --source-checkpoint s1_prequery_d80k \
    --target-episode "$FAMILY" --mode postcomp --threshold 100000
  have postcomp_100k || { echo "FATAL: postcomp build failed"; exit 1; }
fi

echo "=== DONE — family $FAMILY checkpoints ==="
ls -1 "$CK"
