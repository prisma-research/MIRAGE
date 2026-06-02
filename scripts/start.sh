#!/usr/bin/env bash
# start.sh — Start a GroundingBench experiment run.
#
# Usage:
#   bash scripts/start.sh                        # all 2160 cells, 20 workers
#   bash scripts/start.sh --n-trials 20          # smoke test: first 20 cells
#   bash scripts/start.sh --max-workers 10       # fewer workers
#   bash scripts/start.sh --scenarios S1 S3      # specific scenarios
#   bash scripts/start.sh --dry-run              # preview cells without running
#
# All extra args are forwarded to experiment_runner.
# Must be run from the GroundingBench/ directory.

set -euo pipefail

LOG_FILE="/tmp/gb_run_$(date +%Y%m%dT%H%M%S).log"

# Defaults (can be overridden via args)
SCENARIOS="S1 S2 S3"
MAX_WORKERS=20
EXTRA_ARGS=()

# Forward all args directly to the runner
python -m harness.experiment_runner \
    --scenarios S1 S2 S3 \
    --max-workers "$MAX_WORKERS" \
    "$@" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started experiment (PID: $PID)"
echo "Log: $LOG_FILE"
echo ""
echo "Monitor:  tail -f $LOG_FILE"
echo "Progress: bash scripts/progress.sh"
echo "Stop:     bash scripts/stop.sh"
