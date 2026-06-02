#!/usr/bin/env bash
# progress.sh — Show current experiment progress.
# Must be run from the GroundingBench/ directory.

set -euo pipefail

# Trial counts
total=$(ls logs/trials/*.json 2>/dev/null | wc -l | tr -d ' ')
s1=$(ls logs/trials/*_S1.json 2>/dev/null | wc -l | tr -d ' ')
s2=$(ls logs/trials/*_S2.json 2>/dev/null | wc -l | tr -d ' ')
s3=$(ls logs/trials/*_S3.json 2>/dev/null | wc -l | tr -d ' ')
errors=$(grep -c '"error"' logs/errors.jsonl 2>/dev/null || echo 0)

echo "=== GroundingBench Progress ==="
echo "Completed trials: $total  (S1=$s1  S2=$s2  S3=$s3)"
echo "Errors:           $errors"
echo ""

# Running process
if pid=$(pgrep -f "experiment_runner" 2>/dev/null | head -1); then
    echo "Experiment runner: RUNNING (PID $pid)"
    # Find most recent log
    log=$(ls -t /tmp/gb_run_*.log 2>/dev/null | head -1 || echo "")
    if [ -n "$log" ]; then
        echo "Log: $log"
        echo ""
        echo "--- Last 10 log lines ---"
        tail -10 "$log"
    fi
else
    echo "Experiment runner: NOT RUNNING"
fi
