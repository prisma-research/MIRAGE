#!/usr/bin/env bash
# stop.sh — Kill everything related to a MIRAGE experiment run.
# Usage: bash stop.sh

set -euo pipefail

echo "=== Stopping MIRAGE experiment ==="

# 1. Kill experiment runner
if pids=$(pgrep -f "experiment_runner" 2>/dev/null); then
    echo "Killing experiment_runner (PIDs: $pids)..."
    kill -9 $pids 2>/dev/null || true
else
    echo "No experiment_runner process found."
fi

# 2. Kill orphaned openclaw processes
for pattern in "openclaw-agent" "openclaw-gateway"; do
    if pids=$(pgrep -f "$pattern" 2>/dev/null); then
        echo "Killing $pattern (PIDs: $pids)..."
        kill -9 $pids 2>/dev/null || true
    fi
done

sleep 1

# 3. Remove all gb_* trial agents from openclaw config
echo "Cleaning up trial agents from openclaw registry..."
deleted=0
while IFS= read -r agent_id; do
    openclaw agents delete "$agent_id" --force 2>/dev/null && echo "  deleted: $agent_id" && ((deleted++)) || true
done < <(openclaw agents list 2>/dev/null | grep -E '^\- gb_' | awk '{print $2}' || true)

if [ "$deleted" -eq 0 ]; then
    echo "  No gb_* agents found in registry."
else
    echo "  Removed $deleted agent(s)."
fi

echo "=== Done ==="
