#!/usr/bin/env bash
# eval.sh — Run evaluation summary on completed trials.
#
# Usage:
#   bash scripts/eval.sh           # one-shot summary
#   bash scripts/eval.sh --watch   # watch mode: re-summarise every 50 new trials
#
# Output saved to results/summary_N.json and results/summary_N.md
# Must be run from the MIRAGE/ directory.

set -euo pipefail

python results/eval_summary.py "$@"

echo ""
echo "Results saved to results/"
ls -lt results/summary_*.md 2>/dev/null | head -3 || echo "(no summaries yet)"
