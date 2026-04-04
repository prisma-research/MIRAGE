"""
extract_qual_cases.py — Pull representative qualitative cases from locked trial runs.

Usage:
    python scripts/extract_qual_cases.py \
        --run-dir logs/trials_paper_main_gpt5_0322 \
        --run-dir logs/trials_paper_main_qwen3vl32b_0322 \
        --output results/paper_tables/qual_cases.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from results.eval_summary import load_all_trials
from scripts.aggregate_paper_results import load_run_config


def _model_label(run_dir: Path) -> str:
    cfg = load_run_config(run_dir)
    model = cfg.get("model", "unknown")
    return model.split("/")[-1] if "/" in model else model


def _preview(text: str | None, limit: int = 220) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:limit]


def _record(model: str, trial: dict) -> dict:
    return {
        "model": model,
        "trial_id": trial.get("trial_id"),
        "scenario": trial.get("scenario"),
        "artifact_id": trial.get("artifact_id"),
        "R_path": trial.get("R_path"),
        "I_strict": trial.get("I_strict"),
        "S_hat": trial.get("S_hat"),
        "D": trial.get("D"),
        "created_at": trial.get("created_at"),
        "response_preview": _preview(trial.get("response_text")),
        "tool_trace": trial.get("tool_trace", [])[:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract qualitative case candidates from trial runs")
    parser.add_argument("--run-dir", action="append", type=Path, required=True,
                        help="Trial run directory (repeatable)")
    parser.add_argument("--output", type=Path, default=Path("results/paper_tables/qual_cases.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    cases: dict[str, list[dict]] = {
        "gpt5_s3_success": [],
        "qwen_s3_refusal": [],
        "wrong_retrieval": [],
        "s2_silent_hgr": [],
    }

    for run_dir in args.run_dir:
        model = _model_label(run_dir)
        trials = load_all_trials(run_dir)

        for trial in trials:
            scenario = trial.get("scenario")
            r_path = trial.get("R_path")
            i_strict = trial.get("I_strict")
            s_hat = trial.get("S_hat")
            d = trial.get("D", 0) or 0

            if model == "gpt-5" and scenario == "S3" and r_path == "R_tool" and i_strict == 1 and s_hat == 1.0 and d == 0:
                cases["gpt5_s3_success"].append(_record(model, trial))
            if model == "qwen3-vl-32b-instruct" and scenario == "S3" and d == 1:
                cases["qwen_s3_refusal"].append(_record(model, trial))
            if scenario == "S3" and r_path == "R_tool" and i_strict == 0:
                cases["wrong_retrieval"].append(_record(model, trial))
            if scenario == "S2" and r_path == "R_none" and d == 0:
                cases["s2_silent_hgr"].append(_record(model, trial))

    # Keep a small stable slice for each bucket.
    trimmed = {k: sorted(v, key=lambda x: (x["model"], x.get("created_at") or ""))[:5] for k, v in cases.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trimmed, indent=2))
    print(f"[saved] {args.output}")
    for name, items in trimmed.items():
        print(f"{name}: {len(items)}")


if __name__ == "__main__":
    main()
