"""
C0 vs C2 matched comparison — chart_image, 5 seeds each.

Usage:
    cd GroundingBench
    python -m harness.run_c0_vs_c2
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from pipeline.planters.archive.chart_image_planter import ChartImagePlanter
from pipeline.queries.reference_templates import make_reference_query
from harness.trial_runner import build_trial
from harness.trial_log import TrialConfig, load_trials
from harness.scorer_pipeline import score_trial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).parent.parent / "logs" / "trials"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Matched configs: same seeds, styles, depths for both C0 and C2
MATCHED_CONFIGS = [
    {"seed": 10, "style": "definite",      "depth": 0},
    {"seed": 11, "style": "demonstrative",  "depth": 1},
    {"seed": 12, "style": "temporal",        "depth": 0},
    {"seed": 13, "style": "definite",        "depth": 2},
    {"seed": 14, "style": "demonstrative",   "depth": 1},
]


async def run_one(condition: str, cfg: dict, planter: ChartImagePlanter) -> dict:
    """Run a single trial and return summary dict."""
    spec = planter.make_plant_spec(seed=cfg["seed"])
    config = TrialConfig(
        plant_type="chart_image",
        history_depth=cfg["depth"],
        reference_style=cfg["style"],
        seed=cfg["seed"],
        citation_force_condition=condition,
    )
    query = make_reference_query(cfg["style"], "chart_image", spec.artifact_id, seed=cfg["seed"])

    trial = await build_trial(
        config=config,
        plant_prompt=spec.plant_prompt,
        artifact_id=spec.artifact_id,
        artifact_ocr_keywords=spec.artifact_ocr_keywords,
        query=query,
        output_dir=LOGS_DIR,
    )

    trial = score_trial(
        trial,
        artifact_repr=spec.artifact_repr,
        run_axis3=True,
        n_judges=1,
        output_dir=str(LOGS_DIR),
    )

    n_claims = 0
    if isinstance(trial.S_raw, dict):
        judges = trial.S_raw.get("judges", [])
        if judges:
            n_claims = judges[0].get("n_claims", 0)

    return {
        "condition": condition,
        "seed": cfg["seed"],
        "style": cfg["style"],
        "d": cfg["depth"],
        "xi": cfg["interference"],
        "plant": trial.planting_path,
        "R": trial.R_path,
        "I": trial.I_strict,
        "S": trial.S_hat,
        "D": trial.D,
        "T_resp": trial.T_response,
        "T": trial.T,
        "F": trial.F,
        "n_claims": n_claims,
        "profile": trial.workspace_profile,
        "pre_hash": trial.pre_hash,
        "post_hash": trial.post_hash,
    }


async def main():
    planter = ChartImagePlanter()
    results = []

    # Run C0 trials first, then C2 (strictly serial)
    for condition in ["C0", "C2"]:
        print(f"\n{'='*80}")
        print(f"  CONDITION: {condition}")
        print(f"{'='*80}")
        for i, cfg in enumerate(MATCHED_CONFIGS):
            print(f"\n--- {condition} Trial {i+1}/5 (seed={cfg['seed']}, style={cfg['style']}, d={cfg['depth']}, xi={cfg['interference']}) ---")
            r = await run_one(condition, cfg, planter)
            results.append(r)
            print(f"  plant={r['plant']} R={r['R']} I={r['I']} S={r['S']} D={r['D']} T={r['T']} F={r['F']} claims={r['n_claims']}")

    # Print comparison table
    print(f"\n{'='*100}")
    print("C0 vs C2 COMPARISON — chart_image, matched seeds")
    print(f"{'='*100}")
    header = f"{'cond':>4} {'seed':>4} {'style':<15} {'d':>2} {'xi':>2} {'plant':<13} {'R':<7} {'I':>1} {'S':>5} {'D':>1} {'T':>1} {'F':>1} {'claims':>6}"
    print(header)
    print("-" * 100)

    c0_results = [r for r in results if r["condition"] == "C0"]
    c2_results = [r for r in results if r["condition"] == "C2"]

    for r in c0_results:
        print(f"  {r['condition']:>2} {r['seed']:>4} {r['style']:<15} {r['d']:>2} {r['xi']:>2} {r['plant']:<13} {r['R']:<7} {r['I']:>1} {str(r['S']):>5} {r['D']:>1} {r['T']:>1} {r['F']:>1} {r['n_claims']:>6}")
    print("-" * 100)
    for r in c2_results:
        print(f"  {r['condition']:>2} {r['seed']:>4} {r['style']:<15} {r['d']:>2} {r['xi']:>2} {r['plant']:<13} {r['R']:<7} {r['I']:>1} {str(r['S']):>5} {r['D']:>1} {r['T']:>1} {r['F']:>1} {r['n_claims']:>6}")

    # Aggregate stats
    print(f"\n{'='*60}")
    print("AGGREGATE")
    print(f"{'='*60}")
    for label, group in [("C0", c0_results), ("C2", c2_results)]:
        n = len(group)
        r_none = sum(1 for r in group if r["R"] == "R_none") / n
        r_tool = sum(1 for r in group if r["R"] == "R_tool") / n
        r_boot = sum(1 for r in group if r["R"] == "R_boot") / n
        f_rate = sum(1 for r in group if r["F"] == 1) / n
        t_rate = sum(1 for r in group if r["T"] == 1) / n
        d_rate = sum(1 for r in group if r["D"] == 1) / n
        plant_none = sum(1 for r in group if r["plant"] == "none") / n

        print(f"  {label}: R_none={r_none:.0%}  R_tool={r_tool:.0%}  R_boot={r_boot:.0%}  "
              f"T={t_rate:.0%}  F={f_rate:.0%}  D={d_rate:.0%}  plant_none={plant_none:.0%}")

    # Stratify by planting_path
    print(f"\n{'='*60}")
    print("STRATIFIED BY PLANTING_PATH")
    print(f"{'='*60}")
    for plant_val in ["pre_compaction_flush", "none"]:
        for label, group in [("C0", c0_results), ("C2", c2_results)]:
            sub = [r for r in group if r["plant"] == plant_val]
            if not sub:
                continue
            n = len(sub)
            r_none = sum(1 for r in sub if r["R"] == "R_none") / n
            r_tool = sum(1 for r in sub if r["R"] == "R_tool") / n
            f_rate = sum(1 for r in sub if r["F"] == 1) / n
            print(f"  {label} | plant={plant_val:<13} n={n}  R_none={r_none:.0%}  R_tool={r_tool:.0%}  F={f_rate:.0%}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
