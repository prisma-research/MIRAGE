"""
OOG Standardized Batch — 4 models × 5 trials = 20 trials

Protocol: VLM plants (Doubao Vision) → target model retrieves (S3)
All trials: C0, d=0, chart_image, definite, Axis 3 enabled

Usage:
    cd GroundingBench
    python -m harness.run_oog_batch
"""

from __future__ import annotations

import asyncio
import json
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

MODELS = [
    ("DeepSeek V3", "volcengine/deepseek-v3-250324"),
    ("GLM-4.7", "volcengine/glm-4-7-251222"),
    ("Kimi K2", "volcengine/kimi-k2-250905"),
    ("Doubao 2.0 Pro", "volcengine/doubao-seed-2-0-pro-260215"),
]

SEEDS_PER_MODEL = [3000, 3001, 3002, 3003, 3004]  # 5 per model


async def run_model_batch(model_name: str, model_id: str, seeds: list[int]) -> list[dict]:
    """Run 5 trials for one model, return summary dicts."""
    planter = ChartImagePlanter()
    results = []

    for seed in seeds:
        spec = planter.make_plant_spec(seed=seed)
        config = TrialConfig(
            plant_type="chart_image",
            history_depth=0,
            reference_style="definite",
            seed=seed,
        )
        query = make_reference_query("definite", "chart_image", spec.artifact_id, seed=seed)

        logger.info("--- %s seed=%d ---", model_name, seed)

        trials = await build_trial(
            config=config,
            plant_prompt=spec.plant_prompt,
            artifact_id=spec.artifact_id,
            artifact_ocr_keywords=spec.artifact_ocr_keywords,
            query=query,
            output_dir=LOGS_DIR,
            scenarios=["S3"],
            image_path=spec.image_path,
            retrieval_model=model_id,
        )

        for t in trials:
            artifact_repr = t.actual_artifact_repr or spec.artifact_repr
            t = score_trial(
                t,
                artifact_repr=artifact_repr,
                run_axis3=True,
                n_judges=1,
                output_dir=str(LOGS_DIR),
            )

            # Verify S_hat is not null
            if t.S_hat is None:
                logger.warning("S_hat is null for %s seed=%d — F will be 0", model_name, seed)

            r = {
                "model": model_name,
                "seed": seed,
                "trial_id": t.trial_id,
                "R": t.R_path,
                "I": t.I_strict,
                "S_hat": t.S_hat,
                "D": t.D,
                "T_resp": t.T_response,
                "T": t.T,
                "F": t.F,
                "plant": t.planting_path,
                "backend": t.theta.backend,
                "actual_repr_len": len(t.actual_artifact_repr),
                "s_hat_null": t.S_hat is None,
            }
            results.append(r)
            logger.info(
                "  %s: R=%s I=%s S=%s D=%s F=%s plant=%s s_hat_null=%s",
                t.scenario, r["R"], r["I"], r["S_hat"], r["D"], r["F"], r["plant"], r["s_hat_null"],
            )

    return results


async def main():
    all_results = []

    for model_name, model_id in MODELS:
        print(f"\n{'='*60}")
        print(f"  {model_name} ({model_id})")
        print(f"{'='*60}")
        results = await run_model_batch(model_name, model_id, SEEDS_PER_MODEL)
        all_results.extend(results)

    # ═══════════════════════════════════════════════
    # RESULTS TABLE
    # ═══════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("FULL RESULTS — OOG Standardized Batch")
    print(f"{'='*90}")
    print(f"{'model':<18} {'seed':>4} {'R':<8} {'I':>1} {'S_hat':>7} {'D':>1} {'T':>1} {'F':>1} {'plant':<14} {'S_null':>6}")
    print("-" * 90)
    for r in all_results:
        s_str = str(r["S_hat"]) if r["S_hat"] is not None else "NULL"
        print(f"{r['model']:<18} {r['seed']:>4} {r['R']:<8} {r['I']:>1} {s_str:>7} {r['D']:>1} {r['T']:>1} {r['F']:>1} {r['plant']:<14} {str(r['s_hat_null']):>6}")

    # ═══════════════════════════════════════════════
    # OOG CALCULATION
    # ═══════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("OOG CALCULATION (verified, Axis 3 enabled)")
    print(f"{'='*90}")
    print(f"{'Model':<18} {'N':>3} {'E[T]':>6} {'E[F]':>6} {'OOG':>6} {'R_none%':>8} {'S_null':>6} {'CGA':>6}")
    print("-" * 70)

    for model_name, _ in MODELS:
        group = [r for r in all_results if r["model"] == model_name]
        N = len(group)
        if N == 0:
            continue

        E_T = sum(1 for r in group if r["T_resp"] == 1) / N
        E_F = sum(1 for r in group if r["F"] == 1) / N
        OOG = E_T - E_F
        R_none = sum(1 for r in group if r["R"] == "R_none") / N
        S_null = sum(1 for r in group if r["s_hat_null"])
        CGA = E_F

        print(f"{model_name:<18} {N:>3} {E_T:>6.2f} {E_F:>6.2f} {OOG:>6.2f} {R_none:>7.0%} {S_null:>6} {CGA:>6.2f}")

    # Data quality check
    print(f"\n{'='*60}")
    print("DATA QUALITY CHECKS")
    print(f"{'='*60}")
    plant_none = sum(1 for r in all_results if r["plant"] == "none")
    s_hat_null = sum(1 for r in all_results if r["s_hat_null"])
    total = len(all_results)
    print(f"Total trials: {total}")
    print(f"planting_path=none: {plant_none}/{total} {'⚠️ INVALID' if plant_none > 0 else '✅'}")
    print(f"S_hat=null: {s_hat_null}/{total} {'⚠️ F unreliable' if s_hat_null > 0 else '✅'}")

    return all_results


if __name__ == "__main__":
    asyncio.run(main())
