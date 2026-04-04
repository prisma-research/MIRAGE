"""
Pilot Run — end-to-end verification of the MIRAGE pipeline.

Run this before committing to the full 36,000-trial experiment.
Artifacts are loaded from the pre-generated manifest (same as experiment_runner).

Usage:
    cd MIRAGE
    python -m harness.pilot_runner

Pilot trials:
  1. screenshot,     history_depth=2, definite  — verify R detection
  2. chart_image,    history_depth=4, temporal  — verify chart path
  3. ui_state_image, history_depth=0, demonstrative + CitationForce C1
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

from pipeline.planters.base_planter import PlantSpec
from pipeline.queries.reference_templates import make_reference_query, make_grounded_query
from harness.trial_runner import build_trial
from harness.trial_log import TrialConfig, load_trials
from harness.scorer_pipeline import score_trial
from mitigations.citation_force.bootstrap_hook import (
    inject_citation_force_into_bootstrap,
    remove_citation_force_from_bootstrap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LOGS_DIR      = Path(__file__).parent.parent / "logs" / "trials"
MANIFEST_PATH = Path(__file__).parent.parent / "data" / "generated_images" / "manifest.json"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _load_manifest_entry(plant_type: str, artifact_idx: int = 0) -> dict:
    """Load the Nth entry of a given plant_type from the manifest."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found at {MANIFEST_PATH}. "
            "Run: cd MIRAGE && python -m data.generate_dataset"
        )
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        entries = json.load(f)

    pool = [e for e in entries if e.get("plant_type") == plant_type and e.get("image_exists")]
    if not pool:
        raise ValueError(f"No image-valid entries for plant_type='{plant_type}' in manifest")
    if artifact_idx >= len(pool):
        raise IndexError(f"artifact_idx={artifact_idx} out of range (pool size={len(pool)})")

    return pool[artifact_idx]


def _entry_to_spec(entry: dict) -> PlantSpec:
    raw_path = entry.get("image_path")
    image_path = str(MANIFEST_PATH.parent.parent / raw_path) if raw_path else None
    return PlantSpec(
        artifact_id=entry["artifact_id"],
        plant_type=entry["plant_type"],
        plant_prompt=entry["plant_prompt"],
        plant_prompt_ai=entry["plant_prompt_ai"],
        artifact_ocr_keywords=entry.get("ocr_keywords", []),
        artifact_repr=entry.get("artifact_repr", ""),
        image_path=image_path,
        metadata=entry.get("metadata", {}),
        content_query=entry.get("content_query", ""),
    )


async def run_pilot_trial(
    plant_type: str,
    reference_style: str,
    history_depth: int,
    artifact_idx: int = 0,
    citation_force_condition: str = "C0",
    trial_label: str = "",
) -> dict:
    """Run a single pilot trial end-to-end."""
    logger.info(
        "=== Pilot trial: %s (plant=%s, art_idx=%d, style=%s, depth=%d, CF=%s) ===",
        trial_label, plant_type, artifact_idx, reference_style, history_depth, citation_force_condition,
    )

    entry = _load_manifest_entry(plant_type, artifact_idx)
    spec = _entry_to_spec(entry)

    seed = abs(hash((plant_type, artifact_idx, history_depth, reference_style, citation_force_condition))) % 100_000

    config = TrialConfig(
        plant_type=plant_type,
        history_depth=history_depth,
        reference_style=reference_style,
        seed=seed,
        citation_force_condition=citation_force_condition,
    )

    if spec.content_query:
        query = make_grounded_query(reference_style, plant_type, spec.content_query, seed=seed)
    else:
        query = make_reference_query(reference_style, plant_type, spec.artifact_id, seed=seed)

    logger.info("Artifact ID: %s", spec.artifact_id)
    logger.info("Query: %s", query)

    if citation_force_condition != "C0":
        logger.info("Injecting CitationForce (condition=%s)", citation_force_condition)
        inject_citation_force_into_bootstrap(condition=citation_force_condition)

    try:
        trials = await build_trial(
            config=config,
            plant_prompt=spec.plant_prompt_ai,
            artifact_id=spec.artifact_id,
            artifact_ocr_keywords=spec.artifact_ocr_keywords,
            query=query,
            output_dir=LOGS_DIR,
        )

        run_axis3 = bool(os.environ.get("LLM_API_KEY"))
        scored = []
        for t in trials:
            scored.append(score_trial(
                t,
                artifact_repr=spec.artifact_repr,
                run_axis3=run_axis3,
                n_judges=1,
                output_dir=str(LOGS_DIR),
            ))

        # Use the S3 trial as the primary report (fallback: last trial)
        trial = next((t for t in scored if t.scenario == "S3"), scored[-1]) if scored else None
        if trial is None:
            return {"status": "error", "error": "no trials returned", "trial_label": trial_label}

        logger.info("Trial %s outcome: %s", trial.trial_id, trial.outcome)
        logger.info(
            "  planting_path=%s, R_path=%s, I_strict=%s, S_hat=%s, F=%s",
            trial.planting_path, trial.R_path, trial.I_strict, trial.S_hat, trial.F,
        )
        logger.info(
            "  compaction=%s, cf_reinjected=%s",
            trial.compaction_during_session_b, trial.citation_force_reinjected,
        )

        return {
            "trial_id": trial.trial_id,
            "outcome": trial.outcome,
            "planting_path": trial.planting_path,
            "R_path": trial.R_path,
            "I_strict": trial.I_strict,
            "I_empty_return": trial.I_empty_return,
            "S_hat": trial.S_hat,
            "F": trial.F,
            "D": trial.D,
            "compaction": trial.compaction_during_session_b,
            "cf_reinjected": trial.citation_force_reinjected,
            "status": "ok",
        }

    except Exception as exc:
        logger.exception("Trial failed: %s", exc)
        return {"status": "error", "error": str(exc), "trial_label": trial_label}

    finally:
        if citation_force_condition != "C0":
            remove_citation_force_from_bootstrap()


async def main():
    """Run all pilot trials."""
    results = []

    # Pilot 1: screenshot, standard grounding trial — verify S1 filler reads like Alex's real tasks
    r1 = await run_pilot_trial(
        plant_type="screenshot",
        reference_style="definite",
        history_depth=2,
        artifact_idx=0,
        trial_label="pilot_1_screenshot",
    )
    results.append(("pilot_1", r1))

    # Pilot 2: chart_image with temporal reference — verify chart path and diverse filler
    r2 = await run_pilot_trial(
        plant_type="chart_image",
        reference_style="temporal",
        history_depth=4,
        artifact_idx=0,
        trial_label="pilot_2_chart",
    )
    results.append(("pilot_2", r2))

    # Pilot 3: screenshot with demonstrative reference + CitationForce C1
    r3 = await run_pilot_trial(
        plant_type="screenshot",
        reference_style="demonstrative",
        history_depth=0,
        artifact_idx=1,
        citation_force_condition="C1",
        trial_label="pilot_3_screenshot_cf",
    )
    results.append(("pilot_3", r3))

    # Print summary
    print("\n" + "=" * 60)
    print("PILOT SUMMARY")
    print("=" * 60)
    for label, r in results:
        if r.get("status") == "ok":
            print(f"{label}: {r['outcome']}")
            print(f"  planting={r['planting_path']} R={r['R_path']} "
                  f"I={r['I_strict']} S={r['S_hat']} F={r['F']} D={r['D']}")
        else:
            print(f"{label}: ERROR — {r.get('error')}")

    print("\nVERIFICATION CHECKS:")
    p1 = results[0][1]
    if p1.get("status") == "ok":
        checks = [
            ("pilot_1 planting_path != none", p1["planting_path"] != "none"),
            ("pilot_1 R_path is valid", p1["R_path"] in ("R_boot", "R_tool", "R_context", "heuristic_R_context", "R_none")),
        ]
        for name, passed in checks:
            print(f"  {'✓' if passed else '✗'} {name}")

    trials = load_trials(LOGS_DIR)
    print(f"\nTotal trials saved: {len(trials)}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
