"""
Smoke test: run S1 + S2 for 1 manifest artifact with DeepSeek backbone.

Usage:
    cd MIRAGE
    python -m harness.run_s1_s2_smoke
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LOGS_DIR      = Path(__file__).parent.parent / "logs" / "trials"
MANIFEST_PATH = Path(__file__).parent.parent / "data" / "generated_images" / "manifest.json"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _load_first_entry(plant_type: str) -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        entries = json.load(f)
    pool = [e for e in entries if e.get("plant_type") == plant_type and e.get("image_exists")]
    if not pool:
        raise ValueError(f"No image-valid entries for plant_type='{plant_type}'")
    return pool[0]


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


async def main():
    plant_type = "screenshot"
    entry = _load_first_entry(plant_type)
    spec = _entry_to_spec(entry)

    seed = abs(hash((plant_type, 0, 2, "definite", "C0"))) % 100_000

    config = TrialConfig(
        plant_type=plant_type,
        history_depth=2,
        reference_style="definite",
        seed=seed,
        citation_force_condition="C0",
    )

    if spec.content_query:
        query = make_grounded_query("definite", plant_type, spec.content_query, seed=seed)
    else:
        query = make_reference_query("definite", plant_type, spec.artifact_id, seed=seed)

    logger.info("Artifact: %s", spec.artifact_id)
    logger.info("Query: %s", query)
    logger.info("Scenarios: S1 + S2")

    trials = await build_trial(
        config=config,
        plant_prompt=spec.plant_prompt_ai,
        artifact_id=spec.artifact_id,
        artifact_ocr_keywords=spec.artifact_ocr_keywords,
        query=query,
        output_dir=LOGS_DIR,
        scenarios=["S1", "S2"],
        image_path=spec.image_path,
    )

    run_axis3 = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print("\n" + "=" * 60)
    print("S1 + S2 SMOKE RESULTS")
    print("=" * 60)

    for trial in trials:
        trial = score_trial(
            trial,
            artifact_repr=spec.artifact_repr,
            run_axis3=run_axis3,
            n_judges=1,
            output_dir=str(LOGS_DIR),
        )
        print(f"\n[{trial.scenario}] {trial.trial_id}")
        print(f"  planting_path        = {trial.planting_path}")
        print(f"  R_path               = {trial.R_path}")
        print(f"  compaction_triggered = {trial.compaction_during_session_b}")
        print(f"  response (first 200) = {trial.response_text[:200]!r}")
        print(f"  outcome              = {trial.outcome}")

    print(f"\nTotal trials saved: {len(load_trials(LOGS_DIR))}")


if __name__ == "__main__":
    asyncio.run(main())
