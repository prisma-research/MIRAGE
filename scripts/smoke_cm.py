"""
Smoke test for Cm (mandatory artifact_recall) condition.

Runs a single S3 trial with citation_force_condition="Cm" and checks:
  1. bootstrap_cf_present = True
  2. BOOTSTRAP.md contains "Mandatory Tool Retrieval"
  3. Condition marker is "Cm"
  4. artifact_recall appears in tool_trace (the key behavioral check)
  5. Trial completes without error

Usage:
    cd MIRAGE
    python -m scripts.smoke_cm
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
from harness.trial_log import TrialConfig
from harness.scorer_pipeline import score_trial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LOGS_DIR      = Path(__file__).parent.parent / "logs" / "trials_smoke_cm"
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
    image_path = str(MANIFEST_PATH.parent.parent.parent / raw_path) if raw_path else None
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

    seed = abs(hash((plant_type, 0, 0, "definite", "Cm"))) % 100_000

    config = TrialConfig(
        plant_type=plant_type,
        history_depth=0,
        reference_style="definite",
        seed=seed,
        citation_force_condition="Cm",
    )

    if spec.content_query:
        query = make_grounded_query("definite", plant_type, spec.content_query, seed=seed)
    else:
        query = make_reference_query("definite", plant_type, spec.artifact_id, seed=seed)

    logger.info("=== Cm SMOKE TEST ===")
    logger.info("Artifact: %s", spec.artifact_id)
    logger.info("Query: %s", query)
    logger.info("Condition: Cm (mandatory artifact_recall)")

    trials = await build_trial(
        config=config,
        plant_prompt=spec.plant_prompt_ai,
        artifact_id=spec.artifact_id,
        artifact_ocr_keywords=spec.artifact_ocr_keywords,
        query=query,
        output_dir=LOGS_DIR,
        scenarios=["S3"],
        image_path=spec.image_path,
    )

    run_axis3 = bool(os.environ.get("LLM_API_KEY"))
    scored = []
    for trial in trials:
        t = score_trial(
            trial,
            artifact_repr=spec.artifact_repr,
            run_axis3=run_axis3,
            n_judges=1,
            output_dir=str(LOGS_DIR),
        )
        scored.append(t)

    s3 = next((t for t in scored if t.scenario == "S3"), scored[-1]) if scored else None
    if s3 is None:
        print("ERROR: No S3 trial returned")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Cm SMOKE TEST RESULTS")
    print("=" * 60)

    print(f"\n  trial_id:              {s3.trial_id}")
    print(f"  scenario:              {s3.scenario}")
    print(f"  cf_condition:          {s3.theta.citation_force_condition}")
    print(f"  bootstrap_cf_present:  {s3.bootstrap_cf_present}")
    print(f"  outcome:               {s3.outcome}")
    print(f"  R_path:                {s3.R_path}")
    print(f"  I_strict:              {s3.I_strict}")
    print(f"  D (refusal):           {s3.D}")
    print(f"  response (first 300):  {s3.response_text[:300]!r}")

    # Tool trace analysis
    tool_names = [t.get("tool") or t.get("name", "") for t in s3.tool_trace]
    has_artifact_recall = "artifact_recall" in tool_names
    has_memory_search = "memory_search" in tool_names
    has_memory_get = "memory_get" in tool_names

    print(f"\n  tool_trace ({len(s3.tool_trace)} calls):")
    for t in s3.tool_trace:
        name = t.get("tool") or t.get("name", "?")
        print(f"    - {name}")

    print(f"\n  artifact_recall called:  {has_artifact_recall}")
    print(f"  memory_search called:    {has_memory_search}")
    print(f"  memory_get called:       {has_memory_get}")

    # Checks
    checks = [
        ("Condition is Cm", s3.theta.citation_force_condition == "Cm"),
        ("bootstrap_cf_present = True", s3.bootstrap_cf_present is True),
        ("Trial completed (has outcome)", s3.outcome is not None),
        ("artifact_recall in tool_trace", has_artifact_recall),
        ("memory_search NOT in tool_trace (compliance)", not has_memory_search),
    ]

    print("\n" + "=" * 60)
    print("CHECKS")
    print("=" * 60)
    for name, ok in checks:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name}")

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    print(f"\n{passed}/{total} checks passed")

    if has_artifact_recall:
        print("\n✓ KEY RESULT: Model called artifact_recall under Cm condition!")
    elif has_memory_search:
        print("\n✗ KEY RESULT: Model used memory_search instead of artifact_recall (non-compliant)")
    else:
        print("\n? KEY RESULT: Model used neither artifact_recall nor memory_search")

    # Save summary
    summary = {
        "trial_id": s3.trial_id,
        "condition": s3.theta.citation_force_condition,
        "bootstrap_cf_present": s3.bootstrap_cf_present,
        "outcome": s3.outcome,
        "R_path": s3.R_path,
        "I_strict": s3.I_strict,
        "D": s3.D,
        "artifact_recall_called": has_artifact_recall,
        "memory_search_called": has_memory_search,
        "tool_names": tool_names,
        "checks_passed": passed,
        "checks_total": total,
    }
    summary_path = LOGS_DIR / "smoke_cm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
