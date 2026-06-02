"""
Smoke test: run S1 + S2 + S3 for 1 manifest artifact, verify v7 architecture.

Checks (native compaction pipeline):
  1.  Planting persisted artifact (planting_path != "none")
  2.  S2 native_compaction_count >= 1 (native compaction fired)
  3.  S2 active_context_view is non-empty (post-compaction context extracted, not [])
  4.  S2 boundary_input_tokens > 0 (provider-reported input at boundary)
  4b. S2 post_hash captured
  4c. S2 manual_flush_fallback_used is recorded (bool)
  5.  S3 startup_exposure_log is a list (bootstrap exposure recorded)
  6.  All R_path values are valid enum members
  7.  S1 R_path == R_context (artifact still in context window)
  8.  S2/S3 R_path in {R_context, R_boot, R_tool, R_none}

Usage:
    cd GroundingBench
    python -m harness.run_s1_s2_s3_smoke
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
from constants import VALID_R_PATHS, EXPERIMENT_MODEL_REGISTRY, S2_CONTEXT_THRESHOLD, RESERVE_TOKENS_FLOOR
from client.usage_proxy import UsageProxy

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


async def main(model: str | None = None):
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
    logger.info("Model: %s", model or "gateway default")
    logger.info("Scenarios: S1 + S2 + S3")

    # Start the usage-fix proxy once for the lifetime of this process.
    # The proxy patches openclaw.json so the gateway daemon routes shubiaobiao
    # traffic through it for every subsequently-created agent.
    # Dynamic port avoids collisions when multiple experiment processes run in parallel.
    # contextWindow override ensures compaction fires at the benchmark target.
    needs_proxy = any(
        spec.get("needsUsageProxy") for spec in EXPERIMENT_MODEL_REGISTRY.values()
    )
    context_window = S2_CONTEXT_THRESHOLD + RESERVE_TOKENS_FLOOR
    proxy_ctx = UsageProxy(context_window_override=context_window) if needs_proxy else None
    if proxy_ctx is not None:
        proxy_ctx.__enter__()
        logger.info(
            "Usage-fix proxy on port %d (openclaw.json patched, contextWindow=%d)",
            proxy_ctx.port, context_window,
        )

    try:
        trials = await build_trial(
            config=config,
            plant_prompt=spec.plant_prompt_ai,
            artifact_id=spec.artifact_id,
            artifact_ocr_keywords=spec.artifact_ocr_keywords,
            query=query,
            output_dir=LOGS_DIR,
            scenarios=["S1", "S2", "S3"],
            image_path=spec.image_path,
            retrieval_model=model,
        )
    finally:
        if proxy_ctx is not None:
            proxy_ctx.__exit__(None, None, None)

    run_axis3 = bool(os.environ.get("ANTHROPIC_API_KEY"))
    scored = {}
    for trial in trials:
        t = score_trial(
            trial,
            artifact_repr=spec.artifact_repr,
            run_axis3=run_axis3,
            n_judges=1,
            output_dir=str(LOGS_DIR),
        )
        scored[t.scenario] = t

    print("\n" + "=" * 60)
    print("S1 + S2 + S3 SMOKE RESULTS")
    print("=" * 60)

    for scenario in ("S1", "S2", "S3"):
        t = scored.get(scenario)
        if t is None:
            print(f"\n[{scenario}] NOT RUN")
            continue
        print(f"\n[{scenario}] {t.trial_id}")
        print(f"  planting_path              = {t.planting_path}")
        print(f"  R_path                     = {t.R_path}")
        print(f"  compaction_triggered       = {t.compaction_during_session_b}")
        print(f"  native_compaction_count    = {t.native_compaction_count}")
        print(f"  boundary_input_tokens      = {t.boundary_input_tokens}")
        print(f"  native_context_window      = {t.native_context_window}")
        print(f"  manual_flush_fallback_used = {t.manual_flush_fallback_used}")
        print(f"  active_context_view        = {len(t.active_context_view)} entries")
        print(f"  startup_exposure_log       = {len(t.startup_exposure_log)} entries")
        print(f"  post_hash                  = {t.post_hash}")
        print(f"  outcome                    = {t.outcome}")
        print(f"  response (first 200)       = {t.response_text[:200]!r}")

    # -----------------------------------------------------------------------
    # Verification checks (per v7 spec)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VERIFICATION CHECKS (v7 architecture)")
    print("=" * 60)

    checks = []

    s1 = scored.get("S1")
    s2 = scored.get("S2")
    s3 = scored.get("S3")

    # Check 1: Planting persisted
    if s1:
        checks.append(("Planting: artifact persisted (planting_path != none)",
                        s1.planting_path != "none"))

    # Check 2: S2 native compaction fired (compactionCount >= 1)
    if s2:
        checks.append(("S2: native_compaction_count >= 1 (native compaction fired)",
                        (s2.native_compaction_count or 0) >= 1))

    # Check 3: S2 active_context_view populated (key fix — was always [])
    if s2:
        checks.append(("S2: active_context_view is non-empty (post-compaction context extracted)",
                        len(s2.active_context_view) > 0))

    # Check 4: S2 boundary_input_tokens recorded
    if s2:
        checks.append(("S2: boundary_input_tokens > 0 (provider-reported input at boundary)",
                        (s2.boundary_input_tokens or 0) > 0))

    # Check 4b: S2 has a non-None post_hash (compaction snapshot was captured).
    if s2:
        checks.append(("S2: post_hash captured (compaction snapshot recorded)",
                        s2.post_hash is not None))
    # S2 and S3 must share the same pre-compaction baseline.
    if s2 and s3:
        checks.append(("S2/S3: pre_hash matches (same pre-compaction baseline)",
                        s2.pre_hash == s3.pre_hash))

    # Check 4c: manual_flush_fallback_used is recorded (True or False — never None)
    if s2:
        checks.append(("S2: manual_flush_fallback_used is recorded (bool, not None)",
                        isinstance(s2.manual_flush_fallback_used, bool)))

    # Check 5: S3 startup_exposure_log is a list (may be empty if no bootstrap)
    if s3:
        checks.append(("S3: startup_exposure_log is a list",
                        isinstance(s3.startup_exposure_log, list)))

    # Check 6: All R_paths are valid
    for scenario, t in scored.items():
        checks.append((f"{scenario}: R_path is valid enum ({t.R_path})",
                        t.R_path in VALID_R_PATHS))

    # Check 7: S1 R_path ∈ {R_context, heuristic_R_context} (artifact still in context window).
    # chart_image → exact R_context; screenshot → heuristic_R_context (image observability limit).
    if s1:
        checks.append(("S1: R_path indicates in-context access (R_context or heuristic_R_context)",
                        s1.R_path in ("R_context", "heuristic_R_context")))

    # Check 8: S2/S3 R_paths in expected set
    for scenario, t in [("S2", s2), ("S3", s3)]:
        if t:
            checks.append((f"{scenario}: R_path in valid set",
                            t.R_path in VALID_R_PATHS))

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)

    for name, ok in checks:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name}")

    print(f"\n{passed}/{total} checks passed")

    if passed < total:
        print("\nFAILED checks indicate harness issues — review trial_runner.py")
    else:
        print("\nAll checks passed — S1/S2/S3 pipeline verified (S2 compaction snapshot + diverging S3 workspace)")

    print(f"\nTotal trials saved to {LOGS_DIR}: {len(load_trials(LOGS_DIR))}")


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="S1+S2+S3 smoke test")
    _parser.add_argument("--model", type=str, default=None, help="Model for retrieval agent")
    _args = _parser.parse_args()
    asyncio.run(main(model=_args.model))
