"""
Smoke test: verify OpenClaw-native compaction pipeline.

Tests that the native compaction pipeline works end-to-end:
  1. Create agent with context_window_override=45000 (trigger target ~25k)
  2. Plant artifact, inject filler turns
  3. Hard assert: sessions.json.compactionCount >= 1
  4. Hard assert: get_effective_input_tokens() returns non-None
  5. Hard assert: artifact found in workspace memory (native memoryFlush or fallback)
  6. Diagnostic: log JSONL compaction-related customType entries
  7. Record boundary_input_tokens and verify approximately 25k (within tolerance)
  8. Record whether manual_flush_fallback_used was needed
  9. Send query, verify R_path classification

Usage:
    cd MIRAGE
    python -m harness.run_native_compaction_smoke
    python -m harness.run_native_compaction_smoke --model shubiaobiao/gpt-5
"""

from __future__ import annotations

import argparse
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
from client.openclaw_client import OpenClawClient
from constants import VALID_R_PATHS, RESERVE_TOKENS_FLOOR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LOGS_DIR      = Path(__file__).parent.parent / "logs" / "trials"
MANIFEST_PATH = Path(__file__).parent.parent / "data" / "generated_images" / "manifest.json"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Native trigger target: ~25k tokens → contextWindow = 25000 + 20000 = 45000
TRIGGER_TARGET = 25_000
CONTEXT_WINDOW_OVERRIDE = TRIGGER_TARGET + RESERVE_TOKENS_FLOOR


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
        history_depth=0,
        reference_style="definite",
        seed=seed,
        context_threshold=TRIGGER_TARGET,
        citation_force_condition="C0",
    )

    if spec.content_query:
        query = make_grounded_query("definite", plant_type, spec.content_query, seed=seed)
    else:
        query = make_reference_query("definite", plant_type, spec.artifact_id, seed=seed)

    logger.info("Artifact: %s", spec.artifact_id)
    logger.info("Query: %s", query)
    logger.info("Trigger target: ~%dk tokens (contextWindow=%d)", TRIGGER_TARGET // 1000, CONTEXT_WINDOW_OVERRIDE)
    logger.info("Scenarios: S2 + S3 (native compaction)")

    trials = await build_trial(
        config=config,
        plant_prompt=spec.plant_prompt_ai,
        artifact_id=spec.artifact_id,
        artifact_ocr_keywords=spec.artifact_ocr_keywords,
        query=query,
        output_dir=LOGS_DIR,
        scenarios=["S2", "S3"],
        image_path=spec.image_path,
        retrieval_model=model,
        context_threshold=TRIGGER_TARGET,
    )

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
    print("NATIVE COMPACTION SMOKE RESULTS")
    print("=" * 60)

    for scenario in ("S2", "S3"):
        t = scored.get(scenario)
        if t is None:
            print(f"\n[{scenario}] NOT RUN")
            continue
        print(f"\n[{scenario}] {t.trial_id}")
        print(f"  planting_path              = {t.planting_path}")
        print(f"  R_path                     = {t.R_path}")
        print(f"  native_compaction_count    = {t.native_compaction_count}")
        print(f"  boundary_input_tokens      = {t.boundary_input_tokens}")
        print(f"  native_context_window      = {t.native_context_window}")
        print(f"  manual_flush_fallback_used = {t.manual_flush_fallback_used}")
        print(f"  compaction_triggered       = {t.compaction_during_session_b}")
        print(f"  active_context_view        = {len(t.active_context_view)} entries")
        print(f"  outcome                    = {t.outcome}")
        print(f"  response (first 200)       = {t.response_text[:200]!r}")

    # -----------------------------------------------------------------------
    # Verification checks (native compaction pipeline)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VERIFICATION CHECKS (native compaction pipeline)")
    print("=" * 60)

    checks = []
    s2 = scored.get("S2")
    s3 = scored.get("S3")

    # Check 1: Native compaction fired (hard assert)
    if s2:
        checks.append(("S2: native_compaction_count >= 1 (native compaction fired)",
                        (s2.native_compaction_count or 0) >= 1))

    # Check 2: get_effective_input_tokens() returned non-None (hard assert)
    if s2:
        checks.append(("S2: boundary_input_tokens is not None (provider reports input + cacheRead)",
                        s2.boundary_input_tokens is not None))

    # Check 3: Artifact found in workspace memory (hard assert)
    if s2:
        checks.append(("S2: planting_path != none (artifact persisted in memory)",
                        s2.planting_path != "none"))

    # Check 4: boundary_input_tokens approximately target (within 50% tolerance)
    if s2 and s2.boundary_input_tokens:
        lower = TRIGGER_TARGET * 0.5
        upper = TRIGGER_TARGET * 2.0
        in_range = lower <= s2.boundary_input_tokens <= upper
        checks.append((
            f"S2: boundary_input_tokens ~{TRIGGER_TARGET//1000}k "
            f"(actual={s2.boundary_input_tokens:,}, range={lower:,.0f}-{upper:,.0f})",
            in_range,
        ))

    # Check 5: manual_flush_fallback_used is recorded (not None)
    if s2:
        checks.append(("S2: manual_flush_fallback_used is bool (explicitly recorded)",
                        isinstance(s2.manual_flush_fallback_used, bool)))

    # Check 6: native_context_window matches expected override
    if s2:
        checks.append((
            f"S2: native_context_window == {CONTEXT_WINDOW_OVERRIDE} (benchmark-registered)",
            s2.native_context_window == CONTEXT_WINDOW_OVERRIDE,
        ))

    # Check 7: R_path values are valid
    for scenario, t in scored.items():
        checks.append((f"{scenario}: R_path is valid enum ({t.R_path})",
                        t.R_path in VALID_R_PATHS))

    # Diagnostic: log JSONL compaction entries (not asserted)
    print("\n  [DIAGNOSTIC] JSONL compaction-related entries (not asserted):")
    if s2:
        try:
            client = OpenClawClient(agent_id=s2.workspace_profile)
            jsonl = client.get_session_jsonl(s2.session_b_id)
            compaction_entries = [
                e for e in jsonl
                if e.get("customType", "") in ("compaction", "compact", "context-compaction")
                or (e.get("type") == "custom" and "compact" in str(e.get("data", {})).lower())
            ]
            print(f"    Found {len(compaction_entries)} compaction-related JSONL entries")
            for ce in compaction_entries[:3]:
                print(f"    - type={ce.get('type')} customType={ce.get('customType')}")
        except Exception as e:
            print(f"    Could not read JSONL: {e}")

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)

    print()
    for name, ok in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}")

    print(f"\n{passed}/{total} checks passed")

    if passed < total:
        print("\nFAILED checks indicate native compaction pipeline issues")
        sys.exit(1)
    else:
        print("\nAll checks passed — native compaction pipeline verified")

    print(f"\nTotal trials saved to {LOGS_DIR}: {len(load_trials(LOGS_DIR))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Native compaction smoke test")
    parser.add_argument("--model", type=str, default=None, help="Model to test (e.g. shubiaobiao/gpt-5)")
    args = parser.parse_args()
    asyncio.run(main(model=args.model))
