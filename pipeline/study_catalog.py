"""
Artifact catalog and pilot-set utilities for the visual-evidence-recovery study.

Builds from:
  - data/generated_images/manifest.json
  - configs/annotations/paper_main_50_50_annotations.csv

Provides:
  - Artifact catalog with visual_type, confusion_group, and study metadata
  - Pilot-set selection with confusion-group coverage
  - Question bank schema validation (no generation — bank is manually authored)

The study question bank is curated at: configs/study/question_bank_curated.json

Usage:
    cd MIRAGE
    python -m pipeline.study_catalog build-catalog
    python -m pipeline.study_catalog select-pilot
    python -m pipeline.study_catalog validate-bank
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

GB_ROOT = Path(__file__).parent.parent

MANIFEST_PATH = GB_ROOT / "data" / "generated_images" / "manifest.json"
ANNOTATIONS_PATH = GB_ROOT / "configs" / "annotations" / "paper_main_50_50_annotations.csv"
CATALOG_OUTPUT = GB_ROOT / "configs" / "study" / "artifact_catalog.json"
PILOT_OUTPUT = GB_ROOT / "configs" / "study" / "pilot_set.json"


# ============================================================================
# visual_type taxonomy
# ============================================================================
#
# Derived from existing metadata. Kept small and defensible.
#
#   standalone_chart  — single chart image (all ChartQA)
#   app_ui            — native application UI (ScreenSpot: ios, android, macos, windows)
#   web_ui            — web-based interface (ScreenSpot: tool, shop, forum, gitlab)
#

_APP_PLATFORMS = {"ios", "android", "macos", "windows"}
_WEB_PLATFORMS = {"tool", "shop", "forum", "gitlab"}


def _classify_visual_type(entry: dict) -> str:
    if entry["plant_type"] == "chart_image":
        return "standalone_chart"
    platform = entry.get("metadata", {}).get("platform", "")
    if platform in _APP_PLATFORMS:
        return "app_ui"
    if platform in _WEB_PLATFORMS:
        return "web_ui"
    return "app_ui"  # default for screenshots without platform


# ============================================================================
# confusion_group assignment
# ============================================================================
#
# Confusion groups partition artifacts so wrong-source probes pair targets
# with plausible distractors from the same group.
#
# For charts (ChartQA): group by semantic domain extracted from the query.
# For screenshots (ScreenSpot): group by platform family.
#

_FINANCIAL_KEYWORDS = {
    "revenue", "sales", "income", "profit", "earnings", "spending",
    "expenditure", "cost", "budget", "market", "gdp", "export", "import",
    "investment", "debt", "tax", "fee", "royalt", "advertising", "ad",
}
_DEMO_KEYWORDS = {
    "population", "people", "subscriber", "user", "member", "employee",
    "worker", "patient", "caregiver", "household", "reader", "fan",
}
_RATE_KEYWORDS = {
    "rate", "percent", "share", "ratio", "proportion", "growth", "inflation",
}


def _classify_confusion_group(entry: dict) -> str:
    if entry["plant_type"] == "chart_image":
        query = entry.get("content_query", "").lower()
        if any(k in query for k in _FINANCIAL_KEYWORDS):
            return "chart_financial"
        if any(k in query for k in _DEMO_KEYWORDS):
            return "chart_demographic"
        if any(k in query for k in _RATE_KEYWORDS):
            return "chart_rate"
        return "chart_other"
    # Screenshots: group by platform family
    platform = entry.get("metadata", {}).get("platform", "")
    if platform in {"ios", "android"}:
        return "ui_mobile"
    if platform in {"macos", "windows"}:
        return "ui_desktop"
    return "ui_web"


# ============================================================================
# Build artifact catalog
# ============================================================================

def build_catalog(
    manifest_path: Path = MANIFEST_PATH,
    annotations_path: Path = ANNOTATIONS_PATH,
) -> list[dict]:
    """Build unified artifact catalog from manifest + annotations."""
    with manifest_path.open() as f:
        manifest = json.load(f)

    # Load annotations if available
    annotations: dict[str, dict] = {}
    if annotations_path.exists():
        with annotations_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotations[row["artifact_id"]] = row

    catalog = []
    for entry in manifest:
        if not entry.get("image_exists"):
            continue

        aid = entry["artifact_id"]
        ann = annotations.get(aid, {})
        meta = entry.get("metadata", {})

        record = {
            "artifact_id": aid,
            "source_dataset": meta.get("dataset", entry["plant_type"]),
            "plant_type": entry["plant_type"],
            "visual_type": _classify_visual_type(entry),
            "confusion_group": _classify_confusion_group(entry),
            "image_path": entry.get("image_path", ""),
            "content_query": entry.get("content_query", ""),
            "seed_answer": str(meta.get("answer", "")),
            "ocr_keywords": entry.get("ocr_keywords", []),
            "plant_prompt": entry.get("plant_prompt", ""),
            "plant_prompt_ai": entry.get("plant_prompt_ai", ""),
            "artifact_repr": entry.get("artifact_repr", ""),
            # From annotations (if available)
            "difficulty": ann.get("difficulty", ""),
            "query_type": ann.get("query_type", ""),
            "shortcut_risk": ann.get("shortcut_risk", ""),
            "chart_answer_type": ann.get("chart_answer_type", ""),
            # Answerability
            "answerability_capability": (
                "numeric_recall" if meta.get("answer", "") and
                str(meta.get("answer", "")).replace(".", "").replace("-", "").isdigit()
                else "descriptive_recall"
            ),
            "metadata": meta,
        }
        catalog.append(record)

    return catalog


# ============================================================================
# Select pilot artifact set
# ============================================================================

def select_pilot_set(
    catalog: list[dict],
    n_charts: int = 4,
    n_screenshots: int = 2,
    seed: int = 42,
) -> dict:
    """Select a small pilot set ensuring confusion-group coverage.

    Returns a dict with:
      - artifacts: list of selected artifact records
      - confusion_groups: mapping from group → artifact_ids
      - rationale: why these were chosen
    """
    rng = random.Random(seed)

    # Filter to annotated artifacts with images and answers
    charts = [a for a in catalog
              if a["plant_type"] == "chart_image"
              and a["seed_answer"]
              and a["difficulty"]]
    screenshots = [a for a in catalog
                   if a["plant_type"] == "screenshot"
                   and a["difficulty"]]

    # Group charts by confusion_group — pick from groups with ≥2 members
    chart_groups: dict[str, list[dict]] = defaultdict(list)
    for c in charts:
        chart_groups[c["confusion_group"]].append(c)

    # Prefer chart_financial (best for wrong-source), then others
    selected_charts = []
    for group_name in ["chart_financial", "chart_demographic", "chart_rate", "chart_other"]:
        group = chart_groups.get(group_name, [])
        if len(group) >= 2 and len(selected_charts) < n_charts:
            rng.shuffle(group)
            # Take pairs
            needed = min(n_charts - len(selected_charts), len(group))
            needed = max(needed, 2)  # at least a pair
            selected_charts.extend(group[:needed])

    selected_charts = selected_charts[:n_charts]

    # Screenshots: pick a same-group pair so wrong-source probes are possible.
    # Group by confusion_group, prefer groups with ≥2 members.
    ss_groups: dict[str, list[dict]] = defaultdict(list)
    for s in screenshots:
        ss_groups[s["confusion_group"]].append(s)

    selected_screenshots = []
    for group_name in ["ui_mobile", "ui_desktop", "ui_web"]:
        group = ss_groups.get(group_name, [])
        if len(group) >= 2 and len(selected_screenshots) < n_screenshots:
            rng.shuffle(group)
            selected_screenshots.extend(group[:n_screenshots - len(selected_screenshots)])

    # Fallback: if no group had ≥2, take from any group
    if len(selected_screenshots) < n_screenshots:
        rng.shuffle(screenshots)
        for s in screenshots:
            if s not in selected_screenshots and len(selected_screenshots) < n_screenshots:
                selected_screenshots.append(s)

    selected = selected_charts + selected_screenshots

    # Build confusion groups from selected
    groups: dict[str, list[str]] = defaultdict(list)
    for a in selected:
        groups[a["confusion_group"]].append(a["artifact_id"])

    return {
        "artifacts": selected,
        "confusion_groups": dict(groups),
        "n_charts": len(selected_charts),
        "n_screenshots": len(selected_screenshots),
        "rationale": (
            f"Selected {len(selected_charts)} charts from annotated set with "
            f"confusion-group coverage (financial pairs for wrong-source probes) + "
            f"{len(selected_screenshots)} screenshots from the same confusion group "
            f"(enabling screenshot wrong-source probes)."
        ),
    }


# ============================================================================
# Question bank validation (no generation — bank is manually authored)
# ============================================================================

def validate_question_bank(bank_path: Path) -> dict:
    """Validate a question bank file. Supports both schemas:
    - Legacy free-form (requires expected_answerability, gold_keywords, ...)
    - Constrained-output (requires response_format, gold_support, gold_source, ...)
    Auto-detects schema from the first question's fields.
    """
    with bank_path.open() as f:
        data = json.load(f)

    qs = data["questions"] if isinstance(data, dict) else data
    errors = []

    # Shared required fields
    _SHARED = [
        "question_id", "family", "target_artifact_id", "visual_type",
        "confusion_group", "prompt", "paraphrase_group_id",
    ]
    # Schema-specific fields
    _FREEFORM_EXTRA = ["expected_answerability", "gold_answer", "gold_keywords", "judge_notes"]
    _CONSTRAINED_EXTRA = ["response_format", "gold_support", "gold_source", "gold_answer", "answer_type"]

    # Auto-detect schema
    is_constrained = bool(qs and "response_format" in qs[0])
    required_fields = _SHARED + (_CONSTRAINED_EXTRA if is_constrained else _FREEFORM_EXTRA)

    families = {"target_present", "wrong_source", "unanswerable"}

    for i, q in enumerate(qs):
        for f in required_fields:
            if f not in q:
                errors.append(f"q[{i}] ({q.get('question_id', '?')}): missing field '{f}'")
        if q.get("family") not in families:
            errors.append(f"q[{i}]: invalid family '{q.get('family')}'")
        if not q.get("prompt", "").strip():
            errors.append(f"q[{i}]: empty prompt")

    from collections import Counter
    by_fam = Counter(q["family"] for q in qs)
    by_art = Counter(q["target_artifact_id"] for q in qs)
    pgs = Counter(q.get("paraphrase_group_id", "") for q in qs)
    unique_prompts = len(set(q["prompt"] for q in qs))

    return {
        "n_questions": len(qs),
        "schema": "constrained" if is_constrained else "freeform",
        "by_family": dict(by_fam),
        "by_artifact": dict(by_art),
        "n_paraphrase_groups": len(pgs),
        "n_unique_prompts": unique_prompts,
        "n_duplicate_prompts": len(qs) - unique_prompts,
        "errors": errors,
    }


CURATED_BANK_PATH = GB_ROOT / "configs" / "study" / "question_bank_curated.json"


# ============================================================================
# CLI
# ============================================================================
# Template-based question generation has been removed from this module.
# The study question bank is now manually authored at:
#   configs/study/question_bank_curated.json
# Use validate-bank to check schema/counts.
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Study artifact catalog utilities. "
        "Question generation is no longer supported here — use the curated bank.",
    )
    sub = parser.add_subparsers(dest="command")

    p1 = sub.add_parser("build-catalog", help="Build artifact catalog from manifest + annotations")
    p1.add_argument("--output", default=str(CATALOG_OUTPUT))

    p2 = sub.add_parser("select-pilot", help="Select pilot artifact set from catalog")
    p2.add_argument("--catalog", default=str(CATALOG_OUTPUT))
    p2.add_argument("--n-charts", type=int, default=4)
    p2.add_argument("--n-screenshots", type=int, default=2)
    p2.add_argument("--output", default=str(PILOT_OUTPUT))
    p2.add_argument("--seed", type=int, default=42)

    p3 = sub.add_parser("validate-bank", help="Validate a curated question bank")
    p3.add_argument("--bank", default=str(CURATED_BANK_PATH))

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "build-catalog":
        catalog = build_catalog()
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(catalog, f, indent=2)
        by_vt = defaultdict(int)
        by_cg = defaultdict(int)
        for a in catalog:
            by_vt[a["visual_type"]] += 1
            by_cg[a["confusion_group"]] += 1
        print(f"Catalog: {len(catalog)} artifacts")
        print(f"  visual_type: {dict(by_vt)}")
        print(f"  confusion_group: {dict(by_cg)}")

    elif args.command == "select-pilot":
        catalog = json.load(open(args.catalog))
        pilot = select_pilot_set(catalog, args.n_charts, args.n_screenshots, args.seed)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(pilot, f, indent=2)
        print(f"Pilot set: {len(pilot['artifacts'])} artifacts")
        print(f"  Confusion groups: {pilot['confusion_groups']}")

    elif args.command == "validate-bank":
        result = validate_question_bank(Path(args.bank))
        print(f"Question bank: {result['n_questions']} questions")
        print(f"  By family: {result['by_family']}")
        print(f"  By artifact: {result['by_artifact']}")
        print(f"  Paraphrase groups: {result['n_paraphrase_groups']}")
        print(f"  Unique prompts: {result['n_unique_prompts']}")
        if result["errors"]:
            print(f"  ERRORS ({len(result['errors'])}):")
            for e in result["errors"][:10]:
                print(f"    {e}")
        else:
            print(f"  No schema errors")


if __name__ == "__main__":
    main()
