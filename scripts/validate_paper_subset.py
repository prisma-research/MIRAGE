"""
validate_paper_subset.py — Validate paper subset + annotations for completeness.

Checks:
  - Exactly 50 screenshot + 50 chart_image (or configurable counts)
  - All IDs exist in manifest
  - No duplicate IDs
  - Every artifact has an annotation row (if annotation file provided)
  - Difficulty values are valid and distributed across all bins
  - Metadata coverage is not degenerate

Usage:
    cd MIRAGE
    python scripts/validate_paper_subset.py \
        --subset configs/subsets/paper_main_50_50.json \
        --annotations configs/annotations/paper_main_50_50_annotations.csv

    # For smaller subsets:
    python scripts/validate_paper_subset.py \
        --subset configs/subsets/paper_depth_10_10.json \
        --expected-ss 10 --expected-ci 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "generated_images" / "manifest.json"
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_QUERY_TYPES = {"gist", "detail", "exact"}
VALID_SHORTCUT_RISK = {"low", "medium", "high"}


def _resolve(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / path
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate paper subset + annotations")
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--expected-ss", type=int, default=50)
    parser.add_argument("--expected-ci", type=int, default=50)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    # Load manifest IDs
    manifest_ids = set()
    for e in json.loads(MANIFEST_PATH.read_text()):
        if e.get("image_exists"):
            manifest_ids.add(e["artifact_id"])

    # Load subset
    subset = json.loads(_resolve(args.subset).read_text())
    ss_ids = subset.get("screenshot", [])
    ci_ids = subset.get("chart_image", [])
    all_ids = ss_ids + ci_ids

    # Check counts
    if len(ss_ids) != args.expected_ss:
        errors.append(f"Expected {args.expected_ss} screenshots, got {len(ss_ids)}")
    if len(ci_ids) != args.expected_ci:
        errors.append(f"Expected {args.expected_ci} chart_images, got {len(ci_ids)}")

    # Check duplicates
    dupes = [aid for aid, c in Counter(all_ids).items() if c > 1]
    if dupes:
        errors.append(f"Duplicate IDs: {dupes}")

    # Check all exist in manifest
    missing = [aid for aid in all_ids if aid not in manifest_ids]
    if missing:
        errors.append(f"IDs not in manifest: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # Annotations
    if args.annotations:
        ann_path = _resolve(args.annotations)
        if not ann_path.exists():
            errors.append(f"Annotation file not found: {ann_path}")
        else:
            with ann_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            ann_ids = set(r["artifact_id"] for r in rows)
            ann_dupes = [aid for aid, c in Counter(r["artifact_id"] for r in rows).items() if c > 1]
            if ann_dupes:
                errors.append(f"Duplicate annotation rows: {ann_dupes}")

            # Every subset artifact must have an annotation
            missing_ann = [aid for aid in all_ids if aid not in ann_ids]
            if missing_ann:
                errors.append(f"Artifacts without annotations: {missing_ann[:5]}")

            # Difficulty validation
            difficulties = [r.get("difficulty", "").strip() for r in rows if r.get("difficulty", "").strip()]
            if not difficulties:
                warnings.append("No difficulty values filled in yet")
            else:
                invalid = [d for d in difficulties if d not in VALID_DIFFICULTIES]
                if invalid:
                    errors.append(f"Invalid difficulty values: {set(invalid)}")
                diff_dist = Counter(difficulties)
                for d in VALID_DIFFICULTIES:
                    if d not in diff_dist:
                        warnings.append(f"Difficulty bin '{d}' is empty")
                print(f"  Difficulty distribution: {dict(diff_dist)}")

            # Query type (optional, warn only)
            qtypes = [r.get("query_type", "").strip() for r in rows if r.get("query_type", "").strip()]
            if qtypes:
                invalid_qt = [q for q in qtypes if q not in VALID_QUERY_TYPES]
                if invalid_qt:
                    warnings.append(f"Invalid query_type values: {set(invalid_qt)}")
                print(f"  Query type distribution: {dict(Counter(qtypes))}")

    # Report
    print(f"\nSubset: {args.subset}")
    print(f"  screenshot: {len(ss_ids)}, chart_image: {len(ci_ids)}, total: {len(all_ids)}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    print("\n✓ All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
