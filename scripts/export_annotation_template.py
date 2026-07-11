"""
export_annotation_template.py — Generate annotation CSV template for paper subset.

Reads the manifest and a paper subset JSON, emits a CSV with prefilled metadata
and blank columns for manual annotation (difficulty, query_type, shortcut_risk).

Usage:
    cd MIRAGE
    python scripts/export_annotation_template.py \
        --subset configs/subsets/paper_main_50_50.json \
        --output configs/annotations/paper_main_50_50_annotations.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "generated_images" / "manifest.json"

ANNOTATION_COLUMNS = [
    "artifact_id",
    "plant_type",
    "dataset",
    "platform",
    "data_type",
    "chart_answer_type",
    "difficulty",
    "query_type",
    "shortcut_risk",
    "notes",
]


def _is_numeric(val: str) -> bool:
    try:
        float(str(val).replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Export annotation CSV template")
    parser.add_argument("--subset", type=str, required=True,
                        help="Paper subset JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path")
    args = parser.parse_args()

    # Load manifest
    entries_by_id = {}
    for e in json.loads(MANIFEST_PATH.read_text()):
        if e.get("image_exists"):
            entries_by_id[e["artifact_id"]] = e

    # Load subset
    subset_path = Path(args.subset)
    if not subset_path.is_absolute():
        subset_path = Path(__file__).parent.parent / subset_path
    subset = json.loads(subset_path.read_text())

    # Collect IDs in order
    ids = []
    for pt in ("screenshot", "chart_image"):
        ids.extend(subset.get(pt, []))

    # Build rows
    rows = []
    for aid in ids:
        entry = entries_by_id.get(aid)
        if not entry:
            print(f"WARNING: {aid} not found in manifest")
            continue
        meta = entry.get("metadata", {})
        pt = entry["plant_type"]

        row = {
            "artifact_id": aid,
            "plant_type": pt,
            "dataset": meta.get("dataset", ""),
            "platform": meta.get("platform", "") if pt == "screenshot" else "",
            "data_type": meta.get("data_type", "") if pt == "screenshot" else "",
            "chart_answer_type": (
                "numeric" if pt == "chart_image" and _is_numeric(meta.get("answer", ""))
                else "text" if pt == "chart_image"
                else ""
            ),
            "difficulty": "",
            "query_type": "",
            "shortcut_risk": "",
            "notes": "",
        }
        rows.append(row)

    # Write CSV
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows → {output}")


if __name__ == "__main__":
    main()
