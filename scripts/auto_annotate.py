"""
auto_annotate.py — Heuristic auto-annotation for paper subset artifacts.

Populates difficulty, query_type, and shortcut_risk in the annotation CSV
based on artifact metadata signals.

Heuristics:
  Screenshots:
    - difficulty: based on instruction word count + icon vs text
      (icons are harder to describe from memory, short instructions are vaguer)
    - query_type: gist (short instruction ≤3 words), detail (4-6), exact (>6)
    - shortcut_risk: icon→low (hard to guess), text+short→high (generic labels)

  Charts:
    - difficulty: based on answer precision and query complexity
      (exact numeric with decimals → hard, simple percentage → easy)
    - query_type: gist (asks about trend/comparison), detail (specific value),
      exact (precise number)
    - shortcut_risk: simple round numbers → high, precise decimals → low

Usage:
    cd GroundingBench
    python scripts/auto_annotate.py \
        --subset configs/subsets/paper_main_50_50.json \
        --annotations configs/annotations/paper_main_50_50_annotations.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "generated_images" / "manifest.json"


def _is_numeric(val: str) -> bool:
    try:
        float(str(val).replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def _answer_precision(answer: str) -> int:
    """Count decimal places in a numeric answer."""
    answer = str(answer).replace(",", "")
    if "." in answer:
        return len(answer.split(".")[-1])
    return 0


def _word_count(text: str) -> int:
    return len(text.strip().split())


def annotate_screenshot(meta: dict) -> dict:
    instruction = meta.get("instruction", "")
    data_type = meta.get("data_type", "text")
    wc = _word_count(instruction)

    # difficulty
    if data_type == "icon" and wc <= 4:
        difficulty = "hard"
    elif data_type == "icon" and wc <= 7:
        difficulty = "medium"
    elif data_type == "icon":
        difficulty = "easy"
    elif wc <= 3:
        difficulty = "medium"
    elif wc <= 6:
        difficulty = "easy"
    else:
        difficulty = "easy"

    # query_type
    if wc <= 3:
        query_type = "gist"
    elif wc <= 6:
        query_type = "detail"
    else:
        query_type = "exact"

    # shortcut_risk: text with short instructions → high (generic UI labels are guessable)
    if data_type == "text" and wc <= 3:
        shortcut_risk = "high"
    elif data_type == "icon":
        shortcut_risk = "low"
    else:
        shortcut_risk = "medium"

    return {"difficulty": difficulty, "query_type": query_type, "shortcut_risk": shortcut_risk}


def annotate_chart(meta: dict) -> dict:
    query = meta.get("query", "")
    answer = str(meta.get("answer", ""))
    wc = _word_count(query)
    is_num = _is_numeric(answer)
    precision = _answer_precision(answer)

    # difficulty
    if is_num and precision >= 2:
        difficulty = "hard"
    elif is_num and precision == 1:
        difficulty = "medium"
    elif is_num and int(float(answer.replace(",", ""))) % 10 == 0:
        difficulty = "easy"
    elif not is_num:
        difficulty = "medium"
    else:
        difficulty = "medium"

    # query_type
    trend_words = {"trend", "increase", "decrease", "change", "compare", "difference", "more", "less", "most", "least"}
    query_lower = query.lower()
    if any(w in query_lower for w in trend_words):
        query_type = "gist"
    elif wc <= 10:
        query_type = "detail"
    else:
        query_type = "exact"

    # shortcut_risk: round easy numbers are guessable
    if is_num and precision == 0 and float(answer.replace(",", "")) % 10 == 0:
        shortcut_risk = "high"
    elif is_num and precision >= 2:
        shortcut_risk = "low"
    else:
        shortcut_risk = "medium"

    return {"difficulty": difficulty, "query_type": query_type, "shortcut_risk": shortcut_risk}


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate paper subset")
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--annotations", type=str, required=True)
    args = parser.parse_args()

    # Load manifest
    entries_by_id = {}
    for e in json.loads(MANIFEST_PATH.read_text()):
        if e.get("image_exists"):
            entries_by_id[e["artifact_id"]] = e

    # Load existing annotation CSV
    ann_path = Path(args.annotations)
    if not ann_path.is_absolute():
        ann_path = Path(__file__).parent.parent / ann_path
    with ann_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Annotate
    updated = 0
    for row in rows:
        aid = row["artifact_id"]
        entry = entries_by_id.get(aid)
        if not entry:
            continue
        meta = entry.get("metadata", {})
        pt = entry["plant_type"]

        if pt == "screenshot":
            ann = annotate_screenshot(meta)
        else:
            ann = annotate_chart(meta)

        for field in ("difficulty", "query_type", "shortcut_risk"):
            if not row.get(field, "").strip():
                row[field] = ann[field]
                updated += 1

    # Write back
    with ann_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {updated} fields across {len(rows)} rows → {ann_path}")

    # Print distribution
    from collections import Counter
    for field in ("difficulty", "query_type", "shortcut_risk"):
        dist = Counter(r[field] for r in rows if r[field])
        print(f"  {field}: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
