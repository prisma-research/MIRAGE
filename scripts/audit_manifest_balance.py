"""
audit_manifest_balance.py — Summarize manifest metadata for balanced subset selection.

Usage:
    cd MIRAGE
    python scripts/audit_manifest_balance.py
    python scripts/audit_manifest_balance.py --subset configs/subsets/paper_main_50_50.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "generated_images" / "manifest.json"


def load_entries(subset_file: str | None = None) -> list[dict]:
    entries = json.loads(MANIFEST_PATH.read_text())
    valid = [e for e in entries if e.get("image_exists")]

    if subset_file:
        p = Path(subset_file)
        if not p.is_absolute():
            p = Path(__file__).parent.parent / p
        subset = json.loads(p.read_text())
        ids = set()
        for pt in ("screenshot", "chart_image"):
            ids.update(subset.get(pt, []))
        valid = [e for e in valid if e["artifact_id"] in ids]

    return valid


def audit(entries: list[dict]) -> None:
    by_type: dict[str, list[dict]] = {}
    for e in entries:
        by_type.setdefault(e["plant_type"], []).append(e)

    print(f"Total entries: {len(entries)}\n")

    for pt, items in sorted(by_type.items()):
        print(f"=== {pt} ({len(items)}) ===")
        meta_keys = set()
        for e in items:
            meta_keys.update(e.get("metadata", {}).keys())

        if pt == "screenshot":
            platforms = Counter(e["metadata"].get("platform", "?") for e in items)
            data_types = Counter(e["metadata"].get("data_type", "?") for e in items)
            datasets = Counter(e["metadata"].get("dataset", "?") for e in items)

            print(f"  datasets:   {dict(datasets)}")
            print(f"  platforms:  {dict(sorted(platforms.items()))}")
            print(f"  data_types: {dict(data_types)}")

            print(f"\n  platform × data_type:")
            combos = Counter(
                (e["metadata"]["platform"], e["metadata"]["data_type"]) for e in items
            )
            for (p, d), c in sorted(combos.items()):
                print(f"    {p:10s} {d:5s}: {c}")
        else:
            datasets = Counter(e["metadata"].get("dataset", "?") for e in items)
            print(f"  datasets: {dict(datasets)}")
            # Check answer types
            answers = [e["metadata"].get("answer", "") for e in items]
            numeric = sum(1 for a in answers if _is_numeric(a))
            print(f"  answer types: numeric={numeric}, text={len(answers)-numeric}")

        print()


def _is_numeric(val: str) -> bool:
    try:
        float(str(val).replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit manifest balance")
    parser.add_argument("--subset", type=str, default=None,
                        help="Optional subset JSON to filter to")
    args = parser.parse_args()
    entries = load_entries(args.subset)
    audit(entries)


if __name__ == "__main__":
    main()
