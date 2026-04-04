#!/usr/bin/env python3
"""Select curated 50+50 paper subset from the manifest.

Produces:
  configs/subsets/paper_main_50_50.json   (50 screenshot + 50 chart_image)
  configs/subsets/paper_depth_10_10.json  (first 10 of each from main)
  configs/subsets/paper_s3_10_10.json     (last 10 of each from main)
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from datetime import date
from pathlib import Path

SEED = 42
MANIFEST = Path(__file__).resolve().parent.parent / "data" / "generated_images" / "manifest.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "configs" / "subsets"

N_SCREENSHOT = 50
N_CHART = 50
MIN_PER_PLATFORM = 2


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except (ValueError, AttributeError):
        return False


def select_screenshots(entries: list[dict], n: int, rng: random.Random) -> list[str]:
    """Stratified sample across platforms, balanced icon/text within each."""
    # Group by platform
    by_platform: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        plat = e["metadata"]["platform"]
        dtype = e["metadata"]["data_type"]
        by_platform[plat][dtype].append(e["artifact_id"])

    platforms = sorted(by_platform.keys())
    pool_sizes = {p: sum(len(v) for v in by_platform[p].values()) for p in platforms}
    total_pool = sum(pool_sizes.values())

    # Proportional allocation with minimum guarantee
    raw_alloc = {p: max(MIN_PER_PLATFORM, round(n * pool_sizes[p] / total_pool)) for p in platforms}

    # Adjust to hit exactly n
    allocated = sum(raw_alloc.values())
    if allocated != n:
        diff = n - allocated
        # Sort platforms by pool size descending to add/remove from largest first
        ranked = sorted(platforms, key=lambda p: pool_sizes[p], reverse=True)
        idx = 0
        while diff != 0:
            p = ranked[idx % len(ranked)]
            if diff > 0:
                raw_alloc[p] += 1
                diff -= 1
            elif diff < 0 and raw_alloc[p] > MIN_PER_PLATFORM:
                raw_alloc[p] -= 1
                diff += 1
            idx += 1

    selected: list[str] = []
    for plat in platforms:
        k = raw_alloc[plat]
        icons = list(by_platform[plat].get("icon", []))
        texts = list(by_platform[plat].get("text", []))
        rng.shuffle(icons)
        rng.shuffle(texts)

        # Balance icon/text roughly 50/50
        n_icon = k // 2
        n_text = k - n_icon

        # If one pool is too small, take more from the other
        if n_icon > len(icons):
            n_icon = len(icons)
            n_text = k - n_icon
        if n_text > len(texts):
            n_text = len(texts)
            n_icon = k - n_text

        picked = icons[:n_icon] + texts[:n_text]
        selected.extend(picked)

    # Final sort for stability
    selected.sort()
    return selected


def select_charts(entries: list[dict], n: int, rng: random.Random) -> list[str]:
    """Random sample with variety in answer types (numeric vs text)."""
    numeric = [e for e in entries if _is_numeric(str(e["metadata"].get("answer", "")))]
    text_ans = [e for e in entries if not _is_numeric(str(e["metadata"].get("answer", "")))]

    rng.shuffle(numeric)
    rng.shuffle(text_ans)

    # Proportion in pool
    total = len(numeric) + len(text_ans)
    n_numeric = round(n * len(numeric) / total)
    n_text = n - n_numeric

    # Ensure we don't exceed pool
    if n_text > len(text_ans):
        n_text = len(text_ans)
        n_numeric = n - n_text
    if n_numeric > len(numeric):
        n_numeric = len(numeric)
        n_text = n - n_numeric

    picked = [e["artifact_id"] for e in numeric[:n_numeric]] + \
             [e["artifact_id"] for e in text_ans[:n_text]]
    picked.sort()
    return picked


def write_subset(path: Path, desc: str, rule: str, ss: list[str], ci: list[str]) -> None:
    obj = {
        "description": desc,
        "created_at": date.today().isoformat(),
        "selection_rule": rule,
        "primary_reference_style": "definite",
        "screenshot": ss,
        "chart_image": ci,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    print(f"  Wrote {path}  ({len(ss)} screenshot + {len(ci)} chart_image)")


def main() -> None:
    with open(MANIFEST) as f:
        manifest = json.load(f)

    screenshots = [e for e in manifest if e["plant_type"] == "screenshot"]
    charts = [e for e in manifest if e["plant_type"] == "chart_image"]
    print(f"Manifest: {len(screenshots)} screenshots, {len(charts)} chart_images")

    rng = random.Random(SEED)

    ss_ids = select_screenshots(screenshots, N_SCREENSHOT, rng)
    ci_ids = select_charts(charts, N_CHART, rng)

    print(f"\nSelected: {len(ss_ids)} screenshots, {len(ci_ids)} chart_images")

    # Verify platform distribution
    ss_lookup = {e["artifact_id"]: e for e in screenshots}
    from collections import Counter
    plat_counts = Counter(ss_lookup[aid]["metadata"]["platform"] for aid in ss_ids)
    dtype_counts = Counter(ss_lookup[aid]["metadata"]["data_type"] for aid in ss_ids)
    print(f"  Platforms: {dict(sorted(plat_counts.items()))}")
    print(f"  Data types: {dict(sorted(dtype_counts.items()))}")

    ci_lookup = {e["artifact_id"]: e for e in charts}
    num_count = sum(1 for aid in ci_ids if _is_numeric(str(ci_lookup[aid]["metadata"].get("answer", ""))))
    print(f"  Chart answers: numeric={num_count}, text={len(ci_ids) - num_count}")

    # --- Main subset ---
    write_subset(
        OUT_DIR / "paper_main_50_50.json",
        "Paper main benchmark -- 50 screenshot + 50 chart_image",
        "stratified by platform/data_type (screenshot) and random (chart), seed=42",
        ss_ids,
        ci_ids,
    )

    # --- Depth subset (first 10 of each) ---
    write_subset(
        OUT_DIR / "paper_depth_10_10.json",
        "Paper depth subset -- 10 screenshot + 10 chart_image (first 10 from main)",
        "first 10 from paper_main_50_50 per category",
        ss_ids[:10],
        ci_ids[:10],
    )

    # --- S3 subset (last 10 of each) ---
    write_subset(
        OUT_DIR / "paper_s3_10_10.json",
        "Paper S3 subset -- 10 screenshot + 10 chart_image (last 10 from main)",
        "last 10 from paper_main_50_50 per category",
        ss_ids[-10:],
        ci_ids[-10:],
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
