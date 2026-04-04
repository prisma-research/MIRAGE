"""
数据集生成脚本：从真实数据集采样图片 + 对话 + manifest。

用法:
    cd MIRAGE
    python -m pipeline.generate_dataset

输出:
    data/generated_images/screenshot/  — GUI/UI/web 截图
    data/generated_images/chart/       — 图表图片
    data/generated_images/manifest.json — 数据集清单（~1,000 条）

数据集来源:
    ScreenSpot  → screenshot   (500)  — GUI/UI/web 截图，含 instruction 注释
    ChartQA     → chart_image  (500)  — 真实图表，含 factual Q&A
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.planters.datasets.screenspot_planter import ScreenSpotPlanter
from pipeline.planters.datasets.chartqa_planter import ChartQAPlanter

N = 500  # samples per dataset
BENCH_ROOT = Path(__file__).resolve().parent.parent  # MIRAGE/


def main() -> int:
    planters = [
        ScreenSpotPlanter(),
        ChartQAPlanter(),
    ]

    manifest = []

    for planter in planters:
        dataset_name = type(planter).__name__
        print(f"\n{'='*60}")
        print(f"生成 {dataset_name} × {N}")
        print(f"{'='*60}")

        try:
            specs = planter.make_plant_specs(n=N, seed=42)
        except Exception as exc:
            print(f"  [ERROR] {dataset_name} failed: {exc}")
            continue

        for spec in specs:
            image_exists = bool(spec.image_path and Path(spec.image_path).exists())
            entry = {
                "artifact_id": spec.artifact_id,
                "plant_type": spec.plant_type,
                "image_path": (
                    str(Path(spec.image_path).relative_to(BENCH_ROOT))
                    if spec.image_path else None
                ),
                "image_exists": image_exists,
                "plant_prompt": spec.plant_prompt,
                "plant_prompt_ai": spec.plant_prompt_ai,
                "artifact_repr": (
                    spec.artifact_repr[:200] + "..."
                    if len(spec.artifact_repr) > 200
                    else spec.artifact_repr
                ),
                "ocr_keywords": spec.artifact_ocr_keywords,
                "content_query": spec.content_query,
                "metadata": spec.metadata,
            }
            manifest.append(entry)

            status = "✅" if image_exists else "❌"
            print(f"  [{status}] {spec.artifact_id}")
            if spec.content_query:
                print(f"      Query: {spec.content_query[:80]}")

    # Save manifest
    manifest_path = Path(__file__).resolve().parent.parent / "data" / "generated_images" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"数据集生成完成！")
    print(f"{'='*60}")

    total = len(manifest)
    ok = sum(1 for e in manifest if e["image_exists"])
    print(f"总数: {total}")
    print(f"图片生成成功: {ok}/{total}")
    print(f"Manifest: {manifest_path}")

    type_counts = Counter(e["plant_type"] for e in manifest)
    for t, c in sorted(type_counts.items()):
        ok_c = sum(1 for e in manifest if e["plant_type"] == t and e["image_exists"])
        print(f"  {t}: {ok_c}/{c}")

    assert "page_image" not in type_counts
    assert "ui_state_image" not in type_counts
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
