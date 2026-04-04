"""
ChartQA → chart_image planter.

HF ID: ahmed-masry/ChartQA
Uses the 'query' field as content_query (simple factual Q&A over real charts).
"""

from __future__ import annotations

from pathlib import Path

from pipeline.planters.base_planter import BasePlanter, PlantSpec
from pipeline.planters.datasets._hf_loader import get_samples, save_image_from_sample, extract_keywords

IMAGES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "generated_images" / "chart"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

HF_ID = "ahmed-masry/ChartQA"
HF_SPLIT = "train"  # 18,294 samples


class ChartQAPlanter(BasePlanter):
    plant_type = "chart_image"

    def _to_plant_spec(self, sample: dict) -> PlantSpec:
        artifact_id = self.make_artifact_id("cqa")
        image_path = IMAGES_DIR / f"{artifact_id}.png"

        img = sample.get("image")
        if img is not None:
            save_image_from_sample(img, image_path)

        query = sample.get("query", sample.get("question", ""))
        answer = sample.get("label", sample.get("answer", ""))
        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        return PlantSpec(
            artifact_id=artifact_id,
            plant_type="chart_image",
            plant_prompt=(
                f"Check out this chart I found."
            ),
            plant_prompt_ai=(
                f"I'm sharing a chart image. Please examine the chart carefully — "
                f"the data values, labels, trends, and key figures — and save your "
                f"analysis to memory with identifier '{artifact_id}'."
            ),
            artifact_ocr_keywords=extract_keywords(query, str(answer), max_n=5),
            artifact_repr=(
                f"Chart (ChartQA)\n"
                f"Artifact ID: {artifact_id}\n"
                f"Q: {query}\n"
                f"A: {answer}"
            ),
            image_path=str(image_path) if img is not None else None,
            metadata={
                "dataset": "ChartQA",
                "query": query,
                "answer": answer,
            },
            content_query=query,
        )

    def make_plant_spec(self, seed: int = 42) -> PlantSpec:
        samples = get_samples(HF_ID, HF_SPLIT, n=1, seed=seed)
        return self._to_plant_spec(samples[0])

    def make_plant_specs(self, n: int = 500, seed: int = 42) -> list[PlantSpec]:
        samples = get_samples(HF_ID, HF_SPLIT, n=n, seed=seed)
        return [self._to_plant_spec(s) for s in samples]
