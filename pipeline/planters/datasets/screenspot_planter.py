"""
ScreenSpot → screenshot planter.

Fields: file_name, bbox, instruction, data_type ("text"|"icon"), data_source, image.
Covers mobile, desktop, and web GUI screenshots.
Uses the 'instruction' field as content_query (e.g. "Click the search bar").
"""

from __future__ import annotations

from pathlib import Path

from pipeline.planters.base_planter import BasePlanter, PlantSpec
from pipeline.planters.datasets._hf_loader import get_samples, save_image_from_sample, extract_keywords

IMAGES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "generated_images" / "screenshot"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


class ScreenSpotPlanter(BasePlanter):
    plant_type = "screenshot"
    HF_ID = "rootsautomation/ScreenSpot"
    HF_SPLIT = "test"

    def _to_plant_spec(self, sample: dict) -> PlantSpec:
        artifact_id = self.make_artifact_id("ss")
        image_path = IMAGES_DIR / f"{artifact_id}.png"
        save_image_from_sample(sample["image"], image_path)

        instruction = sample.get("instruction", "")
        bbox = sample.get("bbox", [])
        platform = sample.get("data_source", "")
        data_type = sample.get("data_type", "")

        return PlantSpec(
            artifact_id=artifact_id,
            plant_type="screenshot",
            plant_prompt=(
                f"Here's a screenshot from {platform} — take a look."
            ),
            plant_prompt_ai=(
                f"I'm sharing a GUI screenshot. Please examine the interface — "
                f"labels, interactive components, layout, and any visible text — "
                f"and save your analysis to memory with identifier '{artifact_id}'."
            ),
            artifact_ocr_keywords=extract_keywords(instruction, max_n=5),
            artifact_repr=(
                f"GUI Screenshot (ScreenSpot)\n"
                f"Artifact ID: {artifact_id}\n"
                f"Instruction: {instruction}\n"
                f"Element type: {data_type}\n"
                f"Bounding box: {bbox}\n"
                f"Platform: {platform}"
            ),
            image_path=str(image_path),
            metadata={
                "dataset": "ScreenSpot",
                "data_type": data_type,
                "instruction": instruction,
                "bbox": bbox,
                "platform": platform,
            },
            content_query=instruction,
        )

    def make_plant_spec(self, seed: int = 42) -> PlantSpec:
        samples = get_samples(self.HF_ID, self.HF_SPLIT, n=1, seed=seed)
        return self._to_plant_spec(samples[0])

    def make_plant_specs(self, n: int = 500, seed: int = 42) -> list[PlantSpec]:
        samples = get_samples(self.HF_ID, self.HF_SPLIT, n=n, seed=seed)
        return [self._to_plant_spec(s) for s in samples]
