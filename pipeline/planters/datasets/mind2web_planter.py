"""
Mind2Web → screenshot planter.

Uses the processed_data split which has direct screenshot bytes.
Uses the 'task' field as content_query.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.planters.base_planter import BasePlanter, PlantSpec
from pipeline.planters.datasets._hf_loader import get_samples, save_image_from_sample, extract_keywords

IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "generated_images" / "screenshot"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


class Mind2WebPlanter(BasePlanter):
    plant_type = "screenshot"
    HF_ID = "osunlp/Mind2Web"
    HF_SPLIT = "train"  # processed_data is the main split name

    def _to_plant_spec(self, sample: dict) -> PlantSpec:
        artifact_id = self.make_artifact_id("m2w")
        image_path = IMAGES_DIR / f"{artifact_id}.png"

        # Mind2Web screenshot field: may be 'screenshot', 'image', or similar
        screenshot = (
            sample.get("screenshot")
            or sample.get("image")
            or sample.get("raw_html_screenshot")
        )
        if screenshot is not None:
            save_image_from_sample(screenshot, image_path)

        website = sample.get("website", sample.get("domain", ""))
        task = sample.get("confirmed_task", sample.get("task", ""))
        action_reprs = sample.get("action_reprs", [])
        if not isinstance(action_reprs, list):
            action_reprs = []

        content_query = (
            f"What element should be interacted with to complete: {task}?"
            if task else ""
        )

        return PlantSpec(
            artifact_id=artifact_id,
            plant_type="screenshot",
            plant_prompt=(
                f"I'm sharing a web screenshot. Please examine the page content, "
                f"interactive elements, and layout. Save your analysis to memory "
                f"with identifier '{artifact_id}'."
            ),
            artifact_ocr_keywords=extract_keywords(task, website, max_n=5),
            artifact_repr=(
                f"Web Screenshot (Mind2Web)\n"
                f"Artifact ID: {artifact_id}\n"
                f"Site: {website}\n"
                f"Task: {task}\n"
                f"Actions: {action_reprs[:3]}"
            ),
            image_path=str(image_path) if screenshot is not None else None,
            metadata={
                "dataset": "Mind2Web",
                "website": website,
                "task": task,
                "action_reprs": action_reprs[:3],
            },
            content_query=content_query,
        )

    def make_plant_spec(self, seed: int = 42) -> PlantSpec:
        samples = get_samples(self.HF_ID, self.HF_SPLIT, n=1, seed=seed)
        return self._to_plant_spec(samples[0])

    def make_plant_specs(self, n: int = 500, seed: int = 42) -> list[PlantSpec]:
        samples = get_samples(self.HF_ID, self.HF_SPLIT, n=n, seed=seed)
        return [self._to_plant_spec(s) for s in samples]
