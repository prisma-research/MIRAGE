"""
GUI-World → screenshot planter (all platforms).

HF ID: Gyrojeff/GUI-World
Uses the 'question' field as content_query.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.planters.base_planter import BasePlanter, PlantSpec
from pipeline.planters.datasets._hf_loader import get_samples, save_image_from_sample, extract_keywords

_SCREENSHOT_DIR = Path(__file__).resolve().parent.parent.parent / "generated_images" / "screenshot"
_SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

HF_ID = "Gyrojeff/GUI-World"
HF_SPLIT = "train"


class GUIWorldPlanter(BasePlanter):
    plant_type = "screenshot"

    def _to_plant_spec(self, sample: dict) -> PlantSpec:
        artifact_id = self.make_artifact_id("gw")

        platform = sample.get("platform", sample.get("source", "")).lower()
        image_path = _SCREENSHOT_DIR / f"{artifact_id}.png"

        img = sample.get("image") or sample.get("screenshot")
        if img is not None:
            save_image_from_sample(img, image_path)

        task_description = sample.get("task_description", sample.get("task", ""))
        question = sample.get("question", "")
        answer = sample.get("answer", sample.get("answers", ""))
        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        return PlantSpec(
            artifact_id=artifact_id,
            plant_type="screenshot",
            plant_prompt=(
                f"I'm sharing a GUI screenshot ({platform}). Please examine the interface, "
                f"content, and interactive elements. Save your analysis to memory with "
                f"identifier '{artifact_id}'."
            ),
            artifact_ocr_keywords=extract_keywords(task_description, str(answer), max_n=5),
            artifact_repr=(
                f"GUI Screenshot (GUI-World)\n"
                f"Artifact ID: {artifact_id}\n"
                f"Platform: {platform}\n"
                f"Task: {task_description}\n"
                f"Q: {question}\n"
                f"A: {answer}"
            ),
            image_path=str(image_path) if img is not None else None,
            metadata={
                "dataset": "GUI-World",
                "platform": platform,
                "task_description": task_description,
                "question": question,
                "answer": answer,
            },
            content_query=question,
        )

    def make_plant_spec(self, seed: int = 42) -> PlantSpec:
        samples = get_samples(HF_ID, HF_SPLIT, n=1, seed=seed)
        return self._to_plant_spec(samples[0])

    def make_plant_specs(self, n: int = 500, seed: int = 42) -> list[PlantSpec]:
        samples = get_samples(HF_ID, HF_SPLIT, n=n, seed=seed)
        return [self._to_plant_spec(s) for s in samples]
