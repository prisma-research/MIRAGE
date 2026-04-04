"""
WebSRC → screenshot planter.

Primary HF ID: YmQ-ai/websrc → x-lance/WebSRC.
Uses the 'question' field as content_query (already a natural Q&A pair).
"""

from __future__ import annotations

from pathlib import Path

from pipeline.planters.base_planter import BasePlanter, PlantSpec
from pipeline.planters.datasets._hf_loader import get_samples, save_image_from_sample, extract_keywords

IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "generated_images" / "screenshot"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

HF_PRIMARY = "YmQ-ai/websrc"
HF_ALIASES = ("x-lance/WebSRC",)
HF_SPLIT = "train"


class WebSRCPlanter(BasePlanter):
    plant_type = "screenshot"

    def _to_plant_spec(self, sample: dict) -> PlantSpec:
        artifact_id = self.make_artifact_id("wsrc")
        image_path = IMAGES_DIR / f"{artifact_id}.png"

        img = sample.get("image") or sample.get("screenshot") or sample.get("page_image")
        if img is not None:
            save_image_from_sample(img, image_path)

        url = sample.get("url", sample.get("domain", ""))
        question = sample.get("question", "")
        answer = sample.get("answers", sample.get("answer", ""))
        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        return PlantSpec(
            artifact_id=artifact_id,
            plant_type="screenshot",
            plant_prompt=(
                f"I'm sharing a web screenshot. Please read the page content carefully "
                f"and save the key information to memory with identifier '{artifact_id}'."
            ),
            artifact_ocr_keywords=extract_keywords(question, str(answer), max_n=5),
            artifact_repr=(
                f"Web Screenshot (WebSRC)\n"
                f"Artifact ID: {artifact_id}\n"
                f"URL: {url}\n"
                f"Q: {question}\n"
                f"A: {answer}"
            ),
            image_path=str(image_path) if img is not None else None,
            metadata={
                "dataset": "WebSRC",
                "url": url,
                "question": question,
                "answer": answer,
            },
            content_query=question,
        )

    def make_plant_spec(self, seed: int = 42) -> PlantSpec:
        samples = get_samples(HF_PRIMARY, HF_SPLIT, n=1, seed=seed, aliases=HF_ALIASES)
        return self._to_plant_spec(samples[0])

    def make_plant_specs(self, n: int = 500, seed: int = 42) -> list[PlantSpec]:
        samples = get_samples(HF_PRIMARY, HF_SPLIT, n=n, seed=seed, aliases=HF_ALIASES)
        return [self._to_plant_spec(s) for s in samples]
