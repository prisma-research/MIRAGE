"""
Rico → ui_state_image planter.

Primary HF IDs: square/rico → Multimodal-Fatima/Rico.
Falls back to local zip download if both HF IDs fail.
Uses a UI-description content_query (Rico has no explicit Q&A).
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from pipeline.planters.base_planter import BasePlanter, PlantSpec
from pipeline.planters.datasets._hf_loader import (
    get_samples,
    save_image_from_sample,
    extract_keywords,
    get_rico_samples_local,
    CACHE_DIR,
)

logger = logging.getLogger(__name__)

IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "generated_images" / "ui"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

HF_PRIMARY = "Multimodal-Fatima/Rico"
HF_ALIASES = ("bigai-nlco/Rico",)
HF_SPLIT = "train"


def _extract_text_nodes(node: dict, results: list[str]) -> None:
    """Recursively collect non-empty text values from a UI tree node."""
    text = node.get("text") or node.get("content-desc") or ""
    if isinstance(text, str) and len(text) > 1:
        results.append(text.strip())
    for child in node.get("children", []):
        _extract_text_nodes(child, results)


class RicoPlanter(BasePlanter):
    plant_type = "ui_state_image"

    def _to_plant_spec_hf(self, sample: dict) -> PlantSpec:
        artifact_id = self.make_artifact_id("rico")
        image_path = IMAGES_DIR / f"{artifact_id}.png"

        img = sample.get("image") or sample.get("screenshot")
        if img is not None:
            save_image_from_sample(img, image_path)

        package_name = sample.get("package_name", sample.get("app_name", ""))
        ui_category = sample.get("ui_category", sample.get("category", ""))
        root_node = sample.get("ui_tree", {})
        text_nodes: list[str] = []
        if isinstance(root_node, dict):
            _extract_text_nodes(root_node, text_nodes)

        return self._build_spec(artifact_id, image_path, package_name, ui_category, text_nodes)

    def _to_plant_spec_local(self, sample: dict) -> PlantSpec:
        artifact_id = self.make_artifact_id("rico")
        image_path = IMAGES_DIR / f"{artifact_id}.png"

        src_path = sample.get("image_path")
        if src_path and not image_path.exists():
            try:
                img = Image.open(src_path)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                img.convert("RGB").save(str(image_path), format="PNG")
            except Exception as exc:
                logger.warning("Could not copy Rico image %s: %s", src_path, exc)
                image_path = None

        package_name = sample.get("package_name", "")
        ui_category = sample.get("ui_category", "")
        nodes = sample.get("nodes", [])
        text_nodes: list[str] = []
        for node in nodes:
            if isinstance(node, dict):
                _extract_text_nodes(node, text_nodes)

        return self._build_spec(artifact_id, image_path, package_name, ui_category, text_nodes)

    def _build_spec(
        self,
        artifact_id: str,
        image_path: Path | None,
        package_name: str,
        ui_category: str,
        text_nodes: list[str],
    ) -> PlantSpec:
        top_texts = [t for t in text_nodes if t][:5]
        content_query = (
            f"What elements are visible in this {ui_category or 'Android'} app screen?"
        )
        return PlantSpec(
            artifact_id=artifact_id,
            plant_type="ui_state_image",
            plant_prompt=(
                f"I'm sharing an Android UI screenshot. Please examine the interface elements, "
                f"labels, and layout. Save your analysis to memory with identifier '{artifact_id}'."
            ),
            artifact_ocr_keywords=extract_keywords(*top_texts, max_n=5),
            artifact_repr=(
                f"Android UI (Rico)\n"
                f"Artifact ID: {artifact_id}\n"
                f"App: {package_name}\n"
                f"Category: {ui_category}\n"
                f"Elements: {top_texts}"
            ),
            image_path=str(image_path) if image_path else None,
            metadata={
                "dataset": "Rico",
                "package_name": package_name,
                "ui_category": ui_category,
                "top_text_nodes": top_texts,
            },
            content_query=content_query,
        )

    def _get_hf_samples(self, n: int, seed: int) -> list[dict] | None:
        try:
            return get_samples(HF_PRIMARY, HF_SPLIT, n=n, seed=seed, aliases=HF_ALIASES)
        except Exception as exc:
            logger.warning("Rico HF load failed (%s); falling back to local zip.", exc)
            return None

    def make_plant_spec(self, seed: int = 42) -> PlantSpec:
        samples = self._get_hf_samples(1, seed)
        if samples is not None:
            return self._to_plant_spec_hf(samples[0])
        local = get_rico_samples_local(n=1, seed=seed)
        return self._to_plant_spec_local(local[0])

    def make_plant_specs(self, n: int = 500, seed: int = 42) -> list[PlantSpec]:
        samples = self._get_hf_samples(n, seed)
        if samples is not None:
            return [self._to_plant_spec_hf(s) for s in samples]
        local = get_rico_samples_local(n=n, seed=seed)
        return [self._to_plant_spec_local(s) for s in local]
