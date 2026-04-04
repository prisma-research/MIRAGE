"""
Base class for artifact planters.

A planter generates:
  1. A plant_prompt / plant_prompt_ai: user messages for Session A.
     - plant_prompt: natural language version ("Here's a screenshot...")
     - plant_prompt_ai: AI-instructional version with explicit memory-save instruction
  2. An artifact_id: unique identifier for the artifact.
  3. artifact_ocr_keywords: keywords that should appear in the artifact content.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PlantSpec:
    """Specification for planting an artifact."""
    artifact_id: str
    plant_type: str
    plant_prompt: str          # natural language — how a real user would share this
    plant_prompt_ai: str       # AI-instructional — explicit memory-save instruction
    artifact_ocr_keywords: list[str]
    artifact_repr: str = ""   # 文本表示，用于 Axis 3 评分
    image_path: str | None = None  # 生成的图片路径（None 表示纯文本 artifact）
    metadata: dict = field(default_factory=dict)
    content_query: str = ""  # 数据集衍生的事实性问题，用于 Session B（空字符串时回退到通用模板）


class BasePlanter(ABC):
    """Abstract base class for artifact planters."""

    plant_type: str = "base"

    def make_artifact_id(self, prefix: str = "") -> str:
        short = uuid.uuid4().hex[:8]
        return f"{prefix or self.plant_type}_{short}"

    @abstractmethod
    def make_plant_spec(self, seed: int = 42) -> PlantSpec:
        """Generate a PlantSpec for one trial."""
        ...
