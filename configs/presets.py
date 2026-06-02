"""
Named experiment presets for GroundingBench paper runs.

Each preset defines a fixed experimental configuration (scenarios, reference styles,
conditions, history depths, and artifact subset). CLI args override individual fields.

Usage in experiment_runner.py:
    spec = _resolve_run_spec(args)  # merges preset + CLI overrides
"""

from __future__ import annotations

from pathlib import Path

# Preset definitions — keys must match --preset CLI choices
PRESETS: dict[str, dict] = {
    # ── Pilot presets (8+8 artifacts, debug/smoke) ───────────────────────
    "main_backbone": {
        "subset_file": "configs/subsets/main_artifacts.json",
        "scenarios": ["S1", "S2", "S3"],
        "reference_styles": ["definite", "demonstrative"],
        "conditions": ["C0"],
        # No history_depths: treated as [0] implicitly (main_backbone does not vary depth)
    },
    "s1_depth": {
        "subset_file": "configs/subsets/depth_artifacts.json",
        "scenarios": ["S1"],
        "reference_styles": ["definite"],   # single style to control trial count
        "conditions": ["C0"],
        "history_depths": [0, 16_000, 32_000],
    },
    "mitigation_s3": {
        "subset_file": "configs/subsets/mitigation_artifacts.json",
        "scenarios": ["S3"],
        "reference_styles": ["definite", "demonstrative"],
        "conditions": ["C0", "C3"],
        # No history_depths: treated as [0] implicitly
    },

    # ── Paper presets (curated 50+50 or 10+10) ──────────────────────────
    "paper_main": {
        "subset_file": "configs/subsets/paper_main_50_50.json",
        "annotation_file": "configs/annotations/paper_main_50_50_annotations.csv",
        "scenarios": ["S1", "S2", "S3"],
        "reference_styles": ["definite"],
        "conditions": ["C0"],
        # 100 artifacts × 1 style × 1 condition × 3 scenarios = 300 cells/model
    },
    "paper_s1_depth": {
        "subset_file": "configs/subsets/paper_depth_10_10.json",
        "annotation_file": "configs/annotations/paper_main_50_50_annotations.csv",
        "scenarios": ["S1"],
        "reference_styles": ["definite"],
        "conditions": ["C0"],
        "history_depths": [0, 16_000, 32_000],
        # 20 artifacts × 1 style × 1 condition × 3 depths = 60 cells/model
    },
    "paper_s3_mitigation": {
        "subset_file": "configs/subsets/paper_s3_10_10.json",
        "annotation_file": "configs/annotations/paper_main_50_50_annotations.csv",
        "scenarios": ["S3"],
        "reference_styles": ["definite"],
        "conditions": ["C0", "C3", "Cm"],
        # 20 artifacts × 1 style × 3 conditions = 60 cells/model
    },

    # ── Compaction threshold sensitivity sweep ──────────────────────────
    "paper_s2s3_compaction": {
        "subset_file": "configs/subsets/paper_compaction_40.json",
        "annotation_file": "configs/annotations/paper_main_50_50_annotations.csv",
        "scenarios": ["S2", "S3"],
        "reference_styles": ["definite"],
        "conditions": ["C0"],
        "context_thresholds": [25_000, 50_000, 75_000, 100_000],
        # 40 artifacts × 1 style × 1 condition × 4 thresholds = 160 cells/model
        # Each cell produces S2 + S3 = 320 trials/model
    },
}

PRESET_NAMES = list(PRESETS.keys())


def resolve_subset_path(subset_file: str) -> Path:
    """Resolve a preset-relative subset_file path to an absolute Path."""
    p = Path(subset_file)
    if not p.is_absolute():
        # Relative paths are resolved from the GroundingBench package root
        p = Path(__file__).parent.parent / p
    return p
