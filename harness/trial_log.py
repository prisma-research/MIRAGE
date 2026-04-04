"""
GroundingTrial Pydantic model and serialization helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class TrialConfig(BaseModel):
    """Experimental configuration for a single trial (θ in the paper)."""

    plant_type: Literal["screenshot", "chart_image"]
    history_depth: int = Field(
        ge=0,
        description="Token target for S1 filler depth (0 = no filler). S2/S3 ignore this field.",
    )
    reference_style: Literal["definite", "demonstrative", "temporal"]
    backend: str = ""  # auto-detected from OpenClaw config if empty
    seed: int = 42
    context_threshold: int = 50_000
    citation_force_condition: Literal["C0", "C1", "C2", "C3", "Cm"] = "C0"
    """
    C0 = no mitigation (control)
    C1 = CitationForce system prompt only
    C2 = CitationForce + artifact_recall tool
    C3 = C2 + compaction re-injection
    Cm = Mandatory artifact_recall tool routing (stronger prompt + tool available)
    """


class GroundingTrial(BaseModel):
    """Full record for a single grounding trial."""

    # Identity
    trial_id: str
    theta: TrialConfig
    scenario: Literal["S1", "S2", "S3"] = "S3"

    # Artifact
    artifact_id: str
    artifact_ocr_keywords: list[str] = Field(default_factory=list)

    # Query
    query: str

    # Session A outcome
    planting_path: Literal["explicit_write", "pre_compaction_flush", "none"]

    # Session observations
    system_prompt_at_init: str = ""
    tool_trace: list[dict] = Field(default_factory=list)
    startup_exposure_log: list[dict] = Field(default_factory=list)  # pre-query startup exposure (R_boot detection)
    active_context_view: list[dict] = Field(default_factory=list)  # pre-query active context turns (R_context detection)
    response_text: str = ""

    # Harness-side artifact visibility signal (set at trial construction time, before scoring)
    # in_context    : harness knows definitionally that the artifact was in the model's active
    #                 context window at query time (S1 — same session, no compaction, no reset).
    # out_of_context: harness knows definitionally that the artifact was NOT in context
    #                 (S3 — fresh session; no conversation history exists).
    # unknown       : harness cannot determine visibility from logs alone (S2 — compaction
    #                 may or may not have evicted the artifact; no per-message drop signal).
    artifact_context_status: Literal["in_context", "out_of_context", "unknown"] = "unknown"

    # Axis 1 — Retrieval path (four-way + one heuristic variant)
    # heuristic_R_context: screenshot trials where artifact_context_status == "unknown" and
    #   text-log keyword evidence suggests in-context access. Not exact; reported separately
    #   from R_context (which requires either chart_image modality or in_context status).
    R_path: Literal["R_context", "heuristic_R_context", "R_boot", "R_tool", "R_none"] | None = None
    R_binary: int | None = None  # 1 if R_context / heuristic_R_context / R_boot / R_tool, else 0

    # Axis 2 — Artifact identification
    I_strict: int | None = None       # 1 if artifact_id in tool source_paths
    I_set: int | None = None          # 1 if any retrieved artifact matches
    I_empty_return: bool | None = None  # True if memory_get returned ""

    # Axis 3 — Answer support
    S_hat: float | str | None = None  # in {0, 0.5, 1} or "indeterminate"
    S_raw: Any = None           # raw axis3 output before rounding

    # Composite outcomes
    D: int | None = None   # 1 if agent refused / said "I don't have that"
    T_response: int | None = None  # 1 if agent provided any non-refusal response
    T: int | None = None   # 1 if response contains ≥1 artifact-dependent claim
    F_strict: int | None = None   # Primary grounded-success (T=1 AND R=1 AND I=1 AND S=1.0)
    F_relaxed: int | None = None  # Secondary robustness check (T=1 AND R=1 AND I=1 AND S≥0.5)
    outcome: str | None = None  # human-readable summary

    # Compaction tracking (CitationForce)
    compaction_during_session_b: bool = False
    citation_force_reinjected: bool | None = None

    # CitationForce workspace verification (set by harness, NOT derived from system_prompt_at_init)
    bootstrap_cf_present: bool = False
    """True if BOOTSTRAP.md in the trial workspace contained the CitationForce section."""

    # Artifact ground truth (agent's actual memory content for Axis 3)
    actual_artifact_repr: str = ""

    # Native context accounting fields
    boundary_input_tokens: int | None = None       # provider-reported effective input tokens at compaction/query boundary
    native_compaction_count: int | None = None     # sessions.json compactionCount
    native_context_window: int | None = None       # benchmark-registered contextWindow for this trial
    manual_flush_fallback_used: bool = False       # True if native memoryFlush failed and manual prompt was sent

    # Workspace isolation
    workspace_profile: str = ""       # agent_id used for this trial
    snapshot_id: str = ""             # baseline snapshot version
    pre_hash: str = ""                # hash of memory/ before Session A
    post_hash: str = ""               # hash of memory/ after Session A

    # Timestamps
    created_at: str = ""
    session_a_id: str = ""
    session_b_id: str = ""


def save_trial(trial: GroundingTrial, output_dir: str | Path = "logs/trials") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{trial.trial_id}.json"
    path.write_text(trial.model_dump_json(indent=2))
    return path


def load_trials(output_dir: str | Path = "logs/trials") -> list[GroundingTrial]:
    output_dir = Path(output_dir)
    trials = []
    for path in sorted(output_dir.glob("*.json")):
        try:
            trials.append(GroundingTrial.model_validate_json(path.read_text()))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Failed to load %s: %s", path, e)
    return trials
