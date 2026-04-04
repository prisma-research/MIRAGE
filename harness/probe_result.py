"""
Probe result data model for checkpoint-based experiments.

Each ProbeResult records the outcome of a single branch probe:
one checkpoint restored, one question asked, one answer captured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ProbeResult(BaseModel):
    """Full record for a single branch probe."""

    # --- Identity ---
    probe_id: str                                   # unique per probe run
    episode_id: str                                 # which episode produced the checkpoint
    checkpoint_id: str                              # which checkpoint was restored
    question_id: str                                # which question was asked
    experiment: Literal["depth", "compaction", "reset"]
    continuation: Literal["same_session", "fresh_session"]

    # --- Checkpoint context ---
    checkpoint_kind: str = ""                       # s1_prequery / postcomp
    depth_label: int | None = None                  # 0 / 16000 / 32000 (S1)
    compaction_threshold: int | None = None         # 50000 / 100000 / 150000 (postcomp)
    effective_input_tokens: int | None = None       # provider-reported eit at checkpoint
    native_compaction_count: int | None = None      # compactionCount at checkpoint

    # --- Question context ---
    target_artifact_id: str | None = None           # which artifact the question is about
    query_type: str = ""                            # target_present / wrong_source / unanswerable
    visual_type: str = ""                           # standalone_chart / app_ui / web_ui
    confusion_group: str = ""                       # chart_financial / ui_mobile / etc.
    paraphrase_group_id: str = ""                   # groups variants of the same base question
    prompt: str = ""                                # the actual question sent
    gold_answer: str | None = None                  # expected answer (if any)
    expected_answerability: str = ""                 # answerable / unanswerable / partially_answerable

    # --- Raw response ---
    response_text: str = ""                         # model's response
    session_id: str = ""                            # session used for the probe
    branch_id: str = ""                             # branch state dir name

    # --- Transcript ---
    jsonl_entry_count: int = 0                      # entries in session JSONL after probe
    tool_trace: list[dict] = Field(default_factory=list)  # tool calls during the probe turn

    # --- Same-session proof (S1/S2 only) ---
    jsonl_count_before: int | None = None
    jsonl_count_after: int | None = None
    session_id_matches_checkpoint: bool | None = None

    # --- Retrieval path classification (auto-scored) ---
    R_path: Literal[
        "R_context", "R_boot", "R_tool", "R_none"
    ] | None = None
    R_binary: int | None = None                     # 1 if any retrieval path, else 0

    # --- Constrained-output parsed fields ---
    # Extracted from the model's structured SUPPORT/SOURCE/ANSWER response.
    parsed_support: str | None = None               # TARGET / OTHER_ARTIFACT / NOT_PRESENT
    parsed_source: str | None = None                # artifact_id or NONE (raw from model)
    parsed_answer: str | None = None                # short value or NONE
    canonical_source: str | None = None             # post-normalization source used for comparison
    parse_success: bool | None = None               # True if structured output was parsed

    # --- Gold fields (from question bank) ---
    gold_support: str | None = None                 # TARGET / OTHER_ARTIFACT / NOT_PRESENT
    gold_source: str | None = None                  # artifact_id or NONE
    answer_type: str = ""                           # numeric / year / string / none

    # --- Primary metrics (constrained-output scoring) ---
    support_correct: int | None = None              # 1 if parsed_support == gold_support
    answer_correct: int | None = None               # 1 if parsed_answer matches gold (with normalization)
    source_correct: int | None = None               # 1 if parsed_source == gold_source

    # --- Legacy free-form metrics (kept for backward compat) ---
    target_grounded_correct: int | None = None
    wrong_source: int | None = None
    abstention: int | None = None
    abstention_correct: int | None = None
    support_score: float | None = None
    judge_notes: str = ""

    # --- Metadata ---
    model: str = ""
    created_at: str = ""
    duration_seconds: float | None = None


def save_probe_result(result: ProbeResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.probe_id}.json"
    path.write_text(result.model_dump_json(indent=2))
    return path


def load_probe_results(output_dir: Path) -> list[ProbeResult]:
    results = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name.startswith("run_") or path.name == "summary.json":
            continue
        try:
            results.append(ProbeResult.model_validate_json(path.read_text()))
        except Exception:
            pass
    return results
