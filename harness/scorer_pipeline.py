"""
Scorer Pipeline — applies all three axes to a GroundingTrial.

After the trial harness captures Session A + B data, this pipeline
computes R_path, I_strict/I_set, S_hat, D, T, F, and outcome.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from constants import RETRIEVAL_TOOL_NAMES
from harness.trial_log import GroundingTrial, save_trial
from scorers.axis1_rpath import classify_r_path, r_binary
from scorers.axis2_identification import summarize_identification
from scorers.axis3.ensemble import score_axis3_detailed
from mitigations.citation_force.signals import analyze_response_signals

logger = logging.getLogger(__name__)


def score_trial(
    trial: GroundingTrial,
    artifact_repr: str = "",
    run_axis3: bool = True,
    api_key: str | None = None,
    n_judges: int = 3,
    output_dir: str | None = None,
) -> GroundingTrial:
    """
    Apply all three axes to a trial and return the updated GroundingTrial.

    Args:
        trial: The trial to score (must have system_prompt_at_init, tool_trace, response_text).
        artifact_repr: Text/OCR representation of the artifact for Axis 3 scoring.
        run_axis3: Whether to run the LLM-based Axis 3 scorer.
        api_key: API key for Axis 3 (Volcengine).
        n_judges: Number of judge calls for Axis 3 ensemble.
        output_dir: If set, save updated trial to this directory.
    """
    # Axis 1: R-path classification.
    #
    # For screenshot trials the harness-side artifact_context_status drives the decision
    # before the text-keyword classifier runs, because image content is not observable
    # from text logs alone:
    #
    #   in_context (S1): harness guarantees the artifact was in the active context window
    #     at query time. If no retrieval tool was called after the query, assign R_context
    #     directly — the text classifier is bypassed to avoid false negatives from sparse
    #     OCR coverage.
    #
    #   out_of_context (S3): harness guarantees the artifact was NOT in context. Skip the
    #     keyword-based R_context check entirely; run the classifier without active_context_view
    #     so the result falls through to R_boot / R_tool / R_none correctly.
    #
    #   unknown (S2): harness cannot confirm visibility. Run the full classifier; if it
    #     returns R_context, downgrade to heuristic_R_context (clearly marked as uncertain).
    #
    # chart_image trials always use the full classifier (text evidence is exact for those).
    if trial.theta.plant_type == "screenshot":
        status = trial.artifact_context_status
        if status == "in_context":
            # Harness-confirmed: artifact was visible. R_context unless agent called a
            # retrieval tool after the query (which would override in-context access).
            has_post_query_retrieval = any(
                e.get("kind") == "call" and e.get("name") in RETRIEVAL_TOOL_NAMES
                for e in trial.tool_trace
            )
            r_path = "R_none" if has_post_query_retrieval else "R_context"
            # Still check R_tool / R_boot for the retrieval case
            if has_post_query_retrieval:
                r_path = classify_r_path(
                    system_prompt=trial.system_prompt_at_init,
                    tool_trace=trial.tool_trace,
                    artifact_id=trial.artifact_id,
                    artifact_ocr_keywords=trial.artifact_ocr_keywords,
                    startup_exposure_log=trial.startup_exposure_log,
                    active_context_view=None,  # suppress context view; it's already in_context
                )
        elif status == "out_of_context":
            # Harness-confirmed not in context — suppress active_context_view
            r_path = classify_r_path(
                system_prompt=trial.system_prompt_at_init,
                tool_trace=trial.tool_trace,
                artifact_id=trial.artifact_id,
                artifact_ocr_keywords=trial.artifact_ocr_keywords,
                startup_exposure_log=trial.startup_exposure_log,
                active_context_view=None,
            )
        else:  # unknown (S2)
            r_path = classify_r_path(
                system_prompt=trial.system_prompt_at_init,
                tool_trace=trial.tool_trace,
                artifact_id=trial.artifact_id,
                artifact_ocr_keywords=trial.artifact_ocr_keywords,
                startup_exposure_log=trial.startup_exposure_log,
                active_context_view=trial.active_context_view,
            )
            if r_path == "R_context":
                r_path = "heuristic_R_context"
    else:
        r_path = classify_r_path(
            system_prompt=trial.system_prompt_at_init,
            tool_trace=trial.tool_trace,
            artifact_id=trial.artifact_id,
            artifact_ocr_keywords=trial.artifact_ocr_keywords,
            startup_exposure_log=trial.startup_exposure_log,
            active_context_view=trial.active_context_view,
        )
    trial.R_path = r_path
    trial.R_binary = r_binary(r_path)

    # Axis 2: Identification
    # R_context: I=1 by definition — artifact was in the active context window (harness-confirmed
    # for screenshots via in_context status; exact for chart_image via text-log evidence).
    if r_path == "R_context":
        trial.I_strict = 1
        trial.I_set = 1
        trial.I_empty_return = False
    # heuristic_R_context (screenshot): image may have been in context window but we cannot
    # confirm from text logs — run the identification scorer rather than auto-setting I=1.
    # R_boot: I=1 by definition if startup_exposure_log contained artifact evidence
    # (R_boot was already confirmed by axis1 — artifact was in startup_exposure_log)
    elif r_path == "R_boot":
        trial.I_strict = 1
        trial.I_set = 1
        trial.I_empty_return = False
    else:
        i_scores = summarize_identification(trial.tool_trace, trial.artifact_id)
        trial.I_strict = i_scores["I_strict"]
        trial.I_set = i_scores["I_set"]
        trial.I_empty_return = i_scores["I_empty_return"]

    # Refusal / D flag (from signal analysis)
    signals = analyze_response_signals(trial.response_text)
    trial.D = 1 if signals["is_refusal"] else 0

    # Axis 3: Answer support (optional, requires Claude API)
    if run_axis3 and trial.response_text and not signals["is_refusal"]:
        try:
            key = api_key or os.environ.get("LLM_API_KEY")
            axis3_detail = score_axis3_detailed(
                query=trial.query,
                response_text=trial.response_text,
                artifact_repr=artifact_repr,
                artifact_description=f"{trial.theta.plant_type} artifact",
                n_judges=n_judges,
                api_key=key,
            )
            trial.S_hat = axis3_detail["final_score"]
            trial.S_raw = axis3_detail
        except Exception as exc:
            logger.warning("Axis 3 scoring failed: %s", exc)
            trial.S_hat = None
            trial.S_raw = {"error": str(exc)}
    elif signals["is_refusal"]:
        trial.S_hat = 0.0

    # Composite: T_response (any non-refusal response) and T (artifact-relevant claim present)
    trial.T_response = 0 if trial.D == 1 else 1

    # T=1: response contains at least one artifact-dependent claim (from Axis 3)
    n_claims = 0
    if isinstance(trial.S_raw, dict):
        judges = trial.S_raw.get("judges", [])
        if judges:
            n_claims = judges[0].get("n_claims", 0)
    trial.T = 1 if (trial.T_response == 1 and n_claims >= 1) else 0

    # F_strict=1: primary metric — T=1 AND R=1 AND I_strict=1 AND S_hat=1.0 (fully supported)
    s_hat_strict = isinstance(trial.S_hat, (int, float)) and trial.S_hat == 1.0
    if trial.T_response == 1 and trial.R_binary == 1 and trial.I_strict == 1 and s_hat_strict:
        trial.F_strict = 1
    else:
        trial.F_strict = 0

    # F_relaxed=1: secondary robustness check — T=1 AND R=1 AND I_strict=1 AND S_hat≥0.5
    s_hat_relaxed = isinstance(trial.S_hat, (int, float)) and trial.S_hat >= 0.5
    if trial.T_response == 1 and trial.R_binary == 1 and trial.I_strict == 1 and s_hat_relaxed:
        trial.F_relaxed = 1
    else:
        trial.F_relaxed = 0

    # Human-readable outcome
    trial.outcome = _summarize_outcome(trial)

    if output_dir:
        save_trial(trial, output_dir)

    return trial


def _summarize_outcome(trial: GroundingTrial) -> str:
    parts = []
    if trial.D == 1:
        parts.append("REFUSED")
    else:
        parts.append(f"R={trial.R_path}")
        if trial.I_strict is not None:
            parts.append(f"I={trial.I_strict}")
        if trial.I_empty_return:
            parts.append("I_empty=T")
        if trial.S_hat is not None:
            parts.append(f"S={trial.S_hat}")
        parts.append(f"F_strict={trial.F_strict}")
        parts.append(f"F_relaxed={trial.F_relaxed}")
    return " | ".join(parts)
