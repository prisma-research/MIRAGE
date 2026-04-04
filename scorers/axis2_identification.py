"""
Axis 2 — Artifact Identification Scorer

Given the tool trace from Session B, determine whether the agent correctly
identified and retrieved the target artifact.

Scores:
  1        : tool returned non-empty content AND artifact_id in source_paths
  0        : tool returned non-empty content BUT artifact_id absent from source_paths
  "empty"  : tool returned empty text (memory_get with hallucinated/missing path)
"""

from __future__ import annotations

from typing import Literal


ScoreI = Literal[1, 0, "empty"]


def score_identification(
    tool_trace: list[dict],
    artifact_id: str,
) -> ScoreI:
    """
    Strict single-artifact identification score (I^(1)).

    Searches for the most recent retrieval call + result pair in the trace.
    """
    # Find all retrieval results (including "read" tool for OpenClaw)
    RETRIEVAL_NAMES = {"memory_search", "artifact_recall", "memory_get", "memory_recall", "read"}
    retrieval_results = [
        e for e in tool_trace
        if e.get("kind") == "result"
        and e.get("name") in RETRIEVAL_NAMES
    ]

    if not retrieval_results:
        return 0

    # Check if any retrieval returned the artifact
    for result in retrieval_results:
        if is_empty_return(result):
            continue  # Skip empty returns for now

        text = result.get("text", "")
        details = result.get("details") or {}

        # Check source_paths in details
        source_paths = details.get("source_paths") or details.get("results", [])
        if isinstance(source_paths, list):
            for sp in source_paths:
                if artifact_id in str(sp):
                    return 1
        elif isinstance(source_paths, str) and artifact_id in source_paths:
            return 1

        # Check in text content
        if artifact_id in text:
            return 1

    # All retrieval results were non-empty but didn't contain artifact_id
    # Check for empty returns
    for result in retrieval_results:
        if is_empty_return(result):
            return "empty"

    return 0


def score_identification_set(
    tool_trace: list[dict],
    artifact_id: str,
    candidate_artifact_ids: list[str],
) -> ScoreI:
    """
    Set-based identification score I^(k).

    Returns 1 if artifact_id is among the retrieved set.
    """
    for result in tool_trace:
        if result.get("kind") != "result":
            continue
        if result.get("name") not in {"memory_search", "artifact_recall", "memory_get", "read"}:
            continue
        if is_empty_return(result):
            continue

        text = result.get("text", "")
        details = result.get("details") or {}
        source_paths = details.get("source_paths") or details.get("results", [])

        all_sources = str(source_paths) + text
        if artifact_id in all_sources:
            return 1

    # Check for empty
    for result in tool_trace:
        if result.get("kind") == "result" and is_empty_return(result):
            return "empty"

    return 0


def is_empty_return(tool_result: dict) -> bool:
    """
    Detect empty memory_get return: {"text": "", "path": ...}.

    An empty return indicates the agent called memory_get with a path that
    doesn't exist (hallucinated artifact path).
    """
    if tool_result.get("kind") != "result":
        return False
    text = tool_result.get("text", "")
    details = tool_result.get("details") or {}

    # Explicit empty text field
    if text == "" or text is None:
        return True

    # Check details structure: {"text": "", ...}
    if isinstance(details, dict):
        if details.get("text") == "" or details.get("content") == "":
            return True
        # memory_get result with empty body
        if "path" in details and not details.get("text"):
            return True

    return False


def summarize_identification(tool_trace: list[dict], artifact_id: str) -> dict:
    """Return a summary dict of all Axis 2 scores."""
    i_strict = score_identification(tool_trace, artifact_id)
    return {
        "I_strict": i_strict,
        "I_set": i_strict,  # same for single-artifact trials
        "I_empty_return": any(
            is_empty_return(e) for e in tool_trace if e.get("kind") == "result"
        ),
    }
