"""
Axis 1 — R-Path Classifier (four-way)

Classifies how (if at all) the artifact was accessed:
  R_context : artifact content in pre-query conversation history, no retrieval (S1)
  R_boot    : artifact found in pre-query startup exposure (fresh-session bootstrap-mediated access)
  R_tool    : agent called memory_search / artifact_recall after the query turn
  R_none    : no legitimate access path active
"""

from __future__ import annotations

import re
from typing import Literal

from constants import RETRIEVAL_TOOL_NAMES, MEMORY_PATH_PATTERNS


def classify_r_path(
    system_prompt: str,
    tool_trace: list[dict],
    artifact_id: str,
    artifact_ocr_keywords: list[str],
    startup_exposure_log: list[dict] | None = None,
    active_context_view: list[dict] | None = None,
    min_keyword_matches: int = 2,
) -> Literal["R_context", "R_boot", "R_tool", "R_none"]:
    """
    Four-way R-path classification, applied in priority order:
    R_context > R_boot > R_tool > R_none.

    Args:
        system_prompt: System prompt / BOOTSTRAP.md text injected at session start (for R_boot).
        tool_trace: Post-query tool call/result dicts from Session B only.
        artifact_id: The artifact's unique identifier.
        artifact_ocr_keywords: OCR keywords from the artifact.
        startup_exposure_log: Pre-query bootstrap tool-result entries (for R_boot).
        active_context_view: Pre-query conversation turns (for R_context).
    """
    keywords = [artifact_id] + [kw for kw in artifact_ocr_keywords if kw]
    # R_context uses only OCR content keywords — artifact_id appears in planting metadata
    # ("saved under id ss_…") which is always present, making it a false positive signal.
    ocr_keywords = [kw for kw in artifact_ocr_keywords if kw]

    # Step 1: R_context — artifact content in pre-query conversation, no retrieval calls
    if active_context_view and _check_r_context(active_context_view, tool_trace, ocr_keywords, min_keyword_matches):
        return "R_context"

    # Step 2: R_boot — artifact in pre-query startup exposure or system prompt
    # system_prompt carries BOOTSTRAP.md content which may contain artifact memory.
    if startup_exposure_log or system_prompt:
        if _check_r_boot(startup_exposure_log or [], keywords, system_prompt):
            return "R_boot"

    # Step 3: R_tool — retrieval call after query
    if _check_r_tool(tool_trace, artifact_id, artifact_ocr_keywords):
        return "R_tool"

    # Step 4: residual
    return "R_none"


def _check_r_context(
    active_context_view: list[dict],
    tool_trace: list[dict],
    keywords: list[str],
    min_keyword_matches: int = 2,
) -> bool:
    """R_context: artifact in active context view AND no retrieval tool calls.

    Requires at least min(min_keyword_matches, len(keywords)) distinct keyword
    matches across all turns to avoid false positives from coincidental single-word
    matches (e.g. generic OCR terms like "page", "enter" in unrelated turns).
    """
    # Must have NO retrieval tool calls anywhere in the trace
    has_retrieval = any(
        entry.get("kind") == "call" and entry.get("name") in RETRIEVAL_TOOL_NAMES
        for entry in tool_trace
    )
    if has_retrieval:
        return False

    # Require at least min(min_keyword_matches, len(keywords)) distinct keyword hits
    threshold = min(min_keyword_matches, len(keywords)) if keywords else 0
    if threshold == 0:
        return False

    matched_keywords: set[str] = set()
    for turn in active_context_view:
        text = turn.get("text", "")
        if not text:
            content = turn.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text += c.get("text", "")
            elif isinstance(content, str):
                text = content
        for kw in keywords:
            if kw and kw in text:
                matched_keywords.add(kw)
        if len(matched_keywords) >= threshold:
            return True
    return False


def _check_r_boot(
    startup_exposure_log: list[dict],
    keywords: list[str],
    system_prompt: str = "",
) -> bool:
    """R_boot: artifact_id or OCR keyword in pre-query startup exposure.

    Checks both the structured startup_exposure_log entries (tool results
    from bootstrap) and the system_prompt / BOOTSTRAP.md text, since the
    agent may receive artifact content through either channel.
    """
    # Check system prompt (BOOTSTRAP.md content is injected here)
    if system_prompt:
        sp_lower = system_prompt.lower()
        for kw in keywords:
            if kw and kw.lower() in sp_lower:
                return True

    # Check structured exposure log entries (bootstrap tool call results)
    for result in startup_exposure_log:
        text = result.get("text", "")
        text_lower = text.lower()
        for kw in keywords:
            if kw and kw.lower() in text_lower:
                return True
    return False


def _check_r_tool(
    tool_trace: list[dict],
    artifact_id: str,
    artifact_ocr_keywords: list[str],
) -> bool:
    """R_tool: retrieval call after query."""
    for entry in tool_trace:
        if entry.get("kind") != "call":
            continue
        name = entry.get("name", "")
        if name in RETRIEVAL_TOOL_NAMES:
            return True
        if name == "read":
            input_data = entry.get("input") or {}
            path = input_data.get("path", "") if isinstance(input_data, dict) else str(input_data)
            if any(pat in path for pat in MEMORY_PATH_PATTERNS):
                return True

    # Also check tool results for artifact_id
    for entry in tool_trace:
        if entry.get("kind") == "result":
            text = entry.get("text", "")
            if artifact_id in text:
                return True

    return False


def _artifact_in_text(artifact_id: str, keywords: list[str], text: str) -> bool:
    """Check if artifact_id or any OCR keyword appears in text."""
    if not text:
        return False
    if artifact_id in text:
        return True
    for kw in keywords:
        if kw and kw.lower() in text.lower():
            return True
    return False


def extract_tool_calls_after_query(tool_trace: list[dict]) -> list[dict]:
    """Return all tool calls (the trace already contains only post-query calls)."""
    return tool_trace


def extract_tool_calls(
    jsonl: list[dict],
    after_entry_id: str | None = None,
) -> list[dict]:
    """Filter tool calls from a raw JSONL trace."""
    result = []
    include = after_entry_id is None
    for entry in jsonl:
        if not include:
            if entry.get("id") == after_entry_id:
                include = True
            continue

        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        role = msg.get("role", "")

        if role == "assistant":
            for c in msg.get("content", []):
                if isinstance(c, dict) and c.get("type") in ("toolCall", "tool_use"):
                    result.append({
                        "kind": "call",
                        "tool_use_id": c.get("id"),
                        "name": c.get("name"),
                        "input": c.get("arguments") or c.get("input"),
                        "entry_id": entry.get("id"),
                    })
        elif role == "toolResult":
            content = msg.get("content", [])
            text = ""
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text += c.get("text", "")
            elif isinstance(content, str):
                text = content
            result.append({
                "kind": "result",
                "tool_use_id": msg.get("toolCallId"),
                "name": msg.get("toolName"),
                "text": text,
                "is_error": msg.get("isError", False),
                "entry_id": entry.get("id"),
            })

    return result


def r_binary(r_path: Literal["R_context", "heuristic_R_context", "R_boot", "R_tool", "R_none"]) -> int:
    """Convert R_path label to binary: 1 if any access path (including heuristic), 0 if none."""
    return 0 if r_path == "R_none" else 1
