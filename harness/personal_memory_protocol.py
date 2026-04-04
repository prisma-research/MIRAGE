"""Prompt builder and response parser for the T × M personal memory study.

Output protocol:
    DECISION=<label>
    MEMORY_USED=YES|NO
    RATIONALE=<short text>

Primary scoring depends only on DECISION.
"""

from __future__ import annotations
import re


def build_task_prompt(case: dict, *, text_only: bool = False) -> str:
    """Combine case task prompt with structured output protocol.

    Args:
        case: Case dict with task_prompt, decision_space, and optionally page_spec.
        text_only: If True and page_spec exists, append a textual description
            of the page content so text-only debug runs can distinguish
            page-grounded cases (e.g., T3 in-scope vs out-of-scope).
    """
    labels = " | ".join(case["decision_space"])
    parts = [case["task_prompt"]]

    if text_only and case.get("page_spec"):
        ps = case["page_spec"]
        facts = ", ".join(f"{k}={v}" for k, v in ps.items())
        parts.append(f"\n[Current page context: {facts}]")

    parts.append(
        f"\n\nReply with exactly:\n"
        f"DECISION=<one of: {labels}>\n"
        f"MEMORY_USED=YES|NO\n"
        f"RATIONALE=<one sentence explaining your choice>"
    )
    return "".join(parts)


def _strip_markdown(s: str) -> str:
    """Strip common markdown formatting wrappers from a string.

    Handles: **bold**, *italic*, `inline code`, combinations thereof.
    Preserves the inner text.
    """
    # Strip bold/italic: **...**  *...*  ***...***
    s = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', s)
    # Strip inline code: `...`
    s = re.sub(r'`([^`]+)`', r'\1', s)
    return s.strip()


def normalize_decision(raw: str, decision_space: list[str]) -> str | None:
    """Case-insensitive match of raw decision against valid labels.

    Handles: extra whitespace, underscores vs hyphens vs spaces,
    surrounding quotes, multi-word labels, markdown wrappers.
    """
    if not raw:
        return None
    cleaned = _strip_markdown(raw)
    cleaned = cleaned.strip().strip("'\"").upper().replace("-", "_").replace(" ", "_")
    for label in decision_space:
        if cleaned == label.upper():
            return label
    return None


def parse_response(text: str, decision_space: list[str]) -> dict:
    """Parse DECISION, MEMORY_USED, RATIONALE from agent response.

    Robust to markdown formatting (**bold**, `code`, *italic*) around
    field names and values. Strips wrappers before matching.

    Returns dict with keys:
        decision:     str | None   (normalized label from decision_space)
        memory_used:  bool | None
        rationale:    str | None
        parse_ok:     bool
    """
    result = {"decision": None, "memory_used": None, "rationale": None, "parse_ok": False}

    if not text:
        return result

    # Pre-clean: strip markdown from the entire response so that
    # **DECISION=AISLE** becomes DECISION=AISLE before regex matching
    cleaned = _strip_markdown(text)

    # DECISION — capture until end of line to handle multi-word labels
    m_dec = re.search(r'DECISION\s*=\s*(.+)', cleaned, re.IGNORECASE)
    if m_dec:
        raw_decision = m_dec.group(1).strip()
        # Strip trailing MEMORY_USED if on same line
        raw_decision = re.split(r'\s+MEMORY_USED\s*=', raw_decision, flags=re.IGNORECASE)[0].strip()
        # Strip any remaining markdown from the value itself
        raw_decision = _strip_markdown(raw_decision)
        result["decision"] = normalize_decision(raw_decision, decision_space)

    # MEMORY_USED
    m_mem = re.search(r'MEMORY_USED\s*=\s*(YES|NO)', cleaned, re.IGNORECASE)
    if m_mem:
        result["memory_used"] = m_mem.group(1).upper() == "YES"

    # RATIONALE
    m_rat = re.search(r'RATIONALE\s*=\s*(.+)', cleaned, re.IGNORECASE)
    if m_rat:
        result["rationale"] = _strip_markdown(m_rat.group(1).strip())

    result["parse_ok"] = result["decision"] is not None
    return result
