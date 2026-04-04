"""Memory file writer for the T × M personal memory study.

Writes compact user-state MEMORY.md to the agent workspace.
v1 uses root MEMORY.md only (no memory/*.md hierarchy).
Content is user-state (preferences, rules, facts), not chat summary.
"""

from __future__ import annotations
from pathlib import Path

# Map M-condition labels to case bank field names
_CONDITION_FIELDS = {
    "M0": "memory_none",
    "M1": "memory_correct",
    "M2": "memory_stale",
    "M3": "memory_irrelevant",
}


def select_memory_text(case: dict, condition: str) -> str:
    """Return the MEMORY.md content string for a given M-condition."""
    field = _CONDITION_FIELDS[condition]
    return case[field]


def write_memory_md(workspace: Path, memory_text: str) -> None:
    """Write compact user-state to root MEMORY.md.

    This is the only memory file used in v1. No daily notes or
    memory/*.md hierarchy — keeps the variable clean.

    Args:
        workspace: Agent workspace path (e.g. ~/.openclaw/workspace-<agent_id>)
        memory_text: Full MEMORY.md content in user-state style
    """
    md_path = workspace / "MEMORY.md"
    md_path.write_text(memory_text, encoding="utf-8")
