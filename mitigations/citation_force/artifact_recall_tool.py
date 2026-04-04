"""
CitationForce — artifact_recall Tool

Server-side wrapper over memory_search that:
1. Returns source_paths alongside snippets (for deterministic Axis 2 scoring)
2. Can be registered as a custom tool in OpenClaw via the plugin SDK

This module defines the tool schema and a Python implementation that wraps
OpenClaw's memory_search gateway call.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# Tool schema in Anthropic tool-use format
ARTIFACT_RECALL_TOOL_SCHEMA = {
    "name": "artifact_recall",
    "description": (
        "Retrieve a specific artifact from memory by ID or keyword. "
        "Returns the artifact content along with its source path for citation. "
        "Use this instead of memory_search when you need to cite an artifact in your response."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "The artifact ID or path to retrieve (e.g., 'screenshot_abc123').",
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional additional keywords to search for the artifact.",
            },
        },
        "required": [],
    },
}


def artifact_recall(
    artifact_id: str = "",
    keywords: list[str] | None = None,
    workspace_dir: Path | None = None,
) -> dict:
    """
    Python implementation of artifact_recall.

    Searches workspace memory files for the artifact and returns:
    {
        "text": "<artifact content>",
        "source_paths": ["<path1>", ...],
        "artifact_id": "<matched_id>",
        "found": true/false
    }
    """
    from pathlib import Path as P

    ws = workspace_dir or (P.home() / ".openclaw" / "workspace")

    search_terms = []
    if artifact_id:
        search_terms.append(artifact_id)
    if keywords:
        search_terms.extend(keywords)

    if not search_terms:
        return {"text": "", "source_paths": [], "artifact_id": artifact_id, "found": False}

    results = []
    for md_file in _iter_memory_files(ws):
        content = md_file.read_text(errors="replace")
        for term in search_terms:
            if term and term in content:
                # Extract surrounding context
                idx = content.find(term)
                snippet = content[max(0, idx - 200) : idx + 500].strip()
                results.append({
                    "path": str(md_file),
                    "snippet": snippet,
                })
                break  # one match per file

    if results:
        combined_text = "\n\n---\n\n".join(
            f"[Source: {r['path']}]\n{r['snippet']}" for r in results
        )
        return {
            "text": combined_text,
            "source_paths": [r["path"] for r in results],
            "artifact_id": artifact_id,
            "found": True,
        }

    return {
        "text": "",
        "source_paths": [],
        "artifact_id": artifact_id,
        "found": False,
    }


def _iter_memory_files(ws: Path):
    """Yield all markdown files in the workspace that may contain artifacts."""
    # MEMORY.md
    memory_md = ws / "MEMORY.md"
    if memory_md.exists():
        yield memory_md

    # Daily memory files
    memory_dir = ws / "memory"
    if memory_dir.exists():
        for f in sorted(memory_dir.glob("*.md"), reverse=True):
            yield f

    # Other top-level md files
    for f in ws.glob("*.md"):
        if f.name not in ("MEMORY.md",):
            yield f
