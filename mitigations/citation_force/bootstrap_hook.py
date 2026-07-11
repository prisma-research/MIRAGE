"""
CitationForce — Bootstrap Hook

Injects a citation-force constraint section into OpenClaw's agent bootstrap
(BOOTSTRAP.md) to require the agent to cite source paths when referencing artifacts.

Conditions:
  C0: No mitigation (control)
  C1: System prompt constraint only (CITATION_FORCE_CONSTRAINT)
  C2: C1 + artifact_recall tool available via plugin
  C3: C2 + compaction re-injection
  Cm: Mandatory artifact_recall routing (MANDATORY_TOOL_CONSTRAINT) + tool available

The artifact_recall tool is packaged as an OpenClaw plugin
(openclaw_artifact_recall_plugin.ts) and condition-gated via a workspace marker
file. The plugin exposes artifact_recall for C2, C3, and Cm conditions.

Note: models may still prefer memory_search over artifact_recall even when both
are available. Cm uses stronger prompt language explicitly prohibiting
memory_search, but compliance is an empirical question, not a guarantee.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

WORKSPACE_DIR = Path.home() / ".openclaw" / "workspace"
OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"

CITATION_FORCE_CONSTRAINT = """\
## CitationForce Constraint

When referencing any artifact, file, screenshot, chart, or image from a previous
session or from memory, you MUST:

1. Use the `artifact_recall` tool (if available) or `memory_search` to retrieve
   the artifact explicitly.
2. In your response, include the source path or identifier of the artifact:
   > [Source: <artifact_id_or_path>]
3. Do NOT describe or paraphrase artifacts from memory without explicit retrieval.
4. If you cannot retrieve the artifact, say explicitly:
   "I was unable to locate the artifact. Please provide it again."

This constraint exists to ensure grounding fidelity — your responses must be
traceable to specific, retrievable artifacts.
"""

MANDATORY_TOOL_CONSTRAINT = """\
## CitationForce Constraint — Mandatory Tool Retrieval

When the user asks about any artifact, file, screenshot, chart, or image from a
previous session or from memory, you MUST follow this exact protocol:

1. ALWAYS call the `artifact_recall` tool FIRST with the artifact identifier or
   relevant keywords. Do NOT use `memory_search` or `read` to look for artifacts.
   The `artifact_recall` tool is the ONLY approved retrieval method for artifact
   queries under this protocol.
2. After calling `artifact_recall`, include the source path in your response:
   > [Source: <artifact_id_or_path>]
3. Do NOT describe or paraphrase artifacts without first calling `artifact_recall`.
4. If `artifact_recall` returns no results, say explicitly:
   "I was unable to locate the artifact. Please provide it again."

IMPORTANT: Using `memory_search` instead of `artifact_recall` is a protocol
violation. You must use `artifact_recall` as your retrieval tool.
"""

BOOTSTRAP_EXTRA_SECTION_HEADER = "## CitationForce Constraint"
CONDITION_MARKER_FILE = ".mirage_citation_force_condition"


def inject_citation_force_into_bootstrap(
    workspace_dir: Path | None = None,
    condition: str = "C1",
) -> Path:
    """
    Write the CitationForce constraint to BOOTSTRAP.md in the workspace.

    OpenClaw's bootstrap-extra-files hook reads BOOTSTRAP.md and includes
    it in the system prompt.

    Also writes a MIRAGE-local condition marker so the optional
    artifact_recall plugin can expose the tool only for C2/C3 workspaces
    without contaminating C0/C1 controls.

    Returns the path to BOOTSTRAP.md.
    """
    ws = workspace_dir or WORKSPACE_DIR
    bootstrap_path = ws / "BOOTSTRAP.md"
    marker_path = ws / CONDITION_MARKER_FILE

    existing = ""
    if bootstrap_path.exists():
        existing = bootstrap_path.read_text()

    marker_path.write_text(condition.strip())

    # Avoid duplicate injection
    if BOOTSTRAP_EXTRA_SECTION_HEADER in existing:
        return bootstrap_path

    # Select constraint text based on condition
    constraint = MANDATORY_TOOL_CONSTRAINT if condition == "Cm" else CITATION_FORCE_CONSTRAINT

    # Append constraint section
    updated = existing.rstrip() + "\n\n" + constraint
    bootstrap_path.write_text(updated)
    return bootstrap_path


def remove_citation_force_from_bootstrap(workspace_dir: Path | None = None) -> None:
    """Remove CitationForce section from BOOTSTRAP.md (cleanup after trial)."""
    ws = workspace_dir or WORKSPACE_DIR
    bootstrap_path = ws / "BOOTSTRAP.md"
    marker_path = ws / CONDITION_MARKER_FILE
    if not bootstrap_path.exists():
        return

    text = bootstrap_path.read_text()
    idx = text.find(BOOTSTRAP_EXTRA_SECTION_HEADER)
    if idx == -1:
        return

    cleaned = text[:idx].rstrip() + "\n"
    bootstrap_path.write_text(cleaned)
    if marker_path.exists():
        marker_path.unlink()


def build_citation_force_system_prompt(base_prompt: str = "") -> str:
    """Return a system prompt that includes the CitationForce constraint."""
    return base_prompt.rstrip() + "\n\n" + CITATION_FORCE_CONSTRAINT


def check_citation_force_in_system_prompt(system_prompt: str) -> bool:
    """Return True if the CitationForce constraint is present in the system prompt."""
    return BOOTSTRAP_EXTRA_SECTION_HEADER in system_prompt or "CitationForce" in system_prompt


def read_citation_force_condition_marker(workspace_dir: Path | None = None) -> str | None:
    """Return the MIRAGE-local CitationForce condition marker if present."""
    ws = workspace_dir or WORKSPACE_DIR
    marker_path = ws / CONDITION_MARKER_FILE
    if not marker_path.exists():
        return None
    value = marker_path.read_text().strip()
    return value or None
