"""
Planting + S1 / S2 / S3 harness with per-trial workspace isolation.

Orchestrates:
  1. Create isolated agent with clean workspace (snapshot restore)
  2. Planting: inject artifact via natural interaction, observe planting_path
  3. Reindex memory
  4. S1 test (same session, in-context) / S2 test (compaction) / S3 test (session reset)
  5. Cleanup: delete the isolated agent
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from client.openclaw_client import OpenClawClient, OPENCLAW_STATE_DIR, DEFAULT_AGENT_ID
from harness.trial_log import GroundingTrial, TrialConfig, save_trial
from harness.filler_turns import get_filler_turns
from constants import (
    S2_CONTEXT_THRESHOLD,
    COMPACTION_TRIGGER_MAX,
    COMPACTION_POLL_TIMEOUT,
    VLM_MODEL,
    SNAPSHOT_ID,
    RESERVE_TOKENS_FLOOR,
    EXPERIMENT_MODEL_REGISTRY,
)

logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).parent.parent / "logs" / "trials"

# Serialise `openclaw agents add/delete` calls per state dir — concurrent writes
# to the same openclaw.json corrupt the agent registry.
_AGENT_REGISTRY_LOCKS: dict[str, asyncio.Lock] = {}

def _registry_lock(state_dir: Path | None = None) -> asyncio.Lock:
    key = str(state_dir or OPENCLAW_STATE_DIR)
    if key not in _AGENT_REGISTRY_LOCKS:
        _AGENT_REGISTRY_LOCKS[key] = asyncio.Lock()
    return _AGENT_REGISTRY_LOCKS[key]

# Baseline snapshot: clean workspace state used to initialize each trial
BASELINE_SNAPSHOT_DIR = Path(__file__).parent.parent / "baseline_snapshot"


# ---------------------------------------------------------------------------
# Workspace isolation
# ---------------------------------------------------------------------------

def _compute_workspace_hash(workspace: Path) -> str:
    """Compute a hash of all .md files in the workspace for contamination check.

    Covers MEMORY.md, memory/*.md, and any other .md files the agent may create.
    """
    h = hashlib.sha256()
    if not workspace.exists():
        return h.hexdigest()[:16]
    for f in sorted(workspace.rglob("*.md")):
        # Use path relative to workspace for determinism
        h.update(str(f.relative_to(workspace)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def create_baseline_snapshot(source_workspace: Path | None = None) -> Path:
    """
    Create a baseline snapshot from the default workspace.

    The snapshot contains AGENTS.md, SOUL.md, USER.md, IDENTITY.md,
    TOOLS.md, HEARTBEAT.md, BOOTSTRAP.md and an empty memory/ directory.
    No artifacts, no MEMORY.md content, no daily memory files.
    """
    source = source_workspace or (OPENCLAW_STATE_DIR / "workspace")
    BASELINE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy system files (not memory content)
    system_files = [
        "AGENTS.md", "SOUL.md", "USER.md", "IDENTITY.md",
        "TOOLS.md", "HEARTBEAT.md", "BOOTSTRAP.md",
    ]
    for fname in system_files:
        src = source / fname
        if src.exists():
            shutil.copy2(src, BASELINE_SNAPSHOT_DIR / fname)

    # Create empty memory directory
    (BASELINE_SNAPSHOT_DIR / "memory").mkdir(exist_ok=True)

    logger.info("Baseline snapshot created at %s", BASELINE_SNAPSHOT_DIR)
    return BASELINE_SNAPSHOT_DIR


async def _setup_trial_agent(
    trial_id: str,
    model: str | None = None,
    workspace_source: Path | None = None,
    state_dir: Path | None = None,
) -> str:
    """
    Create an isolated OpenClaw agent for this trial.

    Args:
        trial_id: Used to generate a unique agent_id.
        model: Optional model override (e.g. "volcengine/deepseek-v3-250324").
        workspace_source: If provided, copy workspace from this path instead of baseline snapshot.
        state_dir: OpenClaw state dir (gateway isolation). Defaults to OPENCLAW_STATE_DIR.

    Returns the agent_id.

    Note: Provider baseUrl (including proxy routing) and contextWindow are
    inherited from the global openclaw.json.  Use UsageProxy context manager
    at the process level to patch the global config before creating agents.
    """
    effective_state_dir = state_dir or OPENCLAW_STATE_DIR
    agent_id = f"gb_{hashlib.sha256(trial_id.encode()).hexdigest()[:12]}"
    workspace_path = effective_state_dir / f"workspace-{agent_id}"

    # Restore workspace from source or baseline snapshot
    source = workspace_source or BASELINE_SNAPSHOT_DIR
    if not source.exists():
        if workspace_source:
            raise RuntimeError(f"Workspace source not found: {source}")
        create_baseline_snapshot()
        source = BASELINE_SNAPSHOT_DIR

    if workspace_path.exists():
        shutil.rmtree(workspace_path)
    shutil.copytree(source, workspace_path)

    # Register agent with OpenClaw (with optional model override).
    # Serialised per state dir — concurrent writes to the same openclaw.json corrupt the registry.
    cmd = ["openclaw", "agents", "add", agent_id, "--workspace", str(workspace_path)]
    if model:
        cmd.extend(["--model", model])
    cli_env = {**os.environ, "OPENCLAW_STATE_DIR": str(effective_state_dir)}
    async with _registry_lock(effective_state_dir):
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=cli_env,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode().strip() or stdout.decode().strip()
            if "already exists" not in err.lower():
                raise RuntimeError(f"Failed to create agent {agent_id}: {err}")

    # Provider baseUrl and contextWindow are inherited from the global
    # openclaw.json, patched by UsageProxy.__enter__() before any agents
    # are created.  No per-agent models.json write needed.

    logger.info(
        "Trial agent created: %s (workspace=%s)",
        agent_id, workspace_path,
    )
    return agent_id


async def _teardown_trial_agent(agent_id: str, state_dir: Path | None = None) -> None:
    """Delete the trial agent and its workspace."""
    effective_state_dir = state_dir or OPENCLAW_STATE_DIR
    workspace_path = effective_state_dir / f"workspace-{agent_id}"
    cli_env = {**os.environ, "OPENCLAW_STATE_DIR": str(effective_state_dir)}

    async with _registry_lock(effective_state_dir):
        proc = await asyncio.create_subprocess_exec(
            "openclaw", "agents", "delete", agent_id, "--force",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=cli_env,
        )
        await proc.communicate()

    if workspace_path.exists():
        shutil.rmtree(workspace_path, ignore_errors=True)

    logger.info("Trial agent cleaned up: %s", agent_id)


async def _reindex_memory(agent_id: str = DEFAULT_AGENT_ID, state_dir: Path | None = None) -> None:
    """Force reindex memory files so memory_search reflects Planting writes."""
    effective_state_dir = state_dir or OPENCLAW_STATE_DIR
    cli_env = {**os.environ, "OPENCLAW_STATE_DIR": str(effective_state_dir)}
    proc = await asyncio.create_subprocess_exec(
        "openclaw", "memory", "index", "--force", "--agent", agent_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=cli_env,
    )
    await proc.communicate()
    logger.info("Memory index updated (agent=%s)", agent_id)


# ---------------------------------------------------------------------------
# Artifact record (produced by planters)
# ---------------------------------------------------------------------------

@dataclass
class ArtifactRecord:
    artifact_id: str
    plant_type: str
    ocr_keywords: list[str]
    session_a_id: str
    planting_path: Literal["explicit_write", "pre_compaction_flush", "none"] = "none"
    raw_session_a_messages: list[dict] = field(default_factory=list)
    jsonl_entry_count_after_a: int = 0  # split point for S1 / S2 / S3
    actual_artifact_repr: str = ""  # agent's actual memory content (for Axis 3)


# ---------------------------------------------------------------------------
# Retrieval result (S1 / S2 / S3)
# ---------------------------------------------------------------------------

@dataclass
class SessionBResult:
    session_b_id: str
    system_prompt_at_init: str
    tool_trace: list[dict]
    response_text: str
    compaction_during_session_b: bool
    citation_force_reinjected: bool | None
    startup_exposure_log: list[dict] = field(default_factory=list)
    active_context_view: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Planting path detection
# ---------------------------------------------------------------------------

def observe_planting_path(
    artifact_id: str,
    workspace_path: Path,
) -> Literal["explicit_write", "pre_compaction_flush", "none"]:
    """
    Check workspace for artifact_id to determine how it was persisted.
    """
    # Check MEMORY.md
    memory_md = workspace_path / "MEMORY.md"
    if memory_md.exists() and artifact_id in memory_md.read_text():
        return "pre_compaction_flush"

    # Check daily memory files (match by filename OR content)
    memory_dir = workspace_path / "memory"
    if memory_dir.exists():
        for md_file in sorted(memory_dir.glob("*.md"), reverse=True):
            if artifact_id in md_file.name or artifact_id in md_file.read_text():
                return "pre_compaction_flush"

    # Check other workspace files (match by filename OR content)
    for md_file in workspace_path.glob("**/*.md"):
        if artifact_id in md_file.name or artifact_id in md_file.read_text():
            return "explicit_write"

    return "none"


def extract_actual_artifact_repr(
    artifact_id: str,
    workspace_path: Path,
) -> str:
    """Read the agent's actual memory content for artifact_id after Planting.

    This is what the agent wrote to memory files — used as artifact_repr for
    Axis 3 scoring instead of the planter's pre-canned text, to avoid
    format mismatch (e.g. CSV vs human-readable numbers).
    """
    # Check MEMORY.md
    memory_md = workspace_path / "MEMORY.md"
    if memory_md.exists():
        text = memory_md.read_text()
        if artifact_id in text:
            return text

    # Check daily memory files
    memory_dir = workspace_path / "memory"
    if memory_dir.exists():
        for md_file in sorted(memory_dir.glob("*.md"), reverse=True):
            text = md_file.read_text()
            if artifact_id in text:
                return text

    # Check other workspace files
    for md_file in workspace_path.glob("**/*.md"):
        text = md_file.read_text()
        if artifact_id in text:
            return text

    return ""


# ---------------------------------------------------------------------------
# Planting runner
# ---------------------------------------------------------------------------

async def run_planting(
    config: TrialConfig,
    plant_prompt: str,
    artifact_id: str,
    artifact_ocr_keywords: list[str],
    agent_id: str = DEFAULT_AGENT_ID,
    image_path: str | None = None,
    state_dir: Path | None = None,
) -> ArtifactRecord:
    """Run the Planting turn: inject artifact via natural interaction, observe planting_path."""
    async with OpenClawClient(agent_id=agent_id, state_dir=state_dir) as client:
        session_id = await client.start_session(reason="new")
        logger.info("Planting started: %s (agent=%s)", session_id, agent_id)

        # Copy image into the agent workspace so OpenClaw's media allowlist accepts it.
        # The gateway only permits reads from paths under the state dir (workspace/, agents/, etc.).
        # The source image lives outside those dirs, causing "not under an allowed directory" errors.
        local_image_path = image_path
        if image_path:
            dest = client.workspace_dir / Path(image_path).name
            shutil.copy2(image_path, dest)
            local_image_path = str(dest)
            logger.debug("Image copied to workspace: %s", dest)

        response = await client.send_message(
            plant_prompt, session_id=session_id, image_path=local_image_path,
        )
        logger.debug("Planting response: %s", str(response)[:200])

        await asyncio.sleep(2.0)

        planting_path = observe_planting_path(artifact_id, client.workspace_dir)
        jsonl = client.get_session_jsonl(session_id)

        # Extract what agent actually saved to memory (for Axis 3 artifact_repr)
        actual_repr = extract_actual_artifact_repr(artifact_id, client.workspace_dir)
        if actual_repr:
            logger.info("Extracted actual artifact repr from memory (%d chars)", len(actual_repr))

        # Record JSONL entry count as split point for S1 / S2 / S3
        jsonl_count = len(jsonl)
        logger.info("Planting complete: %d JSONL entries (split point for S1/S2/S3)", jsonl_count)

        return ArtifactRecord(
            artifact_id=artifact_id,
            plant_type=config.plant_type,
            ocr_keywords=artifact_ocr_keywords,
            session_a_id=session_id,
            planting_path=planting_path,
            raw_session_a_messages=jsonl,
            jsonl_entry_count_after_a=jsonl_count,
            actual_artifact_repr=actual_repr,
        )


# ---------------------------------------------------------------------------
# S1 test runner (same session as Planting — no reset)
# ---------------------------------------------------------------------------

async def run_s1_test(
    config: TrialConfig,
    artifact: ArtifactRecord,
    query: str,
    agent_id: str = DEFAULT_AGENT_ID,
    state_dir: Path | None = None,
) -> SessionBResult:
    """Run S1 test in the SAME session as Planting (no reset).

    The artifact is still in the active context window.
    Expected R_path = R_context (no retrieval needed).
    """
    async with OpenClawClient(agent_id=agent_id, state_dir=state_dir) as client:
        # Use the SAME session as Planting (no start_session call)
        session_id = artifact.session_a_id
        logger.info("S1 test in session: %s (agent=%s)", session_id, agent_id)

        # Record split point: everything after this is S1
        s1_split = artifact.jsonl_entry_count_after_a

        # Inject filler turns until effective input tokens reach the absolute target.
        # history_depth is an absolute effective-input-tokens target (input + cacheRead).
        # Same base seed as S2/S3 compaction phase — S1 uses the prefix of that stream.
        filler_seed = config.seed
        if config.history_depth > 0:
            filler_pool = get_filler_turns(n=50, seed=filler_seed, category="substantive")
            for msg in itertools.cycle(filler_pool):
                eit = client.get_effective_input_tokens(session_id)
                if eit is None:
                    raise RuntimeError(
                        "S1 filler: provider input/cacheRead unavailable — cannot proceed"
                    )
                if eit >= config.history_depth:
                    break
                await client.send_message(msg, session_id=session_id)

        # Send reference query
        response = await client.send_message(query, session_id=session_id)
        response_text = _extract_response_text(response)

        # Read full JSONL and slice to S1 portion
        full_jsonl = client.get_session_jsonl(session_id)
        s1_jsonl = full_jsonl[s1_split:]
        logger.info("S1 JSONL split: total=%d, session_a=%d, s1=%d", len(full_jsonl), s1_split, len(s1_jsonl))

        tool_trace = _extract_tool_calls(s1_jsonl)

        # Build active context view: everything the model could see at query time
        # (planting + all filler turns), excluding the query turn itself.
        # Using full_jsonl[:s1_split] was wrong — it omitted filler, making depth irrelevant.
        pre_query_jsonl = _slice_before_query(full_jsonl, query)
        active_context_view = _extract_active_context_view(pre_query_jsonl)

        return SessionBResult(
            session_b_id=session_id,
            system_prompt_at_init="",  # No new system prompt in S1
            tool_trace=tool_trace,
            response_text=response_text,
            compaction_during_session_b=False,
            citation_force_reinjected=None,
            startup_exposure_log=[],
            active_context_view=active_context_view,
        )


# ---------------------------------------------------------------------------
# S2 test runner (same session as S1 — context-threshold compaction)
# ---------------------------------------------------------------------------

# Imported from constants.py: S2_CONTEXT_THRESHOLD


async def run_s2_test(
    config: TrialConfig,
    artifact: ArtifactRecord,
    query: str,
    agent_id: str,
    session_id: str,
    pre_query_jsonl: list[dict],
    compaction_verified: bool,
    state_dir: Path | None = None,
) -> SessionBResult:
    """Run S2 query in the existing compaction session (no new session started).

    Receives the pre-existing session from _run_native_compaction_phase and sends
    the reference query, reusing the compacted context already in that session.
    active_context_view is populated from pre_query_jsonl so R_context scoring
    can detect the artifact in the compaction summary.
    """
    async with OpenClawClient(agent_id=agent_id, state_dir=state_dir) as client:
        logger.info("S2 query in existing session: %s (agent=%s)", session_id, agent_id)

        # Send reference query into the existing compaction session
        response = await client.send_message(query, session_id=session_id)
        response_text = _extract_response_text(response)

        full_jsonl = client.get_session_jsonl(session_id)
        # Only attribute post-query tool calls to S2
        post_query_jsonl = full_jsonl[len(pre_query_jsonl):]
        tool_trace = _extract_tool_calls(post_query_jsonl)

        active_context_view = _extract_post_compaction_context(pre_query_jsonl)
        citation_force_reinjected = (
            _detect_citation_force_reinject(full_jsonl) if compaction_verified else None
        )

        return SessionBResult(
            session_b_id=session_id,
            system_prompt_at_init="",
            tool_trace=tool_trace,
            response_text=response_text,
            compaction_during_session_b=compaction_verified,
            citation_force_reinjected=citation_force_reinjected,
            startup_exposure_log=[],
            active_context_view=active_context_view,
        )


def _extract_startup_exposure_log(jsonl: list[dict], query_text: str) -> list[dict]:
    """Extract tool results from bootstrap turns that occur before the reference query.

    In an S3 session the agent may execute bootstrap tool calls (exec cat MEMORY.md,
    memory_search) before the user's reference query arrives.  These pre-query results
    are the primary detection signal for R_boot.  We collect all tool-result entries
    that appear before the first user message containing the query text.
    """
    early: list[dict] = []
    for entry in jsonl:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        role = msg.get("role", "")

        # Stop collecting once we reach the user's reference query turn
        if role == "user":
            content = msg.get("content", [])
            text = ""
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text += c.get("text", "")
            elif isinstance(content, str):
                text = content
            if query_text and query_text[:60] in text:
                break

        if role == "toolResult":
            content = msg.get("content", [])
            text = ""
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text += c.get("text", "")
            elif isinstance(content, str):
                text = content
            early.append({
                "kind": "result",
                "tool_use_id": msg.get("toolCallId"),
                "name": msg.get("toolName"),
                "text": text,
            })
    return early


def _extract_active_context_view(jsonl: list[dict]):
    """Extract text content from conversation turns for R_context detection."""
    history = []
    for entry in jsonl:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        role = msg.get("role", "")
        if role in ("user", "assistant"):
            content = msg.get("content", [])
            text = ""
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text += c.get("text", "")
            elif isinstance(content, str):
                text = content
            if text:
                history.append({"role": role, "text": text})
    return history


# ---------------------------------------------------------------------------
# S3 runner (session reset)
# ---------------------------------------------------------------------------

def _slice_before_query(jsonl: list[dict], query: str) -> list[dict]:
    """Return only the JSONL entries that appear before the reference query user turn.

    Used to build S3's active_context_view from bootstrap-only context,
    excluding the model's response (which would cause R_context false positives).
    """
    prefix = query[:60]
    for i, entry in enumerate(jsonl):
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        text = ""
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text += c.get("text", "")
        elif isinstance(content, str):
            text = content
        if prefix in text:
            return jsonl[:i]
    return jsonl  # fallback: query not found, return full list


async def run_s3_test(
    config: TrialConfig,
    artifact: ArtifactRecord,
    query: str,
    agent_id: str = DEFAULT_AGENT_ID,
    state_dir: Path | None = None,
) -> SessionBResult:
    """Run S3 test: reset session (same agent/workspace), send reference query.

    No warmup turns are added so that S2 and S3 differ only in whether the
    session was reset — the sole controlled variable in the paired comparison.
    """
    async with OpenClawClient(agent_id=agent_id, state_dir=state_dir) as client:
        session_id = await client.start_session(reason="new")
        logger.info("S3 session started: %s (agent=%s)", session_id, agent_id)

        system_prompt_at_init = client.get_system_prompt(session_id)

        response = await client.send_message(query, session_id=session_id)
        response_text = _extract_response_text(response)

        # S3 uses a fresh agent — all JSONL entries belong to S3
        session_b_jsonl = client.get_session_jsonl(session_id)

        startup_exposure_log = _extract_startup_exposure_log(session_b_jsonl, query)
        # active_context_view must be pre-query only — using the full session (including
        # the model's response) causes R_context false positives when the response text
        # happens to contain OCR keywords (e.g. through hallucination).
        # S3 expected R_path ∈ {R_boot, R_tool, R_none}; R_context must never fire here.
        pre_query_jsonl = _slice_before_query(session_b_jsonl, query)
        active_context_view = _extract_active_context_view(pre_query_jsonl)
        # tool_trace must be post-query only — bootstrap can issue retrieval calls
        # (e.g. memory_search during startup) before the query turn. Including those
        # in tool_trace would cause R_tool false positives when R_boot does not fire.
        post_query_jsonl = session_b_jsonl[len(pre_query_jsonl):]
        tool_trace = _extract_tool_calls(post_query_jsonl)

        compaction_during_session_b = _detect_compaction(session_b_jsonl)
        citation_force_reinjected = _detect_citation_force_reinject(session_b_jsonl) if compaction_during_session_b else None

        return SessionBResult(
            session_b_id=session_id,
            system_prompt_at_init=system_prompt_at_init,
            tool_trace=tool_trace,
            startup_exposure_log=startup_exposure_log,
            active_context_view=active_context_view,
            response_text=response_text,
            compaction_during_session_b=compaction_during_session_b,
            citation_force_reinjected=citation_force_reinjected,
        )


# ---------------------------------------------------------------------------
# Tool trace / compaction helpers
# ---------------------------------------------------------------------------

def _extract_response_text(payload: dict) -> str:
    """Extract the assistant's text from a chat.send or openclaw agent payload."""
    if not payload:
        return ""

    result = payload.get("result", {})
    if "payloads" in result:
        parts = []
        for p in result["payloads"]:
            if isinstance(p, dict) and p.get("text"):
                parts.append(p["text"])
        if parts:
            return "\n".join(parts)

    if "text" in payload:
        return payload["text"]

    if "content" in payload:
        content = payload["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(c.get("text", ""))
            return " ".join(parts)

    return str(payload)


def _extract_tool_calls(jsonl: list[dict], after_turn_index: int = 0) -> list[dict]:
    """Extract all tool calls and results from JSONL."""
    tool_calls = []
    for entry in jsonl:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        role = msg.get("role", "")

        if role == "assistant":
            for c in msg.get("content", []):
                if not isinstance(c, dict):
                    continue
                if c.get("type") in ("toolCall", "tool_use"):
                    tool_calls.append({
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
            tool_calls.append({
                "kind": "result",
                "tool_use_id": msg.get("toolCallId"),
                "name": msg.get("toolName"),
                "text": text,
                "is_error": msg.get("isError", False),
                "details": msg.get("details"),
                "entry_id": entry.get("id"),
            })

    return tool_calls


def _detect_compaction(jsonl: list[dict]) -> bool:
    for entry in jsonl:
        ct = entry.get("customType", "")
        if ct in ("compaction", "compact", "context-compaction"):
            return True
        if entry.get("type") == "custom":
            data = entry.get("data", {})
            if "compact" in str(data).lower() or "compaction" in str(data).lower():
                return True
    return False


def _detect_citation_force_reinject(jsonl: list[dict]) -> bool | None:
    for entry in jsonl:
        if entry.get("type") == "custom":
            data = entry.get("data", {})
            sections = data.get("postCompactionSections", [])
            if sections:
                for section in sections:
                    if "citation" in str(section).lower() or "CITATION_FORCE" in str(section):
                        return True
                return False
    return None


def _verify_memory_write(workspace_path: Path, artifact_id: str) -> bool:
    """Check that artifact_id appears in workspace memory files post-compaction."""
    candidates = [workspace_path / "MEMORY.md"] + list((workspace_path / "memory").glob("*.md"))
    for path in candidates:
        if path.exists() and artifact_id in path.read_text():
            return True
    return False


def _sync_root_memory_md(workspace_path: Path) -> None:
    """Create/update MEMORY.md at workspace root from memory/*.md content.

    OpenClaw's fresh-session bootstrap reads MEMORY.md at workspace root, but
    the agent writes memory files into the memory/ subdirectory.  This syncs
    all memory/*.md content into a root-level MEMORY.md so the S3 startup
    bootstrap finds artifact content and populates startup_exposure_log
    (enabling R_boot detection).
    """
    memory_dir = workspace_path / "memory"
    if not memory_dir.exists():
        return

    import re as _re
    _date_pat = _re.compile(r'^\d{4}-\d{2}-\d{2}')
    all_files = list(memory_dir.glob("*.md"))
    # Artifact files (non-date-named) first so they aren't truncated by
    # OpenClaw's read limit when the S3 startup bootstrap reads MEMORY.md.
    artifact_files = sorted(f for f in all_files if not _date_pat.match(f.name))
    date_files = sorted(f for f in all_files if _date_pat.match(f.name))
    parts = []
    for md_file in artifact_files + date_files:
        try:
            text = md_file.read_text().strip()
            if text:
                parts.append(f"<!-- {md_file.name} -->\n{text}")
        except OSError:
            pass

    if not parts:
        return

    root_memory = workspace_path / "MEMORY.md"
    root_memory.write_text("\n\n".join(parts) + "\n")
    logger.debug("Synced %d memory files → MEMORY.md (%s)", len(parts), workspace_path)


def _extract_post_compaction_context(jsonl: list[dict]) -> list[dict]:
    """Extract the post-compaction context for S2 active_context_view.

    Finds the last compaction entry, extracts its summary text, then collects
    all user/assistant messages after that entry. Falls back to
    _extract_active_context_view if no compaction is detected.
    """
    compaction_idx = None
    for i, entry in enumerate(jsonl):
        ct = entry.get("customType", "")
        if ct in ("compaction", "compact", "context-compaction"):
            compaction_idx = i
        elif entry.get("type") == "custom":
            data = entry.get("data", {})
            if "compact" in str(data).lower() or "compaction" in str(data).lower():
                compaction_idx = i

    if compaction_idx is None:
        return _extract_active_context_view(jsonl)  # fallback: no compaction found

    context = []
    # Extract compaction summary text
    data = jsonl[compaction_idx].get("data", {})
    summary = (data.get("summary") or data.get("content") or
               data.get("compacted_content") or data.get("postCompactionSummary") or "")
    if summary:
        context.append({"role": "assistant", "text": str(summary)})

    # Extract messages after compaction
    for entry in jsonl[compaction_idx + 1:]:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        role = msg.get("role", "")
        if role in ("user", "assistant"):
            text = ""
            content = msg.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text += c.get("text", "")
            elif isinstance(content, str):
                text = content
            if text:
                context.append({"role": role, "text": text})
    return context


# Imported from constants.py: COMPACTION_TRIGGER_MAX


_MANUAL_FLUSH_PROMPT = (
    "Before we continue, please update your memory files to record everything important "
    "about the artifact you processed in this session — including its artifact ID, full "
    "content details, and any key observations. Use your write/edit tools to update "
    "MEMORY.md and any relevant memory files now."
)


async def force_compaction_and_flush(
    client: OpenClawClient,
    session_id: str,
    workspace_path: Path,
    artifact_id: str,
    poll_timeout: float = COMPACTION_POLL_TIMEOUT,
) -> bool:
    """Flush artifact info to memory files via manual prompt (always).

    Always sends the manual memory-flush prompt regardless of whether native
    compaction fired. This ensures consistent behaviour across all backends
    (gpt-4o-mini, Claude, DeepSeek, etc.) — we never rely on automatic compaction.
    """
    logger.info("Sending manual memory-flush prompt")
    await client.send_message(_MANUAL_FLUSH_PROMPT, session_id=session_id)

    ok = _verify_memory_write(workspace_path, artifact_id)
    logger.info("Manual memory flush complete; memory_write_verified=%s", ok)
    return ok


# ---------------------------------------------------------------------------
# Full trial orchestrator
# ---------------------------------------------------------------------------

# Imported from constants.py: VLM_MODEL


async def _run_native_compaction_phase(
    config: TrialConfig,
    artifact: ArtifactRecord,
    agent_id: str,
    session_id: str,
    filler_offset: int = 0,
    state_dir: Path | None = None,
    max_filler_turns: int = 200,
) -> tuple[bool, list[dict], int, int | None, bool]:
    """Inject filler turns until OpenClaw fires native compaction, return session state.

    contextWindow was already set in openclaw.json by UsageProxy(context_window_override=...).
    OpenClaw fires compaction natively when context approaches contextWindow - reserveTokensFloor.

    No harness-side token estimation involved — we poll sessions.json.compactionCount.

    Returns:
        (compaction_verified, pre_query_jsonl, compaction_count, boundary_input_tokens, manual_flush_fallback_used)
    """
    async with OpenClawClient(agent_id=agent_id, state_dir=state_dir) as client:
        logger.info("Native compaction phase in session: %s (agent=%s)", session_id, agent_id)

        initial_cc = client.get_session_compaction_count(session_id)
        filler_seed = config.seed
        filler_pool = get_filler_turns(n=50, seed=filler_seed, category="substantive")

        turns_injected = 0
        compaction_fired = False
        for msg in itertools.islice(itertools.cycle(filler_pool[filler_offset:]), max_filler_turns):
            await client.send_message(msg, session_id=session_id)
            turns_injected += 1
            current_cc = client.get_session_compaction_count(session_id)
            if current_cc > initial_cc:
                logger.info(
                    "Native compaction fired after %d filler turns (compactionCount: %d → %d)",
                    turns_injected, initial_cc, current_cc,
                )
                compaction_fired = True
                break

        if not compaction_fired:
            raise RuntimeError(
                f"Native compaction did not fire within {max_filler_turns} filler turns "
                f"(session={session_id}, agent={agent_id}, initial_cc={initial_cc})"
            )

        # Record effective input tokens at the compaction boundary
        boundary_input_tokens = client.get_effective_input_tokens(session_id)
        final_cc = client.get_session_compaction_count(session_id)

        # Verify artifact persistence via existing _verify_memory_write
        compaction_verified = _verify_memory_write(client.workspace_dir, artifact.artifact_id)
        manual_flush_fallback_used = False

        if not compaction_verified:
            # Native memoryFlush didn't persist artifact — send manual flush as fallback
            logger.warning(
                "Native memoryFlush did not persist artifact %s — sending manual flush fallback",
                artifact.artifact_id,
            )
            await client.send_message(_MANUAL_FLUSH_PROMPT, session_id=session_id)
            compaction_verified = _verify_memory_write(client.workspace_dir, artifact.artifact_id)
            manual_flush_fallback_used = True
            if not compaction_verified:
                logger.warning("Manual flush fallback also failed for artifact %s", artifact.artifact_id)

        pre_query_jsonl = client.get_session_jsonl(session_id)
        return compaction_verified, pre_query_jsonl, final_cc, boundary_input_tokens, manual_flush_fallback_used


async def build_trial(
    config: TrialConfig,
    plant_prompt: str,
    artifact_id: str,
    artifact_ocr_keywords: list[str],
    query: str,
    output_dir: Path | None = None,
    scenarios: list[str] | None = None,
    image_path: str | None = None,
    retrieval_model: str | None = None,
    citation_force_condition: str = "C0",
    gateway_state_dir: Path | None = None,
    context_threshold: int | None = None,
) -> list[GroundingTrial]:
    """
    Orchestrate: VLM plants artifact → target model retrieves.

    When image_path is provided and retrieval_model differs from VLM:
      1. Create VLM agent → Planting (inject image artifact) → save workspace snapshot
      2. Create retrieval agent (target model) from snapshot → S1/S3 tests
      3. Cleanup both agents

    Args:
        scenarios: Which scenarios to run. Default ["S3"]. Options: ["S1", "S3"] or ["S1"].
        retrieval_model: Model for S1/S2/S3 retrieval. If None, uses gateway default (same model for planting and retrieval).

    Returns:
        List of GroundingTrial records (one per scenario).
    """
    if scenarios is None:
        scenarios = ["S3"]

    sd = gateway_state_dir or OPENCLAW_STATE_DIR
    trial_base_id = f"trial_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Determine planting model: use the retrieval model itself when it supports
    # image input, so that S2 compaction and query run on the same target model.
    # Only fall back to VLM_MODEL when the retrieval model is absent or not
    # image-capable in EXPERIMENT_MODEL_REGISTRY.
    if image_path is not None and retrieval_model is not None:
        retrieval_provider, retrieval_id = retrieval_model.split("/", 1)
        provider_spec = EXPERIMENT_MODEL_REGISTRY.get(retrieval_provider, {})
        model_specs = provider_spec.get("models", [])
        is_image_capable = any(
            m["id"] == retrieval_id and "image" in m.get("input", [])
            for m in model_specs
        )
        if is_image_capable:
            planting_model = retrieval_model
        else:
            planting_model = VLM_MODEL
    else:
        planting_model = None
    use_vlm_planting = planting_model is not None and planting_model != retrieval_model
    config.backend = retrieval_model or ""

    # Auto-detect backend from OpenClaw config if still empty
    if not config.backend:
        try:
            import json as _json
            cfg = _json.load((sd / "openclaw.json").open())
            config.backend = cfg.get("agents", {}).get("defaults", {}).get("model", {}).get("primary", "unknown")
        except Exception:
            config.backend = "unknown"

    logger.info(
        "Starting trial set %s (scenarios=%s, plant_model=%s, retrieval_model=%s)",
        trial_base_id, scenarios, planting_model or "default", config.backend,
    )

    # Compute context_window_override for agents that go through compaction (S2 path).
    # contextWindow = target_trigger_point + RESERVE_TOKENS_FLOOR
    # OpenClaw fires compaction when context approaches contextWindow - reserveTokensFloor.
    effective_context_threshold = context_threshold or S2_CONTEXT_THRESHOLD
    context_window_override = effective_context_threshold + RESERVE_TOKENS_FLOOR
    needs_compaction = "S2" in scenarios or "S3" in scenarios

    # Phase 1: Create planting agent (VLM if image, else default)
    plant_agent_id = await _setup_trial_agent(
        f"{trial_base_id}_plant", model=planting_model, state_dir=sd,
    )
    plant_workspace = sd / f"workspace-{plant_agent_id}"
    pre_hash = _compute_workspace_hash(plant_workspace)

    # Inject CitationForce into the per-trial workspace if requested
    if citation_force_condition != "C0":
        from mitigations.citation_force.bootstrap_hook import inject_citation_force_into_bootstrap
        inject_citation_force_into_bootstrap(
            workspace_dir=sd / f"workspace-{plant_agent_id}",
            condition=citation_force_condition,
        )

    plant_snapshot = None
    s1_agent_id = None
    s3_agent_id = None
    s3_snapshot = None
    trials = []

    try:
        # Planting: inject artifact (using VLM if image provided)
        artifact = await run_planting(
            config, plant_prompt, artifact_id, artifact_ocr_keywords,
            agent_id=plant_agent_id, image_path=image_path, state_dir=sd,
        )
        logger.info("Planting path: %s", artifact.planting_path)

        # Validate planting succeeded
        if artifact.planting_path == "none":
            logger.warning("planting_path=none — artifact not persisted to memory")
        if not artifact.actual_artifact_repr:
            logger.warning("actual_artifact_repr is empty — agent may not have saved artifact")

        # Snapshot post-plant workspace so S1 can branch independently from Session A.
        # S1 gets its own session (re-plant from snapshot); S2/S3 continue session A
        # without S1's Q&A ever appearing in their context.
        plant_snapshot = sd / f"snapshot-{trial_base_id}-plant"
        if plant_snapshot.exists():
            shutil.rmtree(plant_snapshot)
        shutil.copytree(plant_workspace, plant_snapshot)

        # ------------------------------------------------------------------ #
        # Branch coroutines — run concurrently via asyncio.gather             #
        # ------------------------------------------------------------------ #

        async def _s1_branch() -> list[GroundingTrial]:
            nonlocal s1_agent_id
            s1_agent_id = await _setup_trial_agent(
                f"{trial_base_id}_s1", model=planting_model,
                workspace_source=plant_snapshot, state_dir=sd,
                    )
            if citation_force_condition != "C0":
                from mitigations.citation_force.bootstrap_hook import inject_citation_force_into_bootstrap
                inject_citation_force_into_bootstrap(
                    sd / f"workspace-{s1_agent_id}",
                    condition=citation_force_condition,
                )
            # Architectural constraint: S1 must re-plant into its own fresh session.
            # run_s1_test reads the live session's JSONL history to build the active
            # context view (R_context scoring). The snapshot workspace contains the
            # planted memory files but NOT the session JSONL, so there is no way to
            # transfer an existing conversation history to a new agent. Re-planting
            # creates the required live session JSONL from scratch.
            #
            # Experimental implication: S1's planting event is an independent replicate,
            # not a deterministic fork of the shared plant. Any variability in the
            # planting LLM's output is an accepted, documented limitation of the current
            # session model.
            s1_artifact = await run_planting(
                config, plant_prompt, artifact_id, artifact_ocr_keywords,
                agent_id=s1_agent_id, image_path=image_path, state_dir=sd,
            )
            s1_result = await run_s1_test(
                config, s1_artifact, query, agent_id=s1_agent_id, state_dir=sd,
            )
            s1_trial = GroundingTrial(
                trial_id=f"{trial_base_id}_S1",
                theta=config,
                scenario="S1",
                artifact_id=artifact_id,
                artifact_ocr_keywords=artifact_ocr_keywords,
                query=query,
                planting_path=s1_artifact.planting_path,
                system_prompt_at_init=s1_result.system_prompt_at_init,
                tool_trace=s1_result.tool_trace,
                startup_exposure_log=s1_result.startup_exposure_log,
                active_context_view=s1_result.active_context_view,
                response_text=s1_result.response_text,
                compaction_during_session_b=False,
                citation_force_reinjected=None,
                # S1 is definitionally "artifact still in active context" — same session,
                # no compaction forced, no reset. Harness guarantees in_context status.
                artifact_context_status="in_context",
                created_at=datetime.datetime.utcnow().isoformat() + "Z",
                session_a_id=s1_artifact.session_a_id,   # S1's own independent session
                session_b_id=s1_result.session_b_id,
                actual_artifact_repr=s1_artifact.actual_artifact_repr,
                workspace_profile=s1_agent_id,
                snapshot_id=SNAPSHOT_ID,
                pre_hash=pre_hash,
                post_hash=_compute_workspace_hash(sd / f"workspace-{s1_agent_id}"),
            )
            if output_dir:
                save_trial(s1_trial, output_dir)
                logger.info("S1 trial saved: %s", s1_trial.trial_id)
            return [s1_trial]

        async def _s2s3_branch() -> list[GroundingTrial]:
            nonlocal s3_agent_id, s3_snapshot
            # Continue original session A — S1 Q&A never ran here; filler_offset always 0
            compaction_agent_id = plant_agent_id
            compaction_ws = plant_workspace
            compaction_session_id = artifact.session_a_id

            compaction_ok, pre_query_jsonl, native_cc, boundary_eit, manual_fallback = (
                await _run_native_compaction_phase(
                    config, artifact, compaction_agent_id, compaction_session_id,
                    filler_offset=0, state_dir=sd,
                )
            )
            post_compaction_hash = _compute_workspace_hash(compaction_ws)

            # Save post-compaction snapshot for S3 (before S2 query can dirty state)
            if "S3" in scenarios:
                s3_snapshot = sd / f"snapshot-{trial_base_id}"
                if s3_snapshot.exists():
                    shutil.rmtree(s3_snapshot)
                shutil.copytree(compaction_ws, s3_snapshot)

                if config.citation_force_condition != "C0":
                    from mitigations.citation_force.bootstrap_hook import inject_citation_force_into_bootstrap
                    inject_citation_force_into_bootstrap(
                        workspace_dir=s3_snapshot,
                        condition=config.citation_force_condition,
                    )
                    logger.info(
                        "CitationForce injected into S3 snapshot (condition=%s)",
                        config.citation_force_condition,
                    )

            # S2 and S3 run concurrently — different agents/workspaces after snapshot
            _captured_s3_agent: list[str | None] = [None]

            async def _s2_query() -> list[GroundingTrial]:
                if "S2" not in scenarios:
                    return []
                s2_result = await run_s2_test(
                    config, artifact, query,
                    agent_id=compaction_agent_id,
                    session_id=compaction_session_id,
                    pre_query_jsonl=pre_query_jsonl,
                    compaction_verified=compaction_ok,
                    state_dir=sd,
                )
                s2_trial = GroundingTrial(
                    trial_id=f"{trial_base_id}_S2",
                    theta=config,
                    scenario="S2",
                    artifact_id=artifact_id,
                    artifact_ocr_keywords=artifact_ocr_keywords,
                    query=query,
                    planting_path=artifact.planting_path,
                    system_prompt_at_init="",
                    tool_trace=s2_result.tool_trace,
                    startup_exposure_log=[],
                    active_context_view=s2_result.active_context_view,
                    response_text=s2_result.response_text,
                    compaction_during_session_b=s2_result.compaction_during_session_b,
                    citation_force_reinjected=s2_result.citation_force_reinjected,
                    # S2 continues the same session but compaction may have evicted the artifact.
                    # We do not have a per-message drop signal, so status is unknown.
                    artifact_context_status="unknown",
                    # Native context accounting
                    boundary_input_tokens=boundary_eit,
                    native_compaction_count=native_cc,
                    native_context_window=context_window_override,
                    manual_flush_fallback_used=manual_fallback,
                    created_at=datetime.datetime.utcnow().isoformat() + "Z",
                    session_a_id=artifact.session_a_id,
                    session_b_id=s2_result.session_b_id,
                    actual_artifact_repr=artifact.actual_artifact_repr,
                    workspace_profile=compaction_agent_id,
                    snapshot_id=SNAPSHOT_ID,
                    pre_hash=pre_hash,
                    post_hash=post_compaction_hash,
                )
                if output_dir:
                    save_trial(s2_trial, output_dir)
                    logger.info("S2 trial saved: %s", s2_trial.trial_id)
                return [s2_trial]

            async def _s3_setup_and_run() -> list[GroundingTrial]:
                if "S3" not in scenarios:
                    return []
                agent = await _setup_trial_agent(
                    f"{trial_base_id}_s3",
                    model=retrieval_model,
                    workspace_source=s3_snapshot,
                    state_dir=sd,
                            )
                _captured_s3_agent[0] = agent
                # Sync memory/*.md → root MEMORY.md so the S3 startup bootstrap
                # can read artifact content (enabling R_boot detection).
                _sync_root_memory_md(sd / f"workspace-{agent}")
                await _reindex_memory(agent, state_dir=sd)
                s3_result = await run_s3_test(
                    config, artifact, query, agent_id=agent, state_dir=sd,
                )
                s3_trial = GroundingTrial(
                    trial_id=f"{trial_base_id}_S3",
                    theta=config,
                    scenario="S3",
                    artifact_id=artifact_id,
                    artifact_ocr_keywords=artifact_ocr_keywords,
                    query=query,
                    planting_path=artifact.planting_path,
                    system_prompt_at_init=s3_result.system_prompt_at_init,
                    tool_trace=s3_result.tool_trace,
                    startup_exposure_log=s3_result.startup_exposure_log,
                    active_context_view=s3_result.active_context_view,
                    response_text=s3_result.response_text,
                    compaction_during_session_b=s3_result.compaction_during_session_b,
                    citation_force_reinjected=s3_result.citation_force_reinjected,
                    # S3 is a fresh session — no conversation history exists at query time.
                    # The artifact can only be accessed via memory retrieval or bootstrap.
                    artifact_context_status="out_of_context",
                    # Inherit native context accounting from shared compaction phase
                    boundary_input_tokens=boundary_eit,
                    native_compaction_count=native_cc,
                    native_context_window=context_window_override,
                    manual_flush_fallback_used=manual_fallback,
                    created_at=datetime.datetime.utcnow().isoformat() + "Z",
                    session_a_id=artifact.session_a_id,
                    session_b_id=s3_result.session_b_id,
                    actual_artifact_repr=artifact.actual_artifact_repr,
                    workspace_profile=agent,
                    snapshot_id=SNAPSHOT_ID,
                    pre_hash=pre_hash,
                    post_hash=_compute_workspace_hash(sd / f"workspace-{agent}"),
                )
                # Verify CitationForce section presence in workspace BOOTSTRAP.md
                bootstrap_path = sd / f"workspace-{agent}" / "BOOTSTRAP.md"
                if bootstrap_path.exists():
                    from mitigations.citation_force.bootstrap_hook import BOOTSTRAP_EXTRA_SECTION_HEADER
                    s3_trial.bootstrap_cf_present = (
                        BOOTSTRAP_EXTRA_SECTION_HEADER in bootstrap_path.read_text()
                    )
                if output_dir:
                    save_trial(s3_trial, output_dir)
                    logger.info("S3 trial saved: %s", s3_trial.trial_id)
                return [s3_trial]

            inner_results = await asyncio.gather(_s2_query(), _s3_setup_and_run())
            s3_agent_id = _captured_s3_agent[0]  # propagate for finally-block cleanup
            results: list[GroundingTrial] = []
            for r in inner_results:
                results.extend(r)
            return results

        async def _noop() -> list[GroundingTrial]:
            return []

        # Outer gather: S1 branch || S2/S3 branch (independent sessions after plant_snapshot)
        outer_results = await asyncio.gather(
            _s1_branch() if "S1" in scenarios else _noop(),
            _s2s3_branch() if ("S2" in scenarios or "S3" in scenarios) else _noop(),
        )
        for branch in outer_results:
            trials.extend(branch)

        return trials

    finally:
        await _teardown_trial_agent(plant_agent_id, state_dir=sd)
        if s1_agent_id:
            await _teardown_trial_agent(s1_agent_id, state_dir=sd)
        if s3_agent_id:
            await _teardown_trial_agent(s3_agent_id, state_dir=sd)
        if plant_snapshot and plant_snapshot.exists():
            shutil.rmtree(plant_snapshot, ignore_errors=True)
        if s3_snapshot and s3_snapshot.exists():
            shutil.rmtree(s3_snapshot, ignore_errors=True)
