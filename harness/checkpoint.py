"""
Full-state checkpoint infrastructure for S1 / S2 / S3 experiments.

Provides reusable primitives for:
  - Saving the ENTIRE OpenClaw worker state dir as an immutable checkpoint
  - Restoring a checkpoint into a fresh branch state dir
  - Same-session continuation (S1/S2): resume the original session
  - Fresh-session continuation (S3): create a new session from restored state

Checkpoint layout:
    logs/checkpoints/<episode_id>/<checkpoint_id>/
        manifest.json          # metadata (immutable after creation)
        state/                 # full copy of worker state dir

Checkpoint capture definitions
------------------------------

S1 pre-query checkpoints (s1_prequery_d0 / d16k / d32k):

    Capture criterion: the checkpoint is saved immediately AFTER filler
    injection has brought effective_input_tokens (provider-reported
    input + cacheRead from the latest JSONL message entry) to at or
    above the target depth, and BEFORE the reference query is sent.

    The depth labels map to provider-reported effective_input_tokens
    thresholds (NOT token estimates or JSONL line counts):

        d0   — effective_input_tokens in [0, 28000).
               Captured right after artifact planting, before filler.
               In multimodal sessions the system prompt + bootstrap
               alone consume ~14-15k tokens, so d0 typically lands
               at ~20-23k after planting 6 visual artifacts.
        d50k — effective_input_tokens >= 50000.
        d80k — effective_input_tokens >= 80000.
               Pre-compaction absolute token-position milestones.

    No compaction has fired at any S1 checkpoint; the session is
    continuous from planting through filler to the checkpoint moment.

Post-compaction checkpoints (postcomp_50k / 100k / 150k):

    Capture criterion: the checkpoint is saved AFTER a real OpenClaw
    native compaction event has been verified (sessions.json
    compactionCount has increased above its pre-filler value), and
    BEFORE any S2 or S3 query is sent.

    The threshold labels correspond to the contextWindow - reserveTokensFloor
    target at which filler injection stopped and compaction fired:

        postcomp_50k  — compaction_threshold = 50000
        postcomp_100k — compaction_threshold = 100000
        postcomp_150k — compaction_threshold = 150000

    These checkpoints are shared bases for both S2 (same-session) and
    S3 (fresh-session) continuations.

Usage:
    # Save S1
    manifest = save_s1_prequery_checkpoint(
        state_dir=worker_state_dir,
        episode_id="ep_001",
        agent_id="gb_abc123",
        session_id="uuid-...",
        depth_label=0,
        effective_input_tokens=142,
        jsonl_entry_count=4,
    )

    # Save post-compaction
    manifest = save_postcomp_checkpoint(
        state_dir=worker_state_dir,
        episode_id="ep_001",
        agent_id="gb_abc123",
        session_id="uuid-...",
        compaction_threshold=50000,
        native_compaction_count=1,
        boundary_input_tokens=48500,
        compaction_verified=True,
    )

    # Restore + same-session continuation (S1/S2)
    branch = await restore_checkpoint(manifest, branch_id="branch_s1_001")
    result = await continue_same_session(branch, message="What was in that chart?")

    # Restore + fresh-session continuation (S3)
    branch = await restore_checkpoint(manifest, branch_id="branch_s3_001")
    result = await create_fresh_session(branch, message="What was in that chart?")
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from client.openclaw_client import OpenClawClient

logger = logging.getLogger(__name__)

# Default root for checkpoint storage
CHECKPOINTS_ROOT = Path(__file__).parent.parent / "logs" / "checkpoints"


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

@dataclass
class CheckpointManifest:
    """Immutable metadata for a saved checkpoint."""

    checkpoint_id: str
    checkpoint_kind: Literal[
        "s1_prequery",     # S1 pre-query, pre-compaction
        "postcomp",        # post-compaction, pre-query (shared S2/S3 base)
        "custom",          # any other checkpoint
    ]
    episode_id: str                       # run or episode that produced this checkpoint
    source_state_dir: str                 # original worker state dir path
    agent_id: str                         # agent active at checkpoint time
    session_id: str                       # session active at checkpoint time
    created_at: str                       # ISO 8601

    # --- S1 pre-query fields ---
    # effective_input_tokens: provider-reported input + cacheRead at capture time.
    #   This is the AUTHORITATIVE context-size signal: OpenClawClient.get_effective_input_tokens().
    #   For d0 this is whatever the provider reports after planting (typically a few hundred).
    #   For d16k/d32k this is >= the target depth after filler injection.
    effective_input_tokens: int | None = None
    # jsonl_entry_count: number of entries in the session JSONL at capture time.
    #   Used as a secondary integrity check — not the primary depth signal.
    jsonl_entry_count: int | None = None
    # history_depth_label: the nominal depth target (0, 16000, 32000) that the
    #   episode runner was aiming for. Not an exact measurement — use
    #   effective_input_tokens for the precise value.
    history_depth_label: int | None = None

    # --- Post-compaction fields ---
    # compaction_threshold: the contextWindow - reserveTokensFloor target at which
    #   filler injection stopped and compaction was triggered (50000, 100000, 150000).
    compaction_threshold: int | None = None
    # native_compaction_count: sessions.json compactionCount AFTER compaction fired.
    #   Must be > 0; the checkpoint is invalid if this is 0 or None for postcomp kind.
    native_compaction_count: int | None = None
    # boundary_input_tokens: provider-reported effective input tokens at the
    #   compaction boundary (last filler turn before/after compaction).
    boundary_input_tokens: int | None = None
    # compaction_verified: True if _verify_memory_write confirmed the artifact
    #   was persisted to workspace memory files after compaction.
    compaction_verified: bool | None = None

    # --- Integrity ---
    workspace_hash: str = ""              # SHA-256 prefix of workspace .md files

    # --- Resolved paths (set after save) ---
    checkpoint_dir: str = ""              # path to the checkpoint directory
    state_subdir: str = ""                # path to the state/ copy

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> CheckpointManifest:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, manifest_path: Path) -> CheckpointManifest:
        with manifest_path.open() as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Branch state (returned by restore)
# ---------------------------------------------------------------------------

@dataclass
class BranchState:
    """A restored branch ready for continuation."""

    branch_id: str
    branch_state_dir: Path                # new state dir (copy of checkpoint)
    agent_id: str                         # agent_id from the checkpoint
    session_id: str                       # session_id from the checkpoint
    gateway_port: int                     # port the branch gateway is listening on
    gateway_proc: subprocess.Popen | None = None  # managed gateway process
    source_manifest: CheckpointManifest | None = None
    rewrite_summary: dict | None = None   # path rewrite log from restore

    def stop_gateway(self) -> None:
        """Stop the branch gateway process."""
        if self.gateway_proc is not None:
            try:
                self.gateway_proc.terminate()
                self.gateway_proc.wait(timeout=5)
            except Exception:
                try:
                    self.gateway_proc.kill()
                except Exception:
                    pass
            self.gateway_proc = None
            logger.info("Branch gateway stopped: %s (port=%d)", self.branch_id, self.gateway_port)

    def cleanup(self) -> None:
        """Stop gateway and remove branch state dir."""
        self.stop_gateway()
        if self.branch_state_dir.exists():
            shutil.rmtree(self.branch_state_dir, ignore_errors=True)
            logger.info("Branch state dir removed: %s", self.branch_state_dir)


# ---------------------------------------------------------------------------
# Continuation results
# ---------------------------------------------------------------------------

@dataclass
class ContinuationResult:
    """Result of a continuation (same-session or fresh-session)."""

    branch: BranchState
    session_id: str                       # session used for the continuation
    response: dict                        # raw response from send_message
    response_text: str                    # extracted response text
    jsonl: list[dict]                     # full JSONL after continuation

    # --- Same-session proof fields ---
    # These are populated by continue_same_session() to provide hard evidence
    # that the original session was truly reused (not silently replaced).
    jsonl_count_before: int | None = None  # JSONL entry count BEFORE the follow-up
    jsonl_count_after: int | None = None   # JSONL entry count AFTER the follow-up
    jsonl_file_path: str | None = None     # absolute path to the JSONL file used
    session_id_matches_checkpoint: bool | None = None  # True if session_id == checkpoint session_id


# ---------------------------------------------------------------------------
# Workspace hashing (reuses logic from trial_runner)
# ---------------------------------------------------------------------------

def _compute_workspace_hash(workspace: Path) -> str:
    """SHA-256 prefix of all .md files in workspace."""
    h = hashlib.sha256()
    if not workspace.exists():
        return h.hexdigest()[:16]
    for f in sorted(workspace.rglob("*.md")):
        h.update(str(f.relative_to(workspace)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    state_dir: Path,
    checkpoint_id: str,
    checkpoint_kind: Literal["s1_prequery", "postcomp", "custom"],
    episode_id: str,
    agent_id: str,
    session_id: str,
    *,
    # S1 fields
    effective_input_tokens: int | None = None,
    jsonl_entry_count: int | None = None,
    history_depth_label: int | None = None,
    # Post-compaction fields
    compaction_threshold: int | None = None,
    native_compaction_count: int | None = None,
    boundary_input_tokens: int | None = None,
    compaction_verified: bool | None = None,
    # General
    checkpoints_root: Path | None = None,
) -> CheckpointManifest:
    """Save the entire worker state dir as an immutable checkpoint.

    Copies everything under state_dir (openclaw.json, agents/, sessions/,
    workspace-*, logs/, etc.) into:
        <checkpoints_root>/<episode_id>/<checkpoint_id>/state/

    Returns the manifest with resolved paths.
    """
    root = checkpoints_root or CHECKPOINTS_ROOT
    ckpt_dir = root / episode_id / checkpoint_id
    state_copy = ckpt_dir / "state"

    if ckpt_dir.exists():
        raise FileExistsError(
            f"Checkpoint already exists: {ckpt_dir}. "
            "Checkpoints are immutable — use a different checkpoint_id."
        )

    # Compute workspace hash before copy
    workspace_candidates = list(state_dir.glob(f"workspace-{agent_id}"))
    if not workspace_candidates:
        workspace_candidates = list(state_dir.glob("workspace"))
    ws_hash = ""
    if workspace_candidates:
        ws_hash = _compute_workspace_hash(workspace_candidates[0])

    # Full copy of the state dir
    logger.info("Saving checkpoint %s: copying %s → %s", checkpoint_id, state_dir, state_copy)
    shutil.copytree(state_dir, state_copy)

    manifest = CheckpointManifest(
        checkpoint_id=checkpoint_id,
        checkpoint_kind=checkpoint_kind,
        episode_id=episode_id,
        source_state_dir=str(state_dir),
        agent_id=agent_id,
        session_id=session_id,
        created_at=datetime.datetime.utcnow().isoformat() + "Z",
        effective_input_tokens=effective_input_tokens,
        jsonl_entry_count=jsonl_entry_count,
        history_depth_label=history_depth_label,
        compaction_threshold=compaction_threshold,
        native_compaction_count=native_compaction_count,
        boundary_input_tokens=boundary_input_tokens,
        compaction_verified=compaction_verified,
        workspace_hash=ws_hash,
        checkpoint_dir=str(ckpt_dir),
        state_subdir=str(state_copy),
    )

    # Write manifest
    manifest_path = ckpt_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    logger.info(
        "Checkpoint saved: %s (kind=%s, agent=%s, session=%s, eit=%s, hash=%s)",
        checkpoint_id, checkpoint_kind, agent_id, session_id,
        effective_input_tokens, ws_hash,
    )
    return manifest


# ---------------------------------------------------------------------------
# Restore checkpoint → branch
# ---------------------------------------------------------------------------

def _remove_stale_locks(state_dir: Path) -> int:
    """Remove stale .lock files from a state dir. Returns count removed."""
    removed = 0
    for lock_file in state_dir.rglob("*.lock"):
        lock_file.unlink()
        removed += 1
    for lock_file in state_dir.rglob("*.jsonl.lock"):
        lock_file.unlink(missing_ok=True)
        removed += 1
    return removed


def _normalize_device_platform(state_dir: Path) -> int:
    """Rewrite paired-device platform pins to the host platform.

    Checkpoints authored on macOS pin devices as platform="darwin" in
    devices/paired.json. When restored on a linux node, the connecting
    client claims platform="linux", so the gateway flags a
    metadata-upgrade mismatch and demands pairing ("pairing required").
    Headless pairing can't auto-approve a metadata-upgrade, so the client
    falls back to embedded mode and concurrent filler turns deadlock on
    the session .jsonl.lock — compaction never fires. Aligning the pinned
    platform with the host platform removes the mismatch entirely.

    Returns the number of device records updated.
    """
    import sys

    host_platform = sys.platform  # 'linux' / 'darwin' / 'win32' — matches node's process.platform
    updated = 0
    for paired_path in state_dir.rglob("devices/paired.json"):
        try:
            data = json.loads(paired_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        changed = False
        for rec in data.values():
            if isinstance(rec, dict) and rec.get("platform") not in (None, host_platform):
                rec["platform"] = host_platform
                changed = True
                updated += 1
        if changed:
            paired_path.write_text(json.dumps(data, indent=2))
    return updated


def _rewrite_branch_paths(
    branch_state_dir: Path,
    original_state_dir: str,
    new_port: int,
) -> dict:
    """Rewrite all absolute paths in the branch to point at the branch state dir.

    This is the critical isolation step. Without it, the cloned branch would
    still read/write files in the ORIGINAL worker state dir.

    Rewrites:
      1. openclaw.json
         - gateway.port → new_port
         - agents.list[*].workspace → branch path
         - agents.list[*].agentDir → branch path
      2. agents/*/sessions/sessions.json
         - each entry's sessionFile → branch path

    Returns a dict summarizing what was rewritten (for logging/proof).
    """
    old_prefix = original_state_dir.rstrip("/")
    new_prefix = str(branch_state_dir).rstrip("/")
    summary: dict = {"openclaw_json": [], "sessions_json": [], "workspace_data_json": []}

    # --- 1. Rewrite openclaw.json ---
    cfg_path = branch_state_dir / "openclaw.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"openclaw.json not found in {branch_state_dir}")
    with cfg_path.open() as f:
        cfg = json.load(f)

    cfg.setdefault("gateway", {})["port"] = new_port
    summary["openclaw_json"].append(f"gateway.port → {new_port}")

    for agent_entry in cfg.get("agents", {}).get("list", []):
        for field in ("workspace", "agentDir"):
            old_val = agent_entry.get(field)
            if old_val and isinstance(old_val, str) and old_prefix in old_val:
                new_val = old_val.replace(old_prefix, new_prefix)
                agent_entry[field] = new_val
                summary["openclaw_json"].append(
                    f"agents.list[{agent_entry.get('id', '?')}].{field}: "
                    f"{old_val} → {new_val}"
                )

    with cfg_path.open("w") as f:
        json.dump(cfg, f, indent=2)

    # --- 2. Rewrite sessions.json files ---
    for sessions_json in branch_state_dir.rglob("sessions.json"):
        try:
            with sessions_json.open() as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        changed = False
        for key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            sf = entry.get("sessionFile")
            if sf and isinstance(sf, str) and old_prefix in sf:
                new_sf = sf.replace(old_prefix, new_prefix)
                entry["sessionFile"] = new_sf
                summary["sessions_json"].append(
                    f"{sessions_json.relative_to(branch_state_dir)}[{key}].sessionFile: "
                    f"{sf} → {new_sf}"
                )
                changed = True

        if changed:
            with sessions_json.open("w") as f:
                json.dump(data, f, indent=2)

    # --- 3. Rewrite imagePath fields in workspace data json files ---
    # Artifact data files (data/**/*.json) store absolute imagePath fields that
    # point to the original episode state dir. Rewrite them to the branch path
    # so OpenClaw can resolve image attachments during probing.
    summary["workspace_data_json"] = []
    for ws_dir in branch_state_dir.glob("workspace-*"):
        data_dir = ws_dir / "data"
        if not data_dir.is_dir():
            continue
        for data_json in data_dir.rglob("*.json"):
            try:
                with data_json.open() as f:
                    obj = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(obj, dict):
                continue
            ip = obj.get("imagePath")
            if ip and isinstance(ip, str) and old_prefix in ip:
                obj["imagePath"] = ip.replace(old_prefix, new_prefix)
                with data_json.open("w") as f:
                    json.dump(obj, f, indent=2)
                summary["workspace_data_json"].append(
                    f"{data_json.relative_to(branch_state_dir)}: imagePath rewritten"
                )

    return summary


def _find_free_port(start: int = 19800) -> int:
    """Find a single free port starting from `start`."""
    import socket
    p = start
    while p < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                p += 1
    raise RuntimeError("No free port found")


async def restore_checkpoint(
    manifest: CheckpointManifest,
    branch_id: str,
    branch_root: Path | None = None,
    port_start: int = 19800,
    start_gateway: bool = True,
    gateway_timeout: float = 60.0,
) -> BranchState:
    """Restore a checkpoint into a fresh, fully-isolated branch state dir.

    Steps:
      1. Copy checkpoint state/ to a new branch state dir
      2. Rewrite ALL absolute paths to point at the branch (not the original):
         - openclaw.json: gateway.port, agents.list[*].workspace, agents.list[*].agentDir
         - sessions/sessions.json: each entry's sessionFile
      3. Remove stale lock files (*.lock, *.jsonl.lock)
      4. Optionally start an OpenClaw gateway on the branch
      5. Return BranchState with agent_id, session_id, branch dir

    After restore, the branch is fully self-contained: no reads or writes
    go to the original state dir or checkpoint state dir.

    Args:
        manifest: The checkpoint manifest to restore from.
        branch_id: Unique identifier for this branch (used in dir name).
        branch_root: Parent dir for branch state dirs. Defaults to ~/
        port_start: Starting port to scan for a free port.
        start_gateway: Whether to start a gateway process.
        gateway_timeout: Max seconds to wait for gateway readiness.
    """
    source_state = Path(manifest.state_subdir)
    if not source_state.exists():
        raise FileNotFoundError(f"Checkpoint state dir not found: {source_state}")

    # Create branch state dir
    root = branch_root or Path.home()
    branch_state_dir = root / f".openclaw-branch-{branch_id}"
    if branch_state_dir.exists():
        shutil.rmtree(branch_state_dir)

    logger.info("Restoring checkpoint %s → branch %s", manifest.checkpoint_id, branch_state_dir)
    shutil.copytree(source_state, branch_state_dir)

    # Rewrite all absolute paths for full isolation
    port = _find_free_port(port_start)
    rewrite_summary = _rewrite_branch_paths(
        branch_state_dir,
        original_state_dir=manifest.source_state_dir,
        new_port=port,
    )
    n_rewrites = sum(len(v) for v in rewrite_summary.values())
    logger.info(
        "Path rewrite complete: %d changes (openclaw.json=%d, sessions.json=%d, workspace_data=%d)",
        n_rewrites,
        len(rewrite_summary["openclaw_json"]),
        len(rewrite_summary["sessions_json"]),
        len(rewrite_summary["workspace_data_json"]),
    )
    for category, entries in rewrite_summary.items():
        for entry in entries:
            logger.debug("  %s: %s", category, entry)

    # Remove stale locks
    n_locks = _remove_stale_locks(branch_state_dir)
    if n_locks:
        logger.info("Removed %d stale lock files", n_locks)

    # Align paired-device platform pins with the host (macOS-authored
    # checkpoints pin darwin → metadata-upgrade mismatch → pairing required
    # → embedded fallback → session-lock deadlock → compaction never fires).
    n_dev = _normalize_device_platform(branch_state_dir)
    if n_dev:
        logger.info("Normalized %d paired-device platform pin(s) to host", n_dev)

    # Start gateway
    gateway_proc = None
    if start_gateway:
        (branch_state_dir / "logs").mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "OPENCLAW_STATE_DIR": str(branch_state_dir)}
        log_out = (branch_state_dir / "logs" / "gateway.log").open("a")
        log_err = (branch_state_dir / "logs" / "gateway.err.log").open("a")
        gateway_proc = subprocess.Popen(
            # --auth none --allow-unconfigured: without these the branch gateway
            # demands pairing, the CLI falls back to embedded mode, and concurrent
            # embedded filler calls deadlock on the session .jsonl.lock so
            # compaction never fires. Matches the persistent gateway bring-up.
            ["openclaw", "gateway", "run", "--port", str(port),
             "--force", "--auth", "none", "--allow-unconfigured"],
            stdout=log_out,
            stderr=log_err,
            env=env,
        )
        logger.info("Branch gateway starting (port=%d, pid=%d)", port, gateway_proc.pid)

        # Poll for readiness via HTTP (systemctl not available on HPC nodes)
        from urllib.request import urlopen
        from urllib.error import URLError
        deadline = time.monotonic() + gateway_timeout
        ready = False
        while time.monotonic() < deadline:
            try:
                urlopen(f"http://127.0.0.1:{port}/", timeout=2)
                ready = True
                break
            except (URLError, OSError, ConnectionRefusedError):
                pass
            if gateway_proc.poll() is not None:
                raise RuntimeError(
                    f"Branch gateway exited early (code={gateway_proc.returncode}). "
                    f"Check {branch_state_dir / 'logs' / 'gateway.err.log'}"
                )
            await asyncio.sleep(0.5)

        if not ready:
            gateway_proc.kill()
            raise RuntimeError(
                f"Branch gateway did not become ready within {gateway_timeout}s "
                f"(branch={branch_id}, port={port})"
            )
        logger.info("Branch gateway ready: %s (port=%d)", branch_id, port)

    branch = BranchState(
        branch_id=branch_id,
        branch_state_dir=branch_state_dir,
        agent_id=manifest.agent_id,
        session_id=manifest.session_id,
        gateway_port=port,
        gateway_proc=gateway_proc,
        source_manifest=manifest,
        rewrite_summary=rewrite_summary,
    )
    return branch


# ---------------------------------------------------------------------------
# Same-session continuation (S1 / S2)
# ---------------------------------------------------------------------------

async def continue_same_session(
    branch: BranchState,
    message: str,
    image_path: str | None = None,
) -> ContinuationResult:
    """Send a follow-up message in the restored original session.

    Used for S1 (pre-compaction) and S2 (post-compaction) continuations.
    The session_id from the checkpoint is reused — the model sees the
    full conversation history up to the checkpoint point.

    Proof of same-session reuse:
    The result includes jsonl_count_before and jsonl_count_after, read
    from the same JSONL file path. If the follow-up truly appended to
    the original session, jsonl_count_after > jsonl_count_before and
    the file path matches the checkpoint session_id.
    """
    async with OpenClawClient(
        agent_id=branch.agent_id,
        state_dir=branch.branch_state_dir,
        ws_url=f"ws://127.0.0.1:{branch.gateway_port}",  # not the default :18789
    ) as client:
        logger.info(
            "Same-session continuation: session=%s, agent=%s, branch=%s",
            branch.session_id, branch.agent_id, branch.branch_id,
        )

        # Read JSONL BEFORE the follow-up for proof
        jsonl_path = client.sessions_dir / f"{branch.session_id}.jsonl"
        jsonl_before = client.get_session_jsonl(branch.session_id)
        count_before = len(jsonl_before)

        # Verify the JSONL file exists (it should, from the checkpoint)
        if not jsonl_path.exists():
            raise RuntimeError(
                f"Session JSONL not found in branch: {jsonl_path}. "
                "The checkpoint may not contain a valid session for this agent."
            )

        response = await client.send_message(
            message,
            session_id=branch.session_id,
            image_path=image_path,
        )
        response_text = _extract_text(response)

        # Read JSONL AFTER the follow-up
        jsonl_after = client.get_session_jsonl(branch.session_id)
        count_after = len(jsonl_after)

        # Fallback: if CLI returned empty text, extract from JSONL entries
        if not response_text or response_text.startswith("{"):
            response_text = _extract_text_from_jsonl(jsonl_after, count_before)

        # Hard proof: the same file was appended to
        session_id_match = branch.session_id == (
            branch.source_manifest.session_id if branch.source_manifest else branch.session_id
        )

        if count_after <= count_before:
            logger.warning(
                "Same-session proof WEAK: JSONL count did not increase "
                "(before=%d, after=%d). The gateway may have written to a "
                "different file or the response was empty.",
                count_before, count_after,
            )

        logger.info(
            "Same-session proof: session=%s, jsonl_before=%d, jsonl_after=%d, "
            "file=%s, id_match=%s",
            branch.session_id, count_before, count_after,
            jsonl_path, session_id_match,
        )

        return ContinuationResult(
            branch=branch,
            session_id=branch.session_id,
            response=response,
            response_text=response_text,
            jsonl=jsonl_after,
            jsonl_count_before=count_before,
            jsonl_count_after=count_after,
            jsonl_file_path=str(jsonl_path),
            session_id_matches_checkpoint=session_id_match,
        )


# ---------------------------------------------------------------------------
# Fresh-session continuation (S3)
# ---------------------------------------------------------------------------

async def create_fresh_session(
    branch: BranchState,
    message: str,
    image_path: str | None = None,
) -> ContinuationResult:
    """Create a new session from the restored branch state and send a message.

    Used for S3: the agent has the workspace/memory from the checkpoint
    but no conversation history. The model starts fresh with bootstrap
    context only.

    S3 restore procedure (executed in order):
    1. Sync memory/*.md → root MEMORY.md
       Reason: OpenClaw's fresh-session bootstrap reads MEMORY.md at the
       workspace root, but the agent writes memory files into the memory/
       subdirectory. Without this sync, the bootstrap will not find artifact
       content, breaking R_boot detection.

    2. Force memory reindex (openclaw memory index --force)
       Reason: The memory search index in the checkpoint is stale — it was
       built for the original state dir path. The branch has a different
       path, so the index entries would point to non-existent files.
       Reindexing ensures memory_search queries work in the fresh session.

    3. Create fresh session (start_session)
       This triggers the bootstrap flow: OpenClaw reads MEMORY.md,
       BOOTSTRAP.md, and the system files, composing the initial context.

    4. Send the follow-up message into the fresh session.
    """
    workspace_path = branch.branch_state_dir / f"workspace-{branch.agent_id}"

    # Step 1: Sync memory/*.md → root MEMORY.md
    _sync_root_memory_md(workspace_path)

    # Step 2: Force memory reindex
    await _reindex_memory(branch.agent_id, branch.branch_state_dir)

    async with OpenClawClient(
        agent_id=branch.agent_id,
        state_dir=branch.branch_state_dir,
        ws_url=f"ws://127.0.0.1:{branch.gateway_port}",  # not the default :18789
    ) as client:
        logger.info(
            "Fresh-session continuation: agent=%s, branch=%s",
            branch.agent_id, branch.branch_id,
        )

        # Step 3: Create fresh session (triggers bootstrap)
        new_session_id = await client.start_session(reason="new")
        logger.info("Fresh session created: %s", new_session_id)

        # Step 4: Send the follow-up message
        response = await client.send_message(
            message,
            session_id=new_session_id,
            image_path=image_path,
        )
        response_text = _extract_text(response)
        jsonl = client.get_session_jsonl(new_session_id)

        return ContinuationResult(
            branch=branch,
            session_id=new_session_id,
            response=response,
            response_text=response_text,
            jsonl=jsonl,
        )


# ---------------------------------------------------------------------------
# S1 checkpoint capture helpers
# ---------------------------------------------------------------------------

# Accepted S1 depth labels and their valid effective_input_tokens ranges.
# d0 is the low-depth baseline: eit must be below the d0 ceiling.
# All other labels are absolute eit milestones (pre-compaction).
#
# 50k pilot:  d0, d40k
# 100k main:  d0, d50k, d80k
_S1_DEPTH_LABELS = {0, 16_000, 32_000, 40_000, 50_000, 80_000}
# d0 ceiling: in multimodal OpenClaw sessions, the system prompt + bootstrap
# alone consume ~14-15k tokens, and planting 6 visual artifacts with a few
# interleaved work tasks adds another ~6-8k. The d0 checkpoint captures the
# state right after artifact planting but before any depth-pushing filler.
# 28k accommodates the realistic base context with comfortable margin.
# Env-overridable: the combined 12-object modality trunk (original 6 + 6 new) plants
# twice the artifacts before d0, so its d0 sits higher (~30-38k) yet is still shallow
# relative to d50k/d80k. Set S1_D0_EIT_CEILING for that build.
import os as _os
_S1_D0_EIT_CEILING = int(_os.environ.get("S1_D0_EIT_CEILING", "28000"))


def save_s1_prequery_checkpoint(
    state_dir: Path,
    episode_id: str,
    agent_id: str,
    session_id: str,
    depth_label: int,
    effective_input_tokens: int,
    jsonl_entry_count: int,
    native_compaction_count: int,
    checkpoints_root: Path | None = None,
) -> CheckpointManifest:
    """Save an S1 pre-query, pre-compaction checkpoint.

    CAPTURE CRITERION:
    This must be called AFTER filler injection has brought
    effective_input_tokens (provider-reported input + cacheRead) to at
    or above the target depth, and BEFORE the reference query is sent.

    For d0: called right after artifact planting completes, before filler.
    For d50k/d80k (or d16k/d32k/d40k): called after filler pushes eit >= target.

    Enforced invariants:
    1. depth_label must be in _S1_DEPTH_LABELS.
    2. native_compaction_count must be 0 — S1 checkpoints are pre-compaction.
    3. For d0: effective_input_tokens must be < 28000 (prevents mislabeled
       d0 after substantial filler). In multimodal OpenClaw sessions the
       system prompt + bootstrap consume ~14-15k tokens and artifact
       planting adds another ~6-8k, so d0 typically lands at ~20-23k.
    4. For non-d0: effective_input_tokens must be >= depth_label.

    Args:
        depth_label: Nominal target depth. Must be 0, 16000, 32000, or 40000.
        effective_input_tokens: REQUIRED. Provider-reported input + cacheRead
            from OpenClawClient.get_effective_input_tokens().
        jsonl_entry_count: REQUIRED. Number of entries in the session JSONL
            at capture time (secondary integrity check).
        native_compaction_count: REQUIRED. sessions.json compactionCount at
            capture time. Must be 0.

    Raises:
        ValueError: On any invariant violation.
    """
    # Invariant 1: valid depth label
    if depth_label not in _S1_DEPTH_LABELS:
        raise ValueError(
            f"depth_label ({depth_label}) must be one of {sorted(_S1_DEPTH_LABELS)}."
        )

    # Invariant 2: no compaction
    if native_compaction_count != 0:
        raise ValueError(
            f"native_compaction_count ({native_compaction_count}) must be 0 "
            f"for S1 pre-query checkpoints. Compaction has already fired — "
            f"this is not a valid S1 state."
        )

    # Invariant 3: d0 ceiling — prevent mislabeled d0 after heavy filler
    if depth_label == 0 and effective_input_tokens >= _S1_D0_EIT_CEILING:
        raise ValueError(
            f"depth_label=0 (d0) but effective_input_tokens={effective_input_tokens} "
            f">= {_S1_D0_EIT_CEILING}. A d0 checkpoint must be captured before "
            f"substantial filler injection. Use d16k or d32k instead."
        )

    # Invariant 4: d16k/d32k floor
    if depth_label > 0 and effective_input_tokens < depth_label:
        raise ValueError(
            f"effective_input_tokens ({effective_input_tokens}) is below "
            f"depth_label ({depth_label}). Checkpoint would be captured "
            f"before the filler target was reached."
        )

    ckpt_id = f"s1_prequery_d{depth_label // 1000}k" if depth_label > 0 else "s1_prequery_d0"
    return save_checkpoint(
        state_dir=state_dir,
        checkpoint_id=ckpt_id,
        checkpoint_kind="s1_prequery",
        episode_id=episode_id,
        agent_id=agent_id,
        session_id=session_id,
        effective_input_tokens=effective_input_tokens,
        jsonl_entry_count=jsonl_entry_count,
        history_depth_label=depth_label,
        checkpoints_root=checkpoints_root,
    )


# ---------------------------------------------------------------------------
# S2/S3 post-compaction checkpoint capture helpers
# ---------------------------------------------------------------------------

def save_postcomp_checkpoint(
    state_dir: Path,
    episode_id: str,
    agent_id: str,
    session_id: str,
    compaction_threshold: int,
    native_compaction_count: int,
    boundary_input_tokens: int | None,
    compaction_verified: bool,
    checkpoints_root: Path | None = None,
) -> CheckpointManifest:
    """Save a post-compaction, pre-query checkpoint (shared S2/S3 base).

    CAPTURE CRITERION:
    This must be called AFTER all of:
      1. Filler injection drove context to contextWindow - reserveTokensFloor.
      2. OpenClaw native compaction fired (sessions.json compactionCount
         increased above the pre-filler value).
      3. Memory persistence was verified (compaction_verified = True means
         _verify_memory_write confirmed artifact in workspace).
    And BEFORE any S2 or S3 query is sent into this session.

    The checkpoint is a shared base for both S2 (same-session continuation)
    and S3 (fresh-session continuation).

    Args:
        compaction_threshold: Target threshold (50000, 100000, 150000) that
            was used as contextWindow - reserveTokensFloor.
        native_compaction_count: REQUIRED. sessions.json compactionCount
            AFTER compaction fired. Must be > 0.
        boundary_input_tokens: Provider-reported effective input tokens
            at the compaction boundary. May be None if provider did not
            report (logged as a warning).
        compaction_verified: REQUIRED. True if artifact was verified in
            workspace memory files after compaction.

    Raises:
        ValueError: If native_compaction_count <= 0 (no real compaction).
        ValueError: If compaction_verified is False (artifact not persisted).
    """
    if native_compaction_count is None or native_compaction_count <= 0:
        raise ValueError(
            f"native_compaction_count ({native_compaction_count}) must be > 0. "
            "Post-compaction checkpoint requires real compaction to have fired."
        )
    if not compaction_verified:
        raise ValueError(
            "compaction_verified is False. Post-compaction checkpoint requires "
            "verified artifact persistence in workspace memory files. "
            "A checkpoint with unverified memory is not a valid S2/S3 base."
        )
    if boundary_input_tokens is None:
        logger.warning(
            "boundary_input_tokens is None for postcomp checkpoint at threshold=%d. "
            "Provider may not have reported input tokens.",
            compaction_threshold,
        )

    threshold_k = compaction_threshold // 1000
    ckpt_id = f"postcomp_{threshold_k}k"
    return save_checkpoint(
        state_dir=state_dir,
        checkpoint_id=ckpt_id,
        checkpoint_kind="postcomp",
        episode_id=episode_id,
        agent_id=agent_id,
        session_id=session_id,
        compaction_threshold=compaction_threshold,
        native_compaction_count=native_compaction_count,
        boundary_input_tokens=boundary_input_tokens,
        compaction_verified=compaction_verified,
        checkpoints_root=checkpoints_root,
    )


# ---------------------------------------------------------------------------
# Listing / loading helpers
# ---------------------------------------------------------------------------

def list_checkpoints(
    episode_id: str,
    checkpoints_root: Path | None = None,
) -> list[CheckpointManifest]:
    """List all checkpoints for an episode."""
    root = checkpoints_root or CHECKPOINTS_ROOT
    episode_dir = root / episode_id
    if not episode_dir.exists():
        return []
    manifests = []
    for ckpt_dir in sorted(episode_dir.iterdir()):
        manifest_path = ckpt_dir / "manifest.json"
        if manifest_path.exists():
            manifests.append(CheckpointManifest.load(manifest_path))
    return manifests


def load_checkpoint(
    episode_id: str,
    checkpoint_id: str,
    checkpoints_root: Path | None = None,
) -> CheckpointManifest:
    """Load a specific checkpoint manifest."""
    root = checkpoints_root or CHECKPOINTS_ROOT
    manifest_path = root / episode_id / checkpoint_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {manifest_path}")
    return CheckpointManifest.load(manifest_path)


# ---------------------------------------------------------------------------
# S3 restore helpers (memory sync + reindex)
# ---------------------------------------------------------------------------

def _sync_root_memory_md(workspace_path: Path) -> None:
    """Create/update MEMORY.md at workspace root from memory/*.md content.

    OpenClaw's fresh-session bootstrap reads MEMORY.md at workspace root, but
    the agent writes memory files into the memory/ subdirectory. This syncs
    all memory/*.md content into a root-level MEMORY.md so the S3 startup
    bootstrap finds artifact content and populates startup_exposure_log
    (enabling R_boot detection).

    Artifact files (non-date-named) are placed first so they aren't truncated
    by OpenClaw's read limit when the bootstrap reads MEMORY.md.
    """
    import re as _re

    memory_dir = workspace_path / "memory"
    if not memory_dir.exists():
        return

    _date_pat = _re.compile(r'^\d{4}-\d{2}-\d{2}')
    all_files = list(memory_dir.glob("*.md"))
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
    logger.info("Synced %d memory files → MEMORY.md (%s)", len(parts), workspace_path)


async def _reindex_memory(agent_id: str, state_dir: Path) -> None:
    """Force reindex memory files so memory_search reflects current workspace state."""
    cli_env = {**os.environ, "OPENCLAW_STATE_DIR": str(state_dir)}
    proc = await asyncio.create_subprocess_exec(
        "openclaw", "memory", "index", "--force", "--agent", agent_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=cli_env,
    )
    await proc.communicate()
    logger.info("Memory index updated (agent=%s, state_dir=%s)", agent_id, state_dir)


# ---------------------------------------------------------------------------
# Text extraction helper (from trial_runner pattern)
# ---------------------------------------------------------------------------

def _extract_text_from_jsonl(jsonl: list[dict], split: int) -> str:
    """Extract assistant response text from new JSONL entries after `split`.

    Fallback for when ``openclaw agent --json`` returns empty stdout but the
    session JSONL was updated.  Scans entries added after the split point for
    assistant-role content.

    OpenClaw JSONL format: each entry has ``type: "message"`` with a nested
    ``message`` dict containing ``role`` and ``content``.
    """
    new_entries = jsonl[split:]
    parts: list[str] = []
    for entry in new_entries:
        msg = entry.get("message", {})
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        if role != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    # Strip OpenClaw reply markers like [[reply_to_current]]
                    text = text.replace("[[reply_to_current]]", "").strip()
                    if text:
                        parts.append(text)
                elif isinstance(part, str):
                    parts.append(part)
    return "\n".join(parts)


def _extract_text(payload: dict) -> str:
    """Extract assistant text from an openclaw response payload."""
    if not payload:
        return ""
    result = payload.get("result", {})
    if "payloads" in result:
        parts = [p["text"] for p in result["payloads"] if isinstance(p, dict) and p.get("text")]
        if parts:
            return "\n".join(parts)
    if "text" in payload:
        return payload["text"]
    if "content" in payload:
        content = payload["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return str(payload)
