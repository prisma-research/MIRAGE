"""
Episode Checkpoint Runner — builds a long trunk conversation and captures
reusable checkpoint families for downstream branch probes.

This runner does NOT do evaluation. It only:
  1. Starts an isolated worker gateway
  2. Registers an agent and starts a session
  3. Walks through the trunk steps (artifacts, filler, checkpoints, compaction)
  4. Saves full-state checkpoints at each milestone
  5. Writes a run manifest summarizing all checkpoints

Usage:
    cd GroundingBench
    python -m harness.episode_checkpoint_runner \\
        --episode configs/episodes/pilot_episode.json \\
        --model shubiaobiao/gpt-5
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from client.openclaw_client import OpenClawClient
from client.usage_proxy import UsageProxy, patch_openclaw_config
from harness.checkpoint import (
    save_s1_prequery_checkpoint,
    save_postcomp_checkpoint,
    list_checkpoints,
    CHECKPOINTS_ROOT,
)
from harness.filler_turns import get_filler_turns
from constants import RESERVE_TOKENS_FLOOR, EXPERIMENT_MODEL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gateway lifecycle (single-worker, adapted from experiment_runner)
# ---------------------------------------------------------------------------

def _start_episode_gateway(
    state_dir: Path,
    port: int,
    proxy_port: int | None,
    context_window: int,
    model: str | None = None,
) -> subprocess.Popen:
    """Start a single isolated gateway for the episode trunk."""
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "logs").mkdir(parents=True, exist_ok=True)

    main_cfg_path = Path.home() / ".openclaw" / "openclaw.json"
    with main_cfg_path.open() as f:
        cfg = json.load(f)

    # mode=local is required or the gateway refuses to start ("set gateway.mode=local
    # or pass --allow-unconfigured"); the main ~/.openclaw config may not carry it
    # (the persistent gateway is started with --allow-unconfigured), so set it here.
    cfg["gateway"] = {**cfg.get("gateway", {}), "port": port, "mode": "local"}
    cfg.pop("web", None)

    # Agent defaults: compaction in safeguard mode
    agents_cfg = cfg.get("agents", {})
    agents_cfg.pop("list", None)
    defaults = agents_cfg.setdefault("defaults", {})
    defaults["compaction"] = {
        "mode": "safeguard",
        "reserveTokensFloor": RESERVE_TOKENS_FLOOR,
        "memoryFlush": {"enabled": True},
    }
    if model:
        defaults["model"] = {"primary": model}
    cfg["agents"] = agents_cfg

    if "tools" not in cfg:
        cfg["tools"] = {}
    cfg["tools"]["profile"] = "full"

    patch_openclaw_config(
        cfg,
        proxy_port=proxy_port,
        context_window_override=context_window,
        compaction_model=None,  # use the same model (e.g. gpt-5) for compaction
    )

    with (state_dir / "openclaw.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    env = {**os.environ, "OPENCLAW_STATE_DIR": str(state_dir)}
    proc = subprocess.Popen(
        ["openclaw", "gateway", "run", "--port", str(port), "--force", "--allow-unconfigured"],
        stdout=(state_dir / "logs" / "gateway.log").open("a"),
        stderr=(state_dir / "logs" / "gateway.err.log").open("a"),
        env=env,
    )

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["openclaw", "gateway", "status"],
                capture_output=True, timeout=15, env=env,
            )
            if result.returncode == 0:
                logger.info("Gateway ready (port=%d, pid=%d)", port, proc.pid)
                return proc
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.5)

    proc.kill()
    raise RuntimeError(f"Gateway did not start within 60s (port={port})")


async def _register_agent(agent_id: str, state_dir: Path, model: str | None = None) -> str:
    workspace = state_dir / f"workspace-{agent_id}"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "MEMORY.md").write_text("# Memory\n\nNo memories yet.\n")
    (workspace / "memory").mkdir(exist_ok=True)

    default_ws = Path.home() / ".openclaw" / "workspace"
    for fname in ("AGENTS.md", "SOUL.md", "USER.md", "IDENTITY.md",
                   "TOOLS.md", "HEARTBEAT.md", "BOOTSTRAP.md"):
        src = default_ws / fname
        if src.exists():
            shutil.copy2(src, workspace / fname)

    cmd = ["openclaw", "agents", "add", agent_id, "--workspace", str(workspace)]
    if model:
        cmd.extend(["--model", model])
    env = {**os.environ, "OPENCLAW_STATE_DIR": str(state_dir)}
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode().strip() or stdout.decode().strip()
        if "already exists" not in err.lower():
            raise RuntimeError(f"Agent registration failed: {err}")
    logger.info("Agent registered: %s", agent_id)
    return agent_id


def _find_free_port(start: int = 19600) -> int:
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


def _verify_artifact_persistence(
    workspace: Path,
    artifact_id: str,
    ocr_keywords: list[str],
    min_keyword_matches: int = 2,
) -> bool:
    """Check that artifact content was persisted to workspace memory files.

    Uses OCR/content keywords as the persistence signal — NOT the literal
    artifact_id string. The model is not required to copy the artifact_id
    into memory; it only needs to have saved enough of the artifact's actual
    content (chart values, UI labels, etc.) to be retrievable later.

    Falls back to artifact_id check as a secondary signal (some models do
    save it), but the primary criterion is keyword presence.
    """
    keywords = [k.lower() for k in ocr_keywords if k and len(k) >= 3]
    if not keywords:
        # No usable keywords — fall back to artifact_id check
        keywords = [artifact_id.lower()]

    all_text = ""
    for md in [workspace / "MEMORY.md"] + list((workspace / "memory").glob("*.md")):
        if md.exists():
            all_text += " " + md.read_text()

    if not all_text.strip():
        return False

    text_lower = all_text.lower()

    # Primary: keyword matches
    matched = sum(1 for k in keywords if k in text_lower)
    if matched >= min(min_keyword_matches, len(keywords)):
        return True

    # Secondary fallback: artifact_id literal
    if artifact_id.lower() in text_lower:
        return True

    return False


# ---------------------------------------------------------------------------
# Trunk execution
# ---------------------------------------------------------------------------

async def run_episode(
    episode_path: Path,
    model: str,
    episode_id: str | None = None,
    checkpoints_root: Path | None = None,
) -> dict:
    """Execute the trunk conversation and capture checkpoints.

    Returns a run manifest dict summarizing the episode.
    """
    with episode_path.open() as f:
        episode = json.load(f)

    ckpt_root = checkpoints_root or CHECKPOINTS_ROOT
    ep_id = episode_id or episode.get("episode_id", f"ep_{uuid.uuid4().hex[:8]}")
    trunk_steps = episode["trunk"]
    artifact_defs = episode.get("artifacts", {})

    # Resolve image paths relative to GroundingBench root
    gb_root = Path(__file__).parent.parent
    for aid, adef in artifact_defs.items():
        raw = adef.get("image_path")
        if raw:
            adef["_resolved_image_path"] = str(gb_root / raw)
        rawf = adef.get("file_path")
        if rawf:
            adef["_resolved_file_path"] = str(gb_root / rawf)

    # Compute initial contextWindow from the highest compaction threshold in the
    # episode config. This ensures the session can reach all S1 depth milestones
    # (e.g., d50k, d80k) WITHOUT triggering premature compaction.
    # For a 100k design: initial_cw = 100000 + 20000 = 120000.
    max_comp_threshold = 50000  # default fallback
    for step in trunk_steps:
        if step.get("type") == "compaction_wait":
            t = step.get("compaction_threshold", 0)
            if t > max_comp_threshold:
                max_comp_threshold = t
    initial_cw = max_comp_threshold + RESERVE_TOKENS_FLOOR
    logger.info("Initial contextWindow: %d (from max compaction threshold %d + reserve %d)",
                initial_cw, max_comp_threshold, RESERVE_TOKENS_FLOOR)

    # Start proxy if needed
    needs_proxy = any(
        spec.get("needsUsageProxy")
        for spec in EXPERIMENT_MODEL_REGISTRY.values()
    )
    proxy_ctx = UsageProxy(patch_global=False) if needs_proxy else None
    proxy_port = None
    if proxy_ctx:
        proxy_ctx.__enter__()
        proxy_port = proxy_ctx.port
        logger.info("Usage proxy on port %d", proxy_port)

    port = _find_free_port()
    # Point all OpenClawClient instances at THIS episode gateway's port. Without this
    # the client defaults to ws://127.0.0.1:18789; on a login node that accidentally
    # hits the persistent main gateway, but on a compute node (no :18789, no systemd)
    # connect()'s HTTP probe fails and the systemd CLI fallback errors out.
    os.environ["OPENCLAW_WS_URL"] = f"ws://127.0.0.1:{port}"
    state_dir = Path.home() / f".openclaw-episode-{ep_id}"
    if state_dir.exists():
        shutil.rmtree(state_dir)

    gateway_proc = _start_episode_gateway(
        state_dir, port, proxy_port, initial_cw, model=model,
    )

    manifests = []
    run_manifest = {
        "episode_id": ep_id,
        "model": model,
        "episode_config": str(episode_path),
        "started_at": datetime.datetime.utcnow().isoformat() + "Z",
        "state_dir": str(state_dir),
        "checkpoints": [],
    }

    try:
        agent_id = await _register_agent(
            f"ep_{uuid.uuid4().hex[:8]}", state_dir, model=model,
        )
        workspace = state_dir / f"workspace-{agent_id}"

        async with OpenClawClient(agent_id=agent_id, state_dir=state_dir) as client:
            session_id = await client.start_session(reason="new")
            logger.info("Trunk session started: %s", session_id)

            planted_artifacts: list[str] = []

            for i, step in enumerate(trunk_steps):
                step_type = step.get("type")
                if not step_type:
                    continue  # skip comment / section-marker entries
                logger.info("Step %d/%d: %s", i + 1, len(trunk_steps), step_type)

                if step_type == "artifact":
                    aid = step["artifact_id"]
                    prompt = step["prompt"]
                    adef = artifact_defs.get(aid, {})
                    logger.info("  Planting artifact: %s (%s)", aid, adef.get("plant_type", "text"))

                    # Handle image: copy to workspace for OpenClaw media allowlist
                    image_path = None
                    if step.get("image") and adef.get("_resolved_image_path"):
                        src_img = Path(adef["_resolved_image_path"])
                        if src_img.exists():
                            dest_img = workspace / src_img.name
                            shutil.copy2(src_img, dest_img)
                            image_path = str(dest_img)
                            logger.info("  Image copied to workspace: %s", dest_img.name)
                        else:
                            logger.warning("  Image not found: %s", src_img)

                    # Handle non-image file (text/pdf artifact): copy into the
                    # workspace under its name so the model can cite/read the file
                    # (source = filename → artifact_id) and it persists post-compaction.
                    if not step.get("image") and adef.get("_resolved_file_path"):
                        src_f = Path(adef["_resolved_file_path"])
                        if src_f.exists():
                            shutil.copy2(src_f, workspace / src_f.name)
                            logger.info("  File copied to workspace: %s", src_f.name)
                        else:
                            logger.warning("  File not found: %s", src_f)

                    await client.send_message(
                        prompt, session_id=session_id, image_path=image_path,
                    )
                    await asyncio.sleep(2.0)
                    planted_artifacts.append(aid)

                    # Verify persistence using content keywords (not artifact_id)
                    kw = adef.get("ocr_keywords", [])
                    if _verify_artifact_persistence(workspace, aid, kw):
                        logger.info("  Artifact %s content persisted to memory", aid)
                    else:
                        logger.warning("  Artifact %s content NOT found in memory after planting", aid)

                elif step_type == "work_task":
                    prompt = step["prompt"]
                    logger.info("  Work task: %s", prompt[:60])
                    await client.send_message(prompt, session_id=session_id)

                elif step_type == "memory_write":
                    # Explicit pre-compaction intervention: instruct the agent
                    # to write durable memory entries for all planted artifacts.
                    # This is an intervention condition, not the untouched baseline.
                    prompt = step["prompt"]
                    logger.info("  Memory-write intervention: requesting durable memory entries")
                    await client.send_message(prompt, session_id=session_id)
                    await asyncio.sleep(3.0)

                    # Verify each artifact was written
                    written = 0
                    for aid in planted_artifacts:
                        kw = artifact_defs.get(aid, {}).get("ocr_keywords", [])
                        if _verify_artifact_persistence(workspace, aid, kw):
                            written += 1
                            logger.info("    %s: persisted", aid)
                        else:
                            logger.warning("    %s: NOT found after memory-write", aid)
                    logger.info("  Memory-write result: %d/%d artifacts verified",
                                written, len(planted_artifacts))

                elif step_type == "filler":
                    target_eit = step["target_eit"]
                    seed = step.get("seed", 42)
                    pool = get_filler_turns(n=50, seed=seed, category="substantive")
                    for msg in itertools.cycle(pool):
                        eit = client.get_effective_input_tokens(session_id)
                        if eit is not None and eit >= target_eit:
                            break
                        await client.send_message(msg, session_id=session_id)
                    eit = client.get_effective_input_tokens(session_id)
                    logger.info("  Filler complete: eit=%s (target=%d)", eit, target_eit)

                elif step_type == "checkpoint":
                    kind = step["checkpoint_kind"]
                    eit = client.get_effective_input_tokens(session_id)
                    cc = client.get_session_compaction_count(session_id)
                    jsonl_count = len(client.get_session_jsonl(session_id))

                    if kind == "s1_prequery":
                        depth = step["depth_label"]

                        # Runtime guard: fail early with an actionable message
                        # rather than letting the checkpoint invariant raise
                        # a confusing ValueError deep in the save function.
                        from harness.checkpoint import _S1_D0_EIT_CEILING
                        if depth == 0 and eit is not None and eit >= _S1_D0_EIT_CEILING:
                            raise RuntimeError(
                                f"Pre-d0 trunk too long: effective_input_tokens={eit} "
                                f"(ceiling is {_S1_D0_EIT_CEILING}). Shorten or remove "
                                f"pre-d0 work_task steps in the episode config."
                            )

                        m = save_s1_prequery_checkpoint(
                            state_dir=state_dir,
                            episode_id=ep_id,
                            agent_id=agent_id,
                            session_id=session_id,
                            depth_label=depth,
                            effective_input_tokens=eit if eit is not None else 0,
                            jsonl_entry_count=jsonl_count,
                            native_compaction_count=cc,
                            checkpoints_root=ckpt_root,
                        )
                        logger.info("  Saved %s (eit=%s, cc=%d)", m.checkpoint_id, eit, cc)
                    elif kind == "postcomp":
                        threshold = step["compaction_threshold"]
                        verified = all(
                            _verify_artifact_persistence(
                                workspace, aid,
                                artifact_defs.get(aid, {}).get("ocr_keywords", []),
                            )
                            for aid in planted_artifacts
                        )
                        m = save_postcomp_checkpoint(
                            state_dir=state_dir,
                            episode_id=ep_id,
                            agent_id=agent_id,
                            session_id=session_id,
                            compaction_threshold=threshold,
                            native_compaction_count=cc,
                            boundary_input_tokens=eit,
                            compaction_verified=verified,
                            checkpoints_root=ckpt_root,
                        )
                        logger.info("  Saved %s (eit=%s, cc=%d, verified=%s)",
                                    m.checkpoint_id, eit, cc, verified)
                    else:
                        raise ValueError(f"Unknown checkpoint_kind: {kind}")

                    manifests.append(m)
                    run_manifest["checkpoints"].append({
                        "checkpoint_id": m.checkpoint_id,
                        "checkpoint_kind": m.checkpoint_kind,
                        "effective_input_tokens": m.effective_input_tokens,
                        "native_compaction_count": m.native_compaction_count,
                        "compaction_verified": m.compaction_verified,
                    })

                elif step_type == "compaction_wait":
                    threshold = step["compaction_threshold"]
                    max_turns = step.get("max_filler_turns", 40)
                    seed = step.get("seed", 42)

                    # Reconfigure gateway contextWindow if needed
                    target_cw = threshold + RESERVE_TOKENS_FLOOR
                    cfg_path = state_dir / "openclaw.json"
                    with cfg_path.open() as f:
                        cfg = json.load(f)
                    current_cw = None
                    for prov in cfg.get("models", {}).get("providers", {}).values():
                        for mod in prov.get("models", []):
                            current_cw = mod.get("contextWindow")
                            break
                        if current_cw:
                            break
                    if current_cw != target_cw:
                        logger.info("  Reconfiguring contextWindow: %s → %d", current_cw, target_cw)
                        # compaction_model=None: preserve primary model for compaction,
                        # do NOT reintroduce deepseek override.
                        patch_openclaw_config(
                            cfg,
                            context_window_override=target_cw,
                            compaction_model=None,
                        )
                        with cfg_path.open("w") as f:
                            json.dump(cfg, f, indent=2)
                        # Note: OpenClaw reads config on each request, no restart needed

                    initial_cc = client.get_session_compaction_count(session_id)
                    pool = get_filler_turns(n=50, seed=seed, category="substantive")
                    fired = False
                    consecutive_errors = 0
                    for j, msg in enumerate(itertools.islice(itertools.cycle(pool), max_turns)):
                        try:
                            await asyncio.wait_for(
                                client.send_message(msg, session_id=session_id),
                                timeout=180,
                            )
                            consecutive_errors = 0
                        except (asyncio.TimeoutError, Exception) as exc:
                            is_timeout = isinstance(exc, asyncio.TimeoutError)
                            label = "TIMEOUT" if is_timeout else type(exc).__name__
                            logger.warning("  Filler turn %d %s: %s", j + 1, label, exc)
                            cc = client.get_session_compaction_count(session_id)
                            if cc > initial_cc:
                                logger.info("  compaction_fired (detected after %s, cc: %d→%d)",
                                            label, initial_cc, cc)
                                fired = True
                                break
                            consecutive_errors += 1
                            if consecutive_errors >= 5:
                                logger.error("  Too many consecutive filler errors (%d).", consecutive_errors)
                                break
                            logger.info("  Compaction not yet fired (cc=%d). Skipping to next filler.", cc)
                            continue

                        cc = client.get_session_compaction_count(session_id)
                        if cc > initial_cc:
                            logger.info("  compaction_fired after %d turns (cc: %d→%d)",
                                        j + 1, initial_cc, cc)
                            fired = True
                            break
                        if (j + 1) % 5 == 0:
                            eit = client.get_effective_input_tokens(session_id)
                            logger.info("  Filler turn %d: eit=%s, cc=%d", j + 1, eit, cc)

                    if not fired:
                        raise RuntimeError(
                            f"Compaction did not fire within {max_turns} turns "
                            f"at threshold={threshold}"
                        )

                    # Verify artifact persistence using content keywords
                    for aid in planted_artifacts:
                        kw = artifact_defs.get(aid, {}).get("ocr_keywords", [])
                        if not _verify_artifact_persistence(workspace, aid, kw):
                            logger.warning("  Artifact %s content not found after compaction"
                                           " — sending manual flush", aid)
                            await client.send_message(
                                "Please review your memory files and make sure you've saved "
                                "all the important details from the charts and screenshots "
                                "we discussed earlier in this session.",
                                session_id=session_id,
                            )

                else:
                    logger.warning("  Unknown step type: %s (skipping)", step_type)

        run_manifest["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        run_manifest["agent_id"] = agent_id
        run_manifest["session_id"] = session_id
        run_manifest["n_checkpoints"] = len(manifests)

        # Save run manifest
        manifest_dir = ckpt_root / ep_id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "run_manifest.json"
        with manifest_path.open("w") as f:
            json.dump(run_manifest, f, indent=2)
        logger.info("Run manifest saved: %s", manifest_path)

        return run_manifest

    finally:
        gateway_proc.terminate()
        try:
            gateway_proc.wait(timeout=5)
        except Exception:
            gateway_proc.kill()
        if proxy_ctx:
            proxy_ctx.__exit__(None, None, None)
        # Keep state_dir for debugging; user can clean up manually
        logger.info("Gateway stopped. State dir preserved: %s", state_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build trunk conversation and capture checkpoint family",
    )
    parser.add_argument(
        "--episode", type=str, required=True,
        help="Path to episode config JSON",
    )
    parser.add_argument(
        "--model", type=str, default="shubiaobiao/gpt-5",
        help="Model to use (default: shubiaobiao/gpt-5)",
    )
    parser.add_argument(
        "--episode-id", type=str, default=None,
        help="Override episode ID (default: from config)",
    )
    parser.add_argument(
        "--checkpoints-root", type=str, default=None,
        help="Override checkpoint storage root",
    )
    args = parser.parse_args()

    ckpt_root = Path(args.checkpoints_root) if args.checkpoints_root else None
    result = asyncio.run(run_episode(
        episode_path=Path(args.episode),
        model=args.model,
        episode_id=args.episode_id,
        checkpoints_root=ckpt_root,
    ))

    print("\n" + "=" * 60)
    print("EPISODE COMPLETE")
    print("=" * 60)
    print(f"  Episode: {result['episode_id']}")
    print(f"  Model: {result['model']}")
    print(f"  Checkpoints: {result['n_checkpoints']}")
    for c in result["checkpoints"]:
        print(f"    {c['checkpoint_id']} (eit={c['effective_input_tokens']}, "
              f"cc={c['native_compaction_count']})")


if __name__ == "__main__":
    main()
