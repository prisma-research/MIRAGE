"""
Build generic-prewrite post-compaction checkpoint in two phases:

Phase 1: d80k → near_comp_base_95k (custom near-compaction branching base)
  - Restore from pilot_v3_100k / s1_prequery_d80k
  - Inject generic substantive filler to push eit to ~95k
  - Verify native_compaction_count still 0
  - Save as pilot_v3_100k / near_comp_base_95k (checkpoint_kind="custom")
  - This is NOT a standard S1 checkpoint — it is a prewrite branching base.

Phase 2: near_comp_base_95k → generic prewrite → compaction → postcomp
  - Restore from pilot_v3_100k / near_comp_base_95k
  - Send PREWRITE_PROMPT_GENERIC
  - Verify artifact persistence (MEMORY.md and/or memory/*.md)
  - Continue filler until compaction fires (with per-turn hard timeout)
  - Once compaction fires, STOP immediately — no more filler
  - Verify post-compaction persistence
  - Save as pilot_v3_100k_prewrite_generic / postcomp_100k

Usage:
    cd MIRAGE
    python -m harness.build_generic_prewrite_checkpoint --phase 1   # build near-comp base
    python -m harness.build_generic_prewrite_checkpoint --phase 2   # build postcomp
    python -m harness.build_generic_prewrite_checkpoint              # both phases
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from client.openclaw_client import OpenClawClient
from client.usage_proxy import UsageProxy, patch_openclaw_config
from constants import EXPERIMENT_MODEL_REGISTRY, RESERVE_TOKENS_FLOOR
from harness.checkpoint import (
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
    save_postcomp_checkpoint,
    CHECKPOINTS_ROOT,
)
from harness.filler_turns import get_filler_turns
from harness.episode_checkpoint_runner import _verify_artifact_persistence
from harness.resume_postcomp_from_checkpoint import (
    PREWRITE_PROMPT_GENERIC,
    ARTIFACT_KEYWORDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Phase 1 config ──
PHASE1_SOURCE_EPISODE = "pilot_v3_100k"
PHASE1_SOURCE_CHECKPOINT = "s1_prequery_d80k"
# This is a custom branching base, NOT a standard S1 checkpoint.
# It does not go through save_s1_prequery_checkpoint() and its depth invariants.
PHASE1_TARGET_CHECKPOINT = "near_comp_base_95k"
PHASE1_TARGET_EIT = 95_000

# ── Phase 2 config ──
PHASE2_SOURCE_EPISODE = "pilot_v3_100k"
PHASE2_SOURCE_CHECKPOINT = "near_comp_base_95k"
PHASE2_TARGET_EPISODE = "pilot_v3_100k_prewrite_generic"
COMPACTION_THRESHOLD = 100_000

# Hard timeout (seconds) for each filler send_message call.
# If OpenClaw gets stuck in a provider retry loop (e.g. Azure content filter),
# this ensures we regain control and can check compactionCount.
FILLER_TURN_TIMEOUT = 180  # 3 minutes per filler turn
MAX_CONSECUTIVE_ERRORS = 5


def _check_compaction_fired(client, session_id, initial_cc, branch_state_dir):
    """Dual-signal compaction detection.

    Returns (fired: bool, cc: int). Checks both:
      Signal A: sessions.json compactionCount > initial_cc
      Signal B: JSONL contains a type="compaction" event

    Either signal alone is sufficient — avoids race windows where one
    signal updates before the other.
    """
    # Signal A: compactionCount from sessions.json
    cc = client.get_session_compaction_count(session_id)
    if cc > initial_cc:
        return True, cc

    # Signal B: scan JSONL for compaction event (check last 10 lines only for speed)
    import json as _json
    agent_id = client.agent_id
    jsonl_path = (branch_state_dir / "agents" / agent_id
                  / "sessions" / f"{session_id}.jsonl")
    if jsonl_path.exists():
        lines = jsonl_path.read_text().strip().split('\n')
        for line in lines[-10:]:  # only check recent entries
            try:
                entry = _json.loads(line)
                if entry.get('type') == 'compaction':
                    # Signal B fired — re-read cc for consistency
                    cc = client.get_session_compaction_count(session_id)
                    return True, max(cc, 1)
            except _json.JSONDecodeError:
                continue

    return False, cc


async def _start_branch_with_proxy(manifest, branch_id, proxy_port, cw_override=None):
    """Restore checkpoint, patch config, start gateway. Returns branch."""
    branch = await restore_checkpoint(
        manifest, branch_id=branch_id,
        port_start=19700, start_gateway=False,
    )

    target_cw = cw_override or (COMPACTION_THRESHOLD + RESERVE_TOKENS_FLOOR)
    cfg_path = branch.branch_state_dir / "openclaw.json"
    with cfg_path.open() as f:
        cfg = json.load(f)
    patch_openclaw_config(
        cfg, proxy_port=proxy_port,
        context_window_override=target_cw, compaction_model=None,
    )
    with cfg_path.open("w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Config patched: proxy=%s, cw=%d", proxy_port, target_cw)

    env = {**os.environ, "OPENCLAW_STATE_DIR": str(branch.branch_state_dir)}
    (branch.branch_state_dir / "logs").mkdir(parents=True, exist_ok=True)
    gw = subprocess.Popen(
        ["openclaw", "gateway", "run", "--port", str(branch.gateway_port), "--force"],
        stdout=(branch.branch_state_dir / "logs" / "gateway.log").open("a"),
        stderr=(branch.branch_state_dir / "logs" / "gateway.err.log").open("a"),
        env=env,
    )
    branch.gateway_proc = gw
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        try:
            r = subprocess.run(["openclaw", "gateway", "status"],
                               capture_output=True, timeout=15, env=env)
            if r.returncode == 0:
                break
        except subprocess.TimeoutExpired:
            pass
        await asyncio.sleep(0.5)
    logger.info("Gateway ready (port=%d)", branch.gateway_port)
    return branch


async def _send_with_timeout(client, msg, session_id, timeout=FILLER_TURN_TIMEOUT):
    """Send a message with a hard external timeout.

    If OpenClaw is stuck in an internal provider retry loop (e.g. Azure
    content filter 400s), send_message() may never return. This wrapper
    ensures we regain control after `timeout` seconds so the caller can
    check compactionCount and decide whether to salvage or skip.
    """
    return await asyncio.wait_for(
        client.send_message(msg, session_id=session_id),
        timeout=timeout,
    )


async def phase1(proxy_port):
    """Phase 1: d80k → near_comp_base_95k (custom near-compaction branching base)."""
    logger.info("=== PHASE 1: Building near-compaction base checkpoint (d80k → ~95k) ===")

    manifest = load_checkpoint(PHASE1_SOURCE_EPISODE, PHASE1_SOURCE_CHECKPOINT)
    logger.info("Source: %s (eit=%s)", manifest.checkpoint_id, manifest.effective_input_tokens)

    branch = await _start_branch_with_proxy(manifest, "build_near_comp_base", proxy_port)

    try:
        async with OpenClawClient(
            agent_id=branch.agent_id, state_dir=branch.branch_state_dir,
        ) as client:
            sid = branch.session_id

            # Filler loop with per-turn hard timeout and skip-on-error.
            # Phase 1 specific: compaction firing is a FAILURE (we need pre-comp).
            phase1_initial_cc = client.get_session_compaction_count(sid)
            logger.info("Phase 1 initial compactionCount: %d", phase1_initial_cc)
            pool = get_filler_turns(n=50, seed=500, category="substantive")
            consecutive_errors = 0
            for j, msg in enumerate(itertools.islice(itertools.cycle(pool), 40)):
                try:
                    await _send_with_timeout(client, msg, sid)
                    consecutive_errors = 0
                except (asyncio.TimeoutError, Exception) as exc:
                    is_timeout = isinstance(exc, asyncio.TimeoutError)
                    label = "TIMEOUT" if is_timeout else type(exc).__name__
                    logger.warning("  Filler turn %d %s: %s", j + 1, label, exc)

                    # Dual-signal check: compaction during stuck/failed turn?
                    fired, cc = _check_compaction_fired(client, sid, phase1_initial_cc, branch.branch_state_dir)
                    if fired:
                        logger.error("Compaction fired during %s at turn %d (cc=%d). "
                                     "Cannot save pre-comp checkpoint.", label, j + 1, cc)
                        return None

                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.error("Too many consecutive filler errors (%d). Aborting phase 1.",
                                     consecutive_errors)
                        return None
                    logger.info("  Skipping failed filler, trying next (cc=%d).", cc)
                    continue

                eit = client.get_effective_input_tokens(sid)
                # Dual-signal compaction check after successful send
                fired, cc = _check_compaction_fired(client, sid, phase1_initial_cc, branch.branch_state_dir)

                if (j + 1) % 3 == 0 or (eit and eit >= PHASE1_TARGET_EIT):
                    logger.info("  Turn %d: eit=%s, cc=%d", j + 1, eit, cc)

                if fired:
                    logger.error("Compaction fired prematurely at turn %d (cc=%d). Aborting.", j + 1, cc)
                    return None

                if eit and eit >= PHASE1_TARGET_EIT:
                    logger.info("Target eit reached: %d >= %d", eit, PHASE1_TARGET_EIT)
                    break

            # Final dual-signal check
            fired, final_cc = _check_compaction_fired(client, sid, phase1_initial_cc, branch.branch_state_dir)
            if fired:
                logger.error("Compaction fired during filler (cc=%d). Cannot save pre-comp checkpoint.", final_cc)
                return None
            final_eit = client.get_effective_input_tokens(sid)

            logger.info("Saving custom near-comp base checkpoint: eit=%s, cc=%d", final_eit, final_cc)

            jsonl_path = (branch.branch_state_dir / "agents" / branch.agent_id
                         / "sessions" / f"{sid}.jsonl")
            jsonl_count = sum(1 for line in jsonl_path.open() if line.strip()) if jsonl_path.exists() else None

            # checkpoint_kind="custom" — this is NOT a standard S1 depth checkpoint.
            # It is a near-compaction branching base for prewrite experiments.
            m = save_checkpoint(
                state_dir=branch.branch_state_dir,
                checkpoint_id=PHASE1_TARGET_CHECKPOINT,
                checkpoint_kind="custom",
                episode_id=PHASE1_SOURCE_EPISODE,
                agent_id=branch.agent_id,
                session_id=sid,
                effective_input_tokens=final_eit,
                jsonl_entry_count=jsonl_count,
                checkpoints_root=CHECKPOINTS_ROOT,
            )

            print(f"\n{'=' * 60}")
            print("PHASE 1 COMPLETE: NEAR-COMP BRANCHING BASE (custom)")
            print(f"{'=' * 60}")
            print(f"  checkpoint_id:           {m.checkpoint_id}")
            print(f"  checkpoint_kind:         custom (NOT a standard S1)")
            print(f"  episode_id:              {m.episode_id}")
            print(f"  effective_input_tokens:  {m.effective_input_tokens}")
            print(f"  native_compaction_count: 0 (verified)")
            print(f"  manifest:                {m.checkpoint_dir}/manifest.json")
            return m

    finally:
        branch.cleanup()


async def phase2(proxy_port):
    """Phase 2: near_comp_base → generic prewrite → compaction → postcomp."""
    logger.info("=== PHASE 2: Generic prewrite → compaction → postcomp ===")

    manifest = load_checkpoint(PHASE2_SOURCE_EPISODE, PHASE2_SOURCE_CHECKPOINT)
    logger.info("Source: %s (eit=%s)", manifest.checkpoint_id, manifest.effective_input_tokens)

    branch = await _start_branch_with_proxy(manifest, "build_postcomp_generic", proxy_port)
    workspace = branch.branch_state_dir / f"workspace-{branch.agent_id}"

    try:
        async with OpenClawClient(
            agent_id=branch.agent_id, state_dir=branch.branch_state_dir,
        ) as client:
            sid = branch.session_id

            # Step 1: Send generic prewrite
            logger.info("Sending PREWRITE_PROMPT_GENERIC...")
            await _send_with_timeout(client, PREWRITE_PROMPT_GENERIC, sid, timeout=300)
            await asyncio.sleep(5.0)

            # Step 2: Verify after prewrite
            # _verify_artifact_persistence checks BOTH MEMORY.md and memory/*.md.
            # The generic prompt may cause the model to write to MEMORY.md directly
            # (rather than individual memory/ files). This is a valid persistence
            # path — MEMORY.md is the primary file read during fresh-session bootstrap.
            logger.info("Verifying after prewrite (checks MEMORY.md + memory/*.md)...")
            n_ok = 0
            for aid, kw in ARTIFACT_KEYWORDS.items():
                ok = _verify_artifact_persistence(workspace, aid, kw)
                n_ok += ok
                logger.info("  %s: %s", aid, "PERSISTED" if ok else "NOT FOUND")
            logger.info("Prewrite: %d/%d verified", n_ok, len(ARTIFACT_KEYWORDS))

            # Log where content was actually written
            mem_md = workspace / "MEMORY.md"
            mem_dir_files = list((workspace / "memory").glob("*.md")) if (workspace / "memory").is_dir() else []
            logger.info("Persistence locations: MEMORY.md=%d bytes, memory/*.md=%d files",
                        mem_md.stat().st_size if mem_md.exists() else 0, len(mem_dir_files))

            if n_ok == 0:
                logger.error("Zero artifacts persisted after generic prewrite. Aborting.")
                return None

            # Step 3: Filler until compaction
            # INVARIANT: once compactionCount increases, STOP immediately.
            # Each send_message has a hard external timeout (FILLER_TURN_TIMEOUT).
            # On timeout or error: check compaction, salvage if fired, skip otherwise.
            logger.info("Filling toward compaction at %d...", COMPACTION_THRESHOLD)
            initial_cc = client.get_session_compaction_count(sid)
            pool = get_filler_turns(n=50, seed=600, category="substantive")
            fired = False
            salvaged = False
            consecutive_errors = 0

            for j, msg in enumerate(itertools.islice(itertools.cycle(pool), 60)):
                try:
                    await _send_with_timeout(client, msg, sid)
                    consecutive_errors = 0
                except (asyncio.TimeoutError, Exception) as exc:
                    is_timeout = isinstance(exc, asyncio.TimeoutError)
                    exc_label = "TIMEOUT" if is_timeout else type(exc).__name__
                    logger.warning("Filler turn %d %s: %s", j + 1, exc_label, exc)

                    # Dual-signal: check if compaction fired despite the error/timeout
                    comp_fired, cc = _check_compaction_fired(
                        client, sid, initial_cc, branch.branch_state_dir)
                    if comp_fired:
                        logger.info("compaction_fired (detected after %s, cc: %d→%d)",
                                    exc_label, initial_cc, cc)
                        logger.info("postcomp_salvaged_after_provider_error: turn %d", j + 1)
                        fired = True
                        salvaged = True
                        break

                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.error("Too many consecutive filler errors (%d). Aborting.", consecutive_errors)
                        break
                    logger.info("Compaction not yet fired (cc=%d). Skipping to next filler.", cc)
                    continue

                # Normal path: dual-signal compaction check after successful send
                comp_fired, cc = _check_compaction_fired(
                    client, sid, initial_cc, branch.branch_state_dir)
                if comp_fired:
                    logger.info("compaction_fired after %d turns (cc: %d→%d)", j + 1, initial_cc, cc)
                    fired = True
                    break
                if (j + 1) % 5 == 0:
                    eit = client.get_effective_input_tokens(sid)
                    logger.info("  Turn %d: eit=%s, cc=%d", j + 1, eit, cc)

            if not fired:
                logger.error("Compaction did not fire in 60 turns. Aborting.")
                return None

            # Step 4: Verify after compaction
            logger.info("Verifying after compaction (checks MEMORY.md + memory/*.md)...")
            all_ok = True
            for aid, kw in ARTIFACT_KEYWORDS.items():
                ok = _verify_artifact_persistence(workspace, aid, kw)
                if not ok:
                    all_ok = False
                logger.info("  %s: %s", aid, "PERSISTED" if ok else "NOT FOUND")

            manual_flush_used = False
            if not all_ok:
                missing = [a for a, kw in ARTIFACT_KEYWORDS.items()
                           if not _verify_artifact_persistence(workspace, a, kw)]
                logger.warning("Missing after compaction: %s — sending manual_flush_fallback", missing)
                manual_flush_used = True
                try:
                    await _send_with_timeout(
                        client,
                        "Please check your memory files and make sure all important information "
                        "from our earlier conversation is saved. Write or update any entries "
                        "that are missing or incomplete.",
                        sid, timeout=120,
                    )
                    await asyncio.sleep(3.0)
                except (asyncio.TimeoutError, Exception) as exc:
                    logger.warning("manual_flush_fallback failed: %s", exc)

                all_ok = all(
                    _verify_artifact_persistence(workspace, a, kw)
                    for a, kw in ARTIFACT_KEYWORDS.items()
                )
                n_final = sum(
                    _verify_artifact_persistence(workspace, a, kw)
                    for a, kw in ARTIFACT_KEYWORDS.items()
                )
                logger.info("After manual_flush_fallback: %d/%d verified", n_final, len(ARTIFACT_KEYWORDS))

            # Step 5: Save
            eit = client.get_effective_input_tokens(sid)
            cc = client.get_session_compaction_count(sid)
            logger.info("Saving postcomp_100k (eit=%s, cc=%d, verified=%s, salvaged=%s, flush=%s)",
                        eit, cc, all_ok, salvaged, manual_flush_used)

            m = save_postcomp_checkpoint(
                state_dir=branch.branch_state_dir,
                episode_id=PHASE2_TARGET_EPISODE,
                agent_id=branch.agent_id,
                session_id=sid,
                compaction_threshold=COMPACTION_THRESHOLD,
                native_compaction_count=cc,
                boundary_input_tokens=eit,
                compaction_verified=all_ok,
                checkpoints_root=CHECKPOINTS_ROOT,
            )

            print(f"\n{'=' * 60}")
            print("PHASE 2 COMPLETE: GENERIC-PREWRITE POSTCOMP CHECKPOINT")
            print(f"{'=' * 60}")
            print(f"  checkpoint_id:           {m.checkpoint_id}")
            print(f"  episode_id:              {m.episode_id}")
            print(f"  manifest:                {m.checkpoint_dir}/manifest.json")
            print(f"  native_compaction_count: {m.native_compaction_count}")
            print(f"  boundary_input_tokens:   {m.boundary_input_tokens}")
            print(f"  compaction_verified:     {m.compaction_verified}")
            print(f"  prewrite_type:           generic (no answer leakage)")
            print(f"  salvaged:                {salvaged}")
            print(f"  manual_flush_fallback:   {manual_flush_used}")
            return m

    finally:
        branch.cleanup()


async def main(args):
    needs_proxy = any(
        spec.get("needsUsageProxy") for spec in EXPERIMENT_MODEL_REGISTRY.values()
    )
    proxy_ctx = UsageProxy(patch_global=False) if needs_proxy else None
    proxy_port = None
    if proxy_ctx:
        proxy_ctx.__enter__()
        proxy_port = proxy_ctx.port
        logger.info("Usage proxy started on port %d", proxy_port)

    try:
        if args.phase in (None, 1):
            result = await phase1(proxy_port)
            if result is None and args.phase is None:
                logger.error("Phase 1 failed. Stopping.")
                return

        if args.phase in (None, 2):
            await phase2(proxy_port)
    finally:
        if proxy_ctx:
            proxy_ctx.__exit__(None, None, None)
            logger.info("Proxy stopped")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, choices=[1, 2], default=None,
                   help="Run only phase 1 or 2 (default: both)")
    asyncio.run(main(p.parse_args()))
