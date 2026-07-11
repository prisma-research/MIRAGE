"""
Resume from s1_prequery_d80k to build postcomp_100k with pre-compaction memory-write intervention.

Resumes from existing checkpoint — does NOT rebuild the pre-d80k trunk.

Usage:
    cd MIRAGE
    python -m harness.resume_postcomp_from_checkpoint
"""

from __future__ import annotations

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
    save_postcomp_checkpoint,
    CHECKPOINTS_ROOT,
)
from harness.filler_turns import get_filler_turns
from harness.episode_checkpoint_runner import _verify_artifact_persistence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SOURCE_EPISODE = "pilot_v3_100k"
SOURCE_CHECKPOINT = "s1_prequery_d80k"
TARGET_EPISODE = "pilot_v3_100k_prewrite"
COMPACTION_THRESHOLD = 100_000

ARTIFACT_KEYWORDS = {
    "cqa_9f9df328": ["gdp", "new", "jersey", "556", "dollars"],
    "cqa_536c64e4": ["internet", "users", "canada", "35.63"],
    "cqa_3dd0635a": ["poker", "tables", "casinos", "nevada", "313"],
    "cqa_92d6ef9a": ["fan", "cost", "index", "boston", "celtics", "419"],
    "ss_324e28cd": ["fitness", "leelin", "181", "steps"],
    "ss_102dfb65": ["google", "maps", "notifications"],
}

# --- DEPRECATED: leaked prewrite prompt ---
# This prompt enumerates all 6 artifacts with specific answer-bearing values
# (556.73, 35.63M, 920/313, 676.42/419.96, 181cm, etc.), making it unsuitable
# as a fair experimental condition. Retained for historical comparison only.
# Do NOT use for new experiments.
PREWRITE_PROMPT_LEAKED = (
    "I need you to do something important before we continue. I've shared 6 visual items "
    "with you earlier in this conversation — 4 charts and 2 phone screenshots. I need you "
    "to write a compact memory file entry for EACH one right now, using your file-editing tools.\n\n"
    "For each item, create or update a file in your memory/ directory with:\n"
    "- A clear title identifying the artifact\n"
    "- The type (chart or screenshot)\n"
    "- Key data: specific numbers, years, values, names, or labels visible in the artifact\n"
    "- Enough detail that you could answer questions about it later from memory alone\n\n"
    "The 6 items are:\n"
    "1. New Jersey Real GDP chart (2000-2020, line chart, values in billion USD, peak 556.73 in 2019)\n"
    "2. Canada digital users chart (bar chart, Jan 2021, ~35.63M internet users, 4 categories)\n"
    "3. Nevada casino poker tables chart (2000-2020, bar chart, peak 920 in 2010, dropped to 313 in 2020)\n"
    "4. NBA Fan Cost Index chart (2015, horizontal bar, NY Knicks 676.42 highest, Boston Celtics 419.96)\n"
    "5. Android fitness app profile screenshot (user leelin, Male, 181cm, age 22, 2M steps award)\n"
    "6. iOS Google Maps notification settings screenshot (Allow Notifications toggle OFF)\n\n"
    "Please write all 6 memory entries now using your tools. This is important — I'll be asking "
    "about these later and need them saved durably."
)

# --- Active: generic prewrite prompt ---
# Does not enumerate artifacts, does not leak specific values.
# Relies on the model's own judgment about what is worth persisting.
PREWRITE_PROMPT_GENERIC = (
    "Before we continue, please review what we've covered so far and update your durable "
    "memory files with anything likely to matter later.\n\n"
    "Create or update concise, self-contained memory entries for important facts, references, "
    "materials, settings, names, numbers, decisions, and other details that may be useful to "
    "recover in a later session.\n\n"
    "Keep the entries organized, specific, and retrievable, but concise. Use your judgment "
    "about what is important enough to preserve, and write the notes into your memory files "
    "using your normal file-editing tools."
)

# Alias for backward compat — points to leaked version (this script's original behavior).
# New scripts should use PREWRITE_PROMPT_GENERIC explicitly.
PREWRITE_PROMPT = PREWRITE_PROMPT_LEAKED


async def main():
    # Start usage proxy
    needs_proxy = any(
        spec.get("needsUsageProxy") for spec in EXPERIMENT_MODEL_REGISTRY.values()
    )
    proxy_ctx = UsageProxy(patch_global=False) if needs_proxy else None
    proxy_port = None
    if proxy_ctx:
        proxy_ctx.__enter__()
        proxy_port = proxy_ctx.port
        logger.info("Usage proxy started on port %d", proxy_port)

    branch = None
    try:
        manifest = load_checkpoint(SOURCE_EPISODE, SOURCE_CHECKPOINT)
        logger.info("Loaded: %s (eit=%s)", manifest.checkpoint_id, manifest.effective_input_tokens)

        # Restore WITHOUT starting gateway (need to patch config first)
        branch = await restore_checkpoint(
            manifest, branch_id="resume_postcomp_100k",
            port_start=19700, start_gateway=False,
        )
        workspace = branch.branch_state_dir / f"workspace-{branch.agent_id}"

        # Patch config: proxy + contextWindow + no deepseek
        target_cw = COMPACTION_THRESHOLD + RESERVE_TOKENS_FLOOR
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

        # Start gateway
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

        async with OpenClawClient(
            agent_id=branch.agent_id, state_dir=branch.branch_state_dir,
        ) as client:
            sid = branch.session_id

            # Step 1: Prewrite intervention
            logger.info("Sending prewrite intervention...")
            await client.send_message(PREWRITE_PROMPT, session_id=sid)
            await asyncio.sleep(5.0)

            # Step 2: Verify after prewrite
            logger.info("Verifying after prewrite...")
            n_ok = 0
            for aid, kw in ARTIFACT_KEYWORDS.items():
                ok = _verify_artifact_persistence(workspace, aid, kw)
                n_ok += ok
                logger.info("  %s: %s", aid, "PERSISTED" if ok else "NOT FOUND")
            logger.info("Prewrite: %d/%d verified", n_ok, len(ARTIFACT_KEYWORDS))

            if n_ok == 0:
                logger.error("Zero artifacts persisted. Aborting.")
                return

            # Step 3: Filler until compaction
            # Key invariant: once compactionCount increases, STOP immediately.
            # If a filler turn fails (provider 400, timeout, content filter),
            # check compaction first — if it fired, salvage. Otherwise skip
            # that filler and try the next one.
            logger.info("Filling toward compaction at %d...", COMPACTION_THRESHOLD)
            initial_cc = client.get_session_compaction_count(sid)
            pool = get_filler_turns(n=50, seed=300, category="substantive")
            fired = False
            consecutive_errors = 0
            MAX_CONSECUTIVE_ERRORS = 5

            for j, msg in enumerate(itertools.islice(itertools.cycle(pool), 60)):
                try:
                    await asyncio.wait_for(
                        client.send_message(msg, session_id=sid),
                        timeout=180,
                    )
                    consecutive_errors = 0
                except (asyncio.TimeoutError, Exception) as exc:
                    is_timeout = isinstance(exc, asyncio.TimeoutError)
                    label = "TIMEOUT" if is_timeout else type(exc).__name__
                    logger.warning("Filler turn %d %s: %s", j+1, label, exc)
                    cc = client.get_session_compaction_count(sid)
                    if cc > initial_cc:
                        logger.info("compaction_fired (detected after %s, cc: %d→%d)",
                                    label, initial_cc, cc)
                        logger.info("postcomp_salvaged_after_provider_error: turn %d", j+1)
                        fired = True
                        break
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.error("Too many consecutive filler errors (%d). Aborting.", consecutive_errors)
                        break
                    logger.info("Compaction not yet fired (cc=%d). Skipping failed filler, trying next.", cc)
                    continue

                cc = client.get_session_compaction_count(sid)
                if cc > initial_cc:
                    logger.info("compaction_fired after %d turns (cc: %d→%d)", j+1, initial_cc, cc)
                    fired = True
                    break
                if (j+1) % 5 == 0:
                    eit = client.get_effective_input_tokens(sid)
                    logger.info("  Turn %d: eit=%s, cc=%d", j+1, eit, cc)

            if not fired:
                logger.error("Compaction did not fire in 60 turns. Aborting.")
                return

            # Step 4: Verify after compaction
            logger.info("Verifying after compaction...")
            all_ok = True
            for aid, kw in ARTIFACT_KEYWORDS.items():
                ok = _verify_artifact_persistence(workspace, aid, kw)
                if not ok:
                    all_ok = False
                logger.info("  %s: %s", aid, "PERSISTED" if ok else "NOT FOUND")

            if not all_ok:
                missing = [a for a, kw in ARTIFACT_KEYWORDS.items()
                           if not _verify_artifact_persistence(workspace, a, kw)]
                logger.warning("Missing after compaction: %s — sending manual flush", missing)
                await client.send_message(
                    "Please check your memory files and make sure all 6 visual artifacts "
                    "are saved. Write or update any that are missing.", session_id=sid,
                )
                await asyncio.sleep(3.0)
                all_ok = all(
                    _verify_artifact_persistence(workspace, a, kw)
                    for a, kw in ARTIFACT_KEYWORDS.items()
                )
                n_final = sum(
                    _verify_artifact_persistence(workspace, a, kw)
                    for a, kw in ARTIFACT_KEYWORDS.items()
                )
                logger.info("After flush: %d/%d verified", n_final, len(ARTIFACT_KEYWORDS))

            # Step 5: Save
            eit = client.get_effective_input_tokens(sid)
            cc = client.get_session_compaction_count(sid)
            logger.info("Saving postcomp_100k (eit=%s, cc=%d, verified=%s)", eit, cc, all_ok)

            m = save_postcomp_checkpoint(
                state_dir=branch.branch_state_dir,
                episode_id=TARGET_EPISODE,
                agent_id=branch.agent_id,
                session_id=sid,
                compaction_threshold=COMPACTION_THRESHOLD,
                native_compaction_count=cc,
                boundary_input_tokens=eit,
                compaction_verified=all_ok,
                checkpoints_root=CHECKPOINTS_ROOT,
            )

            print(f"\n{'='*60}")
            print("POSTCOMP_100K SAVED")
            print(f"{'='*60}")
            print(f"  checkpoint_id:          {m.checkpoint_id}")
            print(f"  episode_id:             {m.episode_id}")
            print(f"  manifest:               {m.checkpoint_dir}/manifest.json")
            print(f"  native_compaction_count: {m.native_compaction_count}")
            print(f"  boundary_input_tokens:  {m.boundary_input_tokens}")
            print(f"  compaction_verified:    {m.compaction_verified}")
            print(f"  S3 valid:               {'YES' if all_ok else 'NO'}")

    finally:
        if branch:
            branch.cleanup()
        if proxy_ctx:
            proxy_ctx.__exit__(None, None, None)
            logger.info("Proxy stopped")


if __name__ == "__main__":
    asyncio.run(main())
