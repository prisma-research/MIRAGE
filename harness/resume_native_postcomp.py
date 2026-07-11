"""
Resume from an s1_prequery_d80k checkpoint and build a NATIVE postcomp_<N>
checkpoint — paper-matched config (NO prewrite, native summarizer = primary model).

Why this exists: episode_checkpoint_runner builds the whole trunk in one shot, and
its compaction_wait filler loop uses a 180s per-turn timeout. With slow high-context
backbones (e.g. Doubao at ~95k EIT) a single turn can exceed 180s; the orphaned
gateway subprocess keeps the session .jsonl.lock held, so the next turns fail with
"session file locked" and 5 consecutive errors abort the build at the very last step
— after d0/d50k/d80k were already saved. This script resumes from the saved d80k
checkpoint (no rebuild) and uses a longer per-turn timeout to avoid orphaned locks.

NATIVE = no PREWRITE_PROMPT injection, compaction_model=None (runtime uses the
primary agent model as summarizer). This matches the paper GPT-5 config produced by
episode_checkpoint_runner on pilot_episode_100k.json.

Usage:
    cd MIRAGE
    OPENCLAW_TOKEN=none python -m harness.resume_native_postcomp \\
        --source-episode pilot_v3_100k_doubao \\
        --target-episode pilot_v3_100k_doubao \\
        --threshold 100000 --turn-timeout 600
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
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from client.openclaw_client import OpenClawClient
from client.usage_proxy import UsageProxy, patch_openclaw_config
from constants import EXPERIMENT_MODEL_REGISTRY, RESERVE_TOKENS_FLOOR
from harness.checkpoint import (
    load_checkpoint, restore_checkpoint, save_postcomp_checkpoint, CHECKPOINTS_ROOT,
)
from harness.filler_turns import get_filler_turns
from harness.episode_checkpoint_runner import _verify_artifact_persistence

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# 6 main-experiment artifacts + content keywords (from pilot_episode_100k.json),
# used only for an informational post-compaction persistence check.
ARTIFACT_KEYWORDS = {
    "cqa_9f9df328": ["gdp", "new", "jersey", "556", "dollars"],
    "cqa_536c64e4": ["internet", "users", "canada", "35.63"],
    "cqa_3dd0635a": ["poker", "tables", "casinos", "nevada", "313"],
    "cqa_92d6ef9a": ["fan", "cost", "index", "boston", "celtics", "419"],
    "ss_324e28cd": ["week", "summary"],
    "ss_102dfb65": ["google", "maps", "notifications"],
}


async def main(args):
    needs_proxy = any(s.get("needsUsageProxy") for s in EXPERIMENT_MODEL_REGISTRY.values())
    proxy_ctx = UsageProxy(patch_global=False) if needs_proxy else None
    proxy_port = None
    if proxy_ctx:
        proxy_ctx.__enter__()
        proxy_port = proxy_ctx.port
        logger.info("Usage proxy started on port %d", proxy_port)

    branch = None
    try:
        manifest = load_checkpoint(args.source_episode, args.source_checkpoint)
        logger.info("Loaded %s/%s (eit=%s)", args.source_episode,
                    args.source_checkpoint, manifest.effective_input_tokens)

        branch = await restore_checkpoint(
            manifest, branch_id=f"resume_native_postcomp_{args.threshold//1000}k",
            port_start=19700, start_gateway=False,
        )
        workspace = branch.branch_state_dir / f"workspace-{branch.agent_id}"

        target_cw = args.threshold + RESERVE_TOKENS_FLOOR
        cfg_path = branch.branch_state_dir / "openclaw.json"
        with cfg_path.open() as f:
            cfg = json.load(f)
        # NATIVE: compaction_model=None → runtime uses primary agent model as summarizer.
        patch_openclaw_config(cfg, proxy_port=proxy_port,
                              context_window_override=target_cw, compaction_model=None)
        # Restored branch configs lack gateway.mode → gateway refuses to start
        # ("set gateway.mode=local or pass --allow-unconfigured"). Set it explicitly.
        cfg.setdefault("gateway", {})["mode"] = "local"
        if args.model:
            agents = cfg.setdefault("agents", {})
            agents.setdefault("defaults", {}).setdefault("model", {})["primary"] = args.model
            for entry in agents.get("list", []):
                if "model" in entry:
                    entry["model"] = args.model
        with cfg_path.open("w") as f:
            json.dump(cfg, f, indent=2)
        logger.info("Config patched: proxy=%s cw=%d model=%s (NATIVE, no prewrite)",
                    proxy_port, target_cw, args.model)

        env = {**os.environ, "OPENCLAW_STATE_DIR": str(branch.branch_state_dir)}
        (branch.branch_state_dir / "logs").mkdir(parents=True, exist_ok=True)
        gw = subprocess.Popen(
            ["openclaw", "gateway", "run", "--port", str(branch.gateway_port),
             "--force", "--allow-unconfigured"],
            stdout=(branch.branch_state_dir / "logs" / "gateway.log").open("a"),
            stderr=(branch.branch_state_dir / "logs" / "gateway.err.log").open("a"),
            env=env,
        )
        branch.gateway_proc = gw
        # HTTP readiness poll (gateway status is unreliable on this node — no systemd).
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{branch.gateway_port}/", timeout=2)
                break
            except (urllib.error.URLError, OSError, ConnectionRefusedError):
                pass
            if gw.poll() is not None:
                errlog = branch.branch_state_dir / "logs" / "gateway.err.log"
                tail = ""
                if errlog.exists():
                    tail = "\n".join(errlog.read_text().splitlines()[-30:])
                raise RuntimeError(f"Gateway exited early (code={gw.returncode}).\n"
                                   f"--- gateway.err.log tail ---\n{tail}")
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError(f"Gateway did not become ready on port {branch.gateway_port} in 60s")
        logger.info("Gateway ready (port=%d)", branch.gateway_port)

        async with OpenClawClient(agent_id=branch.agent_id,
                                  state_dir=branch.branch_state_dir) as client:
            sid = branch.session_id

            # Filler from d80k toward threshold — NO prewrite step (native).
            logger.info("Filling toward NATIVE compaction at %d (turn-timeout=%ds)...",
                        args.threshold, args.turn_timeout)
            initial_cc = client.get_session_compaction_count(sid)
            pool = get_filler_turns(n=50, seed=300, category="substantive")
            fired = False
            consecutive_errors = 0
            for j, msg in enumerate(itertools.islice(itertools.cycle(pool), args.max_turns)):
                try:
                    await asyncio.wait_for(
                        client.send_message(msg, session_id=sid),
                        timeout=args.turn_timeout,
                    )
                    consecutive_errors = 0
                except (asyncio.TimeoutError, Exception) as exc:
                    label = "TIMEOUT" if isinstance(exc, asyncio.TimeoutError) else type(exc).__name__
                    logger.warning("Filler turn %d %s: %s", j + 1, label, str(exc)[:200])
                    cc = client.get_session_compaction_count(sid)
                    if cc > initial_cc:
                        logger.info("compaction_fired (detected after %s, cc: %d→%d)",
                                    label, initial_cc, cc)
                        fired = True
                        break
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        logger.error("Too many consecutive filler errors (%d). Aborting.",
                                     consecutive_errors)
                        break
                    logger.info("Compaction not yet fired (cc=%d). Skipping to next filler.", cc)
                    # Give an orphaned slow request time to release the session lock.
                    await asyncio.sleep(args.lock_cooldown)
                    continue

                cc = client.get_session_compaction_count(sid)
                if cc > initial_cc:
                    logger.info("compaction_fired after %d turns (cc: %d→%d)",
                                j + 1, initial_cc, cc)
                    fired = True
                    break
                if (j + 1) % 3 == 0:
                    eit = client.get_effective_input_tokens(sid)
                    logger.info("  Turn %d: eit=%s, cc=%d", j + 1, eit, cc)

            if not fired:
                logger.error("Compaction did not fire within %d turns. Aborting.", args.max_turns)
                return

            # Informational post-compaction persistence check (native: do NOT force-flush).
            logger.info("Post-compaction artifact persistence (informational):")
            n_ok = 0
            for aid, kw in ARTIFACT_KEYWORDS.items():
                ok = _verify_artifact_persistence(workspace, aid, kw)
                n_ok += ok
                logger.info("  %s: %s", aid, "found" if ok else "not found")
            logger.info("Persistence: %d/%d artifact ids referenced in memory", n_ok, len(ARTIFACT_KEYWORDS))

            eit = client.get_effective_input_tokens(sid)
            cc = client.get_session_compaction_count(sid)
            logger.info("Saving postcomp_%dk (eit=%s, cc=%d)", args.threshold // 1000, eit, cc)
            m = save_postcomp_checkpoint(
                state_dir=branch.branch_state_dir,
                episode_id=args.target_episode,
                agent_id=branch.agent_id,
                session_id=sid,
                compaction_threshold=args.threshold,
                native_compaction_count=cc,
                boundary_input_tokens=eit,
                compaction_verified=(cc > initial_cc),
                checkpoints_root=CHECKPOINTS_ROOT,
            )
            print(f"\n{'='*60}\nNATIVE POSTCOMP SAVED\n{'='*60}")
            print(f"  checkpoint_id:           {m.checkpoint_id}")
            print(f"  episode_id:              {m.episode_id}")
            print(f"  native_compaction_count: {m.native_compaction_count}")
            print(f"  boundary_input_tokens:   {m.boundary_input_tokens}")
            print(f"  checkpoint_dir:          {m.checkpoint_dir}")
    finally:
        if branch:
            branch.cleanup()
        if proxy_ctx:
            proxy_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Resume d80k → NATIVE postcomp (no prewrite)")
    p.add_argument("--source-episode", default="pilot_v3_100k_doubao")
    p.add_argument("--source-checkpoint", default="s1_prequery_d80k")
    p.add_argument("--target-episode", default="pilot_v3_100k_doubao")
    p.add_argument("--threshold", type=int, default=100000)
    p.add_argument("--turn-timeout", type=int, default=600,
                   help="Per-filler-turn timeout (s). Must exceed the backbone's worst-case "
                        "high-context latency to avoid orphaned session-file locks.")
    p.add_argument("--lock-cooldown", type=float, default=15.0,
                   help="Sleep after a failed turn to let an orphaned request release the lock.")
    p.add_argument("--max-turns", type=int, default=60)
    p.add_argument("--model", default="volcengine/doubao-seed-2-0-pro-260215")
    asyncio.run(main(p.parse_args()))
