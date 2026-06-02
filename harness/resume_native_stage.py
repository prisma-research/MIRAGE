"""
Resumable NATIVE checkpoint stage builder. Restores from a saved checkpoint and
builds the NEXT checkpoint by sending substantive filler — either to a target EIT
(s1 stage) or until native compaction fires (postcomp stage). NO prewrite; native
summarizer = primary agent model.

This is the resumable unit for building a checkpoint family stage-by-stage:
each stage restores the previous saved checkpoint, so a crashed/timed-out job is
resumed simply by re-running the stages whose target checkpoint does not yet exist.

Robustness (hard-won, see memory groundingbench-doubao-runtime-setup):
  - per-turn timeout 600s (Doubao at ~95k EIT exceeds 180s → orphaned session lock)
  - lock-cooldown sleep after a failed turn
  - restored branch configs lack gateway.mode → set gateway.mode=local + --allow-unconfigured

Usage (s1 stage):
    OPENCLAW_TOKEN=none python -m harness.resume_native_stage \\
        --source-episode modality_ext_v1 --source-checkpoint s1_prequery_d0 \\
        --target-episode modality_ext_v1 --mode s1 --target-eit 50000 --depth-label 50000

Usage (postcomp stage):
    OPENCLAW_TOKEN=none python -m harness.resume_native_stage \\
        --source-episode modality_ext_v1 --source-checkpoint s1_prequery_d80k \\
        --target-episode modality_ext_v1 --mode postcomp --threshold 100000
"""
from __future__ import annotations
import argparse, asyncio, itertools, json, logging, os, subprocess, sys, time
import urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from client.openclaw_client import OpenClawClient
from client.usage_proxy import UsageProxy, patch_openclaw_config
from constants import EXPERIMENT_MODEL_REGISTRY, RESERVE_TOKENS_FLOOR
from harness.checkpoint import (
    load_checkpoint, restore_checkpoint,
    save_s1_prequery_checkpoint, save_postcomp_checkpoint, CHECKPOINTS_ROOT,
)
from harness.filler_turns import get_filler_turns

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def main(args):
    needs_proxy = any(s.get("needsUsageProxy") for s in EXPERIMENT_MODEL_REGISTRY.values())
    proxy_ctx = UsageProxy(patch_global=False) if needs_proxy else None
    proxy_port = None
    if proxy_ctx:
        proxy_ctx.__enter__(); proxy_port = proxy_ctx.port
        logger.info("Usage proxy on port %d", proxy_port)

    branch = None
    try:
        manifest = load_checkpoint(args.source_episode, args.source_checkpoint)
        logger.info("Resuming from %s/%s (eit=%s)", args.source_episode,
                    args.source_checkpoint, manifest.effective_input_tokens)

        branch = await restore_checkpoint(
            manifest, branch_id=f"stage_{args.mode}_{args.source_checkpoint}",
            port_start=19700, start_gateway=False)

        # contextWindow: for postcomp must sit above threshold; for s1 keep high so
        # compaction does NOT fire prematurely while filling to target_eit.
        cw = (args.threshold + RESERVE_TOKENS_FLOOR) if args.mode == "postcomp" else 200000
        cfg_path = branch.branch_state_dir / "openclaw.json"
        cfg = json.loads(cfg_path.read_text())
        patch_openclaw_config(cfg, proxy_port=proxy_port,
                              context_window_override=cw, compaction_model=None)
        cfg.setdefault("gateway", {})["mode"] = "local"
        if args.model:
            a = cfg.setdefault("agents", {})
            a.setdefault("defaults", {}).setdefault("model", {})["primary"] = args.model
            for e in a.get("list", []):
                if "model" in e: e["model"] = args.model
        cfg_path.write_text(json.dumps(cfg, indent=2))
        logger.info("Patched: cw=%d model=%s (NATIVE, no prewrite)", cw, args.model)

        env = {**os.environ, "OPENCLAW_STATE_DIR": str(branch.branch_state_dir)}
        (branch.branch_state_dir / "logs").mkdir(parents=True, exist_ok=True)
        gw = subprocess.Popen(
            ["openclaw", "gateway", "run", "--port", str(branch.gateway_port),
             "--force", "--allow-unconfigured"],
            stdout=(branch.branch_state_dir / "logs" / "gateway.log").open("a"),
            stderr=(branch.branch_state_dir / "logs" / "gateway.err.log").open("a"), env=env)
        branch.gateway_proc = gw
        dl = time.monotonic() + 60
        while time.monotonic() < dl:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{branch.gateway_port}/", timeout=2); break
            except (urllib.error.URLError, OSError): pass
            if gw.poll() is not None:
                err = (branch.branch_state_dir / "logs" / "gateway.err.log")
                tail = "\n".join(err.read_text().splitlines()[-20:]) if err.exists() else ""
                raise RuntimeError(f"Gateway exited early (code={gw.returncode}).\n{tail}")
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError("Gateway not ready in 60s")
        logger.info("Gateway ready (port=%d)", branch.gateway_port)
        # Point OpenClawClient at THIS branch gateway's port (default is :18789, which
        # only exists on the login node; on a compute node connect() would fail).
        os.environ["OPENCLAW_WS_URL"] = f"ws://127.0.0.1:{branch.gateway_port}"

        async with OpenClawClient(agent_id=branch.agent_id,
                                  state_dir=branch.branch_state_dir) as client:
            sid = branch.session_id
            initial_cc = client.get_session_compaction_count(sid)
            pool = get_filler_turns(n=50, seed=args.seed, category="substantive")
            target = args.target_eit if args.mode == "s1" else args.threshold
            logger.info("Filling (mode=%s) toward %s (turn-timeout=%ds)...",
                        args.mode, target, args.turn_timeout)
            fired = False
            consecutive_errors = 0
            for j, msg in enumerate(itertools.islice(itertools.cycle(pool), args.max_turns)):
                # s1: stop when eit reaches target. postcomp: stop when compaction fires.
                if args.mode == "s1":
                    eit = client.get_effective_input_tokens(sid)
                    if eit is not None and eit >= args.target_eit:
                        logger.info("Reached target eit=%s >= %d", eit, args.target_eit)
                        break
                try:
                    await asyncio.wait_for(client.send_message(msg, session_id=sid),
                                           timeout=args.turn_timeout)
                    consecutive_errors = 0
                except (asyncio.TimeoutError, Exception) as exc:
                    label = "TIMEOUT" if isinstance(exc, asyncio.TimeoutError) else type(exc).__name__
                    logger.warning("Filler turn %d %s: %s", j + 1, label, str(exc)[:160])
                    cc = client.get_session_compaction_count(sid)
                    if args.mode == "postcomp" and cc > initial_cc:
                        logger.info("compaction_fired (after %s, cc %d→%d)", label, initial_cc, cc)
                        fired = True; break
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        logger.error("Too many consecutive errors. Aborting stage."); return
                    await asyncio.sleep(args.lock_cooldown)
                    continue
                cc = client.get_session_compaction_count(sid)
                if args.mode == "postcomp" and cc > initial_cc:
                    logger.info("compaction_fired after %d turns (cc %d→%d)", j + 1, initial_cc, cc)
                    fired = True; break
                if (j + 1) % 3 == 0:
                    logger.info("  turn %d: eit=%s cc=%d", j + 1,
                                client.get_effective_input_tokens(sid), cc)

            eit = client.get_effective_input_tokens(sid)
            cc = client.get_session_compaction_count(sid)
            jsonl_count = len(client.get_session_jsonl(sid))

            if args.mode == "s1":
                m = save_s1_prequery_checkpoint(
                    state_dir=branch.branch_state_dir, episode_id=args.target_episode,
                    agent_id=branch.agent_id, session_id=sid, depth_label=args.depth_label,
                    effective_input_tokens=eit if eit is not None else 0,
                    jsonl_entry_count=jsonl_count, native_compaction_count=cc,
                    checkpoints_root=CHECKPOINTS_ROOT)
            else:
                if not fired:
                    logger.error("Compaction did not fire. NOT saving postcomp."); return
                m = save_postcomp_checkpoint(
                    state_dir=branch.branch_state_dir, episode_id=args.target_episode,
                    agent_id=branch.agent_id, session_id=sid,
                    compaction_threshold=args.threshold, native_compaction_count=cc,
                    boundary_input_tokens=eit, compaction_verified=(cc > initial_cc),
                    checkpoints_root=CHECKPOINTS_ROOT)
            print(f"\nSTAGE SAVED: {m.checkpoint_id} (episode={m.episode_id}, eit={eit}, cc={cc})")
            print(f"  dir: {m.checkpoint_dir}")
    finally:
        if branch: branch.cleanup()
        if proxy_ctx: proxy_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-episode", required=True)
    p.add_argument("--source-checkpoint", required=True)
    p.add_argument("--target-episode", required=True)
    p.add_argument("--mode", choices=["s1", "postcomp"], required=True)
    p.add_argument("--target-eit", type=int, default=0, help="s1: fill until eit >= this")
    p.add_argument("--depth-label", type=int, default=0, help="s1: depth label for checkpoint name")
    p.add_argument("--threshold", type=int, default=100000, help="postcomp: compaction threshold")
    p.add_argument("--turn-timeout", type=int, default=600)
    p.add_argument("--lock-cooldown", type=float, default=15.0)
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--model", default="volcengine/doubao-seed-2-0-pro-260215")
    asyncio.run(main(p.parse_args()))
