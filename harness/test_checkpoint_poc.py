"""
Proof-of-concept validation for checkpoint infrastructure.

Validates:
  1. S1 invariant enforcement (depth labels, compaction guard, d0 ceiling)
  2. Save s1_prequery_d0 with precise fields + restore + same-session proof
  3. Postcomp invariant enforcement (cc>0, verified=True)
  4. Real compaction integration: drive a session until native compaction fires,
     save a real postcomp checkpoint, restore, prove continuation works

Usage:
    cd GroundingBench
    python -m harness.test_checkpoint_poc [--state-dir PATH] [--episode-id ID]
    python -m harness.test_checkpoint_poc --skip-integration   # skip real compaction test
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
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

from harness.checkpoint import (
    save_s1_prequery_checkpoint,
    save_postcomp_checkpoint,
    restore_checkpoint,
    continue_same_session,
    create_fresh_session,
    load_checkpoint,
    list_checkpoints,
    CheckpointManifest,
    CHECKPOINTS_ROOT,
    _S1_DEPTH_LABELS,
    _S1_D0_EIT_CEILING,
)
from client.openclaw_client import OpenClawClient
from client.usage_proxy import patch_openclaw_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _start_temp_gateway(
    state_dir: Path,
    port: int = 19900,
    context_window: int | None = None,
) -> subprocess.Popen:
    """Start a temporary gateway. If context_window is set, patches all models."""
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "logs").mkdir(parents=True, exist_ok=True)

    main_cfg_path = Path.home() / ".openclaw" / "openclaw.json"
    if not main_cfg_path.exists():
        raise RuntimeError("No ~/.openclaw/openclaw.json found")
    with main_cfg_path.open() as f:
        cfg = json.load(f)
    cfg["gateway"] = {**cfg.get("gateway", {}), "port": port}
    cfg.pop("web", None)
    agents_cfg = cfg.get("agents", {})
    agents_cfg.pop("list", None)
    # Set compaction to safeguard mode with memoryFlush
    defaults = agents_cfg.setdefault("defaults", {})
    defaults["compaction"] = {
        "mode": "safeguard",
        "reserveTokensFloor": 20000,
        "memoryFlush": {"enabled": True},
    }
    cfg["agents"] = agents_cfg

    if context_window:
        patch_openclaw_config(cfg, context_window_override=context_window)

    with (state_dir / "openclaw.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    env = {**os.environ, "OPENCLAW_STATE_DIR": str(state_dir)}
    proc = subprocess.Popen(
        ["openclaw", "gateway", "run", "--port", str(port), "--force"],
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


async def _register_agent(agent_id: str, state_dir: Path) -> str:
    workspace = state_dir / f"workspace-{agent_id}"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "MEMORY.md").write_text("# Memory\n\nNo memories yet.\n")
    (workspace / "memory").mkdir(exist_ok=True)
    default_ws = Path.home() / ".openclaw" / "workspace"
    for fname in ("AGENTS.md", "SOUL.md", "USER.md", "IDENTITY.md", "TOOLS.md", "BOOTSTRAP.md"):
        src = default_ws / fname
        if src.exists():
            shutil.copy2(src, workspace / fname)
    env = {**os.environ, "OPENCLAW_STATE_DIR": str(state_dir)}
    proc = await asyncio.create_subprocess_exec(
        "openclaw", "agents", "add", agent_id, "--workspace", str(workspace),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode().strip() or stdout.decode().strip()
        if "already exists" not in err.lower():
            raise RuntimeError(f"Failed to register agent {agent_id}: {err}")
    return agent_id


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open() if line.strip())


def _scan_for_old_paths(directory: Path, old_prefix: str) -> list[str]:
    hits = []
    for f in directory.rglob("*.json"):
        try:
            if old_prefix in f.read_text():
                hits.append(str(f.relative_to(directory)))
        except OSError:
            pass
    return hits


# ---------------------------------------------------------------------------
# Test 1: S1 invariant enforcement
# ---------------------------------------------------------------------------

async def test_1_s1_invariants() -> None:
    print("\n" + "=" * 70)
    print("TEST 1: S1 invariant enforcement")
    print("=" * 70)

    import tempfile
    d = Path(tempfile.mkdtemp())

    # 1a. Invalid depth_label
    try:
        save_s1_prequery_checkpoint(
            d, "e", "a", "s", depth_label=10000,
            effective_input_tokens=12000, jsonl_entry_count=5,
            native_compaction_count=0,
        )
        assert False, "Should reject invalid depth_label"
    except ValueError as e:
        print(f"  depth_label=10000 rejected: PASS")

    # 1b. Compaction already fired
    try:
        save_s1_prequery_checkpoint(
            d, "e", "a", "s", depth_label=0,
            effective_input_tokens=500, jsonl_entry_count=5,
            native_compaction_count=1,
        )
        assert False, "Should reject cc>0"
    except ValueError as e:
        print(f"  cc=1 (compaction fired) rejected: PASS")

    # 1c. d0 after heavy filler (mislabeled d0)
    try:
        save_s1_prequery_checkpoint(
            d, "e", "a", "s", depth_label=0,
            effective_input_tokens=20000, jsonl_entry_count=30,
            native_compaction_count=0,
        )
        assert False, "Should reject d0 with eit>=16000"
    except ValueError as e:
        print(f"  d0 with eit=20000 rejected: PASS")

    # 1d. d16k with insufficient depth
    try:
        save_s1_prequery_checkpoint(
            d, "e", "a", "s", depth_label=16000,
            effective_input_tokens=12000, jsonl_entry_count=10,
            native_compaction_count=0,
        )
        assert False, "Should reject d16k with eit<16000"
    except ValueError as e:
        print(f"  d16k with eit=12000 rejected: PASS")

    # 1e. Valid d0 (low eit, no compaction)
    # Can't actually save (no real state dir), but verify it gets past validation
    # by catching the FileExistsError/copytree error instead of ValueError
    try:
        save_s1_prequery_checkpoint(
            d, "e", "a", "s", depth_label=0,
            effective_input_tokens=500, jsonl_entry_count=4,
            native_compaction_count=0,
        )
        # If it gets here, the state dir copytree succeeded (unlikely but ok)
    except (FileExistsError, OSError):
        pass  # Expected: fails at copytree, not at validation
    except ValueError:
        assert False, "Valid d0 should not raise ValueError"
    print(f"  valid d0 (eit=500, cc=0) passes validation: PASS")

    shutil.rmtree(d, ignore_errors=True)
    print("  PASS: all 5 S1 invariant checks enforced")


# ---------------------------------------------------------------------------
# Test 2: S1 checkpoint save + same-session proof
# ---------------------------------------------------------------------------

async def test_2_s1_save_and_same_session(
    state_dir: Path,
    agent_id: str,
    session_id: str,
    episode_id: str,
    ckpt_root: Path,
    client: OpenClawClient,
) -> CheckpointManifest:
    print("\n" + "=" * 70)
    print("TEST 2: S1 checkpoint + same-session continuation")
    print("=" * 70)

    eit = client.get_effective_input_tokens(session_id)
    jsonl = client.get_session_jsonl(session_id)
    cc = client.get_session_compaction_count(session_id)
    print(f"  eit={eit}, jsonl_count={len(jsonl)}, cc={cc}")

    manifest = save_s1_prequery_checkpoint(
        state_dir=state_dir, episode_id=episode_id,
        agent_id=agent_id, session_id=session_id,
        depth_label=0, effective_input_tokens=eit if eit is not None else 0,
        jsonl_entry_count=len(jsonl), native_compaction_count=cc,
        checkpoints_root=ckpt_root,
    )
    print(f"  checkpoint: {manifest.checkpoint_id}")

    # Restore + same-session
    orig_jsonl_path = state_dir / "agents" / agent_id / "sessions" / f"{session_id}.jsonl"
    orig_count = _count_jsonl_lines(orig_jsonl_path)

    branch = await restore_checkpoint(
        manifest, branch_id=f"poc_same_{uuid.uuid4().hex[:8]}",
    )

    # Isolation check
    stale = _scan_for_old_paths(branch.branch_state_dir, str(state_dir))
    assert len(stale) == 0, f"Stale paths: {stale}"
    print(f"  isolation: no original paths in branch")

    try:
        result = await continue_same_session(
            branch, message="What files are in your workspace?",
        )
        assert result.session_id == manifest.session_id
        assert result.session_id_matches_checkpoint is True
        assert result.jsonl_count_after > result.jsonl_count_before
        assert result.jsonl_count_before == manifest.jsonl_entry_count
        print(f"  same-session proof: sid match, jsonl {result.jsonl_count_before}→{result.jsonl_count_after}")

        # Original unchanged
        assert _count_jsonl_lines(orig_jsonl_path) == orig_count
        print(f"  original transcript unchanged: {orig_count} lines")
        print("  PASS")
    finally:
        branch.cleanup()

    return manifest


# ---------------------------------------------------------------------------
# Test 3: Postcomp invariant enforcement
# ---------------------------------------------------------------------------

async def test_3_postcomp_invariants() -> None:
    print("\n" + "=" * 70)
    print("TEST 3: Postcomp invariant enforcement")
    print("=" * 70)

    import tempfile
    d = Path(tempfile.mkdtemp())

    # cc=0
    try:
        save_postcomp_checkpoint(
            d, "e", "a", "s", compaction_threshold=50000,
            native_compaction_count=0, boundary_input_tokens=48000,
            compaction_verified=True,
        )
        assert False
    except ValueError:
        print(f"  cc=0 rejected: PASS")

    # verified=False
    try:
        save_postcomp_checkpoint(
            d, "e", "a", "s", compaction_threshold=50000,
            native_compaction_count=1, boundary_input_tokens=48000,
            compaction_verified=False,
        )
        assert False
    except ValueError:
        print(f"  verified=False rejected: PASS")

    shutil.rmtree(d, ignore_errors=True)
    print("  PASS: postcomp invariants enforced")


# ---------------------------------------------------------------------------
# Test 4: Real compaction integration
# ---------------------------------------------------------------------------

async def test_4_real_compaction_integration(
    episode_id: str,
    ckpt_root: Path,
) -> None:
    """Drive a real session until native compaction fires, checkpoint, restore, continue.

    Uses a low contextWindow (25000 + 20000 reserve = 45000 total) so compaction
    fires after just a few filler turns, keeping the test fast.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Real compaction integration")
    print("=" * 70)

    # Set up a gateway with a very low contextWindow to force compaction quickly.
    # contextWindow = 45000 means compaction fires at ~25000 tokens.
    CONTEXT_WINDOW = 45000
    COMPACTION_THRESHOLD = CONTEXT_WINDOW - 20000  # 25000

    state_dir = Path.home() / f".openclaw-compaction-poc-{uuid.uuid4().hex[:8]}"
    proc = _start_temp_gateway(state_dir, port=19910, context_window=CONTEXT_WINDOW)

    try:
        agent_id = f"comp_{uuid.uuid4().hex[:8]}"
        await _register_agent(agent_id, state_dir)
        print(f"  state_dir: {state_dir}")
        print(f"  contextWindow: {CONTEXT_WINDOW}")
        print(f"  compaction_threshold: {COMPACTION_THRESHOLD}")

        async with OpenClawClient(agent_id=agent_id, state_dir=state_dir) as client:
            session_id = await client.start_session(reason="new")
            print(f"  session_id: {session_id}")

            # Plant an artifact for memory persistence verification
            await client.send_message(
                "Please save this to your memory: The artifact ID is AURORA-7-COMPACTION-TEST "
                "and the project codename is DELTA-NINE. Remember these details.",
                session_id=session_id,
            )
            print(f"  artifact planted")

            # Inject filler turns until compaction fires
            from harness.filler_turns import get_filler_turns
            filler_pool = get_filler_turns(n=20, seed=42, category="substantive")

            initial_cc = client.get_session_compaction_count(session_id)
            print(f"  initial compaction_count: {initial_cc}")

            compaction_fired = False
            for i, msg in enumerate(filler_pool):
                await client.send_message(msg, session_id=session_id)
                cc = client.get_session_compaction_count(session_id)
                eit = client.get_effective_input_tokens(session_id)
                print(f"    filler {i+1}: eit={eit}, cc={cc}")
                if cc > initial_cc:
                    compaction_fired = True
                    print(f"  COMPACTION FIRED after {i+1} filler turns (cc: {initial_cc}→{cc})")
                    break

            if not compaction_fired:
                raise RuntimeError(
                    "Native compaction did not fire within 20 filler turns. "
                    "Check contextWindow and model configuration."
                )

            # Record state at compaction boundary
            final_cc = client.get_session_compaction_count(session_id)
            boundary_eit = client.get_effective_input_tokens(session_id)
            jsonl = client.get_session_jsonl(session_id)

            # Verify artifact persistence
            workspace = state_dir / f"workspace-{agent_id}"
            artifact_found = False
            for md_file in list(workspace.rglob("*.md")):
                if "AURORA-7" in md_file.read_text():
                    artifact_found = True
                    break
            print(f"  artifact persisted in memory: {artifact_found}")

            print()
            print("  --- REAL POSTCOMP CHECKPOINT ---")
            print(f"  native_compaction_count: {final_cc}")
            print(f"  boundary_input_tokens: {boundary_eit}")
            print(f"  compaction_verified: {artifact_found}")
            print(f"  jsonl_entry_count: {len(jsonl)}")

            assert final_cc > 0, "compaction_count must be > 0"
            assert artifact_found, "artifact must be persisted for a valid postcomp checkpoint"

            # Save real postcomp checkpoint
            manifest = save_postcomp_checkpoint(
                state_dir=state_dir, episode_id=episode_id,
                agent_id=agent_id, session_id=session_id,
                compaction_threshold=COMPACTION_THRESHOLD,
                native_compaction_count=final_cc,
                boundary_input_tokens=boundary_eit,
                compaction_verified=artifact_found,
                checkpoints_root=ckpt_root,
            )
            print(f"  checkpoint saved: {manifest.checkpoint_id}")

            # Verify manifest round-trip
            loaded = load_checkpoint(episode_id, manifest.checkpoint_id, ckpt_root)
            assert loaded.native_compaction_count == final_cc
            assert loaded.compaction_verified is True
            assert loaded.boundary_input_tokens == boundary_eit
            print(f"  manifest round-trip: OK")

        # --- Restore + same-session continuation (S2 path) ---
        print()
        print("  --- S2 PATH: SAME-SESSION FROM REAL POSTCOMP ---")
        branch_s2 = await restore_checkpoint(
            manifest, branch_id=f"comp_s2_{uuid.uuid4().hex[:8]}",
        )
        stale = _scan_for_old_paths(branch_s2.branch_state_dir, str(state_dir))
        assert len(stale) == 0, f"Stale paths: {stale}"
        print(f"  branch isolation: OK")

        try:
            result = await continue_same_session(
                branch_s2,
                message="What do you know about AURORA-7? Tell me everything.",
            )
            assert result.session_id == session_id
            assert result.jsonl_count_after > result.jsonl_count_before
            print(f"  same-session: sid match, jsonl {result.jsonl_count_before}→{result.jsonl_count_after}")
            print(f"  response (first 120): {result.response_text[:120]}...")
            print(f"  S2 same-session from real postcomp: PASS")
        finally:
            branch_s2.cleanup()

        # --- Restore + fresh-session continuation (S3 path) ---
        print()
        print("  --- S3 PATH: FRESH-SESSION FROM REAL POSTCOMP ---")
        branch_s3 = await restore_checkpoint(
            manifest, branch_id=f"comp_s3_{uuid.uuid4().hex[:8]}",
        )
        stale = _scan_for_old_paths(branch_s3.branch_state_dir, str(state_dir))
        assert len(stale) == 0, f"Stale paths: {stale}"
        print(f"  branch isolation: OK")

        try:
            result = await create_fresh_session(
                branch_s3,
                message="What do you know about AURORA-7? Tell me everything.",
            )
            assert result.session_id != session_id
            assert len(result.jsonl) > 0
            print(f"  fresh session: {result.session_id} (different from original)")
            print(f"  response (first 120): {result.response_text[:120]}...")

            # Verify fresh session lives in branch, not original
            fresh_jsonl = (
                branch_s3.branch_state_dir / "agents" / agent_id
                / "sessions" / f"{result.session_id}.jsonl"
            )
            assert fresh_jsonl.exists(), "Fresh JSONL must be in branch"
            orig_fresh = state_dir / "agents" / agent_id / "sessions" / f"{result.session_id}.jsonl"
            assert not orig_fresh.exists(), "Fresh JSONL must NOT be in original"
            print(f"  fresh JSONL in branch: PASS")
            print(f"  fresh JSONL not in original: PASS")
            print(f"  S3 fresh-session from real postcomp: PASS")
        finally:
            branch_s3.cleanup()

        print()
        print("  PASS: real compaction integration complete")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        if state_dir.exists():
            shutil.rmtree(state_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_poc(args: argparse.Namespace) -> None:
    episode_id = args.episode_id or f"poc_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    ckpt_root = Path(args.checkpoints_root) if args.checkpoints_root else CHECKPOINTS_ROOT

    # Test 1: S1 invariant enforcement (no gateway needed)
    await test_1_s1_invariants()

    # Test 3: Postcomp invariant enforcement (no gateway needed)
    await test_3_postcomp_invariants()

    # Tests 2 + optional 4 need a gateway
    temp_proc = None
    if args.state_dir:
        state_dir = Path(args.state_dir)
    else:
        state_dir = Path.home() / f".openclaw-poc-{uuid.uuid4().hex[:8]}"
        temp_proc = _start_temp_gateway(state_dir, port=19900)

    try:
        agent_id = f"poc_{uuid.uuid4().hex[:8]}"
        await _register_agent(agent_id, state_dir)

        async with OpenClawClient(agent_id=agent_id, state_dir=state_dir) as client:
            session_id = await client.start_session(reason="new")
            await client.send_message(
                "Please remember: Project codename AURORA-7, launch March 15 2025.",
                session_id=session_id,
            )
            # Test 2: S1 save + same-session
            await test_2_s1_save_and_same_session(
                state_dir, agent_id, session_id, episode_id, ckpt_root, client,
            )
    finally:
        if temp_proc:
            temp_proc.terminate()
            try:
                temp_proc.wait(timeout=5)
            except Exception:
                temp_proc.kill()
            if state_dir.exists():
                shutil.rmtree(state_dir, ignore_errors=True)

    # Test 4: Real compaction integration
    if not args.skip_integration:
        await test_4_real_compaction_integration(episode_id, ckpt_root)
    else:
        print("\n  (skipping real compaction integration test)")

    # Summary
    all_ckpts = list_checkpoints(episode_id, ckpt_root)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Episode: {episode_id}")
    print(f"  Checkpoints: {len(all_ckpts)}")
    for m in all_ckpts:
        extras = []
        if m.effective_input_tokens is not None:
            extras.append(f"eit={m.effective_input_tokens}")
        if m.native_compaction_count is not None:
            extras.append(f"cc={m.native_compaction_count}")
        if m.compaction_verified is not None:
            extras.append(f"verified={m.compaction_verified}")
        if m.boundary_input_tokens is not None:
            extras.append(f"b_eit={m.boundary_input_tokens}")
        print(f"    {m.checkpoint_id} [{m.checkpoint_kind}] {', '.join(extras)}")
    print()
    n_tests = 4 if not args.skip_integration else 3
    print(f"  ALL {n_tests} TESTS PASSED")


def main():
    parser = argparse.ArgumentParser(description="Checkpoint POC")
    parser.add_argument("--state-dir", type=str, default=None)
    parser.add_argument("--episode-id", type=str, default=None)
    parser.add_argument("--checkpoints-root", type=str, default=None)
    parser.add_argument(
        "--skip-integration", action="store_true",
        help="Skip the real compaction integration test (test 4)",
    )
    asyncio.run(run_poc(parser.parse_args()))


if __name__ == "__main__":
    main()
