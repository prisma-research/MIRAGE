"""
State-conditioned pilot runner.

Runs full/locate/verify/extract_in_state probes across 5 states,
using the split checkpoint family.

Usage:
    cd MIRAGE
    python -m harness.run_state_conditioned_pilot --max-probes 10   # smoke test
    python -m harness.run_state_conditioned_pilot --max-bq 6        # small validation
    python -m harness.run_state_conditioned_pilot                    # full run
    python -m harness.run_state_conditioned_pilot --analysis-only
"""

from __future__ import annotations
import argparse, asyncio, json, logging, sys, time, uuid, datetime
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from client.usage_proxy import UsageProxy
from constants import EXPERIMENT_MODEL_REGISTRY
from harness.checkpoint import load_checkpoint, restore_checkpoint, CHECKPOINTS_ROOT
from harness.checkpoint import continue_same_session, create_fresh_session
from harness.state_conditioned_scorer import score_probe
from harness.branch_probe_runner import classify_r_path, _extract_tool_trace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

GB_ROOT = Path(__file__).parent.parent
BANK_PATH = GB_ROOT / "configs" / "study" / "state_conditioned_bank.json"
OUTPUT_DIR = GB_ROOT / "logs" / "probes" / "state_conditioned_v1"
MODEL = "shubiaobiao/gpt-5"

# Episode-aware 5-state matrix
DEFAULT_POSTCOMP_EPISODE = "pilot_v3_100k_prewrite_generic"

def _build_states(postcomp_episode: str = DEFAULT_POSTCOMP_EPISODE) -> list[tuple]:
    return [
        ("pilot_v3_100k",      "s1_prequery_d0",    "same_session",  "d0"),
        ("pilot_v3_100k",      "s1_prequery_d50k",  "same_session",  "d50k"),
        ("pilot_v3_100k",      "s1_prequery_d80k",  "same_session",  "d80k"),
        (postcomp_episode,     "postcomp_100k",     "same_session",  "S2"),
        (postcomp_episode,     "postcomp_100k",     "fresh_session", "S3"),
    ]

STATES = _build_states()  # backward compat for direct imports


async def run_one_probe(
    episode_id: str, checkpoint_id: str, continuation: str,
    probe: dict, state_label: str, proxy_port: int | None,
) -> dict:
    """Run one probe, return result dict."""
    manifest = load_checkpoint(episode_id, checkpoint_id)
    pid = f"sc_{state_label}_{probe['probe_id']}_{uuid.uuid4().hex[:4]}"

    if proxy_port is not None:
        branch = await restore_checkpoint(manifest, branch_id=f"br_{pid}",
                                          port_start=19800, start_gateway=False)
        import json as _json, subprocess as _sp, os as _os
        from client.usage_proxy import patch_openclaw_config as _patch
        cfg_path = branch.branch_state_dir / "openclaw.json"
        with cfg_path.open() as f:
            cfg = _json.load(f)
        _patch(cfg, proxy_port=proxy_port, compaction_model=None)
        with cfg_path.open("w") as f:
            _json.dump(cfg, f, indent=2)
        env = {**_os.environ, "OPENCLAW_STATE_DIR": str(branch.branch_state_dir)}
        (branch.branch_state_dir / "logs").mkdir(parents=True, exist_ok=True)
        gw = _sp.Popen(
            ["openclaw", "gateway", "run", "--port", str(branch.gateway_port), "--force"],
            stdout=(branch.branch_state_dir / "logs" / "gateway.log").open("a"),
            stderr=(branch.branch_state_dir / "logs" / "gateway.err.log").open("a"),
            env=env,
        )
        branch.gateway_proc = gw
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            try:
                r = _sp.run(["openclaw", "gateway", "status"],
                            capture_output=True, timeout=15, env=env)
                if r.returncode == 0:
                    break
            except _sp.TimeoutExpired:
                pass
            await asyncio.sleep(0.5)
    else:
        branch = await restore_checkpoint(manifest, branch_id=f"br_{pid}", port_start=19800)

    try:
        t0 = time.monotonic()
        if continuation == "same_session":
            cr = await continue_same_session(branch, message=probe["prompt"])
        else:
            cr = await create_fresh_session(branch, message=probe["prompt"])
        duration = time.monotonic() - t0

        split = cr.jsonl_count_before if cr.jsonl_count_before is not None else 0
        tool_trace = _extract_tool_trace(cr.jsonl, split)
        r_path = classify_r_path(tool_trace, continuation, cr.jsonl)

        scores = score_probe(probe, cr.response_text)

        return {
            "probe_id": pid,
            "original_probe_id": probe["probe_id"],
            "task_type": probe["task_type"],
            "base_question_id": probe["base_question_id"],
            "state": state_label,
            "checkpoint_id": checkpoint_id,
            "continuation": continuation,
            "response_text": cr.response_text,
            "R_path": r_path,
            "duration": round(duration, 1),
            **scores,
        }
    finally:
        branch.cleanup()


def analyze(results: list[dict]) -> dict:
    """Compute State × Task summary."""
    summary = {"n_results": len(results)}

    # State × Task accuracy
    grid: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r.get("parse_success"):
            continue
        state = r["state"]
        task = r["task_type"]
        # Get the primary metric for this task
        if task == "full":
            val = r.get("full_answerable_correct")
        elif task == "locate":
            val = r.get("locate_correct")
        elif task == "verify":
            val = r.get("verify_correct")
        elif task == "extract_in_state":
            val = r.get("extract_correct")
        else:
            continue
        if val is not None:
            grid[state][task].append(val)

    # Build State × Task table
    state_order = ["d0", "d50k", "d80k", "S2", "S3"]
    task_order = ["full", "locate", "verify", "extract_in_state"]
    table = {}
    for s in state_order:
        row = {}
        for t in task_order:
            vals = grid[s][t]
            row[t] = round(sum(vals) / len(vals), 3) if vals else None
        table[s] = row
    summary["state_task_table"] = table

    # Penalties
    penalties = {}
    pairs = [("d0", "d50k"), ("d50k", "d80k"), ("d80k", "S2"), ("S2", "S3")]
    for s1, s2 in pairs:
        for t in task_order:
            v1 = table.get(s1, {}).get(t)
            v2 = table.get(s2, {}).get(t)
            if v1 is not None and v2 is not None:
                penalties[f"{s1}_to_{s2}_{t}"] = round(v1 - v2, 3)
    summary["penalties"] = penalties

    # R_path distribution by state
    rpath_dist: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for r in results:
        if r.get("R_path"):
            rpath_dist[r["state"]][r["R_path"]] += 1
    summary["rpath_by_state"] = {s: dict(d) for s, d in rpath_dist.items()}

    # Parse failure rate
    total = len(results)
    parsed = sum(1 for r in results if r.get("parse_success"))
    summary["parse_failure_rate"] = round(1 - parsed / max(total, 1), 3)

    return summary


def print_analysis(summary: dict):
    print(f"\n{'='*70}")
    print("STATE-CONDITIONED ANALYSIS")
    print(f"{'='*70}")
    print(f"Total results: {summary['n_results']}, parse_failure: {summary['parse_failure_rate']}")

    print(f"\n--- State × Task Accuracy ---")
    table = summary["state_task_table"]
    tasks = ["full", "locate", "verify", "extract_in_state"]
    header = f"{'State':<8}" + "".join(f"{t:<20}" for t in tasks)
    print(header)
    for s in ["d0", "d50k", "d80k", "S2", "S3"]:
        row = table.get(s, {})
        vals = "".join(f"{str(row.get(t, '-')):<20}" for t in tasks)
        print(f"{s:<8}{vals}")

    print(f"\n--- Penalties (higher = worse degradation) ---")
    for name, val in summary.get("penalties", {}).items():
        print(f"  {name}: {val:+.3f}")

    print(f"\n--- R_path by State ---")
    for s in ["d0", "d50k", "d80k", "S2", "S3"]:
        dist = summary.get("rpath_by_state", {}).get(s, {})
        print(f"  {s}: {dict(dist)}")


async def main_async(args):
    if args.analysis_only:
        results = []
        for f in sorted(OUTPUT_DIR.glob("*.jsonl")):
            for line in f.open():
                if line.strip():
                    results.append(json.loads(line))
        if not results:
            print(f"No results in {OUTPUT_DIR}")
            return
        s = analyze(results)
        print_analysis(s)
        with (OUTPUT_DIR / "summary.json").open("w") as f:
            json.dump(s, f, indent=2)
        return

    # Load bank
    bank = json.load(BANK_PATH.open())
    probes = bank["derived_probes"]
    bqs = bank["base_questions"]

    # Subset selection
    if args.max_bq:
        # Pick diverse base questions
        bq_ids = set()
        sup = [bq for bq in bqs if bq["has_support"]]
        unsup = [bq for bq in bqs if not bq["has_support"]]
        # Take from each: chart supported, screenshot supported, unsupported
        for bq in sup:
            if len(bq_ids) < args.max_bq - 2:
                bq_ids.add(bq["base_question_id"])
        for bq in unsup:
            if len(bq_ids) < args.max_bq:
                bq_ids.add(bq["base_question_id"])
        probes = [p for p in probes if p["base_question_id"] in bq_ids]
        print(f"Selected {len(bq_ids)} base questions → {len(probes)} probes")

    if args.max_probes:
        probes = probes[:args.max_probes]
        print(f"Limited to {len(probes)} probes")

    # Start proxy
    needs_proxy = any(s.get("needsUsageProxy") for s in EXPERIMENT_MODEL_REGISTRY.values())
    proxy = UsageProxy(patch_global=False) if needs_proxy else None
    proxy_port = None
    if proxy:
        proxy.__enter__()
        proxy_port = proxy.port
        print(f"Proxy on port {proxy_port}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / "results.jsonl"

    all_results = []
    try:
        for ep, ckpt, cont, state_label in STATES:
            print(f"\n--- State: {state_label} ({ckpt} from {ep}, {cont}) ---")
            for i, probe in enumerate(probes):
                logger.info("  [%d/%d] %s %s", i + 1, len(probes), probe["task_type"], probe["probe_id"])
                try:
                    r = await run_one_probe(ep, ckpt, cont, probe, state_label, proxy_port)
                    all_results.append(r)
                    with results_file.open("a") as f:
                        f.write(json.dumps(r) + "\n")
                    ok = r.get("parse_success", False)
                    logger.info("    → parse=%s, R=%s", ok, r.get("R_path"))
                except Exception as exc:
                    logger.error("    → FAILED: %s", exc)
    finally:
        if proxy:
            proxy.__exit__(None, None, None)

    print(f"\nTotal results: {len(all_results)}")
    s = analyze(all_results)
    print_analysis(s)
    with (OUTPUT_DIR / "summary.json").open("w") as f:
        json.dump(s, f, indent=2)
    print(f"Summary saved to {OUTPUT_DIR / 'summary.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-bq", type=int, default=None, help="Limit base questions")
    p.add_argument("--max-probes", type=int, default=None, help="Limit total probes")
    p.add_argument("--analysis-only", action="store_true")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
