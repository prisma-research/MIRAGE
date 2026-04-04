"""
LEGACY: Early one-shot memory-conditioned personal agent runner.

This was the first prototype runner for static M0/M1/M2/M3 case injection.
It is retained as engineering scaffold but is NOT the canonical T×M study runner.

For the current mainline, see:
    harness/run_personal_memory_tm_study.py  (to be implemented)

Usage (prototype only):
    cd MIRAGE
    python -m harness.run_memory_conditioned_personal --dry-run --max-cases 2
    python -m harness.run_memory_conditioned_personal --run-id smoke_pm_01 --max-cases 1
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from client.openclaw_client import OpenClawClient
from constants import VLM_MODEL
from harness.trial_runner import (
    _setup_trial_agent,
    _teardown_trial_agent,
    _reindex_memory,
)
from harness.personal_memory_protocol import build_task_prompt, parse_response
from harness.personal_memory_io import select_memory_text, write_memory_files
from harness.personal_memory_scorer import score_trial, compute_summary, print_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

GB_ROOT = Path(__file__).parent.parent
BANK_PATH = GB_ROOT / "configs" / "study" / "personal_memory_cases_v1.json"
OUTPUT_ROOT = GB_ROOT / "logs" / "probes" / "personal_memory_v1"
CONDITIONS = ["M0", "M1", "M2", "M3"]


async def run_one_trial(
    case: dict,
    condition: str,
    run_id: str,
    model: str,
) -> dict:
    """Run a single (case, condition) trial.

    Creates a fresh agent, injects memory, sends screenshot + task prompt,
    parses and scores the response, then cleans up.
    """
    trial_id = f"{case['case_id']}_{condition}_{run_id}"
    result = {
        "trial_id": trial_id,
        "case_id": case["case_id"],
        "domain": case.get("domain", "unknown"),
        "condition": condition,
        "model": model,
        "stale_aligned": case.get("stale_aligned_decision"),
        "irrelevant_aligned": case.get("irrelevant_aligned_decision"),
    }

    agent_id = None
    t0 = time.monotonic()

    try:
        # 1. Create isolated agent from baseline snapshot
        agent_id = await _setup_trial_agent(trial_id, model=model)

        async with OpenClawClient(agent_id=agent_id) as client:
            workspace = client.get_workspace_path()

            # 2. Write memory files (daily note → sync to root MEMORY.md)
            memory_text = select_memory_text(case, condition)
            write_memory_files(workspace, memory_text)

            # 3. Reindex memory so memory_search works
            await _reindex_memory(agent_id)

            # 4. Start fresh session
            session_id = await client.start_session()
            logger.info("Session started: %s (agent=%s)", session_id, agent_id)

            # 5. Copy screenshot to workspace (gateway security requirement)
            src_image = GB_ROOT / case["image_path"]
            dst_image = workspace / src_image.name
            shutil.copy2(src_image, dst_image)

            # 6. Send screenshot + task prompt
            prompt = build_task_prompt(case)
            response = await client.send_message(
                prompt, session_id=session_id, image_path=str(dst_image)
            )

            # 7. Extract response text
            response_text = ""
            if isinstance(response, dict):
                # Try common response formats
                content = response.get("content", response.get("message", {}).get("content", ""))
                if isinstance(content, list):
                    response_text = " ".join(
                        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                    )
                elif isinstance(content, str):
                    response_text = content
            elif isinstance(response, str):
                response_text = response

            # If response_text is empty, try reading from session JSONL
            if not response_text.strip():
                jsonl = client.get_session_jsonl(session_id)
                for entry in reversed(jsonl):
                    msg = entry.get("message", {})
                    if msg.get("role") == "assistant":
                        c = msg.get("content", "")
                        if isinstance(c, list):
                            response_text = " ".join(
                                b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
                            )
                        elif isinstance(c, str):
                            response_text = c
                        if response_text.strip():
                            break

            result["response_text"] = response_text

            # 8. Parse and score
            parsed = parse_response(response_text, case["decision_space"])
            scored = score_trial(case, condition, parsed)

            result.update(parsed)
            result.update(scored)

    except Exception as e:
        logger.error("Trial %s failed: %s", trial_id, e)
        result["error"] = str(e)
        result["parse_ok"] = False
        result["correct"] = None

    finally:
        result["duration_s"] = round(time.monotonic() - t0, 1)
        # 9. Cleanup
        if agent_id:
            try:
                await _teardown_trial_agent(agent_id)
            except Exception as e:
                logger.warning("Cleanup failed for %s: %s", agent_id, e)

    return result


async def main_async(args):
    # Load case bank
    bank = json.load(BANK_PATH.open())
    cases = bank["cases"]

    # Filter
    if args.case_ids:
        selected = set(args.case_ids.split(","))
        cases = [c for c in cases if c["case_id"] in selected]
    if args.max_cases:
        cases = cases[:args.max_cases]

    run_id = args.run_id
    run_dir = OUTPUT_ROOT / run_id

    # Analysis-only mode
    if args.analysis_only:
        rf = run_dir / "results.jsonl"
        if not rf.exists():
            print(f"No results found: {rf}")
            return
        results = [json.loads(line) for line in rf.open() if line.strip()]
        summary = compute_summary(results)
        print_summary(summary)
        with (run_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        return

    # Dry run
    if args.dry_run:
        print(f"DRY RUN — {len(cases)} cases x {len(CONDITIONS)} conditions = {len(cases) * len(CONDITIONS)} trials")
        print(f"Model: {args.model}")
        print(f"Output: {run_dir}")
        print(f"\nCases:")
        for c in cases:
            print(f"  {c['case_id']:15s} domain={c['domain']:15s} image={c['image_id']} decisions={c['decision_space']}")
        return

    # Normal run
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id:     {run_id}")
    print(f"output_dir: {run_dir}")
    print(f"model:      {args.model}")
    print(f"cases:      {len(cases)} x {len(CONDITIONS)} conditions = {len(cases) * len(CONDITIONS)} trials")

    rf = run_dir / "results.jsonl"
    all_results = []

    for ci, case in enumerate(cases):
        for condition in CONDITIONS:
            logger.info("[%d/%d] %s x %s", ci * len(CONDITIONS) + CONDITIONS.index(condition) + 1,
                       len(cases) * len(CONDITIONS), case["case_id"], condition)

            result = await run_one_trial(case, condition, run_id, args.model)
            all_results.append(result)

            with rf.open("a") as f:
                f.write(json.dumps(result) + "\n")

            logger.info("  → parse=%s correct=%s decision=%s (gold=%s) %.1fs",
                       result.get("parse_ok"), result.get("correct"),
                       result.get("decision"), result.get("gold"),
                       result.get("duration_s", 0))

    # Summary
    print(f"\nTotal: {len(all_results)}")
    summary = compute_summary(all_results)
    summary["run_id"] = run_id
    summary["model"] = args.model
    summary["n_cases"] = len(cases)
    print_summary(summary)

    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {rf}")
    print(f"Summary: {run_dir / 'summary.json'}")


def main():
    p = argparse.ArgumentParser(description="Memory-conditioned personal agent benchmark")
    p.add_argument("--run-id", default=None,
                   help="Run identifier (default: auto-generated)")
    p.add_argument("--max-cases", type=int, default=None,
                   help="Limit to first N cases")
    p.add_argument("--case-ids", type=str, default=None,
                   help="Comma-separated case IDs to run")
    p.add_argument("--model", default=VLM_MODEL,
                   help=f"Model to use (default: {VLM_MODEL})")
    p.add_argument("--analysis-only", action="store_true",
                   help="Recompute summary from existing results")
    p.add_argument("--dry-run", action="store_true",
                   help="Print trial plan without executing")
    args = p.parse_args()

    if args.run_id is None:
        args.run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
