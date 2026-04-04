"""
T × M Personal Memory Study — Main Runner

Canonical runner for the personal memory quality study.
Iterates over cases × M-conditions (M0/M1/M2/M3), creating a fresh
agent per trial, injecting the appropriate MEMORY.md, and scoring
the structured decision response.

Mitigation pilots (--with-mitigation):
  M2-mit: recency-aware overwrite on M2 memory (T2 cases only = SER subset)
  M3-mit: tagged distractor pruning on M3 memory (T3 out-of-scope only = OPR subset)

Usage:
    cd MIRAGE
    python -m harness.run_personal_memory_tm_study --dry-run --max-cases 2
    python -m harness.run_personal_memory_tm_study --run-id smoke_tm_01 --max-cases 1 --allow-text-only
    python -m harness.run_personal_memory_tm_study --run-id full_01
    python -m harness.run_personal_memory_tm_study --run-id mit_01 --with-mitigation
    python -m harness.run_personal_memory_tm_study --analysis-only --run-id full_01

Case bank: configs/study/personal_memory_tm_cases_v1.json
Output:    logs/probes/personal_memory_tm/<run_id>/
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
from harness.personal_memory_io import select_memory_text, write_memory_md
from harness.personal_memory_scorer import score_trial, compute_summary, print_summary
from pipeline.personal_memory.mitigation_policies import (
    apply_recency_overwrite, apply_relevance_filter,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

GB_ROOT = Path(__file__).parent.parent
BANK_PATH = GB_ROOT / "configs" / "study" / "personal_memory_tm_cases_v1.json"
OUTPUT_ROOT = GB_ROOT / "logs" / "probes" / "personal_memory_tm"
CONDITIONS = ["M0", "M1", "M2", "M3"]

# Mitigation pilot conditions and their targeting rules.
# These MUST match the scorer's diagnostic subsets:
#   M2-mit targets T2 cases only (= SER diagnostic subset)
#   M3-mit targets T3 out-of-scope cases only (= OPR diagnostic subset)
MITIGATION_CONDITIONS = {
    "M2-mit": {"base": "M2"},
    "M3-mit": {"base": "M3"},
}


def _should_run_mitigation(condition: str, case: dict) -> bool:
    """Check if a mitigation condition is relevant to this case.

    Targeting aligns with scorer diagnostic subsets:
      M2-mit: T2 only (SER diagnostic set)
      M3-mit: T3 out-of-scope only (OPR diagnostic set)
    """
    if condition not in MITIGATION_CONDITIONS:
        return True
    if condition == "M2-mit":
        return case.get("task_type") == "T2"
    if condition == "M3-mit":
        return case.get("task_type") == "T3" and case.get("t3_scope") == "out_of_scope"
    return False


def _apply_mitigation(condition: str, case: dict) -> str:
    """Apply mitigation transform to base memory using structured_state.

    Uses case["structured_state"] (generation internals) instead of
    parsing rendered markdown — avoids brittle regex extraction.
    """
    spec = MITIGATION_CONDITIONS[condition]
    base_memory = select_memory_text(case, spec["base"])
    ss = case.get("structured_state", {})

    if condition == "M2-mit":
        # Recency overwrite: replace stale preferences with current ones
        current_prefs = ss.get("current_preferences", {})
        if current_prefs:
            return apply_recency_overwrite(base_memory, current_prefs)
        return base_memory

    elif condition == "M3-mit":
        # Tagged distractor pruning: removes "general"-tagged distractors,
        # keeps domain-adjacent distractors. Does NOT filter background traits.
        domain = case.get("domain", "")
        distractor_items = ss.get("distractor_items", [])
        return apply_relevance_filter(base_memory, domain, distractor_items)

    return base_memory


async def run_one_trial(case: dict, condition: str, run_id: str, model: str,
                        allow_text_only: bool = False) -> dict:
    """Run a single (case, condition) trial with fresh agent."""
    case_id = case["case_id"]
    trial_id = f"{case_id}_{condition}_{run_id}"
    result = {
        "trial_id": trial_id,
        "case_id": case_id,
        "domain": case.get("domain", "unknown"),
        "task_type": case.get("task_type", "?"),
        "condition": condition,
        "model": model,
        "stale_aligned": case.get("stale_aligned_action"),
        "noisy_aligned": case.get("noisy_aligned_action"),
        "t3_scope": case.get("t3_scope"),
    }

    agent_id = None
    t0 = time.monotonic()

    try:
        # 1. Create isolated agent
        agent_id = await _setup_trial_agent(trial_id, model=model)

        async with OpenClawClient(agent_id=agent_id) as client:
            workspace = client.get_workspace_path()

            # 2. Write MEMORY.md
            if condition in MITIGATION_CONDITIONS:
                memory_text = _apply_mitigation(condition, case)
            else:
                memory_text = select_memory_text(case, condition)
            write_memory_md(workspace, memory_text)

            # 3. Reindex memory
            await _reindex_memory(agent_id)

            # 4. Start fresh session
            session_id = await client.start_session()
            logger.info("Session started: %s (agent=%s, condition=%s)",
                       session_id, agent_id, condition)

            # 5. Resolve image, then build prompt accordingly
            src_image = None
            for img_field in ("rendered_image_path", "image_path"):
                if case.get(img_field):
                    candidate = GB_ROOT / case[img_field]
                    if candidate.exists():
                        src_image = candidate
                        break

            if src_image:
                # Multimodal path: plain prompt, image sent separately
                prompt = build_task_prompt(case, text_only=False)
                dst_image = workspace / src_image.name
                shutil.copy2(src_image, dst_image)
                response = await client.send_message(
                    prompt, session_id=session_id, image_path=str(dst_image))
            elif allow_text_only:
                # Text-only fallback: append page_spec as textual observation
                prompt = build_task_prompt(case, text_only=True)
                logger.warning("No image for %s — text-only with page_spec", case_id)
                response = await client.send_message(prompt, session_id=session_id)
            else:
                raise FileNotFoundError(
                    f"No image for case {case_id}. Use --allow-text-only for debug.")

            # 7. Extract response text
            response_text = _extract_text(response, client, session_id)
            result["response_text"] = response_text

            # 8. Parse and score
            parsed = parse_response(response_text, case["decision_space"])
            scored = score_trial(case, condition, parsed)
            result.update(parsed)
            result.update(scored)

    except Exception as e:
        logger.error("Trial %s INFRA FAILURE: %s", trial_id, e)
        result["error"] = str(e)
        result["failure_type"] = "infra"
        result["parse_ok"] = False
        result["correct"] = None
    finally:
        result["duration_s"] = round(time.monotonic() - t0, 1)
        if "failure_type" not in result:
            result["failure_type"] = None
        if agent_id:
            try:
                await _teardown_trial_agent(agent_id)
            except Exception as e:
                logger.warning("Cleanup failed for %s: %s", agent_id, e)

    return result


def _extract_text(response, client, session_id) -> str:
    """Extract text from response dict or fall back to JSONL."""
    text = ""
    if isinstance(response, dict):
        content = response.get("content", response.get("message", {}).get("content", ""))
        if isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text")
        elif isinstance(content, str):
            text = content
    elif isinstance(response, str):
        text = response

    if not text.strip():
        jsonl = client.get_session_jsonl(session_id)
        for entry in reversed(jsonl):
            msg = entry.get("message", {})
            if msg.get("role") == "assistant":
                c = msg.get("content", "")
                if isinstance(c, list):
                    text = " ".join(
                        b.get("text", "") for b in c
                        if isinstance(b, dict) and b.get("type") == "text")
                elif isinstance(c, str):
                    text = c
                if text.strip():
                    break
    return text


async def main_async(args):
    bank = json.load(BANK_PATH.open())
    cases = bank["cases"]

    if args.case_ids:
        selected = set(args.case_ids.split(","))
        cases = [c for c in cases if c["case_id"] in selected]
    if args.task_types:
        selected = set(args.task_types.split(","))
        cases = [c for c in cases if c.get("task_type") in selected]
    if args.max_cases:
        cases = cases[:args.max_cases]

    run_id = args.run_id
    run_dir = OUTPUT_ROOT / run_id

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

    # Build trial plan: (case, condition) pairs
    trial_plan = []
    for case in cases:
        for cond in CONDITIONS:
            trial_plan.append((case, cond))
        if args.with_mitigation:
            for mit_cond in MITIGATION_CONDITIONS:
                if _should_run_mitigation(mit_cond, case):
                    trial_plan.append((case, mit_cond))

    if args.dry_run:
        cond_set = sorted({c for _, c in trial_plan})
        print(f"DRY RUN — {len(trial_plan)} trials ({len(cases)} cases)")
        print(f"Conditions: {cond_set}")
        print(f"Model: {args.model}")
        print(f"Output: {run_dir}")
        for c in cases:
            case_conds = [cond for cc, cond in trial_plan if cc["case_id"] == c["case_id"]]
            print(f"  {c['case_id']:25s} T={c.get('task_type','?'):3s} "
                  f"domain={c.get('domain','?'):10s} conds={case_conds}")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id:     {run_id}")
    print(f"output_dir: {run_dir}")
    print(f"model:      {args.model}")
    print(f"trials:     {len(trial_plan)}")

    rf = run_dir / "results.jsonl"
    all_results = []

    for ti, (case, condition) in enumerate(trial_plan):
        logger.info("[%d/%d] %s × %s", ti + 1, len(trial_plan),
                   case["case_id"], condition)

        result = await run_one_trial(case, condition, run_id, args.model,
                                    allow_text_only=args.allow_text_only)

        if result.get("failure_type") == "infra":
            logger.error("INFRA FAILURE: %s", result.get("error"))
            if not args.allow_infra_failures:
                print(f"\nABORTED: infra failure on {result['trial_id']}. "
                      f"Use --allow-infra-failures to continue.")
                return

        all_results.append(result)
        with rf.open("a") as f:
            f.write(json.dumps(result) + "\n")

        logger.info("  → %sparse=%s correct=%s decision=%s gold=%s %.1fs",
                   "[INFRA] " if result.get("failure_type") == "infra" else "",
                   result.get("parse_ok"), result.get("correct"),
                   result.get("decision"), result.get("gold"),
                   result.get("duration_s", 0))

    print(f"\nTotal: {len(all_results)}")
    summary = compute_summary(all_results)
    summary["run_id"] = run_id
    summary["model"] = args.model
    print_summary(summary)

    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults: {rf}")
    print(f"Summary: {run_dir / 'summary.json'}")


def main():
    p = argparse.ArgumentParser(description="T × M Personal Memory Study Runner")
    p.add_argument("--run-id", default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--case-ids", type=str, default=None)
    p.add_argument("--task-types", type=str, default=None)
    p.add_argument("--model", default=VLM_MODEL)
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-text-only", action="store_true",
                   help="Run without images, using page_spec text fallback (debug only).")
    p.add_argument("--allow-infra-failures", action="store_true",
                   help="Continue past infra failures (recorded separately).")
    p.add_argument("--with-mitigation", action="store_true",
                   help="Add targeted mitigation pilots: M2-mit (T2 = SER subset), "
                        "M3-mit (T3 out-of-scope = OPR subset).")
    args = p.parse_args()

    if args.run_id is None:
        args.run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
