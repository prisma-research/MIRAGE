"""
Pilot Run — 100k main experiment with episode-aware split checkpoint family.

Pre-compaction states (baseline trunk):
  d0       — pilot_v3_100k / s1_prequery_d0
  d50k     — pilot_v3_100k / s1_prequery_d50k
  d80k     — pilot_v3_100k / s1_prequery_d80k

Post-compaction states (prewrite intervention):
  S2@100k  — pilot_v3_100k_prewrite / postcomp_100k (same-session)
  S3@100k  — pilot_v3_100k_prewrite / postcomp_100k (fresh-session)

Usage:
    cd GroundingBench
    python -m harness.run_pilot --probes-only --max-questions 6   # 90-probe validation
    python -m harness.run_pilot --probes-only                     # full 1500 probes
    python -m harness.run_pilot --analysis-only                   # re-analyze
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from harness.episode_checkpoint_runner import run_episode
from harness.branch_probe_runner import run_probe_batch
from harness.probe_analysis import compute_summary, print_summary
from harness.probe_result import load_probe_results
from harness.checkpoint import list_checkpoints, CHECKPOINTS_ROOT

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

GB_ROOT = Path(__file__).parent.parent
QUESTION_BANK_PATH = GB_ROOT / "configs" / "study" / "question_bank_constrained_100k.json"

# Output label for this mixed-condition run
OUTPUT_EPISODE_ID = "pilot_v3_100k_mixed"
MODEL = "shubiaobiao/gpt-5"

# Episode-aware experiment matrix.
# Each state specifies its source episode_id and checkpoint_id.
DEFAULT_POSTCOMP_EPISODE = "pilot_v3_100k_prewrite_generic"

def _build_probe_matrix(postcomp_episode: str = DEFAULT_POSTCOMP_EPISODE) -> list[tuple]:
    return [
        # (source_episode,    checkpoint_id,       continuation,    experiment)
        ("pilot_v3_100k",     "s1_prequery_d0",    "same_session",  "depth"),
        ("pilot_v3_100k",     "s1_prequery_d50k",  "same_session",  "depth"),
        ("pilot_v3_100k",     "s1_prequery_d80k",  "same_session",  "depth"),
        (postcomp_episode,    "postcomp_100k",     "same_session",  "compaction"),
        (postcomp_episode,    "postcomp_100k",     "fresh_session", "reset"),
    ]

PROBE_MATRIX = _build_probe_matrix()  # backward compat


async def phase_validate_bank() -> None:
    print("\n" + "=" * 60)
    print("PHASE: VALIDATE QUESTION BANK")
    print("=" * 60)

    if not QUESTION_BANK_PATH.exists():
        raise FileNotFoundError(str(QUESTION_BANK_PATH))

    from pipeline.study_catalog import validate_question_bank
    result = validate_question_bank(QUESTION_BANK_PATH)
    print(f"  Bank: {result['n_questions']} questions")
    print(f"  By family: {result['by_family']}")
    n_err = len(result['errors'])
    print(f"  Schema: {'OK' if n_err == 0 else f'{n_err} errors'}")
    if result["errors"]:
        raise ValueError(f"Bank has {len(result['errors'])} errors")


async def phase_check_checkpoints() -> None:
    """Verify all required checkpoints exist across their respective episodes."""
    print("\n" + "=" * 60)
    print("PHASE: VERIFY CHECKPOINTS")
    print("=" * 60)

    missing = []
    for source_ep, ckpt_id, _, _ in PROBE_MATRIX:
        existing = list_checkpoints(source_ep)
        existing_ids = {m.checkpoint_id for m in existing}
        if ckpt_id not in existing_ids:
            missing.append(f"{source_ep}/{ckpt_id}")
        else:
            m = next(m for m in existing if m.checkpoint_id == ckpt_id)
            print(f"  {source_ep}/{ckpt_id}: eit={m.effective_input_tokens}, cc={m.native_compaction_count}")

    if missing:
        print(f"  MISSING: {missing}")
        raise RuntimeError(f"Missing checkpoints: {missing}")
    print(f"  All {len(PROBE_MATRIX)} states verified")


async def phase_probes(max_questions: int | None = None) -> None:
    print("\n" + "=" * 60)
    print("PHASE: RUN PROBES")
    print("=" * 60)

    # Start usage proxy — checkpoints reference a dead proxy port from build time
    from client.usage_proxy import UsageProxy
    from constants import EXPERIMENT_MODEL_REGISTRY
    needs_proxy = any(s.get("needsUsageProxy") for s in EXPERIMENT_MODEL_REGISTRY.values())
    proxy_ctx = UsageProxy(patch_global=False) if needs_proxy else None
    proxy_port = None
    if proxy_ctx:
        proxy_ctx.__enter__()
        proxy_port = proxy_ctx.port
        print(f"  Proxy started on port {proxy_port}")

    probes_dir = GB_ROOT / "logs" / "probes" / OUTPUT_EPISODE_ID

    bank = json.load(QUESTION_BANK_PATH.open())
    questions = bank["questions"]

    if max_questions:
        from collections import defaultdict
        limited = []
        for fam in ["target_present", "wrong_source", "unanswerable"]:
            fam_qs = [q for q in questions if q["family"] == fam]
            by_art: dict[str, list[dict]] = defaultdict(list)
            for q in fam_qs:
                by_art[q["target_artifact_id"]].append(q)
            art_ids = sorted(by_art.keys())
            art_idx = {aid: 0 for aid in art_ids}
            fam_count = 0
            while fam_count < max_questions and any(
                art_idx[a] < len(by_art[a]) for a in art_ids
            ):
                for aid in art_ids:
                    if fam_count >= max_questions:
                        break
                    idx = art_idx[aid]
                    if idx < len(by_art[aid]):
                        limited.append(by_art[aid][idx])
                        art_idx[aid] += 1
                        fam_count += 1
        questions = limited
        arts_covered = set(q["target_artifact_id"] for q in questions)
        print(f"  Stratified: {len(questions)} questions ({max_questions}/family, "
              f"{len(arts_covered)} artifacts)")

    temp_bank_path = probes_dir / "_active_bank.json"
    temp_bank_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_bank_path.open("w") as f:
        json.dump({"questions": questions}, f, indent=2)

    all_results = []
    try:
        for source_ep, checkpoint_id, continuation, experiment in PROBE_MATRIX:
            label = f"{experiment}/{checkpoint_id}/{continuation} (from {source_ep})"
            print(f"\n  --- {label} ({len(questions)} probes) ---")

            results = await run_probe_batch(
                episode_id=source_ep,
                checkpoint_id=checkpoint_id,
                question_bank_path=temp_bank_path,
                experiment=experiment,
                continuation=continuation,
                model=MODEL,
                output_dir=probes_dir,
                proxy_port=proxy_port,
            )
            all_results.extend(results)
    finally:
        if proxy_ctx:
            proxy_ctx.__exit__(None, None, None)
            print("  Proxy stopped")

    print(f"\n  Total probes: {len(all_results)}")


def phase_analysis() -> None:
    print("\n" + "=" * 60)
    print("PHASE: ANALYSIS")
    print("=" * 60)

    probes_dir = GB_ROOT / "logs" / "probes" / OUTPUT_EPISODE_ID
    results = load_probe_results(probes_dir)
    if not results:
        print(f"  No results in {probes_dir}")
        return

    summary = compute_summary(results)
    print_summary(summary)

    summary_path = probes_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to {summary_path}")


async def main_async(args: argparse.Namespace) -> None:
    if args.analysis_only:
        phase_analysis()
        return

    await phase_validate_bank()
    await phase_check_checkpoints()

    if not args.build_only:
        await phase_probes(max_questions=args.max_questions)
        phase_analysis()


def main():
    parser = argparse.ArgumentParser(description="Run 100k mixed-condition experiment")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--probes-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
