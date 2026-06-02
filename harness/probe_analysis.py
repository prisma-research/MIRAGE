"""
Probe Analysis — compute summary metrics from saved probe results.

PRIMARY ANALYSIS UNIT: paraphrase group × checkpoint × experiment.
Raw probe-level stats are kept for debugging but do NOT drive headline metrics.

Aggregation levels:
  1. Cell-level:      (paraphrase_group_id, checkpoint_id, experiment) — majority-vote
  2. Experiment-level: aggregate cells by experiment
  3. Checkpoint-level: aggregate cells by checkpoint_id
  4. Artifact-level:   aggregate cells by target_artifact_id
  5. Confusion-group:  aggregate cells by confusion_group
  6. Penalties:        paired comparisons between checkpoints

Usage:
    cd GroundingBench
    python -m harness.probe_analysis --probes-dir logs/probes/pilot_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.probe_result import load_probe_results, ProbeResult


# ---------------------------------------------------------------------------
# Cell: the primary analysis unit
# ---------------------------------------------------------------------------

def _build_cells(results: list[ProbeResult]) -> list[dict]:
    """Aggregate raw probes into cells keyed by (pg_id, checkpoint_id, experiment).

    Within each cell, compute majority-vote for binary metrics.
    Returns list of cell dicts with aggregated metrics.
    """
    buckets: dict[tuple, list[ProbeResult]] = defaultdict(list)
    for r in results:
        key = (r.paraphrase_group_id or r.question_id, r.checkpoint_id, r.experiment)
        buckets[key].append(r)

    cells = []
    for (pg_id, ckpt, exp), probes in buckets.items():
        cell = {
            "paraphrase_group_id": pg_id,
            "checkpoint_id": ckpt,
            "experiment": exp,
            "n_probes": len(probes),
            # Inherit from first probe (stable within a cell)
            "target_artifact_id": probes[0].target_artifact_id,
            "query_type": probes[0].query_type,
            "visual_type": probes[0].visual_type,
            "confusion_group": probes[0].confusion_group,
            "continuation": probes[0].continuation,
        }
        # Store gold/parsed support for conditional metric computation
        cell["gold_support"] = probes[0].gold_support
        # Majority-vote parsed_support: most common non-None value
        parsed_vals = [r.parsed_support for r in probes if r.parsed_support]
        cell["parsed_support"] = max(set(parsed_vals), key=parsed_vals.count) if parsed_vals else None
        # Majority-vote for binary metrics across paraphrase variants
        # Constrained-output metrics (100k experiment)
        cell["support_correct"] = _majority(probes, "support_correct")
        cell["answer_correct"] = _majority(probes, "answer_correct")
        cell["source_correct"] = _majority(probes, "source_correct")
        cell["parse_success"] = _majority(probes, "parse_success")
        # Legacy free-form metrics
        cell["target_grounded_correct"] = _majority(probes, "target_grounded_correct")
        cell["wrong_source"] = _majority(probes, "wrong_source")
        cell["abstention"] = _majority(probes, "abstention")
        cell["abstention_correct"] = _majority(probes, "abstention_correct")
        cell["R_binary"] = _majority(probes, "R_binary")
        # Most common R_path
        paths = [r.R_path for r in probes if r.R_path]
        cell["R_path"] = max(set(paths), key=paths.count) if paths else None
        cells.append(cell)

    return cells


def _majority(probes: list[ProbeResult], field: str) -> int | None:
    """Majority vote for a binary field across probes.

    Returns:
      1    — strict majority of non-None values are 1
      0    — strict majority of non-None values are 0
      None — all None, or exact tie (no clear majority)
    """
    vals = [getattr(r, field) for r in probes if getattr(r, field) is not None]
    if not vals:
        return None
    ones = sum(vals)
    if ones > len(vals) / 2:
        return 1
    elif ones < len(vals) / 2:
        return 0
    else:
        return None  # exact tie → ambiguous


# ---------------------------------------------------------------------------
# Rates from cells
# ---------------------------------------------------------------------------

def _cell_rate(cells: list[dict], field: str) -> float | None:
    vals = [c[field] for c in cells if c.get(field) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def _conditional_metrics(cells: list[dict]) -> dict:
    """Compute conditional metrics split by gold_support.

    Returns:
      support_accuracy: over all cells
      answer_accuracy_given_target: answer_correct for gold_support=TARGET only
      source_accuracy_given_other: source_correct for gold_support=OTHER_ARTIFACT only
      not_present_accuracy: support_correct for gold_support=NOT_PRESENT only
      wrong_source_intrusion_rate: parsed_support==TARGET when gold!=TARGET
      hallucination_rate: answer_correct==0 when gold_support=NOT_PRESENT (model gave non-NONE answer)
      parse_failure_rate: parse_success==0 or None
    """
    target_cells = [c for c in cells if c.get("gold_support") == "TARGET"]
    other_cells = [c for c in cells if c.get("gold_support") == "OTHER_ARTIFACT"]
    absent_cells = [c for c in cells if c.get("gold_support") == "NOT_PRESENT"]

    m: dict = {}
    m["support_accuracy"] = _cell_rate(cells, "support_correct")
    m["answer_accuracy_given_target"] = _cell_rate(target_cells, "answer_correct") if target_cells else None
    m["source_accuracy_given_other"] = _cell_rate(other_cells, "source_correct") if other_cells else None
    m["not_present_accuracy"] = _cell_rate(absent_cells, "support_correct") if absent_cells else None

    # Diagnostics: wrong-source intrusion = model said TARGET when gold != TARGET
    # Uses actual parsed_support, not support_correct as proxy.
    non_target = [c for c in cells if c.get("gold_support") != "TARGET"
                  and c.get("parsed_support") is not None]
    if non_target:
        intrusions = [c for c in non_target if c["parsed_support"] == "TARGET"]
        m["wrong_source_intrusion_rate"] = round(len(intrusions) / len(non_target), 3)
    else:
        m["wrong_source_intrusion_rate"] = None

    if absent_cells:
        halluc = [c for c in absent_cells if c.get("answer_correct") is not None and c["answer_correct"] == 0]
        m["hallucination_rate"] = round(len(halluc) / len(absent_cells), 3)
    else:
        m["hallucination_rate"] = None

    parsed = [c for c in cells if c.get("parse_success") is not None]
    if parsed:
        failures = [c for c in parsed if c["parse_success"] == 0]
        m["parse_failure_rate"] = round(len(failures) / len(parsed), 3)
    else:
        m["parse_failure_rate"] = None

    return m


def _cell_path_dist(cells: list[dict]) -> dict[str, int]:
    d: dict[str, int] = defaultdict(int)
    for c in cells:
        if c.get("R_path"):
            d[c["R_path"]] += 1
    return dict(d)


# ---------------------------------------------------------------------------
# Main summary
# ---------------------------------------------------------------------------

def compute_summary(results: list[ProbeResult]) -> dict:
    """Compute summary metrics. Primary unit = cells (majority-voted paraphrase groups)."""
    if not results:
        return {"error": "no results"}

    cells = _build_cells(results)

    summary: dict = {
        "n_probes": len(results),
        "n_cells": len(cells),
        "analysis_unit": "cell = (paraphrase_group × checkpoint × experiment), majority-voted",
    }

    # --- By experiment (headline metrics) ---
    by_exp: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        by_exp[c["experiment"]].append(c)

    summary["by_experiment"] = {}
    for exp, group in by_exp.items():
        entry = {
            "n_cells": len(group),
            "n_probes": sum(c["n_probes"] for c in group),
        }
        entry.update(_conditional_metrics(group))
        summary["by_experiment"][exp] = entry

    # --- By checkpoint (headline metrics) ---
    by_ckpt: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        by_ckpt[c["checkpoint_id"]].append(c)

    summary["by_checkpoint"] = {}
    for ckpt, group in by_ckpt.items():
        entry = {
            "n_cells": len(group),
            "n_probes": sum(c["n_probes"] for c in group),
            "retrieval_paths": _cell_path_dist(group),
        }
        entry.update(_conditional_metrics(group))
        summary["by_checkpoint"][ckpt] = entry

    # --- Penalties (cell-level) ---
    summary["penalties"] = _compute_penalties(by_ckpt, cells)

    # --- By artifact ---
    by_art: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        if c["target_artifact_id"]:
            by_art[c["target_artifact_id"]].append(c)

    summary["by_artifact"] = {}
    for aid, group in by_art.items():
        pgs = set(c["paraphrase_group_id"] for c in group)
        summary["by_artifact"][aid] = {
            "n_cells": len(group),
            "n_paraphrase_groups": len(pgs),
            "support_accuracy": _cell_rate(group, "support_correct"),
            "answer_accuracy": _cell_rate(group, "answer_correct"),
        }

    # --- By confusion group ---
    by_cg: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        if c["confusion_group"]:
            by_cg[c["confusion_group"]].append(c)

    summary["by_confusion_group"] = {}
    for cg, group in by_cg.items():
        pgs = set(c["paraphrase_group_id"] for c in group)
        ws = [c for c in group if c["query_type"] == "wrong_source"]
        summary["by_confusion_group"][cg] = {
            "n_cells": len(group),
            "n_paraphrase_groups": len(pgs),
            "support_accuracy": _cell_rate(group, "support_correct"),
            "wrong_source_rate": _cell_rate(ws, "wrong_source") if ws else None,
            "n_wrong_source_cells": len(ws),
        }

    # --- Raw probe-level (for debugging) ---
    summary["raw_probe_level"] = {
        "n_probes": len(results),
        "retrieval_path_distribution": _raw_path_dist(results),
    }

    return summary


def _raw_path_dist(results: list[ProbeResult]) -> dict[str, int]:
    d: dict[str, int] = defaultdict(int)
    for r in results:
        if r.R_path:
            d[r.R_path] += 1
    return dict(d)


# ---------------------------------------------------------------------------
# Penalties (cell-level)
# ---------------------------------------------------------------------------

def _compute_penalties(
    by_ckpt: dict[str, list[dict]],
    all_cells: list[dict],
) -> dict:
    """Compute paired comparison penalties.

    Supports both 50k pilot and 100k main experiment checkpoint families.
    Reports whichever comparisons have data.
    """
    penalties = {}

    def _ckpt_rate(ckpt_id: str, field: str = "support_correct") -> float | None:
        group = by_ckpt.get(ckpt_id, [])
        return _cell_rate(group, field)

    d0 = _ckpt_rate("s1_prequery_d0")

    # --- 100k main experiment: d0 → d50k → d80k → S2@100k → S3@100k ---
    d50k = _ckpt_rate("s1_prequery_d50k")
    d80k = _ckpt_rate("s1_prequery_d80k")

    if d0 is not None and d50k is not None:
        penalties["aging_d0_vs_d50k"] = round(d0 - d50k, 3)
    if d50k is not None and d80k is not None:
        penalties["aging_d50k_vs_d80k"] = round(d50k - d80k, 3)
    if d0 is not None and d80k is not None:
        penalties["aging_d0_vs_d80k"] = round(d0 - d80k, 3)

    # Compaction: d80k → S2@100k
    s2_100k_cells = [c for c in all_cells
                     if c["checkpoint_id"] == "postcomp_100k" and c["experiment"] == "compaction"]
    s2_100k_rate = _cell_rate(s2_100k_cells, "support_correct")

    if d80k is not None and s2_100k_rate is not None:
        penalties["compaction_d80k_vs_S2_100k"] = round(d80k - s2_100k_rate, 3)

    # Reset: S2@100k → S3@100k
    s3_100k_cells = [c for c in all_cells
                     if c["checkpoint_id"] == "postcomp_100k" and c["experiment"] == "reset"]
    s3_100k_rate = _cell_rate(s3_100k_cells, "support_correct")

    if s2_100k_rate is not None and s3_100k_rate is not None:
        penalties["reset_S2_vs_S3_100k"] = round(s2_100k_rate - s3_100k_rate, 3)

    # --- 50k pilot (legacy, reported if data present) ---
    d40k = _ckpt_rate("s1_prequery_d40k")
    if d0 is not None and d40k is not None:
        penalties["aging_d0_vs_d40k"] = round(d0 - d40k, 3)

    s2_50k_cells = [c for c in all_cells
                    if c["checkpoint_id"] == "postcomp_50k" and c["experiment"] == "compaction"]
    s2_50k_rate = _cell_rate(s2_50k_cells, "target_grounded_correct")
    if d40k is not None and s2_50k_rate is not None:
        penalties["compaction_d40k_vs_S2_50k"] = round(d40k - s2_50k_rate, 3)

    s3_50k_cells = [c for c in all_cells
                    if c["checkpoint_id"] == "postcomp_50k" and c["experiment"] == "reset"]
    s3_50k_rate = _cell_rate(s3_50k_cells, "target_grounded_correct")
    if s2_50k_rate is not None and s3_50k_rate is not None:
        penalties["reset_S2_vs_S3_50k"] = round(s2_50k_rate - s3_50k_rate, 3)

    return penalties


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def print_summary(summary: dict) -> None:
    print(f"\n{'='*60}")
    print("PROBE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total probes: {summary['n_probes']}, Analysis cells: {summary['n_cells']}")
    print(f"Unit: {summary.get('analysis_unit', 'cell')}")

    print(f"\n--- By Experiment (cell-level) ---")
    for exp, m in summary.get("by_experiment", {}).items():
        print(f"  {exp} ({m['n_cells']} cells):")
        print(f"    support_accuracy:          {m.get('support_accuracy')}")
        print(f"    answer_accuracy|TARGET:    {m.get('answer_accuracy_given_target')}")
        print(f"    source_accuracy|OTHER:     {m.get('source_accuracy_given_other')}")
        print(f"    not_present_accuracy:      {m.get('not_present_accuracy')}")
        if m.get('parse_failure_rate') is not None:
            print(f"    parse_failure_rate:        {m['parse_failure_rate']}")

    print(f"\n--- By Checkpoint (cell-level) ---")
    for ckpt, m in summary.get("by_checkpoint", {}).items():
        print(f"  {ckpt} ({m['n_cells']} cells):")
        print(f"    support_accuracy:          {m.get('support_accuracy')}")
        print(f"    answer_accuracy|TARGET:    {m.get('answer_accuracy_given_target')}")
        print(f"    not_present_accuracy:      {m.get('not_present_accuracy')}")
        if m.get('retrieval_paths'):
            print(f"    retrieval: {m['retrieval_paths']}")

    print(f"\n--- Penalties (cell-level) ---")
    for name, val in summary.get("penalties", {}).items():
        print(f"  {name}: {val:+.3f}")

    print(f"\n--- By Artifact ---")
    for aid, m in summary.get("by_artifact", {}).items():
        print(f"  {aid} ({m['n_cells']} cells, {m['n_paraphrase_groups']} pg): "
              f"support={m.get('support_accuracy', m.get('target_grounded_correct'))}")

    print(f"\n--- By Confusion Group ---")
    for cg, m in summary.get("by_confusion_group", {}).items():
        print(f"  {cg} ({m['n_cells']} cells, {m['n_paraphrase_groups']} pg): "
              f"support={m.get('support_accuracy', m.get('target_grounded_correct'))}, "
              f"ws_cells={m.get('n_wrong_source_cells', 0)}")

    print(f"\n--- Scoring Protocol ---")
    print(f"  Constrained: SUPPORT/SOURCE/ANSWER parsed and scored deterministically")
    print(f"  Primary: support_accuracy (SUPPORT enum match)")
    print(f"  Secondary: answer_accuracy (value match with normalization)")
    print(f"  Secondary: source_accuracy (artifact_id match)")


def main():
    parser = argparse.ArgumentParser(description="Analyze probe results")
    parser.add_argument("--probes-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    results = load_probe_results(Path(args.probes_dir))
    if not results:
        print(f"No probe results found in {args.probes_dir}")
        return
    summary = compute_summary(results)
    print_summary(summary)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
