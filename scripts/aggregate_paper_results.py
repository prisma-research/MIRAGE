"""
aggregate_paper_results.py — Paper-ready table aggregation for GroundingBench.

Aggregates results across multiple experiment run directories into markdown/CSV
tables suitable for inclusion in the MM2026 paper.

Usage:
    python scripts/aggregate_paper_results.py \\
        --backbone-dirs logs/trials_sonnet_main_backbone_0321 \\
                        logs/trials_gpt5_main_backbone_0322 \\
                        logs/trials_gemini_main_backbone_0323 \\
        --depth-dir logs/trials_sonnet_s1_depth_0324 \\
        --mitigation-dir logs/trials_sonnet_mitigation_s3_0325 \\
        --output results/paper_tables/

Produces:
    results/paper_tables/table1_backbone.md + .csv
    results/paper_tables/table2_depth.md + .csv
    results/paper_tables/table3_mitigation.md + .csv

Each input dir MUST contain a valid run_config.json (written by experiment_runner.py).
The script fails fast if run_config.json is missing — do not rely on directory naming.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from results.eval_summary import compute_summary, load_all_trials, is_access_strict, is_access_inclusive


# ---------------------------------------------------------------------------
# Run directory loading
# ---------------------------------------------------------------------------

def load_run_config(run_dir: Path) -> dict:
    """Load run_config.json from run_dir. Raises if missing or invalid."""
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"run_config.json not found in {run_dir}. "
            "This file is required — only pass directories produced by experiment_runner.py."
        )
    with cfg_path.open() as f:
        cfg = json.load(f)
    return cfg


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    """Load run_config.json + all trial dicts from a run directory."""
    cfg = load_run_config(run_dir)
    trials = load_all_trials(run_dir)
    return cfg, trials


# ---------------------------------------------------------------------------
# Summary extraction helpers
# ---------------------------------------------------------------------------

def _get_model_label(cfg: dict) -> str:
    """Return a short model label from run_config."""
    model = cfg.get("model", "unknown")
    # Strip provider prefix (e.g. "shubiaobiao/claude-sonnet-4-6" → "claude-sonnet-4-6")
    return model.split("/")[-1] if "/" in model else model


def _extract_backbone_row(cfg: dict, trials: list[dict]) -> dict:
    """Extract a Table 1 row (backbone comparison) from a run."""
    summary = compute_summary(trials)
    by_sc = summary.get("by_scenario", {})

    row: dict = {
        "model": _get_model_label(cfg),
        "preset": cfg.get("preset", ""),
        "n_trials": summary.get("n_total", 0),
        "R_binary_strict_mean": summary.get("R_binary_strict_mean"),
        "R_binary_inclusive_mean": summary.get("R_binary_inclusive_mean"),
        "I_strict": summary.get("I_strict", ""),
        "F_strict_strict": summary.get("F_strict_strict", 0),
        "F_strict_inclusive": summary.get("F_strict_inclusive", 0),
        "D_refusals": summary.get("D_refusals", 0),
    }

    # Per-scenario R_binary strict
    for sc in ["S1", "S2", "S3"]:
        sc_stats = by_sc.get(sc, {})
        row[f"S{sc[-1]}_R_strict"] = sc_stats.get("R_binary_strict_mean")
        row[f"S{sc[-1]}_R_incl"] = sc_stats.get("R_binary_inclusive_mean")
        row[f"S{sc[-1]}_F_strict"] = sc_stats.get("F_strict_strict", 0)

    return row


def _extract_depth_row(cfg: dict, trials: list[dict], depth: int) -> dict:
    """Extract a Table 2 row (S1 depth study) for a specific history_depth."""
    depth_trials = [
        t for t in trials
        if t.get("scenario") == "S1"
        and (t.get("theta") or {}).get("history_depth") == depth
    ]
    if not depth_trials:
        return {"history_depth": depth, "n": 0}

    summary = compute_summary(depth_trials)
    return {
        "model": _get_model_label(cfg),
        "history_depth": depth,
        "n": len(depth_trials),
        "R_binary_strict_mean": summary.get("R_binary_strict_mean"),
        "F_strict_strict": summary.get("F_strict_strict", 0),
        "I_strict": summary.get("I_strict", ""),
    }


def _extract_mitigation_row(cfg: dict, trials: list[dict], condition: str) -> dict:
    """Extract a Table 3 row (mitigation study) for a specific CF condition."""
    cond_trials = [
        t for t in trials
        if (t.get("theta") or {}).get("citation_force_condition") == condition
    ]
    if not cond_trials:
        return {"condition": condition, "n": 0}

    summary = compute_summary(cond_trials)
    fm = summary.get("failure_modes", {})
    rp = summary.get("R_path") or {}
    return {
        "model": _get_model_label(cfg),
        "condition": condition,
        "n": len(cond_trials),
        "R_binary_strict_mean": summary.get("R_binary_strict_mean"),
        "I_strict": summary.get("I_strict", ""),
        "F_strict_strict": summary.get("F_strict_strict", 0),
        "D_refusals": summary.get("D_refusals", 0),
        "R_tool_count": rp.get("R_tool", 0),
        "R_boot_count": rp.get("R_boot", 0),
        "R_none_count": rp.get("R_none", 0),
        "silent_hgr": fm.get("silent_hgr", 0),
        "wrong_retrieval": fm.get("wrong_retrieval", 0),
        "retrieval_but_unsupported": fm.get("retrieval_but_unsupported", 0),
    }


# ---------------------------------------------------------------------------
# Table writers
# ---------------------------------------------------------------------------

def _write_table(rows: list[dict], columns: list[str], md_path: Path, csv_path: Path, title: str) -> None:
    """Write a list of row dicts to both markdown and CSV."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Markdown
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    md_lines = [f"# {title}", "", header, sep]
    for row in rows:
        cells = [str(row.get(c, "")) if row.get(c) is not None else "—" for c in columns]
        # Format floats
        formatted = []
        for c, cell in zip(columns, cells):
            try:
                v = row.get(c)
                if isinstance(v, float):
                    formatted.append(f"{v:.3f}")
                else:
                    formatted.append(cell)
            except Exception:
                formatted.append(cell)
        md_lines.append("| " + " | ".join(formatted) + " |")
    md_lines.append("")

    md_path.write_text("\n".join(md_lines))
    print(f"  [saved] {md_path}")

    # CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [saved] {csv_path}")


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def build_table1_backbone(backbone_dirs: list[Path], output_dir: Path) -> None:
    """Table 1: per-model backbone comparison across scenarios."""
    rows = []
    for run_dir in backbone_dirs:
        cfg, trials = load_run(run_dir)
        row = _extract_backbone_row(cfg, trials)
        rows.append(row)

    columns = [
        "model", "n_trials",
        "R_binary_strict_mean", "R_binary_inclusive_mean",
        "I_strict", "F_strict_strict", "F_strict_inclusive", "D_refusals",
        "S1_R_strict", "S1_R_incl", "S1_F_strict",
        "S2_R_strict", "S2_R_incl", "S2_F_strict",
        "S3_R_strict", "S3_R_incl", "S3_F_strict",
    ]
    _write_table(
        rows, columns,
        output_dir / "table1_backbone.md",
        output_dir / "table1_backbone.csv",
        "Table 1: Backbone Model Comparison",
    )


def build_table2_depth(depth_dir: Path, output_dir: Path) -> None:
    """Table 2: S1 F_strict × history_depth (0 / 16k / 32k)."""
    cfg, trials = load_run(depth_dir)
    depths = sorted(set(
        (t.get("theta") or {}).get("history_depth", 0)
        for t in trials if t.get("scenario") == "S1"
    ))

    rows = []
    for d in depths:
        row = _extract_depth_row(cfg, trials, d)
        rows.append(row)

    columns = ["model", "history_depth", "n", "R_binary_strict_mean", "F_strict_strict", "I_strict"]
    _write_table(
        rows, columns,
        output_dir / "table2_depth.md",
        output_dir / "table2_depth.csv",
        "Table 2: S1 Depth Study — F_strict × History Depth",
    )


def build_table3_mitigation(mitigation_dir: Path, output_dir: Path) -> None:
    """Table 3: C0 vs C3 × S3 silent_HGR, R_tool, R_boot."""
    cfg, trials = load_run(mitigation_dir)
    conditions = sorted(set(
        (t.get("theta") or {}).get("citation_force_condition", "C0")
        for t in trials
    ))

    rows = []
    for cond in conditions:
        row = _extract_mitigation_row(cfg, trials, cond)
        rows.append(row)

    columns = ["model", "condition", "n", "R_binary_strict_mean",
               "I_strict", "F_strict_strict", "D_refusals",
               "R_tool_count", "R_boot_count", "R_none_count",
               "silent_hgr", "wrong_retrieval", "retrieval_but_unsupported"]
    _write_table(
        rows, columns,
        output_dir / "table3_mitigation.md",
        output_dir / "table3_mitigation.csv",
        "Table 3: Mitigation Study — C0 vs C3",
    )


def build_table_s3_failure_modes(backbone_dirs: list[Path], output_dir: Path) -> None:
    """Table: model × failure mode for S3 trials."""
    from results.eval_summary import compute_failure_modes

    rows = []
    for run_dir in backbone_dirs:
        cfg, trials = load_run(run_dir)
        s3_trials = [t for t in trials if t.get("scenario") == "S3"]
        if not s3_trials:
            continue
        fm = compute_failure_modes(s3_trials)
        row = {"model": _get_model_label(cfg), "n": len(s3_trials)}
        row.update(fm)
        rows.append(row)

    columns = ["model", "n", "silent_hgr", "refusal", "wrong_retrieval",
               "bootstrap_success", "heuristic_context_case", "retrieval_but_unsupported"]
    _write_table(
        rows, columns,
        output_dir / "table_s3_failure_modes.md",
        output_dir / "table_s3_failure_modes.csv",
        "S3 Failure Mode Distribution by Model",
    )


def build_table_by_difficulty(backbone_dirs: list[Path], output_dir: Path,
                              annotation_dir: Path | None = None) -> None:
    """Table: model × scenario × difficulty.

    Difficulty is read from annotation CSV referenced in run_config.json,
    or from a manually specified annotation_dir.
    """
    rows = []
    for run_dir in backbone_dirs:
        cfg, trials = load_run(run_dir)
        model = _get_model_label(cfg)

        # Load annotations
        ann_file = cfg.get("annotation_file")
        annotations: dict[str, dict] = {}
        if ann_file:
            import csv as csv_mod
            ann_path = Path(ann_file)
            if not ann_path.is_absolute():
                ann_path = Path(__file__).parent.parent / ann_path
            if ann_path.exists():
                with ann_path.open(newline="", encoding="utf-8") as f:
                    for r in csv_mod.DictReader(f):
                        annotations[r["artifact_id"]] = r

        if not annotations:
            continue

        for sc in ["S1", "S2", "S3"]:
            for diff in ["easy", "medium", "hard"]:
                subset = [
                    t for t in trials
                    if t.get("scenario") == sc
                    and annotations.get(t.get("artifact_id", ""), {}).get("difficulty") == diff
                ]
                if not subset:
                    continue
                summary = compute_summary(subset)
                rows.append({
                    "model": model,
                    "scenario": sc,
                    "difficulty": diff,
                    "n": len(subset),
                    "R_binary_strict_mean": summary.get("R_binary_strict_mean"),
                    "I_strict": summary.get("I_strict", ""),
                    "F_strict_strict": summary.get("F_strict_strict", 0),
                    "D_refusals": summary.get("D_refusals", 0),
                })

    columns = ["model", "scenario", "difficulty", "n",
               "R_binary_strict_mean", "I_strict", "F_strict_strict", "D_refusals"]
    _write_table(
        rows, columns,
        output_dir / "table_by_difficulty.md",
        output_dir / "table_by_difficulty.csv",
        "Results by Model × Scenario × Difficulty",
    )


def build_table_oog(backbone_dirs: list[Path], output_dir: Path) -> None:
    """Table: model × scenario outcome optimism gap using non-refusal proxy."""
    rows = []
    for run_dir in backbone_dirs:
        cfg, trials = load_run(run_dir)
        model = _get_model_label(cfg)
        summary = compute_summary(trials)

        def _row(label: str, stats: dict) -> dict:
            n = stats.get("n", 0) or 0
            refusals = stats.get("D_refusals", 0) or 0
            f_strict = stats.get("F_strict_strict", 0) or 0
            t_resp_rate = ((n - refusals) / n) if n else None
            f_rate = (f_strict / n) if n else None
            oog = (t_resp_rate - f_rate) if (t_resp_rate is not None and f_rate is not None) else None
            return {
                "model": model,
                "scenario": label,
                "n": n,
                "T_resp_rate": t_resp_rate,
                "F_strict_rate": f_rate,
                "OOG": oog,
            }

        rows.append(_row("overall", {
            "n": summary.get("n_total", 0),
            "D_refusals": summary.get("D_refusals", 0),
            "F_strict_strict": summary.get("F_strict_strict", 0),
        }))
        for sc in ["S1", "S2", "S3"]:
            stats = summary.get("by_scenario", {}).get(sc)
            if stats:
                rows.append(_row(sc, stats))

    columns = ["model", "scenario", "n", "T_resp_rate", "F_strict_rate", "OOG"]
    _write_table(
        rows, columns,
        output_dir / "table_oog.md",
        output_dir / "table_oog.csv",
        "Outcome Optimism Gap by Model × Scenario",
    )


def build_table_modality(summary_paths: list[tuple[str, Path]], output_dir: Path) -> None:
    """Table: modality × model breakdown from summary JSONs."""
    rows = []
    for model_label, summary_path in summary_paths:
        with summary_path.open() as f:
            data = json.load(f)
        for modality in ["screenshot", "chart_image"]:
            mod_data = data.get("by_modality", {}).get(modality)
            if not mod_data:
                continue
            rows.append({
                "modality": modality.replace("_", " ").title(),
                "model": model_label,
                "n": mod_data.get("n", 0),
                "I_strict": mod_data.get("I_strict", ""),
                "F_strict_strict": mod_data.get("F_strict_strict", 0),
                "D_refusals": mod_data.get("D_refusals", 0),
            })

    columns = ["modality", "model", "n", "I_strict", "F_strict_strict", "D_refusals"]
    _write_table(
        rows, columns,
        output_dir / "table_modality.md",
        output_dir / "table_modality.csv",
        "Results by Modality × Model",
    )


def build_table_paired(csv_paths: list[tuple[str, Path]], output_dir: Path) -> None:
    """Table: paired artifact comparison across two models by scenario.

    Joins per-trial CSVs on (artifact_id, scenario) and classifies into:
    both succeed / model-A only / model-B only / both fail (based on F_strict).
    """
    import csv as csv_mod

    # Load per-trial data
    model_trials: dict[str, dict[tuple[str, str], int]] = {}
    model_labels = []
    for label, csv_path in csv_paths:
        model_labels.append(label)
        trials: dict[tuple[str, str], int] = {}
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv_mod.DictReader(f):
                key = (row["artifact_id"], row["scenario"])
                trials[key] = int(row["F_strict"])
        model_trials[label] = trials

    if len(model_labels) != 2:
        print("  [warn] Paired table requires exactly 2 models", file=sys.stderr)
        return

    m_a, m_b = model_labels
    trials_a, trials_b = model_trials[m_a], model_trials[m_b]
    all_keys = set(trials_a.keys()) | set(trials_b.keys())

    rows = []
    for sc in ["S1", "S2", "S3"]:
        sc_keys = [k for k in all_keys if k[1] == sc]
        both_ok = sum(1 for k in sc_keys if trials_a.get(k, 0) == 1 and trials_b.get(k, 0) == 1)
        a_only = sum(1 for k in sc_keys if trials_a.get(k, 0) == 1 and trials_b.get(k, 0) == 0)
        b_only = sum(1 for k in sc_keys if trials_a.get(k, 0) == 0 and trials_b.get(k, 0) == 1)
        both_fail = sum(1 for k in sc_keys if trials_a.get(k, 0) == 0 and trials_b.get(k, 0) == 0)
        rows.append({
            "scenario": sc,
            "both_succeed": both_ok,
            f"{m_a}_only": a_only,
            f"{m_b}_only": b_only,
            "both_fail": both_fail,
            "total": len(sc_keys),
        })

    # Overall row
    all_both_ok = sum(r["both_succeed"] for r in rows)
    all_a_only = sum(r[f"{m_a}_only"] for r in rows)
    all_b_only = sum(r[f"{m_b}_only"] for r in rows)
    all_both_fail = sum(r["both_fail"] for r in rows)
    rows.append({
        "scenario": "Overall",
        "both_succeed": all_both_ok,
        f"{m_a}_only": all_a_only,
        f"{m_b}_only": all_b_only,
        "both_fail": all_both_fail,
        "total": all_both_ok + all_a_only + all_b_only + all_both_fail,
    })

    columns = ["scenario", "both_succeed", f"{m_a}_only", f"{m_b}_only", "both_fail", "total"]
    _write_table(
        rows, columns,
        output_dir / "table_paired.md",
        output_dir / "table_paired.csv",
        f"Paired Artifact Comparison: {m_a} vs {m_b}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate GroundingBench paper tables")
    parser.add_argument(
        "--backbone-dirs", nargs="*", type=Path, default=[],
        metavar="DIR",
        help="Run directories for backbone model comparison (one per model)",
    )
    parser.add_argument(
        "--depth-dir", type=Path, default=None,
        help="Run directory for S1 depth study",
    )
    parser.add_argument(
        "--mitigation-dir", type=Path, default=None,
        help="Run directory for S3 mitigation study",
    )
    parser.add_argument(
        "--summary-pairs", nargs="*", default=[],
        metavar="LABEL:PATH",
        help="Model summaries as label:path pairs for modality table (e.g. gpt-5:results/.../summary.json)",
    )
    parser.add_argument(
        "--trial-csv-pairs", nargs="*", default=[],
        metavar="LABEL:PATH",
        help="Per-trial CSVs as label:path pairs for paired table (e.g. gpt-5:results/paper_main_gpt5_0322.csv)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/paper_tables"),
        help="Output directory for table files (default: results/paper_tables/)",
    )
    args = parser.parse_args()

    if not args.backbone_dirs and args.depth_dir is None and args.mitigation_dir is None \
       and not args.summary_pairs and not args.trial_csv_pairs:
        parser.error("Provide at least one of --backbone-dirs, --depth-dir, --mitigation-dir, --summary-pairs, or --trial-csv-pairs")

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backbone_dirs:
        print(f"\nBuilding Table 1 (backbone) from {len(args.backbone_dirs)} runs…")
        try:
            build_table1_backbone(args.backbone_dirs, output_dir)
        except Exception as exc:
            print(f"  [error] Table 1 failed: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"\nBuilding S3 failure mode table…")
        try:
            build_table_s3_failure_modes(args.backbone_dirs, output_dir)
        except Exception as exc:
            print(f"  [warn] S3 failure mode table failed: {exc}", file=sys.stderr)

        print(f"\nBuilding by-difficulty table…")
        try:
            build_table_by_difficulty(args.backbone_dirs, output_dir)
        except Exception as exc:
            print(f"  [warn] By-difficulty table failed: {exc}", file=sys.stderr)

        print(f"\nBuilding OOG table…")
        try:
            build_table_oog(args.backbone_dirs, output_dir)
        except Exception as exc:
            print(f"  [warn] OOG table failed: {exc}", file=sys.stderr)

    if args.depth_dir:
        print(f"\nBuilding Table 2 (depth) from {args.depth_dir}…")
        try:
            build_table2_depth(args.depth_dir, output_dir)
        except Exception as exc:
            print(f"  [error] Table 2 failed: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.mitigation_dir:
        print(f"\nBuilding Table 3 (mitigation) from {args.mitigation_dir}…")
        try:
            build_table3_mitigation(args.mitigation_dir, output_dir)
        except Exception as exc:
            print(f"  [error] Table 3 failed: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.summary_pairs:
        pairs = []
        for sp in args.summary_pairs:
            label, path_str = sp.split(":", 1)
            pairs.append((label, Path(path_str)))
        print(f"\nBuilding modality table from {len(pairs)} summaries…")
        try:
            build_table_modality(pairs, output_dir)
        except Exception as exc:
            print(f"  [warn] Modality table failed: {exc}", file=sys.stderr)

    if args.trial_csv_pairs:
        pairs = []
        for tp in args.trial_csv_pairs:
            label, path_str = tp.split(":", 1)
            pairs.append((label, Path(path_str)))
        print(f"\nBuilding paired table from {len(pairs)} CSVs…")
        try:
            build_table_paired(pairs, output_dir)
        except Exception as exc:
            print(f"  [warn] Paired table failed: {exc}", file=sys.stderr)

    print(f"\nDone. Tables written to {output_dir}/")


if __name__ == "__main__":
    main()
