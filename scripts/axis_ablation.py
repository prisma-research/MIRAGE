"""
axis_ablation.py — Axis ablation analysis for the MIRAGE paper.

Shows what each scoring axis (R_path, I_strict, S_hat) captures that the
others don't, using the 600 main-paper trials.

Outputs:
    results/paper_tables/table_axis_ablation.csv   (machine-readable)
    Console output with key numbers

Usage:
    python scripts/axis_ablation.py

    # Or specify custom trial directories:
    python scripts/axis_ablation.py \
        --trial-dirs logs/trials_paper_main_gpt5_0322 \
                     logs/trials_paper_main_qwen3vl32b_0322
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from results.eval_summary import load_all_trials, is_access_strict

# ---------------------------------------------------------------------------
# Default trial directories (600 main-paper trials)
# ---------------------------------------------------------------------------
DEFAULT_TRIAL_DIRS = [
    ROOT / "logs" / "trials_paper_main_gpt5_0322",
    ROOT / "logs" / "trials_paper_main_qwen3vl32b_0322",
]

SCENARIOS = ["S1", "S2", "S3"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_run_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return json.load(f)


def _model_label(cfg: dict) -> str:
    model = cfg.get("model", "unknown")
    return model.split("/")[-1] if "/" in model else model


# ---------------------------------------------------------------------------
# Axis predicates (applied to individual trial dicts)
# ---------------------------------------------------------------------------

def _t_response(t: dict) -> bool:
    """Non-refusal: T_response == 1."""
    return (t.get("T_response") or 0) == 1


def _r_binary(t: dict) -> bool:
    """Legitimate access via strict R_binary (R_context | R_boot | R_tool)."""
    return is_access_strict(t.get("R_path"))


def _i_strict(t: dict) -> bool:
    """Correct artifact identification."""
    return (t.get("I_strict") or 0) == 1


def _s_hat_1(t: dict) -> bool:
    """Full grounding: S_hat == 1.0 (numeric)."""
    s = t.get("S_hat")
    return isinstance(s, (int, float)) and s == 1.0


# ---------------------------------------------------------------------------
# Analysis 1: Progressive filtering table
# ---------------------------------------------------------------------------

def compute_progressive_filter(trials: list[dict], model: str) -> list[dict]:
    """Return rows for the progressive filtering table, broken down by scenario."""
    rows = []

    for scenario in [None] + SCENARIOS:
        if scenario is None:
            subset = trials
            sc_label = "All"
        else:
            subset = [t for t in trials if t.get("scenario") == scenario]
            sc_label = scenario

        n_total = len(subset)
        # Level 1: T_response = 1
        l1 = [t for t in subset if _t_response(t)]
        # Level 2: + R_binary = 1
        l2 = [t for t in l1 if _r_binary(t)]
        # Level 3: + I_strict = 1
        l3 = [t for t in l2 if _i_strict(t)]
        # Level 4: + S_hat = 1.0 (F_strict)
        l4 = [t for t in l3 if _s_hat_1(t)]

        rows.append({
            "model": model,
            "scenario": sc_label,
            "n_total": n_total,
            "pass_T_response": len(l1),
            "pass_T+R": len(l2),
            "pass_T+R+I": len(l3),
            "pass_T+R+I+S (F_strict)": len(l4),
            "rate_T_response": f"{len(l1)/n_total:.3f}" if n_total else "---",
            "rate_T+R": f"{len(l2)/n_total:.3f}" if n_total else "---",
            "rate_T+R+I": f"{len(l3)/n_total:.3f}" if n_total else "---",
            "rate_F_strict": f"{len(l4)/n_total:.3f}" if n_total else "---",
        })

    return rows


# ---------------------------------------------------------------------------
# Analysis 2: Unique failure modes per axis
# ---------------------------------------------------------------------------

def compute_unique_failures(trials: list[dict], model: str) -> list[dict]:
    """Compute trials uniquely caught by each axis."""
    rows = []

    for scenario in [None] + SCENARIOS:
        if scenario is None:
            subset = trials
            sc_label = "All"
        else:
            subset = [t for t in trials if t.get("scenario") == scenario]
            sc_label = scenario

        n = len(subset)

        # Axis 1 catches: T_response=1 but R_binary=0
        # (answered without legitimate access)
        ax1 = [t for t in subset if _t_response(t) and not _r_binary(t)]

        # Axis 2 catches: T_response=1 AND R_binary=1 but I_strict=0
        # (retrieved something but wrong artifact)
        ax2 = [t for t in subset if _t_response(t) and _r_binary(t) and not _i_strict(t)]

        # Axis 3 catches: T_response=1 AND R_binary=1 AND I_strict=1 but S_hat!=1.0
        # (looks correct but not actually grounded)
        ax3 = [t for t in subset
               if _t_response(t) and _r_binary(t) and _i_strict(t) and not _s_hat_1(t)]

        # Also count refusals (D=1 / T_response=0) for completeness
        refusals = [t for t in subset if not _t_response(t)]

        # Full success
        f_strict = [t for t in subset
                    if _t_response(t) and _r_binary(t) and _i_strict(t) and _s_hat_1(t)]

        rows.append({
            "model": model,
            "scenario": sc_label,
            "n": n,
            "refusal (T=0)": len(refusals),
            "Axis1_catches (T=1,R=0)": len(ax1),
            "Axis2_catches (T=1,R=1,I=0)": len(ax2),
            "Axis3_catches (T=1,R=1,I=1,S!=1)": len(ax3),
            "F_strict (all pass)": len(f_strict),
            # Sanity check: these should sum to n
            "check_sum": len(refusals) + len(ax1) + len(ax2) + len(ax3) + len(f_strict),
        })

    return rows


# ---------------------------------------------------------------------------
# Analysis 3: Naive accuracy comparison
# ---------------------------------------------------------------------------

def compute_naive_accuracy(trials: list[dict], model: str) -> list[dict]:
    """Compare success rates under progressively stricter definitions."""
    rows = []

    for scenario in [None] + SCENARIOS:
        if scenario is None:
            subset = trials
            sc_label = "All"
        else:
            subset = [t for t in trials if t.get("scenario") == scenario]
            sc_label = scenario

        n = len(subset)
        if n == 0:
            continue

        # "Naive" = just non-refusal
        naive = sum(1 for t in subset if _t_response(t))
        # "Naive + access" = non-refusal AND legitimate retrieval
        naive_access = sum(1 for t in subset if _t_response(t) and _r_binary(t))
        # "Naive + access + identification"
        naive_access_id = sum(
            1 for t in subset if _t_response(t) and _r_binary(t) and _i_strict(t))
        # Full F_strict
        f_strict = sum(
            1 for t in subset
            if _t_response(t) and _r_binary(t) and _i_strict(t) and _s_hat_1(t))

        rows.append({
            "model": model,
            "scenario": sc_label,
            "n": n,
            "T_response_rate": f"{naive/n:.3f}",
            "T+R_rate": f"{naive_access/n:.3f}",
            "T+R+I_rate": f"{naive_access_id/n:.3f}",
            "F_strict_rate": f"{f_strict/n:.3f}",
            "OOG_T_vs_F": f"{(naive - f_strict)/n:.3f}",
            "OOG_T+R_vs_F": f"{(naive_access - f_strict)/n:.3f}",
        })

    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_section(title: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    # Compute column widths
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)

    # Header
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-+-".join("-" * widths[c] for c in cols))

    # Rows
    for r in rows:
        line = " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols)
        print(line)
    print()


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build union of all keys, preserving insertion order
    seen: dict[str, None] = {}
    for r in rows:
        for k in r:
            if k not in seen:
                seen[k] = None
    cols = list(seen.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Axis ablation analysis for MIRAGE paper")
    parser.add_argument(
        "--trial-dirs", nargs="*", type=Path, default=None,
        help="Run directories with trial JSONs (default: both paper_main dirs)")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "paper_tables",
        help="Output directory (default: results/paper_tables/)")
    args = parser.parse_args()

    trial_dirs = args.trial_dirs or DEFAULT_TRIAL_DIRS
    output_dir = args.output

    # -----------------------------------------------------------------------
    # Load trials per model
    # -----------------------------------------------------------------------
    all_progressive_rows: list[dict] = []
    all_failure_rows: list[dict] = []
    all_naive_rows: list[dict] = []

    for run_dir in trial_dirs:
        if not run_dir.exists():
            print(f"[error] Directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)

        cfg = _load_run_config(run_dir)
        model = _model_label(cfg)
        trials = load_all_trials(run_dir, dedupe=True)
        print(f"Loaded {len(trials)} trials for {model} from {run_dir.name}")

        all_progressive_rows.extend(compute_progressive_filter(trials, model))
        all_failure_rows.extend(compute_unique_failures(trials, model))
        all_naive_rows.extend(compute_naive_accuracy(trials, model))

    # -----------------------------------------------------------------------
    # Console output
    # -----------------------------------------------------------------------
    _print_section(
        "PART 1: Progressive Filtering (how many trials survive each axis)",
        all_progressive_rows,
    )

    _print_section(
        "PART 2: Unique Failure Modes Caught by Each Axis",
        all_failure_rows,
    )

    _print_section(
        "PART 3: Naive Accuracy Comparison (OOG = Outcome Optimism Gap)",
        all_naive_rows,
    )

    # -----------------------------------------------------------------------
    # Verify partition sanity
    # -----------------------------------------------------------------------
    for r in all_failure_rows:
        if r["check_sum"] != r["n"]:
            print(f"[WARN] Partition check failed for {r['model']} / {r['scenario']}: "
                  f"sum={r['check_sum']} != n={r['n']}", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Write combined CSV
    # -----------------------------------------------------------------------
    # Build a single CSV with a "section" column for easy import into LaTeX/plotting
    combined_rows: list[dict] = []
    for r in all_progressive_rows:
        combined_rows.append({"section": "progressive_filter", **r})
    for r in all_failure_rows:
        combined_rows.append({"section": "unique_failures", **r})
    for r in all_naive_rows:
        combined_rows.append({"section": "naive_accuracy", **r})

    csv_path = output_dir / "table_axis_ablation.csv"
    _write_csv(combined_rows, csv_path)

    # -----------------------------------------------------------------------
    # Print key takeaway numbers
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  KEY TAKEAWAY NUMBERS")
    print("=" * 80)

    for r in all_naive_rows:
        if r["scenario"] == "All":
            model = r["model"]
            t_rate = r["T_response_rate"]
            f_rate = r["F_strict_rate"]
            oog_t = r["OOG_T_vs_F"]
            oog_tr = r["OOG_T+R_vs_F"]
            print(f"  {model}:")
            print(f"    Non-refusal (T_response) rate:  {t_rate}")
            print(f"    Full grounding (F_strict) rate:  {f_rate}")
            print(f"    Optimism gap (T vs F):           {oog_t}")
            print(f"    Optimism gap (T+R vs F):         {oog_tr}")

    # Show axis-specific catches for the "All" rows
    print()
    for r in all_failure_rows:
        if r["scenario"] == "All":
            model = r["model"]
            n = r["n"]
            ax1 = r["Axis1_catches (T=1,R=0)"]
            ax2 = r["Axis2_catches (T=1,R=1,I=0)"]
            ax3 = r["Axis3_catches (T=1,R=1,I=1,S!=1)"]
            f_s = r["F_strict (all pass)"]
            ref = r["refusal (T=0)"]
            print(f"  {model} (n={n}): "
                  f"refusals={ref}, "
                  f"Axis1 catches={ax1} ({100*ax1/n:.1f}%), "
                  f"Axis2 catches={ax2} ({100*ax2/n:.1f}%), "
                  f"Axis3 catches={ax3} ({100*ax3/n:.1f}%), "
                  f"F_strict={f_s} ({100*f_s/n:.1f}%)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
