"""
compaction_threshold_sweep.py — Multi-threshold compaction sensitivity analysis.

Analyzes whether MIRAGE's S2 collapse and S3 divergence are robust across
compaction thresholds {25k, 50k, 75k, 100k}, addressing the reviewer concern
that findings may be artifacts of the fixed 50k setting.

Design:
  1. Load & deduplicate threshold-sweep trials (S2 + S3) for each model.
  2. Group by (model, scenario, context_threshold).
  3. Report metrics per group: n, R_strict, I_strict, F_strict, D, R_path dist.
  4. Unpaired Fisher exact tests on F_strict: 25k vs 50k, 50k vs 75k, 50k vs 100k.
  5. Paired artifact-level transition analysis (McNemar-style) for F_strict.
  6. Output CSV + MD tables for the paper.

Usage:
    # Auto-detect sweep runs from logs/:
    python scripts/compaction_threshold_sweep.py

    # Explicit trial dirs:
    python scripts/compaction_threshold_sweep.py \
        --trial-dirs logs/trials_paper_compactsens_gpt5_0324 \
                     logs/trials_paper_compactsens_qwen3vl32b_0324

Output:
    results/paper_tables/table_compaction_threshold_sweep.csv
    results/paper_tables/table_compaction_threshold_sweep.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOGS_DIR = ROOT / "logs"
OUTPUT_DIR = ROOT / "results" / "paper_tables"


# ---------------------------------------------------------------------------
# Auto-detection of sweep trial directories
# ---------------------------------------------------------------------------


def _detect_sweep_dirs() -> list[Path]:
    """Auto-detect compaction-sweep trial directories under logs/.

    Selection criteria (ALL must hold):
      1. Has run_config.json with len(context_thresholds) > 1
      2. Contains at least one trial_*.json file (not empty / smoke-only)
      3. Dir name does NOT contain 'smoke' (explicit smoke runs are excluded)

    De-duplication: when multiple qualifying dirs exist for the same model,
    only the latest (by run_config.started_at, falling back to dir mtime)
    is kept. This prevents stale / partial reruns from being mixed in.
    """
    if not LOGS_DIR.exists():
        return []

    # Collect (model, started_at, dir) tuples
    scored: list[tuple[str, str, Path]] = []

    for d in sorted(LOGS_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("trials_"):
            continue
        # Skip explicit smoke runs
        if "smoke" in d.name:
            continue
        cfg_path = d / "run_config.json"
        if not cfg_path.exists():
            continue
        try:
            data = json.loads(cfg_path.read_text())
        except Exception:
            continue
        thresholds = data.get("context_thresholds", [])
        if len(thresholds) <= 1:
            continue
        # Must have at least one trial JSON
        if not any(d.glob("trial_*.json")):
            continue
        model = data.get("model", "unknown")
        started = data.get("started_at", "")
        scored.append((model, started, d))

    if not scored:
        return []

    # Keep only the latest run per model
    best: dict[str, tuple[str, Path]] = {}
    for model, started, d in scored:
        prev = best.get(model)
        if prev is None or started > prev[0]:
            best[model] = (started, d)

    return [d for _, d in sorted(best.values(), key=lambda x: x[0])]


# ---------------------------------------------------------------------------
# Loading + deduplication
# ---------------------------------------------------------------------------


def _cell_key(t: dict) -> tuple:
    """Dedup key: (backend, artifact_id, scenario, cf_condition,
    reference_style, history_depth, context_threshold)."""
    theta = t.get("theta") or {}
    return (
        theta.get("backend", ""),
        t.get("artifact_id", ""),
        t.get("scenario", ""),
        theta.get("citation_force_condition", ""),
        theta.get("reference_style", ""),
        theta.get("history_depth", 0),
        theta.get("context_threshold", 50_000),
    )


def _pairing_key(t: dict) -> tuple:
    """Key for paired artifact-level threshold comparison.

    Matches trials that differ only in context_threshold:
    (backend, artifact_id, scenario, cf_condition, reference_style, history_depth).
    """
    theta = t.get("theta") or {}
    return (
        theta.get("backend", ""),
        t.get("artifact_id", ""),
        t.get("scenario", ""),
        theta.get("citation_force_condition", ""),
        theta.get("reference_style", ""),
        theta.get("history_depth", 0),
    )


def _model_label(trial_dir: Path) -> str:
    cfg_path = trial_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        m = cfg.get("model", trial_dir.name)
        return m.split("/")[-1] if "/" in m else m
    return trial_dir.name


def load_and_dedupe(trial_dir: Path) -> list[dict]:
    """Load all trial JSONs, keep latest per cell key."""
    raw = []
    for p in sorted(trial_dir.glob("trial_*.json")):
        try:
            t = json.loads(p.read_text())
            raw.append(t)
        except Exception:
            pass
    raw.sort(key=lambda t: t.get("created_at", ""))
    seen: dict[tuple, dict] = {}
    for t in raw:
        seen[_cell_key(t)] = t  # last wins
    deduped = list(seen.values())
    deduped.sort(key=lambda t: t.get("created_at", ""))
    return deduped


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

# Strict R_binary: excludes heuristic_R_context (consistent with eval_summary)
_STRICT_R_PATHS = {"R_context", "R_boot", "R_tool"}


def compute_group_stats(trials: list[dict]) -> dict:
    n = len(trials)
    if n == 0:
        return {
            "n": 0,
            "R_strict_count": 0, "R_strict_rate": "",
            "I_strict_count": 0, "I_strict_rate": "",
            "F_strict_count": 0, "F_strict_rate": "",
            "D_count": 0, "D_rate": "",
            "R_path_dist": "",
        }
    r_strict = sum(1 for t in trials if t.get("R_path") in _STRICT_R_PATHS)
    i_count = sum(1 for t in trials if t.get("I_strict") == 1)
    f_count = sum(1 for t in trials if t.get("F_strict") == 1)
    d_count = sum(1 for t in trials if t.get("D") == 1)

    r_dist: dict[str, int] = defaultdict(int)
    for t in trials:
        r_dist[t.get("R_path") or "R_none"] += 1
    r_path_str = "  ".join(f"{k}={v}" for k, v in sorted(r_dist.items()))

    return {
        "n": n,
        "R_strict_count": r_strict,
        "R_strict_rate": f"{r_strict / n:.3f}",
        "I_strict_count": i_count,
        "I_strict_rate": f"{i_count / n:.3f}",
        "F_strict_count": f_count,
        "F_strict_rate": f"{f_count / n:.3f}",
        "D_count": d_count,
        "D_rate": f"{d_count / n:.3f}",
        "R_path_dist": r_path_str,
    }


# ---------------------------------------------------------------------------
# Fisher exact test (stdlib-only, no scipy dependency)
# ---------------------------------------------------------------------------


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test p-value for a 2x2 table.

    Table:          col1  col2
        row1         a     b
        row2         c     d
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    r1 = a + b
    r2 = c + d
    c1 = a + c

    def hyper_pmf(x: int) -> float:
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    p_obs = hyper_pmf(a)
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    p_val = sum(hyper_pmf(x) for x in range(lo, hi + 1) if hyper_pmf(x) <= p_obs + 1e-12)
    return min(p_val, 1.0)


# ---------------------------------------------------------------------------
# Paired artifact-level transition analysis
# ---------------------------------------------------------------------------


def paired_transition_analysis(
    trials: list[dict],
    ct_a: int,
    ct_b: int,
    metric: str = "F_strict",
) -> dict | None:
    """Paired comparison of a binary metric across two thresholds.

    Pairs trials by (backend, artifact_id, scenario, condition, style, depth),
    comparing the metric value at ct_a vs ct_b for each artifact.

    Returns a dict with:
      n_paired, concordant (both 0 or both 1), discordant_a_only (1 at ct_a, 0 at ct_b),
      discordant_b_only (0 at ct_a, 1 at ct_b), mcnemar_p.

    McNemar's exact test: two-sided binomial test on the discordant pair counts.
    Returns None if no paired data is available.
    """
    # Index trials by (pairing_key, context_threshold)
    by_pair_ct: dict[tuple, dict] = {}
    for t in trials:
        pk = _pairing_key(t)
        ct = (t.get("theta") or {}).get("context_threshold", 50_000)
        by_pair_ct[(pk, ct)] = t

    # Build pairs
    both_0 = 0  # concordant: metric=0 at both thresholds
    both_1 = 0  # concordant: metric=1 at both thresholds
    a_only = 0  # discordant: metric=1 at ct_a, metric=0 at ct_b
    b_only = 0  # discordant: metric=0 at ct_a, metric=1 at ct_b
    n_paired = 0

    paired_keys = set()
    for (pk, ct), t in by_pair_ct.items():
        if ct == ct_a and pk not in paired_keys:
            partner = by_pair_ct.get((pk, ct_b))
            if partner is None:
                continue
            paired_keys.add(pk)
            n_paired += 1
            va = 1 if t.get(metric) == 1 else 0
            vb = 1 if partner.get(metric) == 1 else 0
            if va == 1 and vb == 1:
                both_1 += 1
            elif va == 0 and vb == 0:
                both_0 += 1
            elif va == 1 and vb == 0:
                a_only += 1
            else:
                b_only += 1

    if n_paired == 0:
        return None

    # McNemar's exact test (two-sided binomial on discordant pairs)
    n_disc = a_only + b_only
    if n_disc == 0:
        p = 1.0
    else:
        # Two-sided: P(X >= max(a_only, b_only)) + P(X <= min(a_only, b_only))
        # under H0: X ~ Binomial(n_disc, 0.5)
        k = min(a_only, b_only)
        # P(X <= k) for Binomial(n_disc, 0.5)
        p = 0.0
        for i in range(k + 1):
            p += comb(n_disc, i) / (2 ** n_disc)
        p = min(2.0 * p, 1.0)  # two-sided

    return {
        "n_paired": n_paired,
        "both_0": both_0,
        "both_1": both_1,
        "a_only": a_only,
        "b_only": b_only,
        "n_discordant": n_disc,
        "mcnemar_p": p,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

THRESHOLDS = [25_000, 50_000, 75_000, 100_000]
COMPARISONS = [
    (25_000, 50_000),
    (50_000, 75_000),
    (50_000, 100_000),
]


def run_analysis(trial_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []

    print("=" * 80)
    print("  MULTI-THRESHOLD COMPACTION SENSITIVITY SWEEP")
    print("=" * 80)
    print()
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Pairwise comparisons: {COMPARISONS}")
    print()

    for trial_dir in trial_dirs:
        if not trial_dir.exists():
            print(f"  [SKIP] {trial_dir} not found")
            continue

        model = _model_label(trial_dir)
        all_trials = load_and_dedupe(trial_dir)

        if not all_trials:
            print(f"  [SKIP] {trial_dir} — no trial JSONs found")
            continue

        # Split by scenario
        s2_trials = [t for t in all_trials if t.get("scenario") == "S2"]
        s3_trials = [t for t in all_trials if t.get("scenario") == "S3"]

        print(f"{'=' * 70}")
        print(f"Model: {model}")
        print(f"  Deduped total: {len(all_trials)}  (S2={len(s2_trials)}, S3={len(s3_trials)})")

        # Dedup counts by threshold
        for sc_label, sc_trials in [("S2", s2_trials), ("S3", s3_trials)]:
            ct_counts: dict[int, int] = defaultdict(int)
            for t in sc_trials:
                ct = (t.get("theta") or {}).get("context_threshold", 50_000)
                ct_counts[ct] += 1
            ct_str = "  ".join(f"{k}k={v}" for k, v in sorted(ct_counts.items()))
            print(f"  {sc_label} by threshold: {ct_str}")

        # Per scenario x threshold
        for scenario, trials in [("S2", s2_trials), ("S3", s3_trials)]:
            print(f"\n  [{scenario}] Per-threshold metrics:")
            print(f"    {'Threshold':>10s} {'n':>4s} {'R_strict':>10s} {'I_strict':>10s} "
                  f"{'F_strict':>10s} {'D':>6s}  R_path distribution")

            # Group by threshold
            by_threshold: dict[int, list[dict]] = defaultdict(list)
            for t in trials:
                ct = (t.get("theta") or {}).get("context_threshold", 50_000)
                by_threshold[ct].append(t)

            for ct in THRESHOLDS:
                group = by_threshold.get(ct, [])
                s = compute_group_stats(group)
                print(f"    {ct:>10d} {s['n']:>4d} "
                      f"{s['R_strict_count']:>3d}/{s['n']:>3d} ({s['R_strict_rate']:>5s}) "
                      f"{s['I_strict_count']:>3d}/{s['n']:>3d} ({s['I_strict_rate']:>5s}) "
                      f"{s['F_strict_count']:>3d}/{s['n']:>3d} ({s['F_strict_rate']:>5s}) "
                      f"{s['D_count']:>3d}/{s['n']:>3d}  {s['R_path_dist']}")

                rows.append({
                    "model": model,
                    "scenario": scenario,
                    "context_threshold": ct,
                    **s,
                })

            # --- Unpaired Fisher exact tests (F_strict) ---
            print(f"\n    Unpaired Fisher exact tests (F_strict):")
            for ct_a, ct_b in COMPARISONS:
                ga = by_threshold.get(ct_a, [])
                gb = by_threshold.get(ct_b, [])
                if not ga or not gb:
                    print(f"      {ct_a} vs {ct_b}: SKIPPED (missing data)")
                    continue
                a = sum(1 for t in ga if t.get("F_strict") == 1)
                b = len(ga) - a
                c = sum(1 for t in gb if t.get("F_strict") == 1)
                d = len(gb) - c
                p = fisher_exact_2x2(a, b, c, d)
                print(f"      {ct_a} vs {ct_b}:")
                print(f"        {ct_a:>6d}: F=1:{a}  F=0:{b}")
                print(f"        {ct_b:>6d}: F=1:{c}  F=0:{d}")
                print(f"        Fisher p = {p:.4f}")

            # --- Paired artifact-level transition analysis (F_strict) ---
            print(f"\n    Paired artifact-level transition (F_strict, McNemar exact):")
            for ct_a, ct_b in COMPARISONS:
                result = paired_transition_analysis(trials, ct_a, ct_b, metric="F_strict")
                if result is None:
                    print(f"      {ct_a} vs {ct_b}: SKIPPED (no paired data)")
                    continue
                print(f"      {ct_a} -> {ct_b}:")
                print(f"        paired artifacts: {result['n_paired']}")
                print(f"        concordant: both_F=1: {result['both_1']}  both_F=0: {result['both_0']}")
                print(f"        discordant: {ct_a}-only F=1: {result['a_only']}  "
                      f"{ct_b}-only F=1: {result['b_only']}")
                print(f"        McNemar p = {result['mcnemar_p']:.4f}  "
                      f"(n_discordant={result['n_discordant']})")

            # --- Unpaired Fisher exact tests (D, secondary) ---
            print(f"\n    Unpaired Fisher exact tests (D, secondary):")
            for ct_a, ct_b in COMPARISONS:
                ga = by_threshold.get(ct_a, [])
                gb = by_threshold.get(ct_b, [])
                if not ga or not gb:
                    continue
                a = sum(1 for t in ga if t.get("D") == 1)
                b = len(ga) - a
                c = sum(1 for t in gb if t.get("D") == 1)
                d = len(gb) - c
                p = fisher_exact_2x2(a, b, c, d)
                print(f"      {ct_a} vs {ct_b}:  D table [{a},{b}] vs [{c},{d}]  p = {p:.4f}")

        print()

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print("=" * 80)
    print("  INTERPRETATION")
    print("=" * 80)
    print()
    print("Two statistical tests are reported for each threshold comparison:")
    print("  1. Unpaired Fisher exact: compares aggregate F_strict rates across thresholds.")
    print("  2. Paired McNemar exact: tests whether individual artifacts change outcome")
    print("     when the threshold changes, controlling for artifact identity.")
    print()
    print("If all p-values are > 0.05, the S2 collapse and S3 divergence are robust")
    print("across {25k, 50k, 75k, 100k} compaction thresholds.")
    print("The main findings are not artifacts of the specific 50k setting.")
    print()
    print("If any comparison shows p < 0.05, the affected model x scenario is threshold-")
    print("sensitive and the paper claim should be narrowed accordingly.")
    print()

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    "model", "scenario", "context_threshold",
    "n", "R_strict_count", "R_strict_rate",
    "I_strict_count", "I_strict_rate",
    "F_strict_count", "F_strict_rate",
    "D_count", "D_rate",
    "R_path_dist",
]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"CSV written to {path}")


def write_md(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Compaction Threshold Sensitivity Sweep",
        "",
        "Thresholds tested: 25k, 50k, 75k, 100k tokens.",
        "Scenarios: S2 (post-compaction, same session) and S3 (fresh session).",
        "Condition: C0 (no mitigation). Reference style: definite.",
        "",
    ]

    current_model = None
    current_scenario = None

    for i, r in enumerate(rows):
        if r["model"] != current_model:
            current_model = r["model"]
            lines.append(f"## {current_model}")
            lines.append("")
        if r["scenario"] != current_scenario:
            current_scenario = r["scenario"]
            lines.append(f"### {current_scenario}")
            lines.append("")
            header = "| Threshold | n | R_strict | I_strict | F_strict | D | R_path distribution |"
            sep = "|---|---|---|---|---|---|---|"
            lines.extend([header, sep])

        ct = r["context_threshold"]
        n = r["n"]
        lines.append(
            f"| {ct:,} | {n} | "
            f"{r['R_strict_count']}/{n} ({r['R_strict_rate']}) | "
            f"{r['I_strict_count']}/{n} ({r['I_strict_rate']}) | "
            f"{r['F_strict_count']}/{n} ({r['F_strict_rate']}) | "
            f"{r['D_count']}/{n} ({r['D_rate']}) | "
            f"{r['R_path_dist']} |"
        )

        # Check if next row switches model or scenario
        if i + 1 < len(rows):
            nr = rows[i + 1]
            if nr["scenario"] != current_scenario or nr["model"] != current_model:
                lines.append("")
                current_scenario = None

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD written to {path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-threshold compaction sensitivity sweep analysis",
    )
    parser.add_argument(
        "--trial-dirs", nargs="+", type=Path, default=None,
        help="Trial directories to analyze (default: auto-detect from logs/)",
    )
    args = parser.parse_args()

    if args.trial_dirs:
        trial_dirs = args.trial_dirs
    else:
        trial_dirs = _detect_sweep_dirs()
        if not trial_dirs:
            print(
                "ERROR: No compaction-sweep trial directories found in logs/.\n"
                "\n"
                "Expected: directories under logs/ whose run_config.json contains\n"
                "multiple context_thresholds, or whose name contains 'compactsens'.\n"
                "\n"
                "Either:\n"
                "  1. Run the sweep first:\n"
                "     python -m harness.experiment_runner --preset paper_s2s3_compaction \\\n"
                "       --model shubiaobiao/gpt-5 --run-id paper_compactsens_gpt5_0324\n"
                "\n"
                "  2. Or specify trial dirs explicitly:\n"
                "     python scripts/compaction_threshold_sweep.py \\\n"
                "       --trial-dirs logs/trials_MY_RUN_1 logs/trials_MY_RUN_2\n",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-detected {len(trial_dirs)} sweep dir(s):")
        for d in trial_dirs:
            print(f"  {d}")
        print()

    rows = run_analysis(trial_dirs)
    if not rows:
        print("No data to write.", file=sys.stderr)
        sys.exit(1)

    write_csv(rows, OUTPUT_DIR / "table_compaction_threshold_sweep.csv")
    write_md(rows, OUTPUT_DIR / "table_compaction_threshold_sweep.md")


if __name__ == "__main__":
    main()
