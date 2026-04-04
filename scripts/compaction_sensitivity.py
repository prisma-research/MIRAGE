"""
compaction_sensitivity.py — Post-boundary context-size sensitivity analysis.

Investigates whether S2/S3 outcomes vary with the context size at the
compaction boundary, using provider-reported `boundary_input_tokens`
(effective input tokens = input + cacheRead) recorded by the harness
when OpenClaw fires native compaction.

The script requires all trials to carry `boundary_input_tokens` and exits
with a clear error if any trial is missing this field.

Design:
  1. Load & deduplicate locked paper trials (S2 + S3).
  2. Read `boundary_input_tokens` from each S2 trial (provider-reported
     effective input tokens at the native compaction boundary).
  3. Stratify S2 by median split (low/high) and quartiles.
  4. Pair each S3 trial with its sibling S2 trial (same snapshot) and
     inherit the S2 proxy bucket — justified because S2/S3 branch from
     the same post-compaction snapshot in the harness.
  5. Report metrics and Fisher exact tests for each model × scenario.

Usage:
    python scripts/compaction_sensitivity.py

Output:
    results/paper_tables/table_compaction_sensitivity.csv
    results/paper_tables/table_compaction_sensitivity.md
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRIAL_DIRS = [
    ROOT / "logs" / "trials_paper_main_gpt5_0322",
    ROOT / "logs" / "trials_paper_main_qwen3vl32b_0322",
]
OUTPUT_DIR = ROOT / "results" / "paper_tables"

# ---------------------------------------------------------------------------
# Loading + deduplication
# ---------------------------------------------------------------------------


def _cell_key(t: dict) -> tuple:
    """Dedup key: (model_backend, artifact_id, scenario, cf_condition,
    reference_style, history_depth)."""
    theta = t.get("theta") or {}
    return (
        theta.get("backend", ""),
        t.get("artifact_id", ""),
        t.get("scenario", ""),
        theta.get("citation_force_condition", ""),
        theta.get("reference_style", ""),
        theta.get("history_depth", 0),
    )


def _pairing_key(t: dict) -> tuple:
    """Cross-scenario pairing key: everything except scenario."""
    theta = t.get("theta") or {}
    return (
        theta.get("backend", ""),
        t.get("artifact_id", ""),
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
# Proxy computation
# ---------------------------------------------------------------------------


def get_boundary_input_tokens(trial: dict) -> int | None:
    """Read provider-reported boundary_input_tokens from the trial record.

    This is the authoritative context-size signal: effective_input_tokens
    (input + cacheRead) recorded at the compaction/query boundary.

    Returns None if the field is missing — callers must handle this explicitly.
    No heuristic fallback (chars // 4) is used; all trials on native-reset
    must have provider-reported boundary_input_tokens.
    """
    bit = trial.get("boundary_input_tokens")
    if bit is not None and bit > 0:
        return bit
    return None


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def compute_group_stats(trials: list[dict]) -> dict:
    n = len(trials)
    if n == 0:
        return {
            "n": 0, "F_strict_count": 0, "F_strict_rate": "",
            "I_strict_count": 0, "I_strict_rate": "",
            "D_count": 0, "D_rate": "",
            "proxy_mean_tokens": "", "proxy_median_tokens": "",
            "token_range": "",
        }
    f_count = sum(1 for t in trials if t.get("F_strict") == 1)
    i_count = sum(1 for t in trials if t.get("I_strict") == 1)
    d_count = sum(1 for t in trials if t.get("D") == 1)
    tokens = [t["_proxy_tokens"] for t in trials]
    return {
        "n": n,
        "F_strict_count": f_count,
        "F_strict_rate": f"{f_count / n:.3f}",
        "I_strict_count": i_count,
        "I_strict_rate": f"{i_count / n:.3f}",
        "D_count": d_count,
        "D_rate": f"{d_count / n:.3f}",
        "proxy_mean_tokens": f"{statistics.mean(tokens):,.0f}",
        "proxy_median_tokens": f"{statistics.median(tokens):,.0f}",
        "token_range": f"{min(tokens):,}–{max(tokens):,}",
    }


def rpath_distribution(trials: list[dict]) -> str:
    c: dict[str, int] = defaultdict(int)
    for t in trials:
        c[t.get("R_path") or "R_none"] += 1
    return "  ".join(f"{k}={v}" for k, v in sorted(c.items()))


# ---------------------------------------------------------------------------
# Fisher exact test (stdlib-only, no scipy dependency)
# ---------------------------------------------------------------------------


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test p-value for a 2×2 table.

    Table:          col1  col2
        row1         a     b
        row2         c     d

    Uses the hypergeometric distribution via math.comb.
    """
    n = a + b + c + d
    r1 = a + b
    r2 = c + d
    c1 = a + c

    def hyper_pmf(x: int) -> float:
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    p_obs = hyper_pmf(a)
    # Two-sided: sum all outcomes as or more extreme (pmf ≤ p_obs)
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    p_val = sum(hyper_pmf(x) for x in range(lo, hi + 1) if hyper_pmf(x) <= p_obs + 1e-12)
    return min(p_val, 1.0)


# ---------------------------------------------------------------------------
# Quartile helpers
# ---------------------------------------------------------------------------


def quartile_boundaries(values: list[int]) -> tuple[int, int, int]:
    s = sorted(values)
    n = len(s)
    return s[n // 4], s[n // 2], s[3 * n // 4]


def assign_quartile(value: int, q1: int, q2: int, q3: int) -> str:
    if value < q1:
        return "Q1"
    elif value < q2:
        return "Q2"
    elif value < q3:
        return "Q3"
    else:
        return "Q4"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis() -> list[dict]:
    rows: list[dict] = []

    print("=" * 80)
    print("  POST-BOUNDARY CONTEXT-SIZE SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()
    print("Proxy variable: boundary_input_tokens (provider-reported input + cacheRead at the")
    print("native compaction boundary). All trials must carry this field; the script stops")
    print("if any are missing.")
    print("S3 trials inherit the proxy bucket from their paired S2 sibling (same snapshot).")
    print()

    # ------------------------------------------------------------------
    # Load and deduplicate per model
    # ------------------------------------------------------------------
    for trial_dir in TRIAL_DIRS:
        if not trial_dir.exists():
            print(f"  [SKIP] {trial_dir} not found")
            continue

        model = _model_label(trial_dir)
        all_trials = load_and_dedupe(trial_dir)

        s2_trials = [t for t in all_trials if t.get("scenario") == "S2"]
        s3_trials = [t for t in all_trials if t.get("scenario") == "S3"]

        print(f"{'=' * 70}")
        print(f"Model: {model}")
        print(f"  Deduped total: {len(all_trials)}  (S2={len(s2_trials)}, S3={len(s3_trials)})")

        # Verify expected counts
        for label, subset, expected in [("S2", s2_trials, 100), ("S3", s3_trials, 100)]:
            if len(subset) != expected:
                print(f"  [ERROR] Expected {expected} {label} trials, got {len(subset)}. Stopping.")
                sys.exit(1)
        print(f"  Dedup verification: 100 S2 + 100 S3 ✓")

        # ------------------------------------------------------------------
        # Compute S2 proxy — require provider-reported boundary_input_tokens
        # ------------------------------------------------------------------
        missing_bit = 0
        for t in s2_trials:
            bit = get_boundary_input_tokens(t)
            if bit is None:
                missing_bit += 1
                t["_proxy_tokens"] = -1
            else:
                t["_proxy_tokens"] = bit
        if missing_bit:
            print(f"  [ERROR] {missing_bit}/{len(s2_trials)} S2 trials missing boundary_input_tokens.")
            print(f"          All native-reset trials must have provider-reported data. Stopping.")
            sys.exit(1)

        tokens = [t["_proxy_tokens"] for t in s2_trials]
        median_tok = statistics.median(tokens)
        print(f"\n  S2 proxy distribution (est. tokens):")
        print(f"    min={min(tokens):,}  max={max(tokens):,}  "
              f"mean={statistics.mean(tokens):,.0f}  "
              f"median={median_tok:,.0f}  "
              f"stdev={statistics.stdev(tokens):,.0f}")

        # Binary split
        for t in s2_trials:
            t["_proxy_bucket"] = "low" if t["_proxy_tokens"] < median_tok else "high"

        # Quartile split
        q1, q2, q3 = quartile_boundaries(tokens)
        for t in s2_trials:
            t["_proxy_quartile"] = assign_quartile(t["_proxy_tokens"], q1, q2, q3)

        # ------------------------------------------------------------------
        # Pair S3 with S2 siblings → inherit proxy
        # ------------------------------------------------------------------
        s2_by_pair_key: dict[tuple, dict] = {}
        for t in s2_trials:
            s2_by_pair_key[_pairing_key(t)] = t

        paired = 0
        unpaired = []
        for t in s3_trials:
            pk = _pairing_key(t)
            sibling = s2_by_pair_key.get(pk)
            if sibling is None:
                unpaired.append(t.get("artifact_id", "?"))
                t["_proxy_tokens"] = -1
                t["_proxy_bucket"] = "unpaired"
                t["_proxy_quartile"] = "unpaired"
            else:
                t["_proxy_tokens"] = sibling["_proxy_tokens"]
                t["_proxy_bucket"] = sibling["_proxy_bucket"]
                t["_proxy_quartile"] = sibling["_proxy_quartile"]
                paired += 1

        print(f"\n  S3 pairing: {paired}/100 paired")
        if unpaired:
            print(f"    [WARNING] Unpaired S3 artifacts: {unpaired}")
        else:
            print(f"    All S3 trials successfully inherited S2 proxy bucket ✓")

        # ------------------------------------------------------------------
        # Report per scenario
        # ------------------------------------------------------------------
        for scenario, trials in [("S2", s2_trials), ("S3", s3_trials)]:
            # Filter out unpaired S3 if any
            usable = [t for t in trials if t.get("_proxy_bucket") != "unpaired"]
            if len(usable) < len(trials):
                print(f"\n  [{scenario}] {len(trials) - len(usable)} unpaired trials excluded")

            low = [t for t in usable if t["_proxy_bucket"] == "low"]
            high = [t for t in usable if t["_proxy_bucket"] == "high"]

            print(f"\n  [{scenario}] Median-split stratification (median={median_tok:,} est. tokens):")
            print(f"    {'Group':<8s} {'n':>4s}  {'F':>4s}  {'F_rate':>7s}  "
                  f"{'I':>4s}  {'I_rate':>7s}  "
                  f"{'D':>4s}  {'D_rate':>7s}  "
                  f"{'mean_tok':>10s}  {'med_tok':>10s}  {'range':>20s}")

            for label, sub in [("low", low), ("high", high)]:
                s = compute_group_stats(sub)
                print(f"    {label:<8s} {s['n']:>4d}  "
                      f"{s['F_strict_count']:>4d}  {s['F_strict_rate']:>7s}  "
                      f"{s['I_strict_count']:>4d}  {s['I_strict_rate']:>7s}  "
                      f"{s['D_count']:>4d}  {s['D_rate']:>7s}  "
                      f"{s['proxy_mean_tokens']:>10s}  {s['proxy_median_tokens']:>10s}  "
                      f"{s['token_range']:>20s}")

            # Fisher exact: F_strict
            a_f = sum(1 for t in low if t.get("F_strict") == 1)
            b_f = len(low) - a_f
            c_f = sum(1 for t in high if t.get("F_strict") == 1)
            d_f = len(high) - c_f
            p_f = fisher_exact_2x2(a_f, b_f, c_f, d_f)

            print(f"\n    Fisher exact (F_strict, low vs. high):")
            print(f"    2×2 table:  low  F=1:{a_f}  F=0:{b_f}")
            print(f"                high F=1:{c_f}  F=0:{d_f}")
            print(f"    p = {p_f:.4f}")

            # Fisher exact: D (secondary)
            a_d = sum(1 for t in low if t.get("D") == 1)
            b_d = len(low) - a_d
            c_d = sum(1 for t in high if t.get("D") == 1)
            d_d = len(high) - c_d
            p_d = fisher_exact_2x2(a_d, b_d, c_d, d_d)

            print(f"\n    Fisher exact (D, low vs. high):")
            print(f"    2×2 table:  low  D=1:{a_d}  D=0:{b_d}")
            print(f"                high D=1:{c_d}  D=0:{d_d}")
            print(f"    p = {p_d:.4f}")

            # R_path distribution
            print(f"\n    R_path distribution:")
            print(f"      low:  {rpath_distribution(low)}")
            print(f"      high: {rpath_distribution(high)}")

            # CSV rows for median split
            for label, sub in [("low", low), ("high", high)]:
                s = compute_group_stats(sub)
                rows.append({
                    "model": model,
                    "scenario": scenario,
                    "stratification": "median_split",
                    "group": label,
                    **s,
                    "fisher_p_value": f"{p_f:.4f}" if label == "low" else "",
                    "notes": f"median={median_tok:,}"
                             + (" (inherited from paired S2)" if scenario == "S3" else ""),
                })

            # Quartile descriptive
            print(f"\n    Quartile stratification (Q1<{q1:,}  Q2<{q2:,}  Q3<{q3:,}):")
            print(f"    {'Q':<4s} {'n':>4s}  {'F':>4s}  {'F_rate':>7s}  "
                  f"{'I':>4s}  {'I_rate':>7s}  "
                  f"{'D':>4s}  {'D_rate':>7s}  {'range':>20s}")

            for q_label in ["Q1", "Q2", "Q3", "Q4"]:
                sub = [t for t in usable if t["_proxy_quartile"] == q_label]
                s = compute_group_stats(sub)
                print(f"    {q_label:<4s} {s['n']:>4d}  "
                      f"{s['F_strict_count']:>4d}  {s['F_strict_rate']:>7s}  "
                      f"{s['I_strict_count']:>4d}  {s['I_strict_rate']:>7s}  "
                      f"{s['D_count']:>4d}  {s['D_rate']:>7s}  "
                      f"{s['token_range']:>20s}")
                rows.append({
                    "model": model,
                    "scenario": scenario,
                    "stratification": "quartile",
                    "group": q_label,
                    **s,
                    "fisher_p_value": "",
                    "notes": f"boundaries: {q1:,}/{q2:,}/{q3:,}"
                             + (" (inherited)" if scenario == "S3" else ""),
                })

        # ------------------------------------------------------------------
        # Discrepancy check vs prior analysis
        # ------------------------------------------------------------------
        print(f"\n  [Discrepancy check]")
        print(f"    Prior script stratified by compaction_during_session_b (= compaction_verified")
        print(f"    in the S2 harness path), which is a different variable than the proxy used here.")
        cv_true = [t for t in s2_trials if t.get("compaction_during_session_b", False)]
        cv_false = [t for t in s2_trials if not t.get("compaction_during_session_b", False)]
        print(f"    compaction_verified=True:  n={len(cv_true)}, "
              f"mean proxy={statistics.mean([t['_proxy_tokens'] for t in cv_true]):,.0f}")
        print(f"    compaction_verified=False: n={len(cv_false)}, "
              f"mean proxy={statistics.mean([t['_proxy_tokens'] for t in cv_false]):,.0f}")
        # Overlap: how many verified=True are in the low bucket?
        cv_true_low = sum(1 for t in cv_true if t["_proxy_bucket"] == "low")
        cv_false_low = sum(1 for t in cv_false if t["_proxy_bucket"] == "low")
        print(f"    compaction_verified=True  in low bucket: {cv_true_low}/{len(cv_true)}")
        print(f"    compaction_verified=False in low bucket: {cv_false_low}/{len(cv_false)}")
        print(f"    The median-split proxy captures context-size variation directly,")
        print(f"    regardless of the compaction_verified flag's semantics.")
        print()

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print("=" * 80)
    print("  INTERPRETATION")
    print("=" * 80)
    print()
    print("Within each compaction threshold regime, provider-reported boundary_input_tokens")
    print("(effective input = input + cacheRead at the native compaction boundary) shows")
    print("natural variation across trials. S3 inherits the same boundary measurement")
    print("from its paired S2 cell (shared post-compaction snapshot).")
    print()
    print("The paper_s2s3_compaction preset sweeps four trigger targets (25k/50k/75k/100k).")
    print("This script analyses within-regime variation for a single threshold; cross-")
    print("threshold comparisons use the context_threshold field in the trial records.")
    print()

    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    "model", "scenario", "stratification", "group",
    "n", "F_strict_count", "F_strict_rate",
    "I_strict_count", "I_strict_rate",
    "D_count", "D_rate",
    "proxy_mean_tokens", "proxy_median_tokens", "token_range",
    "fisher_p_value", "notes",
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
    lines = ["# Compaction Sensitivity Analysis", ""]

    current_model = None
    current_scenario = None
    current_strat = None

    for r in rows:
        if r["model"] != current_model:
            current_model = r["model"]
            lines.append(f"## {current_model}")
            lines.append("")
        if r["scenario"] != current_scenario:
            current_scenario = r["scenario"]
            lines.append(f"### {current_scenario}")
            lines.append("")
        if r["stratification"] != current_strat:
            current_strat = r["stratification"]
            header = (f"| Group | n | F_strict | I_strict | D | "
                      f"Mean tokens | Median tokens | Range | Fisher p |")
            sep = "|---|---|---|---|---|---|---|---|---|"
            lines.extend([header, sep])

        fp = r.get("fisher_p_value", "")
        lines.append(
            f"| {r['group']} | {r['n']} | "
            f"{r['F_strict_count']}/{r['n']} ({r['F_strict_rate']}) | "
            f"{r['I_strict_count']}/{r['n']} ({r['I_strict_rate']}) | "
            f"{r['D_count']}/{r['n']} ({r['D_rate']}) | "
            f"{r['proxy_mean_tokens']} | {r['proxy_median_tokens']} | "
            f"{r['token_range']} | {fp} |"
        )

        # Reset strat if next row switches
        next_idx = rows.index(r) + 1
        if next_idx < len(rows):
            nr = rows[next_idx]
            if nr["stratification"] != current_strat or nr["scenario"] != current_scenario or nr["model"] != current_model:
                lines.append("")
                current_strat = None

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD written to {path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = run_analysis()
    write_csv(rows, OUTPUT_DIR / "table_compaction_sensitivity.csv")
    write_md(rows, OUTPUT_DIR / "table_compaction_sensitivity.md")
