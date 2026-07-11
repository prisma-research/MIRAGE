#!/usr/bin/env python3
"""
Reanalysis of the CitationForce mitigation experiment (C0, Cp, Cm).

Produces paper-ready tables from S3 mitigation trial JSONs:
  - table_mitigation_aggregate.csv  — per-model, per-condition summary
  - table_mitigation_paired.csv     — paired per-artifact transition counts
  - table_mitigation_signals.csv    — grounding_quality breakdown

Three conditions:
  C0  — baseline (no mitigation)
  Cp  — prompt-constraint ablation (internal label "C3")
  Cm  — mandatory artifact_recall tool routing

Usage:
    python scripts/reanalyze_mitigation.py [--gpt5-dir DIR] [--qwen-dir DIR] \
        [--gpt5-cm-dir DIR] [--qwen-cm-dir DIR] [--output DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure MIRAGE is importable
# ---------------------------------------------------------------------------
_GROUNDING_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_GROUNDING_ROOT))

from mitigations.citation_force.signals import analyze_response_signals, detect_citations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_GPT5_DIR    = _GROUNDING_ROOT / "logs" / "trials_paper_s3mit_gpt5_0323v2"
DEFAULT_QWEN_DIR    = _GROUNDING_ROOT / "logs" / "trials_paper_s3mit_qwen3vl32b_0323v2"
DEFAULT_GPT5_CM_DIR = _GROUNDING_ROOT / "logs" / "trials_paper_s3mit_cm_gpt5_0323"
DEFAULT_QWEN_CM_DIR = _GROUNDING_ROOT / "logs" / "trials_paper_s3mit_cm_qwen3vl32b_0323"
DEFAULT_OUTPUT      = _GROUNDING_ROOT / "results" / "paper_tables"

# Display remap: internal condition labels → paper display labels
DISPLAY_REMAP = {"C3": "Cp"}

# Narrow citation regex for paper: only [Source: ...] bracket patterns
NARROW_CITATION_RE = re.compile(r"\[Source:\s*[^\]]+\]", re.IGNORECASE)

# Timeout detection patterns
TIMEOUT_PATTERNS = [
    "Request timed out",
    "timed out before a response",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _display_condition(internal: str) -> str:
    return DISPLAY_REMAP.get(internal, internal)


def detect_timeout(response_text: str) -> bool:
    return any(p in response_text for p in TIMEOUT_PATTERNS)


def narrow_citation_count(response_text: str) -> int:
    """Count [Source: ...] citations only (bracket format)."""
    return len(NARROW_CITATION_RE.findall(response_text))


def _has_artifact_recall(tool_trace: list[dict]) -> bool:
    """Check if artifact_recall was called in the tool trace."""
    return any(
        (t.get("tool") or t.get("name", "")) == "artifact_recall"
        for t in tool_trace
    )


def _has_memory_search(tool_trace: list[dict]) -> bool:
    """Check if memory_search was called in the tool trace."""
    return any(
        (t.get("tool") or t.get("name", "")) == "memory_search"
        for t in tool_trace
    )


def _is_protocol_compliant(trial: dict) -> bool:
    """
    Protocol compliance for Cm: artifact_recall was called AND
    memory_search was NOT the sole retrieval method.
    For C0/Cp: always True (no protocol to comply with).
    """
    cond = _display_condition(trial["theta"]["citation_force_condition"])
    if cond != "Cm":
        return True  # no protocol constraint for C0/Cp
    tt = trial.get("tool_trace", [])
    return _has_artifact_recall(tt)


def _load_trials(trial_dir: Path) -> list[dict]:
    """Load all trial JSONs from a directory (skip run_config.json)."""
    trials = []
    for p in sorted(trial_dir.glob("trial_*.json")):
        trials.append(json.loads(p.read_text()))
    return trials


def _model_label(trial_dir: Path) -> str:
    """Extract short model label from run_config.json."""
    rc = trial_dir / "run_config.json"
    if rc.exists():
        cfg = json.loads(rc.read_text())
        model = cfg.get("model", "")
        name = model.split("/")[-1]
        if "gpt-5" in name.lower():
            return "GPT-5"
        if "qwen3" in name.lower():
            return "Qwen3-VL-32B"
        return name
    return trial_dir.name


def _mcnemar_exact_p(b: int, c: int) -> float:
    """
    Two-sided McNemar's exact test (binomial).
    b = discordant pairs where base=0, treatment=1
    c = discordant pairs where base=1, treatment=0
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(2 * p_tail, 1.0)


def _compute_transitions(clean_pairs, metric_name, get_val):
    """Compute transition counts for a metric on clean pairs."""
    improved = degraded = both_pass = both_fail = 0
    for aid, t_base, t_treat in clean_pairs:
        v0 = get_val(t_base) or 0
        v1 = get_val(t_treat) or 0
        if metric_name == "D":
            # D: improved = base=1 (refused) → treat=0 (no refuse)
            if v0 == 1 and v1 == 0:
                improved += 1
            elif v0 == 0 and v1 == 1:
                degraded += 1
            elif v0 == 0 and v1 == 0:
                both_pass += 1
            else:
                both_fail += 1
        else:
            if v0 == 0 and v1 == 1:
                improved += 1
            elif v0 == 1 and v1 == 0:
                degraded += 1
            elif v0 == 1 and v1 == 1:
                both_pass += 1
            else:
                both_fail += 1
    return {"improved": improved, "degraded": degraded, "both_pass": both_pass, "both_fail": both_fail}


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(
    model_dirs: list[tuple[str, list[Path]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_aggregate_rows = []
    all_paired_rows = []
    all_signal_rows = []

    for model_label, trial_dirs in model_dirs:
        # Merge trials from all directories for this model
        trials = []
        for td in trial_dirs:
            if td.exists():
                trials.extend(_load_trials(td))
        if not trials:
            print(f"WARNING: No trials found for {model_label}")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_label}  ({len(trials)} trials)")
        print(f"{'='*60}")

        # ----- Deduplicate: keep latest trial per (artifact_id, condition) cell -----
        cell_best: dict[tuple[str, str], dict] = {}
        for t in trials:
            cond_internal = t["theta"]["citation_force_condition"]
            cond = _display_condition(cond_internal)
            aid = t["artifact_id"]
            is_to = detect_timeout(t.get("response_text", ""))
            entry = {**t, "_display_cond": cond, "_is_timeout": is_to}

            key = (aid, cond)
            prev = cell_best.get(key)
            if prev is None or t.get("created_at", "") > prev.get("created_at", ""):
                cell_best[key] = entry

        n_dupes = len(trials) - len(cell_best)
        if n_dupes:
            print(f"  (deduplicated {n_dupes} duplicate cells, keeping latest per artifact×condition)")

        # ----- Group by condition and artifact -----
        by_condition: dict[str, list[dict]] = defaultdict(list)
        by_artifact: dict[str, dict[str, dict]] = defaultdict(dict)
        for (aid, cond), entry in sorted(cell_best.items()):
            by_condition[cond].append(entry)
            by_artifact[aid][cond] = entry

        conditions_sorted = sorted(by_condition.keys())

        # ----- Aggregate per condition -----
        print(f"\n--- Aggregate ---")
        for cond in conditions_sorted:
            cond_trials = by_condition[cond]
            n = len(cond_trials)
            n_timeout = sum(1 for t in cond_trials if t["_is_timeout"])
            n_non_timeout = n - n_timeout

            # Core metrics (on all trials)
            R_strict = sum(1 for t in cond_trials if t.get("R_binary") == 1) / n if n else 0
            I_strict_count = sum(1 for t in cond_trials if t.get("I_strict") == 1)
            F_strict_count = sum(1 for t in cond_trials if t.get("F_strict") == 1)
            D_count = sum(1 for t in cond_trials if t.get("D") == 1)

            # Citation count (narrow, on non-timeout non-refusal responses only)
            responded = [
                t for t in cond_trials
                if not t["_is_timeout"] and not t.get("D")
            ]
            cite_count = sum(
                narrow_citation_count(t.get("response_text", ""))
                for t in responded
            )

            # artifact_recall and compliance metrics
            non_timeout_trials = [t for t in cond_trials if not t["_is_timeout"]]
            ar_count = sum(1 for t in non_timeout_trials if _has_artifact_recall(t.get("tool_trace", [])))
            ar_rate = ar_count / len(non_timeout_trials) if non_timeout_trials else 0
            compliance_count = sum(1 for t in non_timeout_trials if _is_protocol_compliant(t))
            compliance_rate = compliance_count / len(non_timeout_trials) if non_timeout_trials else 0

            row = {
                "model": model_label,
                "condition": cond,
                "n": n,
                "n_timeout": n_timeout,
                "n_non_timeout": n_non_timeout,
                "R_strict": round(R_strict, 2),
                "I_strict": f"{I_strict_count}/{n}",
                "F_strict": F_strict_count,
                "D": D_count,
                "citation_count": cite_count,
                "artifact_recall_calls": ar_count,
                "artifact_recall_rate": round(ar_rate, 2),
                "compliance_count": compliance_count,
                "compliance_rate": round(compliance_rate, 2),
            }
            all_aggregate_rows.append(row)
            print(f"  {cond}: n={n}, n_timeout={n_timeout}, R={R_strict:.2f}, "
                  f"I={I_strict_count}/{n}, F={F_strict_count}, D={D_count}, "
                  f"cite={cite_count}, ar_calls={ar_count} ({ar_rate:.0%}), "
                  f"compliance={compliance_count} ({compliance_rate:.0%})")

        # ----- Paired per-artifact analysis -----
        # Run paired analysis for each comparison: C0→Cp, C0→Cm, Cp→Cm
        comparison_pairs = [
            ("C0", "Cp"),
            ("C0", "Cm"),
            ("Cp", "Cm"),
        ]

        for base_cond, treat_cond in comparison_pairs:
            if base_cond not in by_condition or treat_cond not in by_condition:
                continue

            print(f"\n--- Paired Analysis: {base_cond} → {treat_cond} ---")

            n_total = len(by_artifact)
            n_timeout_excluded = 0
            n_missing = 0
            clean_pairs = []

            for aid, cond_map in sorted(by_artifact.items()):
                if base_cond not in cond_map or treat_cond not in cond_map:
                    n_missing += 1
                    continue
                t_base = cond_map[base_cond]
                t_treat = cond_map[treat_cond]
                if t_base["_is_timeout"] or t_treat["_is_timeout"]:
                    n_timeout_excluded += 1
                    continue
                clean_pairs.append((aid, t_base, t_treat))

            n_clean = len(clean_pairs)
            print(f"  Artifacts: n_total={n_total}, n_missing_pair={n_missing}, "
                  f"n_timeout_excluded={n_timeout_excluded}, n_clean_pairs={n_clean}")

            if n_clean == 0:
                continue

            metrics = [
                ("F_strict", lambda t: t.get("F_strict", 0)),
                ("I_strict", lambda t: t.get("I_strict", 0)),
                ("D", lambda t: t.get("D", 0)),
            ]

            paired_row = {
                "model": model_label,
                "comparison": f"{base_cond}→{treat_cond}",
                "n_total": n_total,
                "n_timeout_excluded": n_timeout_excluded,
                "n_clean_pairs": n_clean,
            }

            for metric_name, get_val in metrics:
                trans = _compute_transitions(clean_pairs, metric_name, get_val)
                for k, v in trans.items():
                    paired_row[f"{metric_name}_{k}"] = v
                print(f"  {metric_name}: improved={trans['improved']}, degraded={trans['degraded']}, "
                      f"both_pass={trans['both_pass']}, both_fail={trans['both_fail']}")

            all_paired_rows.append(paired_row)

            # McNemar's test
            print(f"\n  McNemar's Test ({base_cond}→{treat_cond}):")
            for metric_name, get_val in metrics:
                trans = _compute_transitions(clean_pairs, metric_name, get_val)
                if metric_name == "D":
                    b = trans["improved"]
                    c = trans["degraded"]
                else:
                    b = trans["improved"]
                    c = trans["degraded"]
                p_val = _mcnemar_exact_p(b, c)
                print(f"    {metric_name}: b={b}, c={c}, McNemar p={p_val:.4f}")

        # ----- Grounding signal breakdown -----
        print(f"\n--- Grounding Signal Breakdown ---")
        for cond in conditions_sorted:
            quality_counts: dict[str, int] = defaultdict(int)
            cond_trials = by_condition[cond]
            for t in cond_trials:
                resp = t.get("response_text", "")
                if t["_is_timeout"]:
                    quality_counts["timeout"] += 1
                else:
                    signals = analyze_response_signals(resp)
                    quality_counts[signals["grounding_quality"]] += 1

            row = {
                "model": model_label,
                "condition": cond,
                "n": len(cond_trials),
            }
            for q in ["cited", "vague", "refusal", "none", "timeout"]:
                row[q] = quality_counts.get(q, 0)

            # Narrow citation stats (non-timeout, non-refusal only)
            non_to_non_d = [
                t for t in cond_trials
                if not t["_is_timeout"] and not t.get("D")
            ]
            n_responded = len(non_to_non_d)
            n_with_narrow_cite = sum(
                1 for t in non_to_non_d
                if narrow_citation_count(t.get("response_text", "")) > 0
            )

            # artifact_recall usage among non-timeout
            non_timeout = [t for t in cond_trials if not t["_is_timeout"]]
            ar_used = sum(1 for t in non_timeout if _has_artifact_recall(t.get("tool_trace", [])))
            ms_used = sum(1 for t in non_timeout if _has_memory_search(t.get("tool_trace", [])))

            row["n_responded"] = n_responded
            row["n_with_bracket_citation"] = n_with_narrow_cite
            row["artifact_recall_used"] = ar_used
            row["memory_search_used"] = ms_used

            all_signal_rows.append(row)
            parts = ", ".join(f"{q}={quality_counts.get(q, 0)}" for q in ["cited", "vague", "refusal", "none", "timeout"])
            print(f"  {cond}: {parts}  (bracket-cited: {n_with_narrow_cite}/{n_responded}, "
                  f"ar_used: {ar_used}/{len(non_timeout)}, ms_used: {ms_used}/{len(non_timeout)})")

    # ----- Write CSVs -----
    _write_csv(output_dir / "table_mitigation_aggregate.csv", all_aggregate_rows)
    _write_csv(output_dir / "table_mitigation_paired.csv", all_paired_rows)
    _write_csv(output_dir / "table_mitigation_signals.csv", all_signal_rows)

    print(f"\n{'='*60}")
    print(f"Output written to {output_dir}/")
    for name in ["table_mitigation_aggregate.csv", "table_mitigation_paired.csv", "table_mitigation_signals.csv"]:
        print(f"  {name}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    # Collect all keys across all rows
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt5-dir", type=Path, default=DEFAULT_GPT5_DIR,
                        help="GPT-5 C0+Cp trial directory")
    parser.add_argument("--qwen-dir", type=Path, default=DEFAULT_QWEN_DIR,
                        help="Qwen C0+Cp trial directory")
    parser.add_argument("--gpt5-cm-dir", type=Path, default=DEFAULT_GPT5_CM_DIR,
                        help="GPT-5 Cm trial directory")
    parser.add_argument("--qwen-cm-dir", type=Path, default=DEFAULT_QWEN_CM_DIR,
                        help="Qwen Cm trial directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model_dirs = []

    # GPT-5: merge C0+Cp dir and Cm dir
    gpt5_dirs = []
    if args.gpt5_dir.exists():
        gpt5_dirs.append(args.gpt5_dir)
    if args.gpt5_cm_dir.exists():
        gpt5_dirs.append(args.gpt5_cm_dir)
    if gpt5_dirs:
        label = _model_label(args.gpt5_dir) if args.gpt5_dir.exists() else "GPT-5"
        model_dirs.append((label, gpt5_dirs))
    else:
        print("WARNING: No GPT-5 trial directories found")

    # Qwen: merge C0+Cp dir and Cm dir
    qwen_dirs = []
    if args.qwen_dir.exists():
        qwen_dirs.append(args.qwen_dir)
    if args.qwen_cm_dir.exists():
        qwen_dirs.append(args.qwen_cm_dir)
    if qwen_dirs:
        label = _model_label(args.qwen_dir) if args.qwen_dir.exists() else "Qwen3-VL-32B"
        model_dirs.append((label, qwen_dirs))
    else:
        print("WARNING: No Qwen trial directories found")

    if not model_dirs:
        print("ERROR: No trial directories found.")
        sys.exit(1)

    run_analysis(model_dirs, args.output)


if __name__ == "__main__":
    main()
