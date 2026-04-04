"""Scorer for the T × M personal memory study.

Metrics:
    Acc(M0/M1/M2/M3)   per-condition accuracy (by task type and overall)
    Memory Gain         Acc(M1) - Acc(M0)
    Stale Penalty       Acc(M1) - Acc(M2)
    Noise Penalty       Acc(M1) - Acc(M3)
    UC                  Update Consistency (T2: follows latest preference)
    SER                 Stale-Memory Error Rate (T2 × M2 diagnosis)
    OPR                 Over-Personalization Rate (T3 × M3 diagnosis)
"""

from __future__ import annotations
from collections import defaultdict


def resolve_gold_legacy(gold_map: dict, condition: str) -> str | None:
    """Legacy compatibility: resolve gold from old C0/C1/C2-* key format.

    Only use this when loading old-format case banks (personal_memory_cases_v1.json).
    Canonical T×M cases should always have M0/M1/M2/M3 keys directly.
    """
    return (gold_map.get(condition)
            or gold_map.get("C2-correct")
            or gold_map.get("M1")
            or gold_map.get("C1"))


def score_trial(case: dict, condition: str, parsed: dict) -> dict:
    """Score a single (case, condition) trial.

    gold_action must be a per-condition dict: case["gold_action"][condition].
    Canonical conditions are M0/M1/M2/M3. Legacy C0/C1/C2-* keys are NOT
    used by default — if needed, call resolve_gold_legacy() explicitly.
    """
    gold_map = case["gold_action"]
    # Mitigation conditions share gold with their base condition
    _MIT_BASE = {"M2-mit": "M2", "M3-mit": "M3"}
    lookup_key = _MIT_BASE.get(condition, condition)
    gold = gold_map.get(lookup_key)
    if gold is None:
        raise KeyError(
            f"No gold_action for condition '{lookup_key}' (from '{condition}') "
            f"in case '{case.get('case_id')}'. "
            f"Available keys: {list(gold_map.keys())}."
        )
    decision = parsed.get("decision")
    return {
        "correct": decision == gold if parsed["parse_ok"] else None,
        "decision": decision,
        "gold": gold,
        "memory_used": parsed.get("memory_used"),
        "parse_ok": parsed["parse_ok"],
    }


def compute_summary(results: list[dict]) -> dict:
    """Compute T × M aggregate metrics.

    Infra failures (failure_type == "infra") are counted separately and
    excluded from accuracy / parse-failure denominators. Only benchmark
    trials (successful agent execution) are included in metrics.

    Each result dict must have: case_id, condition, task_type, correct,
    decision, gold, parse_ok, domain.
    For SER/OPR: stale_aligned, noisy_aligned, t3_scope fields.
    """
    # Separate infra failures from benchmark trials
    n_infra = sum(1 for r in results if r.get("failure_type") == "infra")
    bench_results = [r for r in results if r.get("failure_type") != "infra"]

    def _acc(trials):
        scored = [t for t in trials if t.get("correct") is not None]
        return round(sum(t["correct"] for t in scored) / len(scored), 3) if scored else None

    # Group by condition (benchmark trials only)
    by_cond = defaultdict(list)
    for r in bench_results:
        by_cond[r["condition"]].append(r)

    # Group by task_type × condition
    by_task_cond = defaultdict(lambda: defaultdict(list))
    for r in bench_results:
        by_task_cond[r.get("task_type", "?")][r["condition"]].append(r)

    # Group by domain × condition
    by_domain_cond = defaultdict(lambda: defaultdict(list))
    for r in bench_results:
        by_domain_cond[r.get("domain", "?")][r["condition"]].append(r)

    # Per-condition accuracy
    cond_acc = {c: _acc(by_cond[c]) for c in ["M0", "M1", "M2", "M3"]}

    # Derived metrics
    def _delta(a, b):
        if a is not None and b is not None:
            return round(a - b, 3)
        return None

    memory_gain = _delta(cond_acc["M1"], cond_acc["M0"])
    stale_penalty = _delta(cond_acc["M1"], cond_acc["M2"])
    noise_penalty = _delta(cond_acc["M1"], cond_acc["M3"])

    # Per task_type × condition
    task_cond_acc = {}
    for tt in ["T1", "T2", "T3"]:
        task_cond_acc[tt] = {c: _acc(by_task_cond[tt][c]) for c in ["M0", "M1", "M2", "M3"]}

    # SER: Stale-memory error rate — T2 only.
    # This is the core diagnostic subset for stale-memory effects.
    # M2-mit pilot targeting also uses T2 only, keeping both aligned.
    case_ids = {r["case_id"] for r in bench_results}
    ser_eligible, ser_errors = 0, 0
    for cid in case_ids:
        m1s = [r for r in bench_results if r["case_id"] == cid and r["condition"] == "M1"
               and r.get("task_type") == "T2"]
        m2s = [r for r in bench_results if r["case_id"] == cid and r["condition"] == "M2"
               and r.get("task_type") == "T2"]
        if not m1s or not m2s:
            continue
        m1, m2 = m1s[0], m2s[0]
        if m1.get("correct") is None or m2.get("correct") is None:
            continue
        if m1["correct"]:
            ser_eligible += 1
            if not m2["correct"] and m2.get("decision") == m2.get("stale_aligned"):
                ser_errors += 1
    ser = round(ser_errors / ser_eligible, 3) if ser_eligible > 0 else None

    # OPR: T3 out-of-scope cases where (M0 or M1 correct) AND M3 wrong
    # AND M3 decision == noisy_aligned
    opr_eligible, opr_errors = 0, 0
    for cid in case_ids:
        m3s = [r for r in bench_results if r["case_id"] == cid and r["condition"] == "M3"
               and r.get("task_type") == "T3" and r.get("t3_scope") == "out_of_scope"]
        if not m3s:
            continue
        m3 = m3s[0]
        if m3.get("correct") is None:
            continue
        # Check baseline
        baselines = [r for r in bench_results if r["case_id"] == cid
                     and r["condition"] in ("M0", "M1") and r.get("correct")]
        if baselines:
            opr_eligible += 1
            if not m3["correct"] and m3.get("decision") == m3.get("noisy_aligned"):
                opr_errors += 1
    opr = round(opr_errors / opr_eligible, 3) if opr_eligible > 0 else None

    # UC: T2 × M1 — does the agent follow the updated preference?
    t2_m1 = [r for r in bench_results if r.get("task_type") == "T2" and r["condition"] == "M1"]
    uc = _acc(t2_m1)

    # Mitigation pilot: before/after comparison on targeted subsets
    mit_summary = {}
    for mit_cond in ["M2-mit", "M3-mit"]:
        mit_trials = [r for r in bench_results if r["condition"] == mit_cond]
        if not mit_trials:
            continue
        base_cond = "M2" if mit_cond == "M2-mit" else "M3"
        # Get matching base trials (same case_ids)
        mit_case_ids = {r["case_id"] for r in mit_trials}
        base_trials = [r for r in bench_results
                       if r["condition"] == base_cond and r["case_id"] in mit_case_ids]
        mit_summary[mit_cond] = {
            "base_condition": base_cond,
            "n_trials": len(mit_trials),
            "acc_before": _acc(base_trials),
            "acc_after": _acc(mit_trials),
            "delta": (_delta(_acc(mit_trials), _acc(base_trials))
                      if _acc(mit_trials) is not None and _acc(base_trials) is not None
                      else None),
        }

    return {
        "n_trials_total": len(results),
        "n_infra_failures": n_infra,
        "n_benchmark_trials": len(bench_results),
        "n_parse_failures": sum(1 for r in bench_results if not r.get("parse_ok", True)),
        "condition_accuracy": cond_acc,
        "memory_gain": memory_gain,
        "stale_penalty": stale_penalty,
        "noise_penalty": noise_penalty,
        "task_condition_accuracy": task_cond_acc,
        "domain_condition_accuracy": {
            dom: {c: _acc(trials) for c, trials in conds.items()}
            for dom, conds in by_domain_cond.items()
        },
        "UC": uc,
        "SER": ser,
        "OPR": opr,
        "mitigation_pilots": mit_summary,
    }


def print_summary(s: dict) -> None:
    """Pretty-print T × M summary."""
    def _f(v):
        return f"{v:.3f}" if v is not None else "  -  "

    print(f"\n{'=' * 65}")
    print("T × M PERSONAL MEMORY STUDY SUMMARY")
    print(f"{'=' * 65}")
    print(f"Total trials: {s.get('n_trials_total', s.get('n_trials', '?'))}")
    print(f"  Infra failures: {s.get('n_infra_failures', 0)} (excluded from metrics)")
    print(f"  Benchmark trials: {s.get('n_benchmark_trials', '?')}, "
          f"parse failures: {s['n_parse_failures']}")

    # Overall accuracy by condition
    print(f"\n--- Accuracy by Memory Condition ---")
    for c in ["M0", "M1", "M2", "M3"]:
        print(f"  {c}: {_f(s['condition_accuracy'].get(c))}")

    # Derived metrics
    print(f"\n--- Derived Metrics ---")
    print(f"  Memory Gain (M1-M0):  {_f(s.get('memory_gain'))}")
    print(f"  Stale Penalty (M1-M2): {_f(s.get('stale_penalty'))}")
    print(f"  Noise Penalty (M1-M3): {_f(s.get('noise_penalty'))}")
    print(f"  UC (T2 update consistency): {_f(s.get('UC'))}")
    print(f"  SER (stale-error rate):     {_f(s.get('SER'))}")
    print(f"  OPR (over-personalization): {_f(s.get('OPR'))}")

    # Per task type
    tca = s.get("task_condition_accuracy", {})
    if tca:
        print(f"\n--- Accuracy by Task × Condition ---")
        header = f"  {'Task':<5s} {'M0':>6s} {'M1':>6s} {'M2':>6s} {'M3':>6s}"
        print(header)
        for tt in ["T1", "T2", "T3"]:
            if tt in tca:
                row = tca[tt]
                print(f"  {tt:<5s} {_f(row.get('M0')):>6s} {_f(row.get('M1')):>6s} "
                      f"{_f(row.get('M2')):>6s} {_f(row.get('M3')):>6s}")

    # Mitigation pilots
    mit = s.get("mitigation_pilots", {})
    if mit:
        print(f"\n--- Mitigation Pilots (targeted before → after) ---")
        for mc, info in mit.items():
            print(f"  {mc}: {info['base_condition']} → {mc} "
                  f"(n={info['n_trials']}) "
                  f"Acc {_f(info['acc_before'])} → {_f(info['acc_after'])} "
                  f"Δ={_f(info['delta'])}")
