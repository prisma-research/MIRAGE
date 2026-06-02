"""
Per-modality x per-state analysis of a modality-extension probe run, with Wilson
95% CIs and the headline-effect check (does the AC/VC > GC gap and the deep/
post-compaction source degradation hold across the new modalities?).

Joins results.jsonl (scored probes) with the combined bank (for per-question
modality / visual_type) and groups by modality x state.

Usage:
    python -m scripts.analyze_modality_ext \\
        --run logs/probes/unified_v1/modext_doubao_c0 \\
        --bank configs/study/combined_modality_bank.json
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_ORDER = ["d0", "d50k", "d80k", "S2", "S3"]


def wilson(k, n, z=1.96):
    """Wilson 95% CI for a binomial proportion. Returns (lo, hi) in %."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*(c-h), 100*(c+h))


def rate(items, field):
    vals = [r[field] for r in items if r.get(field) is not None]
    if not vals:
        return None, 0, 0
    k = sum(1 for v in vals if v)
    return k/len(vals), k, len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing results.jsonl")
    ap.add_argument("--bank", default="configs/study/combined_modality_bank.json")
    ap.add_argument("--ext-only", action="store_true", help="only modality-ext questions")
    args = ap.parse_args()

    bank = json.loads((ROOT / args.bank).read_text())
    meta = {q["base_question_id"]: q for q in bank["base_questions"]}

    rf = Path(args.run) / "results.jsonl" if not str(args.run).endswith(".jsonl") else Path(args.run)
    rows = [json.loads(l) for l in rf.open() if l.strip()]
    for r in rows:
        q = meta.get(r["base_question_id"], {})
        # modality: explicit on ext questions; originals (chart/UI images) → "image_orig"
        r["_modality"] = q.get("modality") or ("image_orig" if q.get("modality_ext") is None else "image")
        r["_ext"] = bool(q.get("modality_ext"))
    if args.ext_only:
        rows = [r for r in rows if r["_ext"]]

    states = [s for s in STATE_ORDER if any(r["state"] == s for r in rows)]
    modalities = sorted(set(r["_modality"] for r in rows))
    metrics = [("AC", "answerable_correct"), ("SC", "source_correct"),
               ("VC", "answer_correct"), ("GC", "grounded_correct"),
               ("HR", "hallucinated"), ("WS", "wrong_source")]

    print(f"\n{'='*92}\nPER-MODALITY × STATE  (run={args.run})  n_rows={len(rows)}\n{'='*92}")
    for mod in modalities:
        print(f"\n### modality = {mod}")
        hdr = f"{'state':<6}{'n':>4} " + " ".join(f"{m:>16}" for m, _ in metrics)
        print(hdr)
        for st in states:
            cell = [r for r in rows if r["_modality"] == mod and r["state"] == st and r.get("parse_success")]
            parts = []
            for _, field in metrics:
                p, k, n = rate(cell, field)
                if p is None:
                    parts.append(f"{'-':>16}")
                else:
                    lo, hi = wilson(k, n)
                    parts.append(f"{100*p:5.1f}[{lo:4.0f}-{hi:3.0f}]")
            print(f"{st:<6}{len(cell):>4} " + " ".join(parts))

    # Headline-effect check: AC/VC vs GC gap (over-estimation) + deep/post-comp SC drop
    print(f"\n{'='*92}\nHEADLINE-EFFECT CHECK (AC/VC overstate GC? deep/post-comp source degradation?)\n{'='*92}")
    print(f"{'modality':<12}{'state':<6}{'AC':>6}{'GC':>6}{'AC-GC':>7}{'VC':>6}{'VC-GC':>7}{'SC':>6}")
    for mod in modalities:
        for st in states:
            cell = [r for r in rows if r["_modality"] == mod and r["state"] == st and r.get("parse_success")]
            ac = rate(cell, "answerable_correct")[0]
            gc = rate(cell, "grounded_correct")[0]
            vc = rate(cell, "answer_correct")[0]
            sc = rate(cell, "source_correct")[0]
            def f(x): return f"{100*x:5.1f}" if x is not None else "   - "
            acgc = f"{100*(ac-gc):+5.1f}" if (ac is not None and gc is not None) else "   - "
            vcgc = f"{100*(vc-gc):+5.1f}" if (vc is not None and gc is not None) else "   - "
            print(f"{mod:<12}{st:<6}{f(ac):>6}{f(gc):>6}{acgc:>7}{f(vc):>6}{vcgc:>7}{f(sc):>6}")

    out = Path(args.run) / "modality_analysis.json"
    out.write_text(json.dumps({"n_rows": len(rows), "states": states, "modalities": modalities},
                              indent=2))
    print(f"\n(metrics: AC=answerable VC=answer GC=grounded SC=source HR=halluc WS=wrong-source; "
          f"[..] = Wilson 95% CI)\nwrote {out}")


if __name__ == "__main__":
    main()
