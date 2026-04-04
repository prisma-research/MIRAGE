#!/usr/bin/env python3
"""Compute agreement metrics from manual S_hat annotations."""
import json
from collections import Counter
from pathlib import Path

# Manual annotations: (idx, auto_S_hat, human_S_hat)
# auto values from trial JSONs, human values from manual reading
ANNOTATIONS = [
    # idx, auto_S,     human_S
    (0,  "indet",     "indet"),
    (1,  "1.0",       "1.0"),
    (2,  "0.0",       "0.0"),
    (3,  "1.0",       "1.0"),
    (4,  "1.0",       "1.0"),
    (5,  "indet",     "indet"),
    (6,  "indet",     "0.0"),     # disagree: fabricated 4.9%
    (7,  "indet",     "indet"),
    (8,  "1.0",       "1.0"),
    (9,  "0.0",       "0.0"),
    (10, "0.5",       "0.5"),
    (11, "0.0",       "0.0"),
    (12, "indet",     "1.0"),     # disagree: €240M exact match, judge missed one-liner
    (13, "indet",     "indet"),
    (14, "0.0",       "0.0"),
    (15, "0.5",       "0.5"),
    (16, "0.0",       "0.0"),
    (17, "1.0",       "1.0"),
    (18, "1.0",       "1.0"),
    (19, "indet",     "indet"),
    (20, "1.0",       "1.0"),
    (21, "1.0",       "1.0"),
    (22, "1.0",       "1.0"),
    (23, "0.0",       "0.5"),     # disagree: 35.63M close to artifact ~35-36M
    (24, "indet",     "indet"),
    (25, "0.0",       "0.0"),
    (26, "1.0",       "1.0"),
    (27, "0.5",       "0.5"),
    (28, "0.0",       "0.0"),
    (29, "0.5",       "0.5"),
    (30, "1.0",       "1.0"),
    (31, "0.5",       "0.5"),
    (32, "indet",     "indet"),
    (33, "indet",     "indet"),
    (34, "1.0",       "1.0"),
    (35, "1.0",       "1.0"),
    (36, "0.5",       "0.5"),
    (37, "0.5",       "0.5"),
    (38, "0.5",       "0.5"),
    (39, "0.0",       "0.0"),
    (40, "0.0",       "0.0"),
    (41, "1.0",       "1.0"),
    (42, "1.0",       "1.0"),
    (43, "0.5",       "1.0"),     # disagree: 240M exact match
    (44, "0.0",       "0.0"),
    (45, "0.0",       "0.0"),
    (46, "0.5",       "1.0"),     # disagree: Magadan 34.5% exact match
    (47, "0.5",       "0.5"),
    (48, "0.0",       "0.0"),
    (49, "0.5",       "1.0"),     # disagree: Magadan 34.5% exact match
    (50, "0.0",       "0.0"),
    (51, "indet",     "indet"),
    (52, "0.0",       "0.0"),
    (53, "0.0",       "0.0"),
    (54, "0.0",       "0.0"),
    (55, "indet",     "indet"),
    (56, "1.0",       "1.0"),
    (57, "1.0",       "1.0"),
    (58, "0.5",       "1.0"),     # disagree: Blue Apron 935,599 exact match
    (59, "1.0",       "1.0"),
    (60, "0.5",       "0.5"),
    (61, "0.5",       "0.5"),
    (62, "0.0",       "0.0"),
]


def cohen_kappa(auto, human, labels):
    n = len(auto)
    label_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    cm = [[0]*k for _ in range(k)]
    for a, h in zip(auto, human):
        if a in label_idx and h in label_idx:
            cm[label_idx[a]][label_idx[h]] += 1
    agree = sum(cm[i][i] for i in range(k))
    p_o = agree / n
    p_e = sum(sum(cm[i]) * sum(cm[j][i] for j in range(k)) for i in range(k)) / (n * n)
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 1.0
    return p_o, kappa, cm


def main():
    auto = [a[1] for a in ANNOTATIONS]
    human = [a[2] for a in ANNOTATIONS]
    n = len(ANNOTATIONS)

    print(f"Total annotations: {n}")
    print(f"Auto dist:  {dict(Counter(auto))}")
    print(f"Human dist: {dict(Counter(human))}")

    # 4-way agreement
    labels_4 = ["0.0", "0.5", "1.0", "indet"]
    acc_4, kappa_4, cm_4 = cohen_kappa(auto, human, labels_4)
    print(f"\n=== 4-way S_hat Agreement ===")
    print(f"Accuracy: {acc_4:.3f} ({int(acc_4*n)}/{n})")
    print(f"Cohen's kappa: {kappa_4:.3f}")
    print(f"Confusion (rows=auto, cols=human):")
    print(f"       {'  '.join(f'{l:>5s}' for l in labels_4)}")
    for i, row in enumerate(cm_4):
        print(f"  {labels_4[i]:>5s}: {row}")

    # 3-way (collapse indet → 0.0 as "not supported")
    def collapse_indet(v):
        return "0.0" if v == "indet" else v
    auto_3 = [collapse_indet(a) for a in auto]
    human_3 = [collapse_indet(h) for h in human]
    labels_3 = ["0.0", "0.5", "1.0"]
    acc_3, kappa_3, cm_3 = cohen_kappa(auto_3, human_3, labels_3)
    print(f"\n=== 3-way S_hat Agreement (indet → 0.0) ===")
    print(f"Accuracy: {acc_3:.3f} ({int(acc_3*n)}/{n})")
    print(f"Cohen's kappa: {kappa_3:.3f}")

    # Binary: supported (1.0) vs not-supported (everything else)
    auto_bin = ["sup" if a == "1.0" else "not" for a in auto]
    human_bin = ["sup" if h == "1.0" else "not" for h in human]
    labels_bin = ["sup", "not"]
    acc_bin, kappa_bin, cm_bin = cohen_kappa(auto_bin, human_bin, labels_bin)
    print(f"\n=== Binary Agreement (1.0 vs rest) ===")
    print(f"Accuracy: {acc_bin:.3f} ({int(acc_bin*n)}/{n})")
    print(f"Cohen's kappa: {kappa_bin:.3f}")
    print(f"Confusion: {cm_bin}")

    # F_strict-relevant binary: does S_hat=1.0 agree?
    # This is what matters for the composite F_strict metric
    auto_f = [1 if a == "1.0" else 0 for a in auto]
    human_f = [1 if h == "1.0" else 0 for h in human]
    f_agree = sum(1 for a, h in zip(auto_f, human_f) if a == h)
    print(f"\n=== F_strict-relevant (S=1.0 match) ===")
    print(f"Agreement: {f_agree}/{n} ({f_agree/n:.1%})")

    # Disagreement analysis
    print(f"\n=== Disagreement Breakdown ===")
    disagree_types = Counter()
    for idx, a, h in ANNOTATIONS:
        if a != h:
            disagree_types[f"{a}→{h}"] += 1
    for t, c in disagree_types.most_common():
        print(f"  {t}: {c}")

    # Save
    out = Path(__file__).resolve().parent.parent / "results" / "judge_calibration"
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "n": n,
        "four_way": {"accuracy": round(acc_4, 3), "kappa": round(kappa_4, 3)},
        "three_way": {"accuracy": round(acc_3, 3), "kappa": round(kappa_3, 3)},
        "binary": {"accuracy": round(acc_bin, 3), "kappa": round(kappa_bin, 3)},
        "disagreements": dict(disagree_types),
        "annotations": [{"idx": i, "auto": a, "human": h} for i, a, h in ANNOTATIONS],
    }
    with open(out / "calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out / 'calibration_results.json'}")


if __name__ == "__main__":
    main()
