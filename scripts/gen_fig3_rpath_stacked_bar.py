"""
gen_fig3_rpath_stacked_bar.py — Generate Figure 3: R_path composition stacked bar chart.

Reads both summary_00300.json files and produces a 2-panel stacked bar chart
(GPT-5 | Qwen3-VL-32B) with 3 bars each (S1, S2, S3) and 5 stacks.

Output: logs/paper/figures/fig3_rpath_composition.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──
BASE = Path(__file__).resolve().parent.parent
GPT5_SUMMARY = BASE / "results" / "paper_main_gpt5_0322" / "summary_00300.json"
QWEN_SUMMARY = BASE / "results" / "paper_main_qwen3vl32b_0322" / "summary_00300.json"
OUT_DIR = BASE / "logs" / "paper" / "figures"
OUT_PDF = OUT_DIR / "fig3_rpath_composition.pdf"

# ── R_path key mapping ──
STACK_KEYS = ["R_context", "heuristic_R_context", "R_tool", "R_boot", "R_none"]
STACK_LABELS = [r"$R_{\rm context}$", r"$R_{\rm heur}$", r"$R_{\rm tool}$",
                r"$R_{\rm boot}$", r"$R_{\rm none}$"]
COLORS = ["#4477AA", "#88CCEE", "#228833", "#EE7733", "#CC3311"]  # colorblind-safe
SCENARIOS = ["S1", "S2", "S3"]


def load_rpath_data(summary_path: Path) -> list[list[int]]:
    """Return a list of 3 lists (one per scenario), each with 5 counts."""
    with summary_path.open() as f:
        data = json.load(f)
    rows = []
    for sc in SCENARIOS:
        rp = data["by_scenario"][sc]["R_path"]
        row = [rp.get(k, 0) for k in STACK_KEYS]
        rows.append(row)
    return rows


def main() -> None:
    gpt5_data = load_rpath_data(GPT5_SUMMARY)
    qwen_data = load_rpath_data(QWEN_SUMMARY)

    # ── Plot ──
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.2), sharey=True)

    for ax, data, title in zip(axes, [gpt5_data, qwen_data],
                                ["GPT-5", "Qwen3-VL-32B"]):
        x = np.arange(len(SCENARIOS))
        bottom = np.zeros(len(SCENARIOS))
        arr = np.array(data)  # shape (3, 5)

        for i in range(len(STACK_KEYS)):
            vals = arr[:, i]
            ax.bar(x, vals, bottom=bottom, width=0.55,
                   color=COLORS[i], label=STACK_LABELS[i], edgecolor="white",
                   linewidth=0.3)
            # Add count labels for non-zero bars
            for j, v in enumerate(vals):
                if v >= 5:
                    ax.text(x[j], bottom[j] + v / 2, str(int(v)),
                            ha="center", va="center", fontsize=6,
                            color="white" if COLORS[i] in ["#4477AA", "#228833", "#CC3311"] else "black")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Trials" if ax == axes[0] else "")

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    print(f"Saved: {OUT_PDF}")
    plt.close(fig)


if __name__ == "__main__":
    main()
