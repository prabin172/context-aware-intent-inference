#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

REPORT_DIR = Path("docs/experiment_reports/final_method_comparison")

PER_CLASS_CSV = REPORT_DIR / "final_per_class_f1.csv"
OUT_PDF = REPORT_DIR / "per_class_f1_full_loso.pdf"

df = pd.read_csv(PER_CLASS_CSV)

PLOT_ORDER = [
    "Nothing",
    "Stand-on-couch",
    "Sit-on-table",
    "Pick-up-backpack",
    "Sit-on-couch",
    "Sit-on-chair",
    "Clean-table",
    "Push-chair",
    "Wear-backpack",
]

DISPLAY_LABELS = {
    "Clean-table": "Clean",
    "Nothing": "Nothing",
    "Pick-up-backpack": "Pick-up",
    "Push-chair": "Push-back",
    "Sit-on-chair": "Sit-on-chair",
    "Sit-on-couch": "Sit-on-couch",
    "Sit-on-table": "Sit-on-table",
    "Stand-on-couch": "Stand",
    "Wear-backpack": "Wear",
}

methods = ["GR", "BF", "E2E"]
colors = {"GR": "#4C97D8", "BF": "#E67E22", "E2E": "#43B581"}
hatches = {"GR": "///", "BF": "\\\\\\", "E2E": "xxx"}

# Slightly compact x spacing between class groups.
x = np.arange(len(PLOT_ORDER)) * 0.90
width = 0.26

fig, ax = plt.subplots(figsize=(8.2, 4.8))

for i, method in enumerate(methods):
    vals = []
    for cls in PLOT_ORDER:
        row = df[(df["Method"] == method) & (df["Class"] == cls)]
        vals.append(float(row["f1_score"].iloc[0]) if not row.empty else 0.0)

    ax.bar(
        x + (i - 1) * width,
        vals,
        width=width,
        color="white",
        edgecolor=colors[method],
        linewidth=1.1,
        hatch=hatches[method],
    )

ax.set_title("Per-Class F1-Score Across Three Methods", fontsize=19)
ax.set_ylabel("F1-Score", fontsize=17)
ax.set_xlabel("Ground Truth Intent Label", fontsize=17)
ax.set_ylim(0, 1.05)

ax.set_xticks(x)
ax.set_xticklabels(
    [DISPLAY_LABELS[c] for c in PLOT_ORDER],
    rotation=28,
    ha="right",
    fontsize=14,
)
ax.tick_params(axis="y", labelsize=14)

# Reduce blank space before the first group and after the last group.
ax.set_xlim(x[0] - 0.45, x[-1] + 0.45)

ax.grid(axis="y", linestyle=":", alpha=0.5)

legend_handles = [
    Patch(facecolor="white", edgecolor="white", alpha=0.0, label="Method"),
    Patch(facecolor="white", edgecolor=colors["GR"], hatch=hatches["GR"], label="GR"),
    Patch(facecolor="white", edgecolor=colors["BF"], hatch=hatches["BF"], label="BF"),
    Patch(facecolor="white", edgecolor=colors["E2E"], hatch=hatches["E2E"], label="E2E"),
]

ax.legend(
    handles=legend_handles,
    ncol=4,
    loc="upper right",
    fontsize=13,
    frameon=True,
    columnspacing=0.8,
    handletextpad=0.45,
    borderpad=0.35,
)

fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.34)
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT_PDF)
