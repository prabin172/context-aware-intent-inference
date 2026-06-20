#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

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

fig, ax = plt.subplots(figsize=(7.0, 3.6))

x = np.arange(len(PLOT_ORDER))
width = 0.24

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

ax.set_title("Per-Class F1-Score Across Three Methods", fontsize=11)
ax.set_ylabel("F1-Score", fontsize=10)
ax.set_xlabel("Ground Truth Intent Label", fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels([DISPLAY_LABELS[c] for c in PLOT_ORDER], rotation=35, ha="right", fontsize=8)
ax.tick_params(axis="y", labelsize=8)
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
    fontsize=8,
    frameon=True,
    columnspacing=1.0,
    handletextpad=0.5,
    borderpad=0.4,
)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

print("Saved:", OUT_PDF)
