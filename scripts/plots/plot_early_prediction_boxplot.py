#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.text import Text
from scipy.stats import ttest_rel, wilcoxon

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

REPORT_DIR = Path("docs/experiment_reports/final_method_comparison")
INPUT_CSV = REPORT_DIR / "final_early_prediction_lead_time.csv"
OUT_PDF = REPORT_DIR / "figure7_early_prediction_boxplot.pdf"

lead = pd.read_csv(INPUT_CSV)

valid = lead[
    (lead["final_prediction_correct"] == True)
    & lead["prediction_lead_time_ms"].notna()
].copy()

paired = valid.pivot_table(
    index=["Subject", "Session", "Global Trial", "Action_Label"],
    columns="Method",
    values="prediction_lead_time_ms",
    aggfunc="first",
).reset_index()

paired = paired.dropna(subset=["GR", "BF", "E2E"]).copy()

gr = paired["GR"]
bf = paired["BF"]
e2e = paired["E2E"]

means = {
    "GR": gr.mean(),
    "BF": bf.mean(),
    "E2E": e2e.mean(),
}

t_bf_gr = ttest_rel(bf, gr)
t_bf_e2e = ttest_rel(bf, e2e)
w_bf_gr = wilcoxon(bf, gr)
w_bf_e2e = wilcoxon(bf, e2e)

print("n_common_successful_trials:", len(paired))
print("GR mean lead:", means["GR"])
print("BF mean lead:", means["BF"])
print("E2E mean lead:", means["E2E"])
print()
print("BF vs GR paired t-test:", t_bf_gr)
print("BF vs GR Wilcoxon:", w_bf_gr)
print("BF vs E2E paired t-test:", t_bf_e2e)
print("BF vs E2E Wilcoxon:", w_bf_e2e)

fig, ax = plt.subplots(figsize=(5.6, 3.6))

bp = ax.boxplot(
    [gr, bf, e2e],
    tick_labels=["GR", "BF", "E2E"],
    patch_artist=True,
    widths=0.5,
    showfliers=False,
)

facecolors = ["#EAF3FF", "#FFF2E6", "#EAF9F1"]
edgecolors = ["#4C97D8", "#E67E22", "#43B581"]
hatches = ["//", "\\\\", "xx"]

for patch, fc, ec, hatch in zip(bp["boxes"], facecolors, edgecolors, hatches):
    patch.set_facecolor(fc)
    patch.set_edgecolor(ec)
    patch.set_hatch(hatch)
    patch.set_linewidth(1.2)

for whisker in bp["whiskers"]:
    whisker.set_color("#555555")
    whisker.set_linewidth(1.0)

for cap in bp["caps"]:
    cap.set_color("#555555")
    cap.set_linewidth(1.0)

for median in bp["medians"]:
    median.set_color("#333333")
    median.set_linewidth(1.2)

ax.set_title("Overall Prediction Lead Time by Method (Box Plot)", fontsize=11)
ax.set_xlabel("Method", fontsize=10)
ax.set_ylabel("Prediction Lead Time (ms)", fontsize=10)
ax.tick_params(axis="both", labelsize=9)
ax.grid(axis="y", linestyle=":", alpha=0.55)

for i, label in enumerate(["GR", "BF", "E2E"], start=1):
    val = means[label]
    ax.text(
        i,
        val + 8,
        f"Avg: {val:.1f} ms",
        ha="center",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2),
        color="#333333",
    )

def add_sig(ax, x1, x2, y, text, h=18):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.9, c="#333333")
    ax.text((x1 + x2) / 2, y + h + 4, text, ha="center", va="bottom", fontsize=9, color="#333333")

ax.set_ylim(100, 950)
ax.set_yticks([100, 300, 500, 700, 900])

add_sig(ax, 1, 2, 845, "***")
add_sig(ax, 2, 3, 845, "***")

ax.text(0.62, 915, "*** p < 0.001", fontsize=8)

for text_obj in fig.findobj(match=Text):
    text_obj.set_fontfamily("Times New Roman")

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")

print()
print("Saved:", OUT_PDF)
