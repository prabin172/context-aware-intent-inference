#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

REPORT_DIR = Path("docs/experiment_reports/final_method_comparison")

METRICS_CSV = REPORT_DIR / "final_ambiguity_metrics_recomputed.csv"
STATS_CSV = REPORT_DIR / "final_ambiguity_stat_tests.csv"

OUT_PDF = REPORT_DIR / "figure9_ambiguity_macro_f1.pdf"
OUT_PNG = REPORT_DIR / "figure9_ambiguity_macro_f1.png"

metrics = pd.read_csv(METRICS_CSV)
stats = pd.read_csv(STATS_CSV)

methods = ["GR", "BF", "E2E"]
values = [
    float(metrics.loc[metrics["Method"] == m, "macro_f1"].iloc[0])
    for m in methods
]

def get_pair_p(comparison):
    row = stats[stats["comparison"] == comparison]
    if row.empty:
        return None
    if "p_holm" in row.columns and pd.notna(row["p_holm"].iloc[0]):
        return float(row["p_holm"].iloc[0])
    return float(row["p_value"].iloc[0])

def sig_label(p):
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."

p_gr_bf = get_pair_p("GR_vs_BF")
p_gr_e2e = get_pair_p("GR_vs_E2E")
p_bf_e2e = get_pair_p("BF_vs_E2E")

fig, ax = plt.subplots(figsize=(6.2, 4.2))

colors = {
    "GR": "#4C97D8",
    "BF": "#E67E22",
    "E2E": "#43B581",
}
hatches = {
    "GR": "//",
    "BF": "\\\\",
    "E2E": "xx",
}

x = range(len(methods))

bars = ax.bar(
    x,
    values,
    width=0.56,
    color="white",
    edgecolor=[colors[m] for m in methods],
    linewidth=1.3,
)

for bar, method in zip(bars, methods):
    bar.set_hatch(hatches[method])

for i, v in enumerate(values):
    ax.text(
        i,
        v + 0.018,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax.set_title("Macro F1-Score on GR Ambiguous Cases (≤ 25.0%)", fontsize=12)
ax.set_xlabel("Method", fontsize=11)
ax.set_ylabel("Macro F1-Score", fontsize=11)
ax.set_xticks(list(x))
ax.set_xticklabels(methods, fontsize=10)
ax.tick_params(axis="y", labelsize=10)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle=":", alpha=0.55)

def add_sig(ax, x1, x2, y, label, h=0.025):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.9, c="black")
    ax.text(
        (x1 + x2) / 2,
        y + h + 0.012,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
    )

# Same style as old Fig. 9, but using current full-LOSO stats.
# Match old Fig. 9 style:
# GR-BF and BF-E2E on the lower row; GR-E2E on the top row.
add_sig(ax, 0, 1, 0.80, sig_label(p_gr_bf))
add_sig(ax, 1, 2, 0.80, sig_label(p_bf_e2e))
add_sig(ax, 0, 2, 0.87, sig_label(p_gr_e2e))

ax.text(-0.36, 0.96, "*** p < 0.001", fontsize=9)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

print("Saved:", OUT_PNG)
print("Saved:", OUT_PDF)
print()
print("Values:")
for m, v in zip(methods, values):
    print(f"{m}: {v:.3f}")
print()
print("Pairwise labels:")
print("GR vs BF:", sig_label(p_gr_bf), p_gr_bf)
print("GR vs E2E:", sig_label(p_gr_e2e), p_gr_e2e)
print("BF vs E2E:", sig_label(p_bf_e2e), p_bf_e2e)
