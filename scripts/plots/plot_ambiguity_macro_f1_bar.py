#!/usr/bin/env python3

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from sklearn.metrics import accuracy_score, f1_score

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

REPORT_DIR = Path("docs/experiment_reports/final_method_comparison")
OUT_PDF = REPORT_DIR / "figure9_ambiguity_macro_f1.pdf"

THRESHOLD = 0.25
METHODS = ["GR", "BF", "E2E"]

spec = importlib.util.spec_from_file_location(
    "final_report",
    "scripts/create_final_method_comparison_report.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

method_dfs = {}
for method, template in mod.METHOD_PATHS.items():
    df = mod.load_method(method, template)
    df = mod.add_predictions(df, onset_ms=2000)
    method_dfs[method] = df

common = mod.merge_common_grid(method_dfs)
amb = common[common["GR_Confidence_Gap"] <= THRESHOLD].copy()

metric_rows = []
for method in METHODS:
    pred_col = f"{method}_Pred_Label"
    metric_rows.append({
        "Method": method,
        "accuracy": accuracy_score(amb["True_Label_Dynamic"], amb[pred_col]),
        "macro_f1": f1_score(
            amb["True_Label_Dynamic"],
            amb[pred_col],
            labels=mod.CLASSES,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            amb["True_Label_Dynamic"],
            amb[pred_col],
            labels=mod.CLASSES,
            average="weighted",
            zero_division=0,
        ),
    })

metrics = pd.DataFrame(metric_rows)

trial_rows = []
for keys, g in amb.groupby(["Subject", "Session", "Global Trial"]):
    subject, session, trial = keys
    for method in METHODS:
        pred_col = f"{method}_Pred_Label"
        trial_rows.append({
            "Subject": subject,
            "Session": session,
            "Global Trial": trial,
            "Method": method,
            "macro_f1": f1_score(
                g["True_Label_Dynamic"],
                g[pred_col],
                labels=mod.CLASSES,
                average="macro",
                zero_division=0,
            ),
        })

per_trial = pd.DataFrame(trial_rows)

arrays = [
    per_trial[per_trial["Method"] == method]["macro_f1"].dropna().to_numpy()
    for method in METHODS
]
h_stat, h_p = kruskal(*arrays)

pairs = [("GR", "BF"), ("GR", "E2E"), ("BF", "E2E")]
raw_pair = []
for a, b in pairs:
    av = per_trial[per_trial["Method"] == a]["macro_f1"].dropna()
    bv = per_trial[per_trial["Method"] == b]["macro_f1"].dropna()
    u_stat, p_val = mannwhitneyu(av, bv, alternative="two-sided")
    raw_pair.append((a, b, u_stat, p_val))

sorted_pair = sorted(enumerate(raw_pair), key=lambda x: x[1][3])
m = len(sorted_pair)
holm = {}
running_max = 0.0
for rank, (idx, (a, b, u_stat, p_val)) in enumerate(sorted_pair, start=1):
    adj = min(1.0, (m - rank + 1) * p_val)
    running_max = max(running_max, adj)
    holm[idx] = running_max

pair_p = {}
for idx, (a, b, u_stat, p_val) in enumerate(raw_pair):
    pair_p[f"{a}_vs_{b}"] = holm[idx]

def sig_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."

values = [
    float(metrics.loc[metrics["Method"] == m, "macro_f1"].iloc[0])
    for m in METHODS
]

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

x = range(len(METHODS))

bars = ax.bar(
    x,
    values,
    width=0.56,
    color="white",
    edgecolor=[colors[m] for m in METHODS],
    linewidth=1.3,
)

for bar, method in zip(bars, METHODS):
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
ax.set_xticklabels(METHODS, fontsize=10)
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

add_sig(ax, 0, 1, 0.80, sig_label(pair_p["GR_vs_BF"]))
add_sig(ax, 0, 2, 0.87, sig_label(pair_p["GR_vs_E2E"]))

ax.text(-0.36, 0.96, "*** p < 0.001", fontsize=9)

fig.tight_layout()
fig.savefig(OUT_PDF, bbox_inches="tight")

print("Common rows:", len(common))
print("Ambiguous rows:", len(amb))
print(f"Ambiguous percent: {100 * len(amb) / len(common):.3f}%")
print()
print("Values:")
for m, v in zip(METHODS, values):
    print(f"{m}: {v:.3f}")
print()
print("Kruskal-Wallis:", h_stat, h_p)
print("Pairwise Holm p-values:")
for k, v in pair_p.items():
    print(k, v)
print()
print("Saved:", OUT_PDF)
