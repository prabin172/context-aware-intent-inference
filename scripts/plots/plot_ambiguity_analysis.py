#!/usr/bin/env python3

from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import kruskal, mannwhitneyu

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

REPORT_DIR = Path("docs/experiment_reports/final_method_comparison")
OUT_PDF = REPORT_DIR / "figure9_ambiguity_macro_f1.pdf"
OUT_PNG = REPORT_DIR / "figure9_ambiguity_macro_f1.png"
OUT_METRICS = REPORT_DIR / "final_ambiguity_metrics_recomputed.csv"
OUT_PER_TRIAL = REPORT_DIR / "final_ambiguity_per_trial_f1.csv"
OUT_STATS = REPORT_DIR / "final_ambiguity_stat_tests.csv"

THRESHOLD = 0.25
METHODS = ["GR", "BF", "E2E"]

# Load helper functions from the final comparison script
spec = importlib.util.spec_from_file_location(
    "final_report",
    "scripts/create_final_method_comparison_report.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("Loading current full-LOSO prediction outputs...")
method_dfs = {}
for method, template in mod.METHOD_PATHS.items():
    df = mod.load_method(method, template)
    df = mod.add_predictions(df, onset_ms=2000)
    method_dfs[method] = df
    print(f"{method}: rows={len(df)}, trials={df[['Subject', 'Session', 'Global Trial']].drop_duplicates().shape[0]}")

print("\nMerging common timestamp grid...")
common = mod.merge_common_grid(method_dfs)

amb = common[common["GR_Confidence_Gap"] <= THRESHOLD].copy()

print(f"Common rows: {len(common)}")
print(f"Ambiguous rows: {len(amb)}")
print(f"Ambiguous percent: {100 * len(amb) / len(common):.3f}%")

# Overall ambiguity metrics
metric_rows = []
for method in METHODS:
    pred_col = f"{method}_Pred_Label"
    metric_rows.append({
        "Method": method,
        "gr_gap_threshold": THRESHOLD,
        "num_ambiguous_rows": len(amb),
        "percent_of_common_rows": 100 * len(amb) / len(common),
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
metrics.to_csv(OUT_METRICS, index=False)

print("\nAmbiguity metrics:")
print(metrics.to_string(index=False))

# Per-trial macro-F1 on ambiguous timestamps
trial_rows = []
key_cols = ["Subject", "Session", "Global Trial"]

for keys, g in amb.groupby(key_cols):
    subject, session, trial = keys

    for method in METHODS:
        pred_col = f"{method}_Pred_Label"
        f1 = f1_score(
            g["True_Label_Dynamic"],
            g[pred_col],
            labels=mod.CLASSES,
            average="macro",
            zero_division=0,
        )
        acc = accuracy_score(g["True_Label_Dynamic"], g[pred_col])

        trial_rows.append({
            "Subject": subject,
            "Session": session,
            "Global Trial": trial,
            "Method": method,
            "num_ambiguous_rows": len(g),
            "accuracy": acc,
            "macro_f1": f1,
        })

per_trial = pd.DataFrame(trial_rows)
per_trial.to_csv(OUT_PER_TRIAL, index=False)

# Statistical tests on per-trial macro-F1 distributions
arrays = [
    per_trial[per_trial["Method"] == method]["macro_f1"].dropna().to_numpy()
    for method in METHODS
]

h_stat, h_p = kruskal(*arrays)

stat_rows = [{
    "test": "Kruskal-Wallis",
    "comparison": "GR_vs_BF_vs_E2E",
    "statistic": h_stat,
    "p_value": h_p,
}]

# Pairwise Mann-Whitney U as fallback-compatible post-hoc test.
# This is not Dunn's test, but it gives pairwise non-parametric comparison.
pairs = [("GR", "BF"), ("GR", "E2E"), ("BF", "E2E")]
raw_pair = []

for a, b in pairs:
    av = per_trial[per_trial["Method"] == a]["macro_f1"].dropna()
    bv = per_trial[per_trial["Method"] == b]["macro_f1"].dropna()

    u_stat, p_val = mannwhitneyu(av, bv, alternative="two-sided")
    raw_pair.append((a, b, u_stat, p_val))

# Holm correction
sorted_pair = sorted(enumerate(raw_pair), key=lambda x: x[1][3])
m = len(sorted_pair)
holm = {}
running_max = 0.0
for rank, (idx, (a, b, u_stat, p_val)) in enumerate(sorted_pair, start=1):
    adj = min(1.0, (m - rank + 1) * p_val)
    running_max = max(running_max, adj)
    holm[idx] = running_max

for idx, (a, b, u_stat, p_val) in enumerate(raw_pair):
    stat_rows.append({
        "test": "Mann-Whitney U post-hoc with Holm correction",
        "comparison": f"{a}_vs_{b}",
        "statistic": u_stat,
        "p_value": p_val,
        "p_holm": holm[idx],
    })

stats = pd.DataFrame(stat_rows)
stats.to_csv(OUT_STATS, index=False)

print("\nStatistical tests:")
print(stats.to_string(index=False))

# Helper for significance marks
def sig_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."

pair_p = {
    row["comparison"]: row.get("p_holm", row["p_value"])
    for _, row in stats.iterrows()
    if "post-hoc" in row["test"]
}

# Plot overall macro-F1 bars like old Fig. 9
plot_vals = metrics.set_index("Method").loc[METHODS, "macro_f1"]

fig, ax = plt.subplots(figsize=(6.2, 4.2))

colors = {"GR": "#4C97D8", "BF": "#E67E22", "E2E": "#43B581"}
hatches = {"GR": "//", "BF": "\\\\", "E2E": "xx"}

x = np.arange(len(METHODS))
bars = ax.bar(
    x,
    plot_vals.values,
    width=0.56,
    color="white",
    edgecolor=[colors[m] for m in METHODS],
    linewidth=1.3,
)

for bar, method in zip(bars, METHODS):
    bar.set_hatch(hatches[method])

for i, val in enumerate(plot_vals.values):
    ax.text(i, val + 0.018, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

ax.set_title("Macro F1-Score on GR Ambiguous Cases (≤ 25.0%)", fontsize=12)
ax.set_ylabel("Macro F1-Score", fontsize=11)
ax.set_xlabel("Method", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(METHODS, fontsize=10)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle=":", alpha=0.55)

def add_sig(ax, x1, x2, y, text, h=0.025):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.9, c="black")
    ax.text((x1+x2)/2, y+h+0.012, text, ha="center", va="bottom", fontsize=10)

# Pairwise brackets
add_sig(ax, 0, 1, 0.76, sig_label(pair_p.get("GR_vs_BF", 1.0)))
add_sig(ax, 0, 2, 0.84, sig_label(pair_p.get("GR_vs_E2E", 1.0)))
add_sig(ax, 1, 2, 0.92, sig_label(pair_p.get("BF_vs_E2E", 1.0)))

ax.text(-0.45, 0.96, "*** p < 0.001", fontsize=9)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

print("\nSaved:", OUT_PNG)
print("Saved:", OUT_PDF)
print("Saved:", OUT_METRICS)
print("Saved:", OUT_PER_TRIAL)
print("Saved:", OUT_STATS)
