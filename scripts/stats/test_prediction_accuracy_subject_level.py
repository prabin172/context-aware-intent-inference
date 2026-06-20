#!/usr/bin/env python3

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

REPORT_DIR = Path("docs/experiment_reports/final_method_comparison")
INPUT = REPORT_DIR / "final_subject_metrics.csv"
OUT_SUMMARY = REPORT_DIR / "final_subject_mean_std_metrics.csv"
OUT_STATS = REPORT_DIR / "final_subject_level_stat_tests.csv"

METHODS = ["GR", "BF", "E2E"]
METRICS = ["accuracy", "macro_f1", "weighted_f1"]

df = pd.read_csv(INPUT)

# Mean/std table
summary = (
    df.groupby("Method")
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
        n_subjects=("Subject", "nunique"),
    )
    .reset_index()
)
summary.to_csv(OUT_SUMMARY, index=False)

print("\nSUBJECT-LEVEL MEAN ± STD")
print(summary.to_string(index=False))

rows = []

def holm_adjust(pvals):
    # pvals: list of (idx, p)
    sorted_items = sorted(pvals, key=lambda x: x[1])
    m = len(sorted_items)
    adjusted = {}
    running_max = 0.0
    for rank, (idx, p) in enumerate(sorted_items, start=1):
        adj = min(1.0, (m - rank + 1) * p)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted

for metric in METRICS:
    pivot = df.pivot(index="Subject", columns="Method", values=metric)
    pivot = pivot[METHODS].dropna()

    stat, p_friedman = friedmanchisquare(
        pivot["GR"],
        pivot["BF"],
        pivot["E2E"],
    )

    rows.append({
        "metric": metric,
        "test": "Friedman",
        "comparison": "GR_vs_BF_vs_E2E",
        "n_subjects": len(pivot),
        "statistic": stat,
        "p_value": p_friedman,
        "p_holm": np.nan,
    })

    raw_pair_pvals = []
    pair_results = []

    for a, b in itertools.combinations(METHODS, 2):
        try:
            w_stat, p_w = wilcoxon(pivot[a], pivot[b], zero_method="wilcox")
        except ValueError:
            w_stat, p_w = np.nan, 1.0

        idx = len(pair_results)
        pair_results.append((a, b, w_stat, p_w))
        raw_pair_pvals.append((idx, p_w))

    adjusted = holm_adjust(raw_pair_pvals)

    for idx, (a, b, w_stat, p_w) in enumerate(pair_results):
        rows.append({
            "metric": metric,
            "test": "Wilcoxon signed-rank",
            "comparison": f"{a}_vs_{b}",
            "n_subjects": len(pivot),
            "statistic": w_stat,
            "p_value": p_w,
            "p_holm": adjusted[idx],
        })

stats = pd.DataFrame(rows)
stats.to_csv(OUT_STATS, index=False)

print("\nSUBJECT-LEVEL STATISTICAL TESTS")
print(stats.to_string(index=False))

print("\nSaved:", OUT_SUMMARY)
print("Saved:", OUT_STATS)
