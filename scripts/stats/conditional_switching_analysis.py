#!/usr/bin/env python3

from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

REPORT_DIR = Path("docs/experiment_reports/final_method_comparison")
OUT_BASELINE = REPORT_DIR / "final_common_grid_baseline_metrics.csv"
OUT_CONDITIONAL = REPORT_DIR / "final_conditional_switching_metrics.csv"

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

spec = importlib.util.spec_from_file_location(
    "final_report",
    "scripts/create_final_method_comparison_report.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("Loading predictions...")
method_dfs = {}
for method, template in mod.METHOD_PATHS.items():
    df = mod.load_method(method, template)
    df = mod.add_predictions(df, onset_ms=2000)
    method_dfs[method] = df
    print(f"{method}: rows={len(df)}, trials={df[['Subject', 'Session', 'Global Trial']].drop_duplicates().shape[0]}")

print("\nMerging common timestamp grid...")
common = mod.merge_common_grid(method_dfs)
print("Common rows:", len(common))
print("Common trials:", common[["Subject", "Session", "Global Trial"]].drop_duplicates().shape[0])

y_true = common["True_Label_Dynamic"]

def metrics_for(pred):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, labels=mod.CLASSES, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, labels=mod.CLASSES, average="weighted", zero_division=0),
    }

baseline_rows = []
for method in ["GR", "BF", "E2E"]:
    m = metrics_for(common[f"{method}_Pred_Label"])
    baseline_rows.append({
        "method": method,
        "num_rows": len(common),
        "num_trials": common[["Subject", "Session", "Global Trial"]].drop_duplicates().shape[0],
        **m,
    })

baseline = pd.DataFrame(baseline_rows)
baseline.to_csv(OUT_BASELINE, index=False)

gr_base = baseline[baseline["method"] == "GR"].iloc[0]
bf_base = baseline[baseline["method"] == "BF"].iloc[0]
e2e_base = baseline[baseline["method"] == "E2E"].iloc[0]

rows = []

for threshold in THRESHOLDS:
    ambiguous = common["GR_Confidence_Gap"] <= threshold
    percent_switched = 100.0 * ambiguous.mean()
    call_reduction = 100.0 - percent_switched

    for fallback in ["BF", "E2E"]:
        pred = np.where(
            ambiguous,
            common[f"{fallback}_Pred_Label"],
            common["GR_Pred_Label"],
        )

        m = metrics_for(pred)

        full_fallback = bf_base if fallback == "BF" else e2e_base

        rows.append({
            "strategy": f"GR_then_{fallback}_if_ambiguous",
            "fallback_method": fallback,
            "gr_gap_threshold": threshold,
            "percent_switched_to_fallback": percent_switched,
            "fallback_call_reduction_vs_always_percent": call_reduction,
            "num_rows": len(common),
            **m,
            "accuracy_gain_vs_gr": m["accuracy"] - gr_base["accuracy"],
            "macro_f1_gain_vs_gr": m["macro_f1"] - gr_base["macro_f1"],
            "weighted_f1_gain_vs_gr": m["weighted_f1"] - gr_base["weighted_f1"],
            "accuracy_gap_to_full_fallback": m["accuracy"] - full_fallback["accuracy"],
            "macro_f1_gap_to_full_fallback": m["macro_f1"] - full_fallback["macro_f1"],
            "weighted_f1_gap_to_full_fallback": m["weighted_f1"] - full_fallback["weighted_f1"],
        })

conditional = pd.DataFrame(rows)
conditional.to_csv(OUT_CONDITIONAL, index=False)

print("\nBaseline metrics on common grid:")
print(baseline.to_string(index=False))

print("\nConditional switching metrics:")
print(conditional.to_string(index=False))

print("\nSaved:", OUT_BASELINE)
print("Saved:", OUT_CONDITIONAL)
