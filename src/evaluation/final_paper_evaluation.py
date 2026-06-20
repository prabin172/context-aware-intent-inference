#!/usr/bin/env python3

from pathlib import Path
import importlib.util
import subprocess
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel, kruskal, mannwhitneyu


REPORT_DIR = Path("docs/experiment_reports/final_method_comparison")
OUT_TXT = REPORT_DIR / "final_evaluation_results.txt"

METHODS = ["GR", "BF", "E2E"]
METRICS = ["accuracy", "macro_f1", "weighted_f1"]
AMBIGUITY_THRESHOLD = 0.25


def load_final_report_module():
    spec = importlib.util.spec_from_file_location(
        "final_report",
        "scripts/create_final_method_comparison_report.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def holm_adjust(pvals):
    sorted_items = sorted(pvals, key=lambda x: x[1])
    m = len(sorted_items)
    adjusted = {}
    running_max = 0.0

    for rank, (idx, p) in enumerate(sorted_items, start=1):
        adj = min(1.0, (m - rank + 1) * p)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max

    return adjusted


def fmt_p(p):
    if pd.isna(p):
        return "nan"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def fmt3(x):
    return f"{float(x):.3f}"


def section(title):
    return "\n" + "=" * 80 + f"\n{title}\n" + "=" * 80 + "\n"


def run_base_final_report():
    print("Running base final comparison report script...")
    subprocess.run(
        ["python", "scripts/create_final_method_comparison_report.py"],
        check=True,
    )


def subject_level_accuracy_stats(lines):
    subject_metrics = pd.read_csv(REPORT_DIR / "final_subject_metrics.csv")

    summary = (
        subject_metrics.groupby("Method")
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

    lines.append(section("OVERALL SUBJECT-LEVEL PERFORMANCE"))
    lines.append(summary.to_string(index=False))
    lines.append("\n\nPaper Table I values:")
    for method in METHODS:
        row = summary[summary["Method"] == method].iloc[0]
        lines.append(
            f"{method}: "
            f"Accuracy {fmt3(row['accuracy_mean'])} ± {fmt3(row['accuracy_std'])}; "
            f"Macro-F1 {fmt3(row['macro_f1_mean'])} ± {fmt3(row['macro_f1_std'])}; "
            f"Weighted-F1 {fmt3(row['weighted_f1_mean'])} ± {fmt3(row['weighted_f1_std'])}"
        )

    rows = []
    for metric in METRICS:
        pivot = subject_metrics.pivot(index="Subject", columns="Method", values=metric)
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

    lines.append(section("SUBJECT-LEVEL STATISTICAL TESTS"))
    lines.append(stats.to_string(index=False))

    return summary, stats


def early_prediction_stats(lines):
    lead_path = REPORT_DIR / "final_early_prediction_lead_time.csv"
    lead = pd.read_csv(lead_path)

    # Use only trials where all three methods produced a sustained correct prediction.
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

    t_bf_gr = ttest_rel(bf, gr)
    t_bf_e2e = ttest_rel(bf, e2e)
    w_bf_gr = wilcoxon(bf, gr)
    w_bf_e2e = wilcoxon(bf, e2e)

    lines.append(section("EARLY PREDICTION LEAD TIME"))
    lines.append(f"Matched successful trials: {len(paired)}")
    lines.append(f"GR mean lead time:  {gr.mean():.3f} ms")
    lines.append(f"BF mean lead time:  {bf.mean():.3f} ms")
    lines.append(f"E2E mean lead time: {e2e.mean():.3f} ms")
    lines.append("")
    lines.append(f"BF vs GR paired t-test: t={t_bf_gr.statistic:.6f}, p={fmt_p(t_bf_gr.pvalue)}")
    lines.append(f"BF vs GR Wilcoxon: W={w_bf_gr.statistic:.6f}, p={fmt_p(w_bf_gr.pvalue)}")
    lines.append(f"BF vs E2E paired t-test: t={t_bf_e2e.statistic:.6f}, p={fmt_p(t_bf_e2e.pvalue)}")
    lines.append(f"BF vs E2E Wilcoxon: W={w_bf_e2e.statistic:.6f}, p={fmt_p(w_bf_e2e.pvalue)}")

def per_class_f1_summary(lines):
    per_class_path = REPORT_DIR / "final_per_class_f1.csv"
    per_class = pd.read_csv(per_class_path)

    lines.append(section("PER-CLASS F1"))
    lines.append("Used for per-class F1 plot.")
    lines.append(per_class.to_string(index=False))


def load_method_predictions(mod):
    method_dfs = {}
    for method, template in mod.METHOD_PATHS.items():
        df = mod.load_method(method, template)
        df = mod.add_predictions(df, onset_ms=2000)
        method_dfs[method] = df
    return method_dfs


def ambiguity_stats(lines, mod, common):
    amb = common[common["GR_Confidence_Gap"] <= AMBIGUITY_THRESHOLD].copy()

    metric_rows = []
    for method in METHODS:
        pred_col = f"{method}_Pred_Label"
        metric_rows.append({
            "Method": method,
            "gr_gap_threshold": AMBIGUITY_THRESHOLD,
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

    trial_rows = []
    key_cols = ["Subject", "Session", "Global Trial"]

    for keys, g in amb.groupby(key_cols):
        subject, session, trial = keys

        for method in METHODS:
            pred_col = f"{method}_Pred_Label"
            trial_rows.append({
                "Subject": subject,
                "Session": session,
                "Global Trial": trial,
                "Method": method,
                "num_ambiguous_rows": len(g),
                "accuracy": accuracy_score(g["True_Label_Dynamic"], g[pred_col]),
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

    stat_rows = [{
        "test": "Kruskal-Wallis",
        "comparison": "GR_vs_BF_vs_E2E",
        "statistic": h_stat,
        "p_value": h_p,
        "p_holm": np.nan,
    }]

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

    for idx, (a, b, u_stat, p_val) in enumerate(raw_pair):
        stat_rows.append({
            "test": "Mann-Whitney U post-hoc with Holm correction",
            "comparison": f"{a}_vs_{b}",
            "statistic": u_stat,
            "p_value": p_val,
            "p_holm": holm[idx],
        })

    stats = pd.DataFrame(stat_rows)

    lines.append(section("AMBIGUITY ANALYSIS"))
    lines.append(f"Common rows: {len(common)}")
    lines.append(f"Ambiguous rows: {len(amb)}")
    lines.append(f"Ambiguous percent: {100 * len(amb) / len(common):.3f}%")
    lines.append("")
    lines.append(metrics.to_string(index=False))
    lines.append("")
    lines.append(stats.to_string(index=False))

    return metrics, stats


def conditional_switching(lines, mod, common):
    y_true = common["True_Label_Dynamic"]

    def metrics_for(pred):
        return {
            "accuracy": accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, labels=mod.CLASSES, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, pred, labels=mod.CLASSES, average="weighted", zero_division=0),
        }

    baseline_rows = []
    for method in METHODS:
        m = metrics_for(common[f"{method}_Pred_Label"])
        baseline_rows.append({
            "method": method,
            "num_rows": len(common),
            "num_trials": common[["Subject", "Session", "Global Trial"]].drop_duplicates().shape[0],
            **m,
        })

    baseline = pd.DataFrame(baseline_rows)

    ambiguous = common["GR_Confidence_Gap"] <= AMBIGUITY_THRESHOLD
    percent_switched = 100.0 * ambiguous.mean()

    rows = []
    for fallback in ["BF", "E2E"]:
        pred = np.where(
            ambiguous,
            common[f"{fallback}_Pred_Label"],
            common["GR_Pred_Label"],
        )

        m = metrics_for(pred)
        rows.append({
            "strategy": f"GR_then_{fallback}_if_ambiguous",
            "fallback_method": fallback,
            "gr_gap_threshold": AMBIGUITY_THRESHOLD,
            "percent_switched_to_fallback": percent_switched,
            "fallback_call_reduction_vs_always_percent": 100.0 - percent_switched,
            "num_rows": len(common),
            **m,
        })

    conditional = pd.DataFrame(rows)

    lines.append(section("CONDITIONAL SWITCHING ANALYSIS"))
    lines.append("Baseline metrics on the switching/common timestamp grid:")
    lines.append(baseline.to_string(index=False))
    lines.append("")
    lines.append(f"GR ambiguity threshold: {AMBIGUITY_THRESHOLD:.2f}")
    lines.append(f"Percent of timestamps triggering fallback: {percent_switched:.3f}%")
    lines.append("")
    lines.append(conditional.to_string(index=False))

    return baseline, conditional


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    run_base_final_report()

    mod = load_final_report_module()

    lines = []
    lines.append("FINAL PAPER EVALUATION RESULTS")
    lines.append("Generated by: src/evaluation/final_paper_evaluation.py")
    lines.append("")

    subject_level_accuracy_stats(lines)
    early_prediction_stats(lines)
    per_class_f1_summary(lines)

    print("Loading prediction outputs for ambiguity and conditional analyses...")
    method_dfs = load_method_predictions(mod)

    print("Merging common timestamp grid...")
    common = mod.merge_common_grid(method_dfs)

    ambiguity_stats(lines, mod, common)
    conditional_switching(lines, mod, common)

    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\nSaved:", OUT_TXT)


if __name__ == "__main__":
    main()
