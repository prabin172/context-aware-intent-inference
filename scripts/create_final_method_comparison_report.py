#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


CLASSES = [
    "Clean-table",
    "Nothing",
    "Pick-up-backpack",
    "Push-chair",
    "Sit-on-chair",
    "Sit-on-couch",
    "Sit-on-table",
    "Stand-on-couch",
    "Wear-backpack",
]

SUBJECTS = ["sub2b", "sub3", "sub4", "sub5", "sub6"]

METHOD_PATHS = {
    "GR": "results/gr/inference_{subject}/gr_predictions_all_test_trials.csv",
    "BF": "results/bayesian_fusion/inference_{subject}/bf_predictions_all_test_trials.csv",
    "E2E": "results/e2e_transformer_matched_gr/inference_{subject}/e2e_predictions_all_test_trials.csv",
}


def norm_name(x: str) -> str:
    return (
        str(x)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("__", "_")
    )


CLASS_NORM_TO_DISPLAY = {norm_name(c): c for c in CLASSES}


CLASS_ALIASES = {
    "Clean-table": ["clean_table"],
    "Nothing": ["nothing"],
    "Pick-up-backpack": ["pick_up_backpack", "pickup_backpack"],
    "Push-chair": ["push_chair"],
    "Sit-on-chair": ["sit_on_chair"],
    "Sit-on-couch": ["sit_on_couch"],
    "Sit-on-table": ["sit_on_table"],
    "Stand-on-couch": ["stand_on_couch"],
    "Wear-backpack": ["wear_backpack"],
}


def clean_label(x: str) -> str:
    n = norm_name(x)
    n = n.replace("pick_up_backpack", "pick_up_backpack")
    n = n.replace("pickup_backpack", "pick_up_backpack")

    for display, aliases in CLASS_ALIASES.items():
        if n == norm_name(display) or n in aliases:
            return display

    if n in CLASS_NORM_TO_DISPLAY:
        return CLASS_NORM_TO_DISPLAY[n]

    return str(x)


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> str | None:
    cols = list(df.columns)
    norm_cols = {norm_name(c): c for c in cols}

    for cand in candidates:
        nc = norm_name(cand)
        if nc in norm_cols:
            return norm_cols[nc]

    for cand in candidates:
        nc = norm_name(cand)
        for c in cols:
            if nc in norm_name(c):
                return c

    if required:
        raise ValueError(f"Could not find required column among candidates: {candidates}\nAvailable columns:\n{cols}")
    return None


def get_time_col(df: pd.DataFrame) -> str:
    return find_col(
        df,
        [
            "Time (ms)",
            "Time_ms",
            "time_ms",
            "timestamp_ms",
            "Normalized_Time",
            "normalized_time",
            "time",
            "timestamp",
        ],
    )


def get_true_label_col(df: pd.DataFrame) -> str:
    return find_col(
        df,
        [
            "True Label",
            "true_label",
            "True_Label",
            "Gesture",
            "label",
            "ground_truth",
            "Ground Truth",
        ],
    )


def get_session_col(df: pd.DataFrame) -> str | None:
    return find_col(df, ["Session", "session"], required=False)


def get_trial_col(df: pd.DataFrame) -> str:
    return find_col(
        df,
        [
            "Global Trial",
            "Global_Trial",
            "global_trial",
            "Trial Number",
            "Trial_Number",
            "trial",
            "Trial",
            "trial_global_idx",
        ],
    )


def find_confidence_columns(df: pd.DataFrame, method: str) -> Dict[str, str]:
    out = {}

    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    for class_name in CLASSES:
        aliases = [norm_name(class_name)] + CLASS_ALIASES[class_name]
        hits = []

        for col in numeric_cols:
            nc = norm_name(col)

            if "pred" in nc or "label" in nc or "trial" in nc or "time" in nc:
                continue

            has_class = any(alias in nc for alias in aliases)
            has_conf_marker = (
                "confidence" in nc
                or "conf" in nc
                or nc.startswith("intent_")
                or nc.startswith("bf_intent_")
            )

            if has_class and has_conf_marker:
                hits.append(col)

        if not hits:
            raise ValueError(
                f"Could not find confidence column for method={method}, class={class_name}\n"
                f"Available columns:\n{list(df.columns)}"
            )

        def score(c: str) -> Tuple[int, int]:
            nc = norm_name(c)
            exact = max(1 if alias in nc else 0 for alias in aliases)
            method_bonus = 1 if norm_name(method) in nc else 0
            return (exact + method_bonus, -len(c))

        hits = sorted(hits, key=score, reverse=True)
        out[class_name] = hits[0]

    return out


def load_method(method: str, path_template: str) -> pd.DataFrame:
    frames = []

    for subject in SUBJECTS:
        path = Path(path_template.format(subject=subject))
        if not path.exists():
            raise FileNotFoundError(f"Missing {method} file: {path}")

        raw = pd.read_csv(path)

        time_col = get_time_col(raw)
        true_col = get_true_label_col(raw)
        trial_col = get_trial_col(raw)
        session_col = get_session_col(raw)
        conf_cols = find_confidence_columns(raw, method)

        df = pd.DataFrame(index=raw.index)
        df["Method"] = method
        df["Subject"] = subject
        df["Session"] = raw[session_col].astype(str) if session_col else subject
        df["Global Trial"] = raw[trial_col]
        df["Time_ms"] = pd.to_numeric(raw[time_col], errors="coerce").round().astype("Int64")
        df["Action_Label"] = raw[true_col].apply(clean_label)

        for class_name, col in conf_cols.items():
            df[class_name] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)

        df = df.dropna(subset=["Time_ms"]).copy()
        df["Time_ms"] = df["Time_ms"].astype(int)

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def add_predictions(df: pd.DataFrame, onset_ms: int) -> pd.DataFrame:
    df = df.copy()
    class_scores = df[CLASSES]
    df["Pred_Label"] = class_scores.idxmax(axis=1)
    sorted_scores = np.sort(class_scores.to_numpy(), axis=1)
    df["Top1_Confidence"] = sorted_scores[:, -1]
    df["Top2_Confidence"] = sorted_scores[:, -2]
    df["Confidence_Gap"] = df["Top1_Confidence"] - df["Top2_Confidence"]
    df["True_Label_Dynamic"] = np.where(df["Time_ms"] < onset_ms, "Nothing", df["Action_Label"])
    return df


def compute_overall_subject_metrics(all_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    subject_rows = []

    for method, g in all_df.groupby("Method"):
        overall_rows.append({
            "Method": method,
            "accuracy": accuracy_score(g["True_Label_Dynamic"], g["Pred_Label"]),
            "macro_f1": f1_score(g["True_Label_Dynamic"], g["Pred_Label"], labels=CLASSES, average="macro", zero_division=0),
            "weighted_f1": f1_score(g["True_Label_Dynamic"], g["Pred_Label"], labels=CLASSES, average="weighted", zero_division=0),
            "num_rows": len(g),
            "num_trials": g[["Subject", "Session", "Global Trial"]].drop_duplicates().shape[0],
        })

        for subject, sg in g.groupby("Subject"):
            subject_rows.append({
                "Method": method,
                "Subject": subject,
                "accuracy": accuracy_score(sg["True_Label_Dynamic"], sg["Pred_Label"]),
                "macro_f1": f1_score(sg["True_Label_Dynamic"], sg["Pred_Label"], labels=CLASSES, average="macro", zero_division=0),
                "weighted_f1": f1_score(sg["True_Label_Dynamic"], sg["Pred_Label"], labels=CLASSES, average="weighted", zero_division=0),
                "num_rows": len(sg),
                "num_trials": sg[["Subject", "Session", "Global Trial"]].drop_duplicates().shape[0],
            })

    return pd.DataFrame(overall_rows), pd.DataFrame(subject_rows)


def compute_per_class_f1(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method, g in all_df.groupby("Method"):
        precision, recall, f1, support = precision_recall_fscore_support(
            g["True_Label_Dynamic"],
            g["Pred_Label"],
            labels=CLASSES,
            zero_division=0,
        )

        for c, p, r, f, s in zip(CLASSES, precision, recall, f1, support):
            rows.append({
                "Method": method,
                "Class": c,
                "precision": p,
                "recall": r,
                "f1_score": f,
                "support": s,
            })

    return pd.DataFrame(rows)


def sustained_first_time(trial: pd.DataFrame, action_label: str, onset_ms: int, stability_ms: int) -> float:
    trial = trial.sort_values("Time_ms").copy()
    trial = trial[trial["Time_ms"] <= onset_ms].copy()

    if trial.empty:
        return np.nan

    t = trial["Time_ms"].to_numpy()
    is_action = (trial["Pred_Label"].to_numpy() == action_label)

    n = len(trial)

    for i in range(n):
        if not is_action[i]:
            continue

        end_time = t[i] + stability_ms
        j = np.searchsorted(t, end_time, side="right")

        if j <= i:
            continue

        if t[j - 1] < end_time:
            continue

        if is_action[i:j].all():
            return float(t[i])

    return np.nan


def compute_early_prediction(all_df: pd.DataFrame, onset_ms: int, stability_ms: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    key_cols = ["Method", "Subject", "Session", "Global Trial"]

    for keys, trial in all_df.groupby(key_cols, sort=False):
        method, subject, session, global_trial = keys
        trial = trial.sort_values("Time_ms").copy()

        action_candidates = trial["Action_Label"]
        action_candidates = action_candidates[action_candidates != "Nothing"]

        if action_candidates.empty:
            continue

        action_label = action_candidates.mode().iloc[0]

        final_idx = (trial["Time_ms"] - onset_ms).abs().idxmin()
        final_pred = trial.loc[final_idx, "Pred_Label"]
        final_correct = final_pred == action_label

        if not final_correct:
            continue

        first_t = sustained_first_time(trial, action_label, onset_ms, stability_ms)
        lead_time = onset_ms - first_t if pd.notna(first_t) else np.nan

        rows.append({
            "Method": method,
            "Subject": subject,
            "Session": session,
            "Global Trial": global_trial,
            "Action_Label": action_label,
            "final_prediction_time_ms": int(trial.loc[final_idx, "Time_ms"]),
            "final_prediction_label": final_pred,
            "final_prediction_correct": final_correct,
            "first_sustained_time_ms": first_t,
            "prediction_lead_time_ms": lead_time,
        })

    detail = pd.DataFrame(rows)

    if detail.empty:
        summary = pd.DataFrame(columns=[
            "Method",
            "num_final_correct_trials",
            "num_trials_with_sustained_prediction",
            "mean_lead_time_ms",
            "median_lead_time_ms",
            "std_lead_time_ms",
            "min_lead_time_ms",
            "max_lead_time_ms",
        ])
        return detail, summary

    summary = (
        detail
        .groupby("Method", dropna=False)
        .agg(
            num_final_correct_trials=("final_prediction_correct", "sum"),
            num_trials_with_sustained_prediction=("prediction_lead_time_ms", "count"),
            mean_lead_time_ms=("prediction_lead_time_ms", "mean"),
            median_lead_time_ms=("prediction_lead_time_ms", "median"),
            std_lead_time_ms=("prediction_lead_time_ms", "std"),
            min_lead_time_ms=("prediction_lead_time_ms", "min"),
            max_lead_time_ms=("prediction_lead_time_ms", "max"),
        )
        .reset_index()
    )

    return detail, summary


def merge_common_grid(method_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    key_cols = ["Subject", "Session", "Global Trial", "Time_ms", "True_Label_Dynamic", "Action_Label"]

    gr = method_dfs["GR"][key_cols + ["Pred_Label", "Confidence_Gap"] + CLASSES].copy()
    gr = gr.rename(columns={
        "Pred_Label": "GR_Pred_Label",
        "Confidence_Gap": "GR_Confidence_Gap",
        **{c: f"GR_{c}" for c in CLASSES},
    })

    bf = method_dfs["BF"][key_cols + ["Pred_Label"] + CLASSES].copy()
    bf = bf.rename(columns={
        "Pred_Label": "BF_Pred_Label",
        **{c: f"BF_{c}" for c in CLASSES},
    })

    e2e = method_dfs["E2E"][key_cols + ["Pred_Label"] + CLASSES].copy()
    e2e = e2e.rename(columns={
        "Pred_Label": "E2E_Pred_Label",
        **{c: f"E2E_{c}" for c in CLASSES},
    })

    merged = gr.merge(bf, on=key_cols, how="inner")
    merged = merged.merge(e2e, on=key_cols, how="inner")
    return merged


def compute_ambiguity(common_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    amb = common_df[common_df["GR_Confidence_Gap"] <= threshold].copy()

    rows = []
    for method in ["GR", "BF", "E2E"]:
        pred_col = f"{method}_Pred_Label"

        rows.append({
            "Method": method,
            "gr_gap_threshold": threshold,
            "num_ambiguous_rows": len(amb),
            "percent_of_common_rows": 100.0 * len(amb) / len(common_df) if len(common_df) else 0.0,
            "accuracy": accuracy_score(amb["True_Label_Dynamic"], amb[pred_col]) if len(amb) else np.nan,
            "macro_f1": f1_score(amb["True_Label_Dynamic"], amb[pred_col], labels=CLASSES, average="macro", zero_division=0) if len(amb) else np.nan,
            "weighted_f1": f1_score(amb["True_Label_Dynamic"], amb[pred_col], labels=CLASSES, average="weighted", zero_division=0) if len(amb) else np.nan,
        })

    return pd.DataFrame(rows)


def compute_conditional_bf(common_df: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    rows = []

    y_true = common_df["True_Label_Dynamic"]

    for threshold in thresholds:
        use_bf = common_df["GR_Confidence_Gap"] <= threshold
        pred = np.where(use_bf, common_df["BF_Pred_Label"], common_df["GR_Pred_Label"])

        rows.append({
            "gr_gap_threshold": threshold,
            "percent_switched_to_bf": 100.0 * use_bf.mean(),
            "accuracy": accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, labels=CLASSES, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, pred, labels=CLASSES, average="weighted", zero_division=0),
            "num_rows": len(common_df),
        })

    return pd.DataFrame(rows)


def write_readme(out_dir: Path) -> None:
    text = """# Final method comparison report

This report compares GR, Bayesian Fusion, and E2E Transformer on the cleaned 9-class split-sit evaluation.

The E2E predictions are evaluated on the same 1029 test trials used for GR and Bayesian Fusion.

Generated files:

- final_overall_metrics.csv
- final_subject_metrics.csv
- final_per_class_f1.csv
- final_early_prediction_lead_time.csv
- final_early_prediction_summary.csv
- final_ambiguity_metrics.csv
- final_conditional_bf_threshold_sweep.csv
- final_common_grid_summary.csv

Notes:

- Dynamic ground truth uses Nothing before the onset threshold and the action label after onset.
- Early prediction lead time is computed as onset time minus the first time the true action remains the top predicted class for the stability window.
- Ambiguous timestamps are defined using the GR confidence gap between the top two predictions.
- Conditional BF uses GR by default and switches to BF when the GR confidence gap is below the threshold.
"""
    (out_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="docs/experiment_reports/final_method_comparison")
    parser.add_argument("--onset_ms", type=int, default=2000)
    parser.add_argument("--lead_onset_ms", type=int, default=2500)
    parser.add_argument("--stability_ms", type=int, default=200)
    parser.add_argument("--ambiguity_threshold", type=float, default=0.25)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions...")
    method_dfs = {}
    for method, template in METHOD_PATHS.items():
        df = load_method(method, template)
        df = add_predictions(df, onset_ms=args.onset_ms)
        method_dfs[method] = df
        print(f"{method}: rows={len(df)}, trials={df[['Subject', 'Session', 'Global Trial']].drop_duplicates().shape[0]}")

    all_df = pd.concat(method_dfs.values(), ignore_index=True)

    print("Computing overall, subject, and per-class metrics...")
    overall, subject = compute_overall_subject_metrics(all_df)
    per_class = compute_per_class_f1(all_df)

    overall.to_csv(out_dir / "final_overall_metrics.csv", index=False)
    subject.to_csv(out_dir / "final_subject_metrics.csv", index=False)
    per_class.to_csv(out_dir / "final_per_class_f1.csv", index=False)

    print("Computing early prediction lead time...")
    lead_detail, lead_summary = compute_early_prediction(
        all_df,
        onset_ms=args.lead_onset_ms,
        stability_ms=args.stability_ms,
    )
    lead_detail.to_csv(out_dir / "final_early_prediction_lead_time.csv", index=False)
    lead_summary.to_csv(out_dir / "final_early_prediction_summary.csv", index=False)

    print("Merging common timestamp grid for ambiguity and conditional BF...")
    common = merge_common_grid(method_dfs)

    common_summary = pd.DataFrame([{
        "num_common_rows": len(common),
        "num_common_trials": common[["Subject", "Session", "Global Trial"]].drop_duplicates().shape[0],
        "ambiguity_threshold": args.ambiguity_threshold,
    }])
    common_summary.to_csv(out_dir / "final_common_grid_summary.csv", index=False)

    print("Computing ambiguity metrics...")
    ambiguity = compute_ambiguity(common, threshold=args.ambiguity_threshold)
    ambiguity.to_csv(out_dir / "final_ambiguity_metrics.csv", index=False)

    print("Computing conditional BF threshold sweep...")
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    conditional = compute_conditional_bf(common, thresholds=thresholds)
    conditional.to_csv(out_dir / "final_conditional_bf_threshold_sweep.csv", index=False)

    write_readme(out_dir)

    print()
    print("Saved final comparison report to:", out_dir)
    print()
    print("Overall metrics:")
    print(overall.to_string(index=False))
    print()
    print("Ambiguity metrics:")
    print(ambiguity.to_string(index=False))
    print()
    print("Conditional BF sweep:")
    print(conditional.to_string(index=False))


if __name__ == "__main__":
    main()
