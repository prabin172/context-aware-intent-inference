import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INTENTS = [
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

DISTANCE_COLUMNS = {
    "chair": "Distance Chair",
    "couch": "Distance Couch",
    "table": "Distance Dining Table",
    "backpack": "Distance Backpack",
}

RELATED_OBJECTS = {
    "Clean-table": {"table"},
    "Nothing": set(),
    "Pick-up-backpack": {"backpack"},
    "Push-chair": {"chair"},
    "Sit-on-chair": {"chair"},
    "Sit-on-couch": {"couch"},
    "Sit-on-table": {"table"},
    "Stand-on-couch": {"couch"},
    "Wear-backpack": {"backpack"},
}

PROXIMITY_LIKELIHOOD_RELATED = {
    "near": 0.70,
    "midway": 0.20,
    "far": 0.08,
    "invisible": 0.02,
}

PROXIMITY_LIKELIHOOD_UNRELATED = 0.25


def confidence_columns_for_intents(df: pd.DataFrame):
    cols = []
    missing = []

    for intent in INTENTS:
        col = f"{intent}_confidence"
        if col in df.columns:
            cols.append(col)
        else:
            missing.append(col)

    if missing:
        raise RuntimeError(f"Missing GR confidence columns: {missing}")

    return cols


def detect_time_column(df: pd.DataFrame):
    candidates = [
        "time_ms",
        "Time (ms)",
        "Time (in ms)",
        "Trial Time (ms)",
        "Trial Time",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise RuntimeError(
        "Could not find a time column. Tried: "
        + ", ".join(candidates)
        + f". Available columns: {list(df.columns)}"
    )


def detect_trial_column(df: pd.DataFrame):
    candidates = [
        "Session Trial",
        "Trial Number",
        "Trial",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise RuntimeError(
        "Could not find a trial column. Tried: "
        + ", ".join(candidates)
        + f". Available columns: {list(df.columns)}"
    )


def classify_proximity(distance, near_threshold: float, midway_threshold: float):
    if pd.isna(distance) or distance <= 0:
        return "invisible"
    if distance < near_threshold:
        return "near"
    if distance < midway_threshold:
        return "midway"
    return "far"


def build_gesture_likelihood(gr_probs: np.ndarray):
    gesture_cpt = np.full((len(INTENTS), len(INTENTS)), 0.05, dtype=float)
    np.fill_diagonal(gesture_cpt, 0.60)

    return gr_probs @ gesture_cpt


def build_object_likelihoods(row, near_threshold: float, midway_threshold: float):
    object_likelihoods = np.ones(len(INTENTS), dtype=float)

    for object_name, distance_col in DISTANCE_COLUMNS.items():
        state = classify_proximity(
            row[distance_col],
            near_threshold=near_threshold,
            midway_threshold=midway_threshold,
        )

        for i, intent in enumerate(INTENTS):
            if object_name in RELATED_OBJECTS[intent]:
                likelihood = PROXIMITY_LIKELIHOOD_RELATED[state]
            else:
                likelihood = PROXIMITY_LIKELIHOOD_UNRELATED

            object_likelihoods[i] *= likelihood

    return object_likelihoods


def normalize_probs(values: np.ndarray):
    total = float(np.sum(values))

    if total <= 0 or not np.isfinite(total):
        return np.full_like(values, 1.0 / len(values), dtype=float)

    return values / total


def compute_bf_row(row, conf_cols, near_threshold: float, midway_threshold: float):
    prior = np.full(len(INTENTS), 1.0 / len(INTENTS), dtype=float)

    gr_probs = row[conf_cols].to_numpy(dtype=float)
    gr_probs = normalize_probs(gr_probs)

    gesture_likelihood = build_gesture_likelihood(gr_probs)
    object_likelihood = build_object_likelihoods(
        row,
        near_threshold=near_threshold,
        midway_threshold=midway_threshold,
    )

    posterior_unnorm = prior * gesture_likelihood * object_likelihood
    posterior = normalize_probs(posterior_unnorm)

    return posterior


def merge_gr_with_proximity(
    gr_df: pd.DataFrame,
    proximity_df: pd.DataFrame,
    session_id: str,
    max_time_diff_ms: float,
):
    prox = proximity_df.copy()

    prox_time_col = detect_time_column(prox)
    prox_trial_col = detect_trial_column(prox)

    prox = prox.rename(
        columns={
            prox_time_col: "proximity_time_ms",
            prox_trial_col: "Session Trial",
        }
    )

    missing_dist = [c for c in DISTANCE_COLUMNS.values() if c not in prox.columns]
    if missing_dist:
        raise RuntimeError(
            f"Missing distance columns in proximity file for {session_id}: {missing_dist}"
        )

    prox["Session"] = session_id
    prox["Session Trial"] = prox["Session Trial"].astype(int)
    prox["proximity_time_ms"] = prox["proximity_time_ms"].astype(float)

    needed_prox_cols = (
        ["Session", "Session Trial", "proximity_time_ms"]
        + list(DISTANCE_COLUMNS.values())
    )
    prox = prox[needed_prox_cols].copy()

    gr = gr_df[gr_df["Session"].astype(str) == session_id].copy()
    if gr.empty:
        return pd.DataFrame()

    gr["Session Trial"] = gr["Session Trial"].astype(int)
    gr["time_ms"] = gr["time_ms"].astype(float)

    merged_parts = []

    for trial_num, gr_trial in gr.groupby("Session Trial", sort=False):
        prox_trial = prox[prox["Session Trial"] == trial_num].copy()

        if prox_trial.empty:
            continue

        gr_trial = gr_trial.sort_values("time_ms").copy()
        prox_trial = prox_trial.sort_values("proximity_time_ms").copy()

        merged = pd.merge_asof(
            gr_trial,
            prox_trial,
            left_on="time_ms",
            right_on="proximity_time_ms",
            direction="nearest",
            tolerance=max_time_diff_ms,
            suffixes=("", "_prox"),
        )

        merged = merged.dropna(subset=["proximity_time_ms"]).copy()
        merged["proximity_time_diff_ms"] = (
            merged["time_ms"] - merged["proximity_time_ms"]
        ).abs()

        merged_parts.append(merged)

    if not merged_parts:
        return pd.DataFrame()

    return pd.concat(merged_parts, ignore_index=True)


def run_subject(args, subject: str):
    gr_path = Path(args.gr_root) / f"inference_{subject}" / "gr_predictions_all_test_trials.csv"

    if not gr_path.exists():
        raise FileNotFoundError(f"Missing GR predictions: {gr_path}")

    print(f"\nSubject: {subject}")
    print(f"Reading GR predictions: {gr_path}")

    gr_df = pd.read_csv(gr_path)
    conf_cols = confidence_columns_for_intents(gr_df)

    subject_output_dir = Path(args.output_root) / f"inference_{subject}"
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    merged_parts = []

    for session_id in sorted(gr_df["Session"].astype(str).unique()):
        prox_path = Path(args.proximity_root) / session_id / "proximity.csv"

        if not prox_path.exists():
            raise FileNotFoundError(f"Missing proximity file for {session_id}: {prox_path}")

        print(f"  Merging session {session_id} with {prox_path}")

        proximity_df = pd.read_csv(prox_path)
        merged_session = merge_gr_with_proximity(
            gr_df=gr_df,
            proximity_df=proximity_df,
            session_id=session_id,
            max_time_diff_ms=args.max_time_diff_ms,
        )

        if merged_session.empty:
            print(f"  Warning: no merged rows for {session_id}")
            continue

        merged_parts.append(merged_session)

    if not merged_parts:
        raise RuntimeError(f"No merged rows for subject {subject}")

    merged_df = pd.concat(merged_parts, ignore_index=True)

    if args.save_merged_inputs:
        merged_path = Path(args.output_root) / f"merged_inputs_{subject}.csv"
        merged_df.to_csv(merged_path, index=False)
        print(f"Saved merged BF inputs: {merged_path}")

    posteriors = np.vstack([
        compute_bf_row(
            row,
            conf_cols=conf_cols,
            near_threshold=args.near_threshold,
            midway_threshold=args.midway_threshold,
        )
        for _, row in merged_df.iterrows()
    ])

    bf_conf_df = pd.DataFrame(
        posteriors,
        columns=[f"{intent}_confidence" for intent in INTENTS],
    )

    predicted_idx = np.argmax(posteriors, axis=1)
    predicted_labels = [INTENTS[i] for i in predicted_idx]
    predicted_conf = posteriors[np.arange(len(posteriors)), predicted_idx]

    metadata_cols = [
        "Subject",
        "Session",
        "Global Trial",
        "Session Trial",
        "Gesture",
        "Split",
        "time_ms",
    ]

    missing_metadata = [c for c in metadata_cols if c not in merged_df.columns]
    if missing_metadata:
        raise RuntimeError(f"Missing metadata columns after merge: {missing_metadata}")

    output_df = merged_df[metadata_cols].copy()
    output_df = pd.concat([output_df.reset_index(drop=True), bf_conf_df], axis=1)
    output_df["predicted_gesture"] = predicted_labels
    output_df["predicted_confidence"] = predicted_conf

    out_path = subject_output_dir / "bf_predictions_all_test_trials.csv"
    output_df.to_csv(out_path, index=False)

    print(f"Saved BF predictions: {out_path}")
    print(f"Rows: {len(output_df)}")
    print(f"Trials: {output_df[['Session', 'Session Trial']].drop_duplicates().shape[0]}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian Fusion using GR predictions and proximity outputs.")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["sub2b", "sub3", "sub4", "sub5", "sub6"],
    )
    parser.add_argument("--gr_root", default="results/gr")
    parser.add_argument("--proximity_root", default="results/proximity")
    parser.add_argument("--output_root", default="results/bayesian_fusion")
    parser.add_argument("--near_threshold", type=float, default=0.8)
    parser.add_argument("--midway_threshold", type=float, default=1.6)
    parser.add_argument("--max_time_diff_ms", type=float, default=5.0)
    parser.add_argument("--save_merged_inputs", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Bayesian Fusion configuration")
    print(f"Intents: {INTENTS}")
    print("Prior: uniform over intents")
    print("Gesture CPT: match=0.60, non-match=0.05")
    print("Related object likelihoods: near=0.70, midway=0.20, far=0.08, invisible=0.02")
    print("Unrelated object likelihood: uniform 0.25")

    outputs = []
    for subject in args.subjects:
        outputs.append(run_subject(args, subject))

    print()
    print("Finished Bayesian Fusion.")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
