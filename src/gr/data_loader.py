from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


TIME_COL = "Time (in ms)"

TRIAL_NUMBER_COL = "Trial Number"
GESTURE_COL = "Gesture"
TRIAL_START_COL = "Start Time (Xsens)"

WINDOW_MS = 500
STRIDE_MS = 250
NUM_WINDOWS = 10
NUM_GESTURE_WINDOWS = 3


def clean_label(label: str, merge_sit: bool = True) -> str:
    label = str(label).strip()

    if merge_sit and label.startswith("Sit"):
        return "Sit"

    return label


def extract_features(window_df: pd.DataFrame) -> np.ndarray:
    """Return joint-angle features only, excluding timestamp."""
    return window_df.drop(columns=[TIME_COL]).to_numpy(dtype=np.float32)


def extract_training_windows_for_trial(
    motion_df: pd.DataFrame,
    trial_row: pd.Series,
    merge_sit: bool = True,
    window_ms: int = WINDOW_MS,
    stride_ms: int = STRIDE_MS,
    num_windows: int = NUM_WINDOWS,
    num_gesture_windows: int = NUM_GESTURE_WINDOWS,
):
    """
    Create fixed GR training windows for one trial.

    Trial-relative windows:
      0-500, 250-750, ..., 2250-2750 ms

    Labels:
      first 7 windows  -> Nothing
      last 3 windows   -> gesture label
    """
    trial_start = float(trial_row[TRIAL_START_COL])
    gesture = clean_label(trial_row[GESTURE_COL], merge_sit=merge_sit)

    segments = []
    labels = []

    for k in range(num_windows):
        win_start = trial_start + k * stride_ms
        win_end = win_start + window_ms

        window_df = motion_df[
            (motion_df[TIME_COL] >= win_start)
            & (motion_df[TIME_COL] < win_end)
        ]

        if window_df.empty:
            continue

        label = gesture if k >= (num_windows - num_gesture_windows) else "Nothing"

        segments.append(extract_features(window_df))
        labels.append(label)

    return segments, labels


def resolve_case_insensitive_file(root: Path, expected_name: str) -> Path:
    direct = root / expected_name
    if direct.exists():
        return direct

    expected_lower = expected_name.lower()
    matches = [p for p in root.glob("*") if p.name.lower() == expected_lower]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise FileNotFoundError(f"Multiple case-insensitive matches for {expected_name}: {matches}")

    raise FileNotFoundError(f"Missing file: {direct}")


def load_session(
    session_id: str,
    joint_angles_root: str | Path = "data/extracted_JointAngles",
    trials_root: str | Path = "data/trials",
    merge_sit: bool = True,
):
    joint_angles_root = Path(joint_angles_root)
    trials_root = Path(trials_root)

    ja_csv = resolve_case_insensitive_file(joint_angles_root, f"{session_id}_ja.csv")
    trials_csv = resolve_case_insensitive_file(trials_root, f"{session_id}_trials.csv")

    motion_df = pd.read_csv(ja_csv)
    trials_df = pd.read_csv(trials_csv)

    if TIME_COL not in motion_df.columns:
        raise ValueError(f"{ja_csv} is missing required column: {TIME_COL}")

    for col in [TRIAL_NUMBER_COL, GESTURE_COL, TRIAL_START_COL]:
        if col not in trials_df.columns:
            raise ValueError(f"{trials_csv} is missing required column: {col}")

    motion_df = motion_df.sort_values(TIME_COL).reset_index(drop=True)

    all_segments = []
    all_labels = []

    for _, trial_row in trials_df.iterrows():
        segments, labels = extract_training_windows_for_trial(
            motion_df=motion_df,
            trial_row=trial_row,
            merge_sit=merge_sit,
        )
        all_segments.extend(segments)
        all_labels.extend(labels)

    return all_segments, all_labels


def discover_sessions(trials_root: str | Path = "data/trials"):
    trials_root = Path(trials_root)
    return sorted(
        p.name.replace("_trials.csv", "")
        for p in trials_root.glob("*_trials.csv")
    )


def load_all_data(
    test_session: str = "sub3",
    joint_angles_root: str | Path = "data/extracted_JointAngles",
    trials_root: str | Path = "data/trials",
    merge_sit: bool = True,
):
    sessions = discover_sessions(trials_root)

    train_segments = []
    train_labels = []
    test_segments = []
    test_labels = []

    for session_id in sessions:
        try:
            segments, labels = load_session(
                session_id=session_id,
                joint_angles_root=joint_angles_root,
                trials_root=trials_root,
                merge_sit=merge_sit,
            )
        except FileNotFoundError as exc:
            print(f"Skipping {session_id}: {exc}")
            continue

        if session_id == test_session:
            test_segments.extend(segments)
            test_labels.extend(labels)
        else:
            train_segments.extend(segments)
            train_labels.extend(labels)

    return (train_segments, train_labels), (test_segments, test_labels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build GR training windows from Xsens joint angles and trial CSVs.")
    parser.add_argument("--joint_angles_root", default="data/extracted_JointAngles")
    parser.add_argument("--trials_root", default="data/trials")
    parser.add_argument("--test_session", default="sub3")
    parser.add_argument("--no_merge_sit", action="store_true", help="Keep Sit-on-chair/couch/table separate instead of merging to Sit.")
    args = parser.parse_args()

    (X_train, y_train), (X_test, y_test) = load_all_data(
        test_session=args.test_session,
        joint_angles_root=args.joint_angles_root,
        trials_root=args.trials_root,
        merge_sit=not args.no_merge_sit,
    )

    print(f"Train segments: {len(X_train)}")
    print(f"Test segments: {len(X_test)}")
    print(f"Train label distribution: {Counter(y_train)}")
    print(f"Test label distribution: {Counter(y_test)}")

    if X_train:
        print(f"Sample train segment shape: {X_train[0].shape}")
    if X_test:
        print(f"Sample test segment shape: {X_test[0].shape}")
