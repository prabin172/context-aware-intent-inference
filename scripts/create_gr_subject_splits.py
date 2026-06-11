from pathlib import Path
from collections import defaultdict

import argparse
import pandas as pd


def subject_from_session(session_id: str) -> str:
    s = session_id.lower()

    if s.startswith("sub4"):
        return "sub4"
    if s.startswith("sub5"):
        return "sub5"
    if s.startswith("sub6"):
        return "sub6"
    if s.startswith("sub2b"):
        return "sub2b"
    if s.startswith("sub3"):
        return "sub3"

    return session_id


def build_subject_trials(subject: str, trials_root: Path) -> pd.DataFrame:
    rows = []
    global_trial = 0

    trial_files = sorted(trials_root.glob("*_trials.csv"))

    for path in trial_files:
        session_id = path.name.replace("_trials.csv", "")

        if subject_from_session(session_id) != subject:
            continue

        df = pd.read_csv(path)

        for _, row in df.iterrows():
            global_trial += 1
            rows.append({
                "Subject": subject,
                "Session": session_id,
                "Global Trial": global_trial,
                "Session Trial": int(row["Trial Number"]),
                "Gesture": row["Gesture"],
                "Object": row.get("Object", ""),
            })

    if not rows:
        raise RuntimeError(f"No trials found for subject: {subject}")

    return pd.DataFrame(rows)


def assign_first_k_per_gesture(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df.copy()
    df["Split"] = "test"

    counts = defaultdict(int)

    for idx, row in df.iterrows():
        gesture = row["Gesture"]

        if counts[gesture] < k:
            df.loc[idx, "Split"] = "calibration"
            counts[gesture] += 1

    return df


def main():
    parser = argparse.ArgumentParser(description="Create GR subject-level calibration/test split manifests.")
    parser.add_argument("--trials_root", default="data/trials")
    parser.add_argument("--output_dir", default="results/gr/splits")
    parser.add_argument("--shots_per_gesture", type=int, default=2)
    parser.add_argument("--subjects", nargs="+", default=["sub2b", "sub3", "sub4", "sub5", "sub6"])
    args = parser.parse_args()

    trials_root = Path(args.trials_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject in args.subjects:
        df = build_subject_trials(subject, trials_root)
        df = assign_first_k_per_gesture(df, args.shots_per_gesture)

        output_path = output_dir / f"{subject}_trial_split.csv"
        df.to_csv(output_path, index=False)

        print(f"\nSubject: {subject}")
        print(f"Saved: {output_path}")
        print(df["Split"].value_counts().to_string())
        print("\nCalibration gesture counts:")
        print(df[df["Split"] == "calibration"]["Gesture"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
