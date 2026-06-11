import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.gr.data_loader import (
    TIME_COL,
    TRIAL_NUMBER_COL,
    resolve_case_insensitive_file,
    extract_training_windows_for_trial,
)


def load_calibration_windows(
    split_csv: Path,
    joint_angles_root: Path,
    trials_root: Path,
    merge_sit: bool = True,
):
    split_df = pd.read_csv(split_csv)
    calibration_df = split_df[split_df["Split"] == "calibration"].copy()

    segments = []
    labels = []

    for session_id, session_split_df in calibration_df.groupby("Session", sort=False):
        ja_csv = resolve_case_insensitive_file(joint_angles_root, f"{session_id}_ja.csv")
        trials_csv = resolve_case_insensitive_file(trials_root, f"{session_id}_trials.csv")

        motion_df = pd.read_csv(ja_csv).sort_values(TIME_COL).reset_index(drop=True)
        trials_df = pd.read_csv(trials_csv)

        for _, split_row in session_split_df.iterrows():
            session_trial = int(split_row["Session Trial"])

            trial_match = trials_df[trials_df[TRIAL_NUMBER_COL] == session_trial]
            if trial_match.empty:
                raise RuntimeError(
                    f"Could not find trial {session_trial} in {trials_csv}"
                )

            trial_row = trial_match.iloc[0]

            trial_segments, trial_labels = extract_training_windows_for_trial(
                motion_df=motion_df,
                trial_row=trial_row,
                merge_sit=merge_sit,
            )

            segments.extend(trial_segments)
            labels.extend(trial_labels)

    return segments, labels


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GR model using target-subject calibration trials.")
    parser.add_argument("--target_subject", required=True)
    parser.add_argument("--base_model_dir", default=None)
    parser.add_argument("--split_csv", default=None)
    parser.add_argument("--joint_angles_root", default="data/extracted_JointAngles")
    parser.add_argument("--trials_root", default="data/trials")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--no_merge_sit", action="store_true")
    args = parser.parse_args()

    target_subject = args.target_subject.lower()

    base_model_dir = Path(args.base_model_dir or f"results/gr/base_{target_subject}")
    split_csv = Path(args.split_csv or f"results/gr/splits/{target_subject}_trial_split.csv")
    output_dir = Path(args.output_dir or f"results/gr/finetuned_{target_subject}")
    joint_angles_root = Path(args.joint_angles_root)
    trials_root = Path(args.trials_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = base_model_dir / f"transformer_gr_leaveout_{target_subject}.keras"
    norm_path = base_model_dir / "normalization_stats.npz"
    encoder_path = base_model_dir / "label_encoder.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing base model: {model_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalization stats: {norm_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing label encoder: {encoder_path}")
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing split CSV: {split_csv}")

    print(f"Loading base model from {model_path}")
    model = load_model(model_path)

    label_encoder = joblib.load(encoder_path)
    norm_stats = np.load(norm_path)
    mean = norm_stats["mean"]
    std = norm_stats["std"]

    print(f"Loading calibration windows from {split_csv}")
    segments, labels = load_calibration_windows(
        split_csv=split_csv,
        joint_angles_root=joint_angles_root,
        trials_root=trials_root,
        merge_sit=not args.no_merge_sit,
    )

    if not segments:
        raise RuntimeError("No calibration segments were found.")

    print(f"Calibration segments: {len(segments)}")
    print(f"Calibration labels: {sorted(set(labels))}")

    normalized = [(seg - mean) / (std + 1e-8) for seg in segments]

    max_len = model.input_shape[1]
    x = pad_sequences(
        normalized,
        maxlen=max_len,
        dtype="float32",
        padding="post",
        value=0.0,
    )
    y = label_encoder.transform(labels)

    print(f"Fine-tuning shape: {x.shape}")

    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        x,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    output_model_path = output_dir / f"transformer_gr_finetuned_{target_subject}.keras"
    model.save(output_model_path)

    # Copy metadata paths by saving lightweight references for clarity.
    metadata = pd.DataFrame([
        {
            "target_subject": target_subject,
            "base_model_dir": str(base_model_dir),
            "split_csv": str(split_csv),
            "calibration_segments": len(segments),
            "output_model": str(output_model_path),
            "normalization_stats": str(norm_path),
            "label_encoder": str(encoder_path),
        }
    ])
    metadata.to_csv(output_dir / "fine_tune_metadata.csv", index=False)

    print(f"Saved fine-tuned model to {output_model_path}")
    print(f"Saved metadata to {output_dir / 'fine_tune_metadata.csv'}")


if __name__ == "__main__":
    main()
