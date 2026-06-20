from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


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

LABEL_TO_INDEX = {label: i for i, label in enumerate(INTENTS)}
INDEX_TO_LABEL = {i: label for label, i in LABEL_TO_INDEX.items()}

SUBJECTS = ["sub2b", "sub3", "sub4", "sub5", "sub6"]

TARGET_IMAGE_SIZE = (224, 224)

RGB_TRANSFORM = T.Compose([
    T.Resize(TARGET_IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

DEPTH_TRANSFORM = T.Compose([
    T.Resize(TARGET_IMAGE_SIZE),
    T.ToTensor(),
])


def subject_from_session(session_id: str) -> str:
    s = session_id.lower()
    if s.startswith("sub4"):
        return "sub4"
    if s.startswith("sub5"):
        return "sub5"
    if s.startswith("sub6"):
        return "sub6"
    if s == "sub2b":
        return "sub2b"
    if s == "sub3":
        return "sub3"
    raise ValueError(f"Unknown session id: {session_id}")


def extract_timestamp_from_png(path: Path) -> float:
    # color_1741547823_473752737.png / depth_...
    parts = path.stem.split("_")
    sec = int(parts[1])
    nsec = int(parts[2])
    return sec * 1000.0 + nsec / 1e6


def find_ja_csv(session_id: str, ja_root: Path) -> Path:
    target = f"{session_id}_ja.csv".lower()
    for p in ja_root.glob("*_ja.csv"):
        if p.name.lower() == target:
            return p
    raise FileNotFoundError(f"Could not find JA CSV for session {session_id} in {ja_root}")


@dataclass
class WindowSpec:
    subject: str
    session: str
    global_trial: int
    session_trial: int
    gesture: str
    split: str
    window_start_ms: float
    window_end_ms: float
    label: str


class E2ETransformerDataset(Dataset):
    """
    Clean E2E Transformer dataset.

    Splits follow GR:
      base_train  -> all split CSVs, subjects != target_subject
      calibration -> target subject split CSV, Split == calibration
      test        -> target subject split CSV, Split == test

    Training/calibration windows:
      10 windows per trial:
      0-500, 250-750, ..., 2250-2750 ms
      first 7 labeled Nothing, last 3 labeled action

    Test windows:
      sliding windows ending from 500 to 2500 ms.
    """

    def __init__(
        self,
        target_subject: str,
        split: str,
        data_root: str = "data",
        split_root: str = "results/gr/splits",
        window_size_ms: float = 500.0,
        train_stride_ms: float = 250.0,
        inference_stride_ms: float = 4.0,
        sequence_len: int = 16,
    ):
        if target_subject not in SUBJECTS:
            raise ValueError(f"Unknown target_subject={target_subject}. Expected one of {SUBJECTS}")

        if split not in {"base_train", "calibration", "test"}:
            raise ValueError("split must be one of: base_train, calibration, test")

        self.target_subject = target_subject
        self.split = split
        self.data_root = Path(data_root)
        self.split_root = Path(split_root)
        self.window_size_ms = float(window_size_ms)
        self.train_stride_ms = float(train_stride_ms)
        self.inference_stride_ms = float(inference_stride_ms)
        self.sequence_len = int(sequence_len)

        self.rgb_transform = RGB_TRANSFORM
        self.depth_transform = DEPTH_TRANSFORM

        self.session_cache = {}
        self.trials_cache = {}
        self.windows = self._build_windows()

    def _load_split_rows(self) -> pd.DataFrame:
        if self.split == "base_train":
            dfs = []
            for p in sorted(self.split_root.glob("*_trial_split.csv")):
                dfs.append(pd.read_csv(p))
            if not dfs:
                raise FileNotFoundError(f"No split CSVs found in {self.split_root}")
            df = pd.concat(dfs, ignore_index=True)
            df = df[df["Subject"].astype(str) != self.target_subject].copy()
            return df

        split_csv = self.split_root / f"{self.target_subject}_trial_split.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Missing split CSV: {split_csv}")

        df = pd.read_csv(split_csv)
        df = df[df["Split"] == self.split].copy()
        return df

    def _load_trials(self, session_id: str) -> pd.DataFrame:
        if session_id in self.trials_cache:
            return self.trials_cache[session_id]

        p = self.data_root / "trials" / f"{session_id}_trials.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing trials CSV: {p}")

        df = pd.read_csv(p)
        self.trials_cache[session_id] = df
        return df

    def _load_session(self, session_id: str):
        if session_id in self.session_cache:
            return self.session_cache[session_id]

        ja_path = find_ja_csv(session_id, self.data_root / "extracted_JointAngles")
        ja = pd.read_csv(ja_path)

        if "Time (in ms)" not in ja.columns:
            raise RuntimeError(f"{ja_path} missing column 'Time (in ms)'")

        times = ja["Time (in ms)"].to_numpy(dtype=float)
        joints = ja.iloc[:, 1:67].to_numpy(dtype=np.float32)

        img_dir = self.data_root / "Synced-color-depthPNG" / session_id
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {img_dir}")

        color_paths = sorted(img_dir.glob("color_*.png"), key=extract_timestamp_from_png)
        depth_paths = sorted(img_dir.glob("depth_*.png"), key=extract_timestamp_from_png)

        if not color_paths:
            raise RuntimeError(f"No color PNGs found in {img_dir}")
        if not depth_paths:
            raise RuntimeError(f"No depth PNGs found in {img_dir}")
        if len(color_paths) != len(depth_paths):
            raise RuntimeError(
                f"Color/depth count mismatch for {session_id}: "
                f"{len(color_paths)} color vs {len(depth_paths)} depth"
            )

        image_times = np.array([extract_timestamp_from_png(p) for p in color_paths], dtype=float)

        session_data = {
            "times": times,
            "joints": joints,
            "color_paths": color_paths,
            "depth_paths": depth_paths,
            "image_times": image_times,
        }

        self.session_cache[session_id] = session_data
        return session_data

    def _build_windows(self):
        rows = self._load_split_rows()
        windows = []

        for _, row in rows.iterrows():
            session = str(row["Session"])
            subject = subject_from_session(session)
            session_trial = int(row["Session Trial"])
            global_trial = int(row["Global Trial"])
            gesture = str(row["Gesture"])
            split_label = str(row["Split"])

            trials_df = self._load_trials(session)
            trial_row = trials_df[trials_df["Trial Number"] == session_trial]
            if trial_row.empty:
                raise RuntimeError(f"Missing trial {session_trial} in data/trials/{session}_trials.csv")

            if self.split in {"base_train", "calibration"}:
                starts = [i * self.train_stride_ms for i in range(10)]
                for i, start_ms in enumerate(starts):
                    end_ms = start_ms + self.window_size_ms
                    label = "Nothing" if i < 7 else gesture

                    windows.append(WindowSpec(
                        subject=subject,
                        session=session,
                        global_trial=global_trial,
                        session_trial=session_trial,
                        gesture=gesture,
                        split=split_label if self.split != "base_train" else "base_train",
                        window_start_ms=float(start_ms),
                        window_end_ms=float(end_ms),
                        label=label,
                    ))
            else:
                end_times = np.arange(
                    self.window_size_ms,
                    2500.0 + 1e-6,
                    self.inference_stride_ms,
                    dtype=float,
                )
                for end_ms in end_times:
                    windows.append(WindowSpec(
                        subject=subject,
                        session=session,
                        global_trial=global_trial,
                        session_trial=session_trial,
                        gesture=gesture,
                        split="test",
                        window_start_ms=float(end_ms - self.window_size_ms),
                        window_end_ms=float(end_ms),
                        label=gesture,
                    ))

        return windows

    def __len__(self):
        return len(self.windows)

    def _load_rgb_depth(self, color_path: Path, depth_path: Path):
        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_bgr is None:
            color_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            depth_raw = np.zeros((224, 224), dtype=np.uint16)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[..., 0]

        rgb = self.rgb_transform(Image.fromarray(color_rgb))
        depth = self.depth_transform(Image.fromarray(depth_raw).convert("L"))
        return rgb, depth

    def __getitem__(self, idx):
        w = self.windows[idx]
        sd = self._load_session(w.session)

        trials_df = self._load_trials(w.session)
        trial_row = trials_df[trials_df["Trial Number"] == w.session_trial].iloc[0]
        trial_start_abs = float(trial_row["Start Time (Xsens)"])

        sample_rel_times = np.linspace(
            w.window_start_ms,
            w.window_end_ms,
            self.sequence_len,
            dtype=float,
        )
        sample_abs_times = trial_start_abs + sample_rel_times

        motion_samples = []
        rgb_samples = []
        depth_samples = []

        times = sd["times"]
        image_times = sd["image_times"]

        for t_abs in sample_abs_times:
            xsens_idx = np.searchsorted(times, t_abs, side="left")
            xsens_idx = min(max(xsens_idx, 0), len(times) - 1)
            motion_samples.append(sd["joints"][xsens_idx])

            img_idx = bisect.bisect_right(image_times, t_abs) - 1
            img_idx = min(max(img_idx, 0), len(image_times) - 1)

            rgb, depth = self._load_rgb_depth(
                sd["color_paths"][img_idx],
                sd["depth_paths"][img_idx],
            )
            rgb_samples.append(rgb)
            depth_samples.append(depth)

        return {
            "rgb": torch.stack(rgb_samples),
            "depth": torch.stack(depth_samples),
            "motion": torch.tensor(np.stack(motion_samples), dtype=torch.float32),
            "label": torch.tensor(LABEL_TO_INDEX[w.label], dtype=torch.long),
            "label_text": w.label,
            "gesture": w.gesture,
            "Subject": w.subject,
            "Session": w.session,
            "Global Trial": w.global_trial,
            "Session Trial": w.session_trial,
            "Split": w.split,
            "time_ms": w.window_end_ms,
        }
