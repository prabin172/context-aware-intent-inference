from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.e2e_transformer.dataset_module import E2ETransformerDataset, INTENTS
from src.e2e_transformer.model import TransformerFusionModel


def main():
    parser = argparse.ArgumentParser(description="Run E2E Transformer inference on target-subject test trials.")
    parser.add_argument("--target_subject", required=True, choices=["sub2b", "sub3", "sub4", "sub5", "sub6"])
    parser.add_argument("--output_root", default="results/e2e_transformer")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sequence_len", type=int, default=16)
    parser.add_argument("--no_pretrained_cnn", action="store_true")
    parser.add_argument("--use_base_model", action="store_true")
    parser.add_argument("--max_windows", type=int, default=None, help="Debug only: run only first N windows.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_base_model:
        model_path = (
            Path(args.output_root)
            / f"base_{args.target_subject}"
            / f"e2e_transformer_base_{args.target_subject}.pth"
        )
        out_dir = Path(args.output_root) / f"inference_base_{args.target_subject}"
    else:
        model_path = (
            Path(args.output_root)
            / f"finetuned_{args.target_subject}"
            / f"e2e_transformer_finetuned_{args.target_subject}.pth"
        )
        out_dir = Path(args.output_root) / f"inference_{args.target_subject}"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    print("E2E inference")
    print("target_subject:", args.target_subject)
    print("device:", device)
    print("model:", model_path)
    print("output:", out_dir)

    dataset = E2ETransformerDataset(
        target_subject=args.target_subject,
        split="test",
        sequence_len=args.sequence_len,
    )

    if args.max_windows is not None:
        dataset_for_loader = Subset(dataset, list(range(min(args.max_windows, len(dataset)))))
    else:
        dataset_for_loader = dataset

    print("test windows:", len(dataset_for_loader))

    loader = DataLoader(
        dataset_for_loader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TransformerFusionModel(
        num_classes=len(INTENTS),
        pretrained_cnn=not args.no_pretrained_cnn,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", unit="batch"):
            rgb = batch["rgb"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            motion = batch["motion"].to(device, non_blocking=True)

            logits = model(rgb, depth, motion)
            probs = F.softmax(logits, dim=1).cpu().numpy()

            pred_idx = probs.argmax(axis=1)

            batch_size = probs.shape[0]

            for i in range(batch_size):
                row = {
                    "Subject": batch["Subject"][i],
                    "Session": batch["Session"][i],
                    "Global Trial": int(batch["Global Trial"][i]),
                    "Session Trial": int(batch["Session Trial"][i]),
                    "Gesture": batch["gesture"][i],
                    "Split": batch["Split"][i],
                    "time_ms": float(batch["time_ms"][i]),
                }

                for j, label in enumerate(INTENTS):
                    row[f"{label}_confidence"] = float(probs[i, j])

                pred_label = INTENTS[int(pred_idx[i])]
                row["predicted_gesture"] = pred_label
                row["predicted_confidence"] = float(probs[i, int(pred_idx[i])])

                rows.append(row)

    out_csv = out_dir / "e2e_predictions_all_test_trials.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print("Saved:", out_csv)
    print("rows:", len(rows))


if __name__ == "__main__":
    main()
