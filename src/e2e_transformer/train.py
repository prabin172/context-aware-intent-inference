from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.e2e_transformer.dataset_module import E2ETransformerDataset, INTENTS
from src.e2e_transformer.model import TransformerFusionModel


def main():
    parser = argparse.ArgumentParser(description="Train E2E Transformer base model with leave-one-subject-out split.")
    parser.add_argument("--target_subject", required=True, choices=["sub2b", "sub3", "sub4", "sub5", "sub6"])
    parser.add_argument("--output_root", default="results/e2e_transformer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sequence_len", type=int, default=16)
    parser.add_argument("--no_pretrained_cnn", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None, help="Debug only: stop after this many batches per epoch.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    out_dir = Path(args.output_root) / f"base_{args.target_subject}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("E2E base training")
    print("target_subject:", args.target_subject)
    print("device:", device)
    print("output:", out_dir)

    dataset = E2ETransformerDataset(
        target_subject=args.target_subject,
        split="base_train",
        sequence_len=args.sequence_len,
    )

    print("training windows:", len(dataset))
    print("classes:", INTENTS)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    model = TransformerFusionModel(
        num_classes=len(INTENTS),
        pretrained_cnn=not args.no_pretrained_cnn,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

        for batch_idx, batch in enumerate(pbar, start=1):
            rgb = batch["rgb"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            motion = batch["motion"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(rgb, depth, motion)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)

            avg_loss = total_loss / max(total_seen, 1)
            acc = total_correct / max(total_seen, 1)

            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

            if args.max_batches is not None and batch_idx >= args.max_batches:
                print(f"Stopping early after {args.max_batches} batches for debug.")
                break

        epoch_loss = total_loss / max(total_seen, 1)
        epoch_acc = total_correct / max(total_seen, 1)

        rows.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "num_samples": total_seen,
        })

        pd.DataFrame(rows).to_csv(out_dir / "training_log.csv", index=False)

        print(f"Epoch {epoch}: loss={epoch_loss:.4f}, accuracy={epoch_acc:.4f}")

    model_path = out_dir / f"e2e_transformer_base_{args.target_subject}.pth"
    torch.save(model.state_dict(), model_path)

    metadata = {
        "target_subject": args.target_subject,
        "classes": INTENTS,
        "sequence_len": args.sequence_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "pretrained_cnn": not args.no_pretrained_cnn,
        "num_train_windows": len(dataset),
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved model:", model_path)
    print("Saved metadata:", out_dir / "metadata.json")


if __name__ == "__main__":
    main()
