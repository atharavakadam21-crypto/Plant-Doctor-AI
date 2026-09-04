"""Train and evaluate the proposed lightweight CNN with attention."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models.lightweight_cnn import build_lightweight_cnn, count_parameters
from preprocessing.dataset_loader import CLASS_NAMES, create_dataloaders


NUM_CLASSES = len(CLASS_NAMES)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights_from_dataset(dataset) -> torch.Tensor:
    counts = np.bincount(dataset.targets, minlength=NUM_CLASSES).astype(np.float64)
    if np.any(counts == 0):
        missing = [CLASS_NAMES[i] for i, c in enumerate(counts) if c == 0]
        raise RuntimeError(f"Training dataset is missing classes: {missing}")
    weights = counts.sum() / (NUM_CLASSES * counts)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return torch.autocast(device_type="cpu", enabled=False)


def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train(train)
    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in tqdm(loader, leave=False, desc="train" if train else "eval"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)

        use_amp = device.type == "cuda" and scaler is not None
        with autocast_context(use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    total = len(loader.dataset)
    return running_loss / total, accuracy_score(y_true, y_pred), y_true, y_pred


def measure_latency(model: nn.Module, device: torch.device, warmup: int = 20, runs: int = 100) -> float:
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / runs


def save_evaluation(
    output_dir: Path,
    y_true: list[int],
    y_pred: list[int],
    model: nn.Module,
    latency_ms: float,
    experiment: str,
) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    parameter_count = count_parameters(model)
    checkpoint_bytes = sum(
        p.numel() * p.element_size() for p in model.state_dict().values()
    )
    metrics = {
        "experiment": experiment,
        "model": "proposed lightweight CNN + SE attention",
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "trainable_parameters": parameter_count,
        "parameter_millions": parameter_count / 1e6,
        "state_dict_bytes": checkpoint_bytes,
        "state_dict_megabytes": checkpoint_bytes / (1024 * 1024),
        "single_image_latency_ms": latency_ms,
        "classes": CLASS_NAMES,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")
    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))),
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(output_dir / "confusion_matrix.csv")


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available.")
    device = torch.device(args.device)

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmented=True,
        seed=args.seed,
    )

    model = build_lightweight_cnn(NUM_CLASSES).to(device)
    weights = class_weights_from_dataset(train_loader.dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
    )

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "best_lightweight_cnn.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = make_scaler(device.type == "cuda" and not args.no_amp)

    history: list[dict[str, float]] = []
    best_val = -math.inf
    stale_epochs = 0

    print("Experiment: proposed_lightweight_cnn_attention")
    print("Training augmentation enabled: True")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Train/val/test: {len(train_loader.dataset)}/"
        f"{len(val_loader.dataset)}/{len(test_loader.dataset)}"
    )
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Class weights: {weights.detach().cpu().numpy().round(4).tolist()}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, train=True
        )
        val_loss, val_acc, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler, train=False
        )
        scheduler.step(val_acc)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} lr={row['learning_rate']:.2e}"
        )

        if val_acc > best_val:
            best_val = val_acc
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "num_classes": NUM_CLASSES,
                    "seed": args.seed,
                    "best_val_accuracy": best_val,
                    "model": "proposed lightweight CNN + SE attention",
                    "experiment": "proposed_lightweight_cnn_attention",
                    "training_augmentation": True,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stop_patience:
                print("Early stopping triggered.")
                break

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, y_true, y_pred = run_epoch(
        model, test_loader, criterion, optimizer, device, scaler, train=False
    )
    latency_ms = measure_latency(model, device)
    save_evaluation(
        output_dir,
        y_true,
        y_pred,
        model,
        latency_ms,
        experiment="proposed_lightweight_cnn_attention",
    )
    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Single-image latency: {latency_ms:.3f} ms")
    print(f"Checkpoint: {checkpoint_path}")
    return {
        "experiment": "proposed_lightweight_cnn_attention",
        "best_val_accuracy": best_val,
        "test_accuracy": test_acc,
        "latency_ms": latency_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
