"""Convenience runner for EfficientNet-B0 augmentation ablation experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from training.efficientnet_b0_train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EfficientNet-B0 with traditional RGB training augmentation (Dataset B)."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Reuse the same baseline implementation but opt into Dataset B's
    # traditional training augmentation. Validation/test remain deterministic.
    import training.efficientnet_b0_train as trainer
    original_create = trainer.create_dataloaders

    def augmented_loaders(*loader_args, **loader_kwargs):
        loader_kwargs["augmented"] = True
        return original_create(*loader_args, **loader_kwargs)

    trainer.create_dataloaders = augmented_loaders
    result = train(args)
    print("Dataset B experiment finished:", result)


if __name__ == "__main__":
    main()
