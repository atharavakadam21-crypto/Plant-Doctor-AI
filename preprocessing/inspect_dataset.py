"""Stage 4 sanity checker for RGB transforms and DataLoaders.

Usage:
    python -m preprocessing.inspect_dataset --dataset dataset
    python -m preprocessing.inspect_dataset --dataset dataset --augmented
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from preprocessing.dataset_loader import CLASS_NAMES, create_datasets, create_dataloaders


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype).view(3, 1, 1)
    image = tensor.cpu() * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augmented", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/preprocessing/train_transform_preview.png"))
    args = parser.parse_args()

    train, validation, test = create_datasets(args.dataset, augmented=args.augmented)
    loaders = create_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmented=args.augmented,
    )
    train_loader, val_loader, test_loader = loaders

    print("Stage 4 dataset inspection")
    print(f"Classes ({len(CLASS_NAMES)}):")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {idx}: {name}")
    print(f"Train samples: {len(train)}")
    print(f"Validation samples: {len(validation)}")
    print(f"Test samples: {len(test)}")
    print(f"Augmentation enabled for train: {args.augmented}")

    images, labels = next(iter(train_loader))
    print(f"Train batch shape: {tuple(images.shape)}")
    print(f"Train labels shape: {tuple(labels.shape)}")
    print(f"Input dtype: {images.dtype}")
    print(f"Input min/max: {images.min().item():.4f} / {images.max().item():.4f}")

    val_images, val_labels = next(iter(val_loader))
    test_images, test_labels = next(iter(test_loader))
    print(f"Validation batch shape: {tuple(val_images.shape)}")
    print(f"Test batch shape: {tuple(test_images.shape)}")
    assert images.shape[1:] == (3, 224, 224)
    assert val_images.shape[1:] == (3, 224, 224)
    assert test_images.shape[1:] == (3, 224, 224)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = min(6, images.shape[0])
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.ravel()
    for i in range(count):
        axes[i].imshow(denormalize(images[i]))
        axes[i].set_title(CLASS_NAMES[int(labels[i])], fontsize=8)
        axes[i].axis("off")
    for i in range(count, len(axes)):
        axes[i].axis("off")
    fig.suptitle(
        "RGB training transform preview — "
        + ("Dataset B (augmented)" if args.augmented else "Dataset A (original)"),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Preview: {args.output}")
    print("Stage 4 sanity checks: PASSED")


if __name__ == "__main__":
    main()
