"""PyTorch Dataset/DataLoader helpers for the prepared PlantVillage split."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from preprocessing.transforms import build_eval_transform, build_train_transform

CLASS_NAMES = [
    "Tomato__healthy",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
    "Potato__healthy",
    "Potato__Early_blight",
    "Potato__Late_blight",
    "Pepper_bell__healthy",
    "Pepper_bell__Bacterial_spot",
]


def _check_classes(folder: ImageFolder) -> None:
    if folder.classes != CLASS_NAMES:
        raise RuntimeError(
            "Class order mismatch. Expected exactly: "
            + ", ".join(CLASS_NAMES)
            + f"; found: {folder.classes}"
        )


def create_datasets(dataset_root: str | Path, augmented: bool = False):
    """Return train, validation and test ImageFolder datasets.

    RGB images are loaded by torchvision and converted to 3-channel RGB.
    Only the training set receives augmentation when augmented=True.
    """
    root = Path(dataset_root)
    train = ImageFolder(root / "train", transform=build_train_transform(augmented))
    validation = ImageFolder(root / "validation", transform=build_eval_transform())
    test = ImageFolder(root / "test", transform=build_eval_transform())
    _check_classes(train)
    _check_classes(validation)
    _check_classes(test)
    return train, validation, test


def create_dataloaders(
    dataset_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 2,
    augmented: bool = False,
    seed: int = 42,
):
    """Create reproducible DataLoaders for one experiment."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train, validation, test = create_datasets(dataset_root, augmented=augmented)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader, test_loader
