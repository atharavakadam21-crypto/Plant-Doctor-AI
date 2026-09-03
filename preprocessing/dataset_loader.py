"""PyTorch Dataset/DataLoader helpers for the prepared PlantVillage split."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from preprocessing.transforms import build_eval_transform, build_train_transform

# Canonical project label order. Model output indices must remain stable even
# though torchvision ImageFolder discovers directory names alphabetically.
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
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


class PlantVillageImageFolder(ImageFolder):
    """ImageFolder with an explicit project-wide class-index mapping.

    ImageFolder sorts class-directory names alphabetically. That is fine for
    discovery, but it is unsafe for a research project whose label order must
    remain identical across training, evaluation, checkpoints, and reports.
    """

    def __init__(self, root: str | Path, transform=None):
        super().__init__(root=root, transform=transform)
        found = set(self.classes)
        expected = set(CLASS_NAMES)
        if found != expected:
            missing = sorted(expected - found)
            unexpected = sorted(found - expected)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise RuntimeError(
                "Prepared dataset classes do not match the project contract: "
                + "; ".join(details)
            )

        # Keep torchvision's samples/images intact, but remap targets into the
        # canonical project order. The dataset must therefore expose a stable
        # class_to_idx matching CLASS_NAMES exactly.
        discovered = dict(self.class_to_idx)
        self.class_to_idx = dict(CLASS_TO_INDEX)
        self.classes = list(CLASS_NAMES)
        self.targets = [CLASS_TO_INDEX[self.classes_from_path(path)] for path, _ in self.samples]
        self.samples = [(path, target) for (path, _), target in zip(self.samples, self.targets)]
        self.imgs = self.samples

        # Preserve the discovered mapping for audit/debugging.
        self.discovered_class_to_idx = discovered

    @staticmethod
    def classes_from_path(path: str | Path) -> str:
        return Path(path).parent.name


# Backwards-compatible helper used by inspection/training code.
def _check_classes(folder: PlantVillageImageFolder) -> None:
    if folder.classes != CLASS_NAMES:
        raise RuntimeError(
            "Class mapping mismatch. Expected exactly: "
            + ", ".join(CLASS_NAMES)
            + f"; found: {folder.classes}"
        )


def create_datasets(dataset_root: str | Path, augmented: bool = False):
    """Return train, validation and test datasets.

    RGB images are loaded as 3-channel images. Only the training set receives
    augmentation when augmented=True. Validation and test remain deterministic.
    """
    root = Path(dataset_root)
    train = PlantVillageImageFolder(
        root / "train", transform=build_train_transform(augmented)
    )
    validation = PlantVillageImageFolder(
        root / "validation", transform=build_eval_transform()
    )
    test = PlantVillageImageFolder(
        root / "test", transform=build_eval_transform()
    )
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
