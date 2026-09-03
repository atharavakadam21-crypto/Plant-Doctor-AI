"""RGB classifier transforms for Plant Doctor AI.

Classifier input is always RGB. Segmentation/severity code is intentionally
separate and must not replace this input with grayscale or binary images.
"""
from __future__ import annotations

from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(augmented: bool = False) -> transforms.Compose:
    """Build the training transform.

    Dataset A (augmented=False) performs only resize/crop/tensor/normalization.
    Dataset B (augmented=True) adds conservative geometric and photometric
    augmentation. Validation and test use build_eval_transform().
    """
    ops: list[transforms.Transform] = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    if augmented:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.15),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03
                ),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def build_eval_transform() -> transforms.Compose:
    """Deterministic transform for validation and test images."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
