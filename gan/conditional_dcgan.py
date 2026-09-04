"""Conditional DCGAN for class-conditional PlantVillage image synthesis.

The GAN is trained at 128x128 RGB resolution. It is an augmentation experiment,
not the classifier itself. Validation/test images are never used for GAN training.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    """Generate RGB images conditioned on one of N class labels."""

    def __init__(self, latent_dim: int = 128, num_classes: int = 8, base_channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        self.project = nn.Sequential(
            nn.Linear(latent_dim * 2, base_channels * 16 * 4 * 4),
            nn.BatchNorm1d(base_channels * 16 * 4 * 4),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embedded = self.label_embedding(labels)
        x = torch.cat([noise, embedded], dim=1)
        x = self.project(x).view(x.size(0), -1, 4, 4)
        return self.net(x)


class ConditionalDiscriminator(nn.Module):
    """Judge real/fake images while conditioning on the class label."""

    def __init__(self, num_classes: int = 8, image_size: int = 128, base_channels: int = 64):
        super().__init__()
        if image_size != 128:
            raise ValueError("This implementation is intentionally fixed at 128x128.")

        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)
        self.features = nn.Sequential(
            nn.Conv2d(4, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 4, 2, 1),
        )
        self.output_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 128, 128):
            raise ValueError(
                "Expected images with shape [B, 3, 128, 128], "
                f"got {tuple(images.shape)}"
            )
        label_map = self.label_embedding(labels).view(labels.size(0), 1, 128, 128)
        x = torch.cat([images, label_map], dim=1)
        x = self.features(x)
        x = self.output_pool(x)
        return x.flatten(1).squeeze(1)


def initialize_weights(module: nn.Module) -> None:
    """DCGAN-style weight initialization."""
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
