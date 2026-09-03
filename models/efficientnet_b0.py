"""EfficientNet-B0 model factory for the Plant Doctor baseline."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet_b0(num_classes: int = 8, pretrained: bool = True) -> nn.Module:
    """Return an EfficientNet-B0 classifier with the requested output classes."""
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
