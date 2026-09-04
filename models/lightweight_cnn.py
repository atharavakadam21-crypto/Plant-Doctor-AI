"""Proposed lightweight CNN with squeeze-and-excitation attention.

The model is intentionally small so it can be compared with EfficientNet-B0
for accuracy, parameter count, model size, and inference latency.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNAct(nn.Sequential):
    """Standard convolution block used for the input stem."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class SEAttention(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class DepthwiseSeparableAttentionBlock(nn.Module):
    """Depthwise-separable convolution block followed by SE attention."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride,
                1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        self.attention = SEAttention(out_channels)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.attention(x)
        if self.use_residual:
            x = x + identity
        return x


class LightweightCNN(nn.Module):
    """Proposed lightweight RGB classifier with channel attention."""

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 24, stride=2),
            DepthwiseSeparableAttentionBlock(24, 32, stride=1),
            DepthwiseSeparableAttentionBlock(32, 48, stride=2),
            DepthwiseSeparableAttentionBlock(48, 48, stride=1),
            DepthwiseSeparableAttentionBlock(48, 64, stride=2),
            DepthwiseSeparableAttentionBlock(64, 64, stride=1),
            DepthwiseSeparableAttentionBlock(64, 96, stride=2),
            DepthwiseSeparableAttentionBlock(96, 96, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.20)
        self.classifier = nn.Linear(96, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "Expected RGB input with shape [B, 3, H, W], "
                f"got {tuple(x.shape)}"
            )
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


def build_lightweight_cnn(num_classes: int = 8) -> LightweightCNN:
    """Construct the proposed lightweight CNN."""
    return LightweightCNN(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
