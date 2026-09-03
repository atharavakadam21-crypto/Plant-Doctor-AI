"""Segmentation quality helpers for multi-image sanity validation.

This module adds conservative reporting only. It does not claim expert lesion
annotations or an agricultural validation standard.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegmentationQuality:
    leaf_area_px: int
    image_area_px: int
    leaf_coverage_percent: float
    lesion_area_px: int
    lesion_coverage_percent_of_leaf: float
    severity_percent: float
    lesion_components: int


def count_components(mask: np.ndarray) -> int:
    binary = (mask > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return 0
    return int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] > 0))


def summarize(mask: np.ndarray, lesion_mask: np.ndarray, severity_percent: float) -> SegmentationQuality:
    leaf_area = int(np.count_nonzero(mask))
    image_area = int(mask.shape[0] * mask.shape[1])
    lesion_area = int(np.count_nonzero(lesion_mask))
    leaf_coverage = (leaf_area / max(1, image_area)) * 100.0
    lesion_coverage = (lesion_area / max(1, leaf_area)) * 100.0
    return SegmentationQuality(
        leaf_area_px=leaf_area,
        image_area_px=image_area,
        leaf_coverage_percent=float(leaf_coverage),
        lesion_area_px=lesion_area,
        lesion_coverage_percent_of_leaf=float(lesion_coverage),
        severity_percent=float(severity_percent),
        lesion_components=count_components(lesion_mask),
    )
