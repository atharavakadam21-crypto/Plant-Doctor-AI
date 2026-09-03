"""Segmentation quality helpers for multi-image sanity validation.

These checks are conservative engineering diagnostics. They do not claim
expert lesion annotations or an agricultural validation standard.
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
    border_contact_percent: float
    solidity: float
    extent: float
    quality_score: float
    quality: str
    usable_for_severity: bool


def count_components(mask: np.ndarray) -> int:
    binary = (mask > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return 0
    return int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] > 0))


def summarize(mask: np.ndarray, lesion_mask: np.ndarray, severity_percent: float) -> SegmentationQuality:
    """Summarize area, geometry, and confidence indicators for one image."""
    leaf_binary = (mask > 0).astype(np.uint8)
    leaf_area = int(leaf_binary.sum())
    image_area = int(mask.shape[0] * mask.shape[1])
    lesion_area = int(np.count_nonzero(lesion_mask))
    leaf_coverage = (leaf_area / max(1, image_area)) * 100.0
    lesion_coverage = (lesion_area / max(1, leaf_area)) * 100.0

    contours, _ = cv2.findContours(leaf_binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or leaf_area == 0:
        return SegmentationQuality(
            leaf_area_px=leaf_area,
            image_area_px=image_area,
            leaf_coverage_percent=float(leaf_coverage),
            lesion_area_px=lesion_area,
            lesion_coverage_percent_of_leaf=float(lesion_coverage),
            severity_percent=float(severity_percent),
            lesion_components=count_components(lesion_mask),
            border_contact_percent=100.0,
            solidity=0.0,
            extent=0.0,
            quality_score=0.0,
            quality="LOW_CONFIDENCE",
            usable_for_severity=False,
        )

    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = contour_area / hull_area if hull_area > 0 else 0.0
    x, y, width, height = cv2.boundingRect(contour)
    extent = contour_area / max(1, width * height)

    h, w = mask.shape[:2]
    border = max(2, int(round(min(h, w) * 0.01)))
    border_pixels = np.concatenate(
        [
            leaf_binary[:border, :].ravel(),
            leaf_binary[-border:, :].ravel(),
            leaf_binary[:, :border].ravel(),
            leaf_binary[:, -border:].ravel(),
        ]
    )
    border_contact = float(border_pixels.sum()) / max(1.0, float(4 * border * min(h, w)))
    border_contact = min(1.0, max(0.0, border_contact))

    # Geometry-based quality score. This is only a flagging heuristic: the
    # project does not have expert leaf masks to turn this into a true metric.
    coverage_term = 1.0 if 0.08 <= leaf_coverage / 100.0 <= 0.75 else 0.0
    solidity_term = float(np.clip((solidity - 0.30) / 0.55, 0.0, 1.0))
    extent_term = float(np.clip((extent - 0.18) / 0.60, 0.0, 1.0))
    border_term = float(np.clip(1.0 - border_contact / 0.40, 0.0, 1.0))
    score = 100.0 * (
        0.35 * coverage_term
        + 0.25 * solidity_term
        + 0.20 * extent_term
        + 0.20 * border_term
    )

    if score >= 75.0:
        quality = "PASS"
    elif score >= 55.0:
        quality = "REVIEW"
    else:
        quality = "LOW_CONFIDENCE"

    return SegmentationQuality(
        leaf_area_px=leaf_area,
        image_area_px=image_area,
        leaf_coverage_percent=float(leaf_coverage),
        lesion_area_px=lesion_area,
        lesion_coverage_percent_of_leaf=float(lesion_coverage),
        severity_percent=float(severity_percent),
        lesion_components=count_components(lesion_mask),
        border_contact_percent=border_contact * 100.0,
        solidity=float(solidity),
        extent=float(extent),
        quality_score=float(score),
        quality=quality,
        usable_for_severity=bool(score >= 55.0),
    )
