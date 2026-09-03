"""Geometry-based diagnostics for heuristic leaf segmentation.

These checks identify suspicious masks; they do not validate disease lesions and
do not claim expert agricultural ground truth.
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
    """Summarize mask geometry and return a conservative quality flag."""
    leaf_binary = (mask > 0).astype(np.uint8)
    leaf_area = int(leaf_binary.sum())
    image_area = int(mask.shape[0] * mask.shape[1])
    lesion_area = int(np.count_nonzero(lesion_mask))
    coverage = leaf_area / max(1, image_area) * 100.0
    lesion_coverage = lesion_area / max(1, leaf_area) * 100.0

    contours, _ = cv2.findContours(leaf_binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or leaf_area == 0:
        return SegmentationQuality(
            leaf_area_px=leaf_area,
            image_area_px=image_area,
            leaf_coverage_percent=coverage,
            lesion_area_px=lesion_area,
            lesion_coverage_percent_of_leaf=lesion_coverage,
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
    solidity = contour_area / hull_area if hull_area else 0.0
    x, y, width, height = cv2.boundingRect(contour)
    extent = contour_area / max(1, width * height)

    h, w = mask.shape[:2]
    edge = max(2, int(round(min(h, w) * 0.01)))
    edge_mask = np.zeros_like(leaf_binary)
    edge_mask[:edge, :] = 1
    edge_mask[-edge:, :] = 1
    edge_mask[:, :edge] = 1
    edge_mask[:, -edge:] = 1
    border_pixels = leaf_binary[edge_mask > 0]
    border_contact = float(border_pixels.mean() * 100.0) if border_pixels.size else 0.0

    # Coverage is only used to flag obviously tiny/all-frame masks. Real leaves
    # may legitimately occupy most of a PlantVillage frame, so there is no hard
    # upper limit such as 75%.
    coverage_score = float(np.clip((coverage - 8.0) / 15.0, 0.0, 1.0))
    all_frame_penalty = float(np.clip((coverage - 96.0) / 4.0, 0.0, 1.0))
    solidity_score = float(np.clip((solidity - 0.45) / 0.45, 0.0, 1.0))
    extent_score = float(np.clip((extent - 0.20) / 0.55, 0.0, 1.0))
    border_score = 1.0 - float(np.clip((border_contact - 75.0) / 25.0, 0.0, 1.0))

    score = 100.0 * (
        0.20 * coverage_score
        + 0.35 * solidity_score
        + 0.30 * extent_score
        + 0.15 * border_score
    )
    score *= 1.0 - 0.35 * all_frame_penalty

    if score >= 70.0:
        quality = "PASS"
    elif score >= 50.0:
        quality = "REVIEW"
    else:
        quality = "LOW_CONFIDENCE"

    # Severity can be used downstream only when the leaf geometry itself is
    # reasonably credible. The exact thresholds are engineering flags.
    usable = bool(score >= 50.0)

    return SegmentationQuality(
        leaf_area_px=leaf_area,
        image_area_px=image_area,
        leaf_coverage_percent=coverage,
        lesion_area_px=lesion_area,
        lesion_coverage_percent_of_leaf=lesion_coverage,
        severity_percent=float(severity_percent),
        lesion_components=count_components(lesion_mask),
        border_contact_percent=border_contact,
        solidity=float(solidity),
        extent=float(extent),
        quality_score=float(score),
        quality=quality,
        usable_for_severity=usable,
    )
