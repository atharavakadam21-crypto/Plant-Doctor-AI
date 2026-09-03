"""Color-space leaf segmentation and heuristic lesion-area estimation.

The RGB image remains the primary classifier input. This module uses HSV/Lab
signals only for analysis: leaf foreground extraction, boundary extraction,
and a heuristic lesion/severity proxy. The lesion masks are not expert-
validated ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class SegmentationResult:
    rgb: np.ndarray
    leaf_mask: np.ndarray
    boundary_mask: np.ndarray
    lesion_mask: np.ndarray
    chlorotic_mask: np.ndarray
    necrotic_mask: np.ndarray
    severity_percent: float


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def make_leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Extract a leaf foreground mask using HSV + Lab signals and morphology."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    _, s, v = cv2.split(hsv)
    _, a, _ = cv2.split(lab)

    # PlantVillage backgrounds are often neutral/low-saturation while leaf
    # tissue is more chromatic. Lab a* adds a green-vs-neutral cue.
    sat_mask = (s >= 35).astype(np.uint8) * 255
    green_a_mask = (a <= 150).astype(np.uint8) * 255
    chromatic = ((s >= 25) & (v >= 25)).astype(np.uint8) * 255

    combined = cv2.bitwise_or(
        cv2.bitwise_and(sat_mask, green_a_mask),
        cv2.bitwise_and(chromatic, sat_mask),
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = _largest_component(combined)

    # Fill internal holes so disease spots do not become holes in the leaf mask.
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(combined)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    return filled


def extract_boundary(leaf_mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Extract the external leaf boundary from the binary foreground mask."""
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(leaf_mask)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(boundary, [contour], -1, 255, thickness=thickness)
    return boundary


def estimate_lesions(
    rgb: np.ndarray, leaf_mask: np.ndarray, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate healthy/chlorotic/necrotic regions using Lab K-Means (k=3).

    Cluster labels are mapped with simple color/brightness heuristics. This is
    a proxy mask for severity and Grad-CAM localization evaluation, not expert
    ground truth.
    """
    leaf_pixels = leaf_mask > 0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    coords = np.column_stack(np.where(leaf_pixels))
    if len(coords) < 30:
        empty = np.zeros(leaf_mask.shape, dtype=np.uint8)
        return empty, empty.copy(), empty.copy()

    pixels = lab[leaf_pixels]
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    necrotic_label = int(np.argmin(centers[:, 0]))
    remaining = [i for i in range(3) if i != necrotic_label]
    chlorotic_label = int(max(remaining, key=lambda i: centers[i, 2]))

    label_image = np.full(leaf_mask.shape, -1, dtype=np.int8)
    label_image[coords[:, 0], coords[:, 1]] = labels

    necrotic = ((label_image == necrotic_label) & leaf_pixels).astype(np.uint8) * 255
    chlorotic = ((label_image == chlorotic_label) & leaf_pixels).astype(np.uint8) * 255
    lesion = cv2.bitwise_or(necrotic, chlorotic)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    necrotic = cv2.morphologyEx(necrotic, cv2.MORPH_OPEN, kernel)
    chlorotic = cv2.morphologyEx(chlorotic, cv2.MORPH_OPEN, kernel)
    lesion = cv2.bitwise_or(necrotic, chlorotic)
    return lesion, chlorotic, necrotic


def segment_and_score(image_path: str | Path, random_state: int = 42) -> SegmentationResult:
    path = Path(image_path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    leaf_mask = make_leaf_mask(rgb)
    boundary_mask = extract_boundary(leaf_mask)
    lesion, chlorotic, necrotic = estimate_lesions(rgb, leaf_mask, random_state)

    leaf_area = int(np.count_nonzero(leaf_mask))
    necrotic_area = int(np.count_nonzero(necrotic))
    chlorotic_area = int(np.count_nonzero(chlorotic))
    if leaf_area == 0:
        severity = 0.0
    else:
        severity = (necrotic_area + 0.5 * chlorotic_area) / leaf_area * 100.0

    return SegmentationResult(
        rgb=rgb,
        leaf_mask=leaf_mask,
        boundary_mask=boundary_mask,
        lesion_mask=lesion,
        chlorotic_mask=chlorotic,
        necrotic_mask=necrotic,
        severity_percent=float(np.clip(severity, 0.0, 100.0)),
    )
