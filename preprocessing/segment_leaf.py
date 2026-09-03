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


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove isolated regions smaller than min_area pixels."""
    if min_area <= 1:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def make_leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Extract a leaf foreground mask using HSV + Lab signals and morphology."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    _, s, v = cv2.split(hsv)
    _, a, _ = cv2.split(lab)

    # Neutral gray/white background is typically low saturation. The Lab a*
    # cue helps retain green tissue while rejecting neutral background pixels.
    saturated = (s >= 30) & (v >= 25)
    green_tissue = (a <= 150) & (s >= 20) & (v >= 20)
    candidate = (saturated | green_tissue).astype(np.uint8) * 255

    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, large_kernel, iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, large_kernel, iterations=1)
    candidate = _largest_component(candidate)

    # Fill texture/vein holes: they belong to the leaf area, not the background.
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(candidate)
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
    """Estimate chlorotic/necrotic regions using Lab K-Means (k=3).

    K-Means supplies three color-region candidates. A candidate is counted as
    disease only when its centroid differs sufficiently from the estimated
    healthy cluster. This prevents a healthy leaf from automatically receiving
    roughly one-third lesion area simply because k=3 was requested.

    The masks are heuristic proxies for severity and Grad-CAM localization,
    not expert-annotated ground truth.
    """
    leaf_pixels = leaf_mask > 0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Cluster an eroded interior so leaf-edge shadows contribute less. The full
    # leaf mask still defines the severity denominator.
    interior_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    interior_mask = cv2.erode(leaf_mask, interior_kernel, iterations=1) > 0
    if int(interior_mask.sum()) < 100:
        interior_mask = leaf_pixels

    coords = np.column_stack(np.where(interior_mask))
    pixels = lab[interior_mask]
    if len(coords) < 30:
        empty = np.zeros(leaf_mask.shape, dtype=np.uint8)
        return empty, empty.copy(), empty.copy()

    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    # Greenest cluster becomes the healthy reference.
    healthy_label = int(np.argmin(centers[:, 1]))
    remaining = [i for i in range(3) if i != healthy_label]

    # Higher b* is the yellow/chlorotic candidate; the remaining candidate is
    # tested for a darker, browner/less-green necrotic signature.
    chlorotic_candidate = int(max(remaining, key=lambda i: centers[i, 2]))
    necrotic_candidate = remaining[0] if remaining[1] == chlorotic_candidate else remaining[1]

    healthy = centers[healthy_label]
    chlorotic_center = centers[chlorotic_candidate]
    necrotic_center = centers[necrotic_candidate]

    # Conservative heuristic gates. These values are engineering parameters,
    # not validated agricultural thresholds.
    chlorotic_valid = (
        chlorotic_center[2] - healthy[2] >= 8.0
        and chlorotic_center[0] - healthy[0] >= -18.0
        and chlorotic_center[1] - healthy[1] >= -8.0
    )
    necrotic_valid = (
        necrotic_center[0] - healthy[0] <= -15.0
        and necrotic_center[1] - healthy[1] >= -5.0
    )

    label_image = np.full(leaf_mask.shape, -1, dtype=np.int8)
    label_image[coords[:, 0], coords[:, 1]] = labels

    if chlorotic_valid:
        chlorotic = ((label_image == chlorotic_candidate) & leaf_pixels).astype(np.uint8) * 255
    else:
        chlorotic = np.zeros_like(leaf_mask)

    if necrotic_valid:
        necrotic = ((label_image == necrotic_candidate) & leaf_pixels).astype(np.uint8) * 255
    else:
        necrotic = np.zeros_like(leaf_mask)

    # Remove very small speckles before calculating severity.
    min_area = max(25, int(np.count_nonzero(leaf_mask) * 0.0003))
    chlorotic = _remove_small_components(chlorotic, min_area)
    necrotic = _remove_small_components(necrotic, min_area)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    chlorotic = cv2.morphologyEx(chlorotic, cv2.MORPH_OPEN, small_kernel)
    necrotic = cv2.morphologyEx(necrotic, cv2.MORPH_OPEN, small_kernel)
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
