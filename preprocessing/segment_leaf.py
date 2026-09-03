"""Color-space leaf segmentation and heuristic lesion-area estimation.

The RGB image remains the primary classifier input. HSV/Lab are used only for
analysis. Lesion masks and severity are heuristic proxies, not expert-annotated
agricultural ground truth.
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
    if min_area <= 1:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def make_leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Extract a leaf mask using adaptive HSV saturation + Lab green cues."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    _, s, v = cv2.split(hsv)
    _, a, _ = cv2.split(lab)

    otsu_threshold, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sat_threshold = max(20, min(int(otsu_threshold), 70))

    saturated = (s >= sat_threshold) & (v >= 20)
    green_tissue = (a <= 150) & (s >= 15) & (v >= 20)
    candidate = (saturated | green_tissue).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
    candidate = _largest_component(candidate)

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(candidate)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    return filled


def extract_boundary(leaf_mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(leaf_mask)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(boundary, [contour], -1, 255, thickness=thickness)
    return boundary


def estimate_lesions(
    rgb: np.ndarray, leaf_mask: np.ndarray, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate lesion candidates with Lab K-Means plus conservative color gates.

    K-Means is used only to propose color regions. The non-healthy clusters are
    counted as lesions only when their centroids differ from a green healthy
    reference. The output is a heuristic proxy, not expert ground truth.
    """
    leaf_pixels = leaf_mask > 0
    empty = np.zeros_like(leaf_mask)
    if int(leaf_pixels.sum()) < 100:
        return empty, empty.copy(), empty.copy()

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    interior_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    interior = cv2.erode(leaf_mask, interior_kernel, iterations=1) > 0
    if int(interior.sum()) < 100:
        interior = leaf_pixels

    coords = np.column_stack(np.where(interior))
    pixels = lab[interior]
    if len(pixels) < 30:
        return empty, empty.copy(), empty.copy()

    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    counts = np.bincount(labels, minlength=3)

    # Prefer a well-supported green cluster as the healthy reference.
    green_candidates = [i for i in range(3) if centers[i, 1] <= np.percentile(centers[:, 1], 66)]
    healthy_label = max(green_candidates, key=lambda i: counts[i]) if green_candidates else int(np.argmax(counts))
    healthy = centers[healthy_label]
    remaining = [i for i in range(3) if i != healthy_label]

    chlorotic_label = max(remaining, key=lambda i: centers[i, 2]) if remaining else None
    necrotic_label = None
    if remaining and chlorotic_label is not None:
        rest = [i for i in remaining if i != chlorotic_label]
        if rest:
            necrotic_label = min(rest, key=lambda i: centers[i, 0])

    # Robust within-leaf spreads make the gates image-adaptive instead of fixed
    # to one photographed leaf.
    mad = np.median(np.abs(pixels - np.median(pixels, axis=0)), axis=0)
    l_gate = max(12.0, float(mad[0]) * 1.8)
    a_gate = max(5.0, float(mad[1]) * 1.0)
    b_gate = max(7.0, float(mad[2]) * 1.0)

    chlorotic_valid = False
    if chlorotic_label is not None:
        c = centers[chlorotic_label]
        chlorotic_valid = (
            c[2] >= healthy[2] + b_gate
            and c[0] >= healthy[0] - l_gate
            and c[1] >= healthy[1] - a_gate
        )

    necrotic_valid = False
    if necrotic_label is not None:
        n = centers[necrotic_label]
        necrotic_valid = (
            (n[0] <= healthy[0] - l_gate and n[1] >= healthy[1] - a_gate)
            or (n[1] >= healthy[1] + a_gate and n[0] <= healthy[0] + 10.0)
        )

    label_image = np.full(leaf_mask.shape, -1, dtype=np.int8)
    label_image[coords[:, 0], coords[:, 1]] = labels

    chlorotic = (
        ((label_image == chlorotic_label) & leaf_pixels).astype(np.uint8) * 255
        if chlorotic_valid and chlorotic_label is not None else empty.copy()
    )
    necrotic = (
        ((label_image == necrotic_label) & leaf_pixels).astype(np.uint8) * 255
        if necrotic_valid and necrotic_label is not None else empty.copy()
    )

    # Keep only spatially meaningful candidate regions.
    min_area = max(20, int(np.count_nonzero(leaf_mask) * 0.0002))
    chlorotic = _remove_small_components(chlorotic, min_area)
    necrotic = _remove_small_components(necrotic, min_area)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    chlorotic = cv2.morphologyEx(chlorotic, cv2.MORPH_OPEN, small_kernel)
    necrotic = cv2.morphologyEx(necrotic, cv2.MORPH_OPEN, small_kernel)
    lesion = cv2.bitwise_and(cv2.bitwise_or(chlorotic, necrotic), leaf_mask)
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
    chlorotic_area = int(np.count_nonzero(chlorotic))
    necrotic_area = int(np.count_nonzero(necrotic))
    severity = 0.0 if leaf_area == 0 else (
        (necrotic_area + 0.5 * chlorotic_area) / leaf_area * 100.0
    )

    return SegmentationResult(
        rgb=rgb,
        leaf_mask=leaf_mask,
        boundary_mask=boundary_mask,
        lesion_mask=lesion,
        chlorotic_mask=chlorotic,
        necrotic_mask=necrotic,
        severity_percent=float(np.clip(severity, 0.0, 100.0)),
    )
