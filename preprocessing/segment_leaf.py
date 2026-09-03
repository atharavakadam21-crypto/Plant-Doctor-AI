"""Leaf segmentation and conservative lesion/severity proxy.

RGB remains the primary classifier input. HSV/Lab and RGB-derived cues are
used only for analysis. Masks are engineering heuristics, not expert ground
truth.
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
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            out[labels == label] = 255
    return out


def _rough_leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Conservative proposal using ExG plus green/chromatic HSV cues."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    rgbf = rgb.astype(np.float32)
    r, g, b = cv2.split(rgbf)

    exg = 2.0 * g - r - b
    exg_u8 = np.clip(exg + 128.0, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    exg_threshold = max(8.0, float(otsu) - 128.0)

    green_hue = (h >= 25) & (h <= 100) & (s >= 35) & (v >= 25)
    strong_green = exg >= exg_threshold
    chromatic_leaf = (s >= 45) & (h >= 15) & (h <= 105) & (v >= 25)

    candidate = (strong_green | (green_hue & chromatic_leaf)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return _largest_component(candidate)


def make_leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Create a conservative leaf mask with constrained GrabCut."""
    rough = _rough_leaf_mask(rgb)
    rough_area = int(np.count_nonzero(rough))
    if rough_area < 100:
        return rough

    h, w = rough.shape
    ys, xs = np.where(rough > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x = max(4, int(0.02 * w))
    pad_y = max(4, int(0.02 * h))
    rx0, ry0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    rx1, ry1 = min(w - 1, x1 + pad_x), min(h - 1, y1 + pad_y)

    mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    mask[ry0:ry1 + 1, rx0:rx1 + 1] = cv2.GC_PR_BGD
    mask[rough > 0] = cv2.GC_PR_FGD

    sure_fg = cv2.erode(
        rough,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    mask[sure_fg > 0] = cv2.GC_FGD

    border = 3
    for sl in (
        (slice(0, border), slice(None)),
        (slice(-border, None), slice(None)),
        (slice(None), slice(0, border)),
        (slice(None), slice(-border, None)),
    ):
        region = mask[sl]
        definite = region == cv2.GC_FGD
        region[:] = np.where(definite, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)
        refined = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    except cv2.error:
        refined = rough.copy()

    allowed = cv2.dilate(
        rough,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    refined = cv2.bitwise_and(refined, allowed)
    refined_area = int(np.count_nonzero(refined))
    ratio = refined_area / max(1, rough_area)
    if ratio < 0.65 or ratio > 1.20:
        refined = rough.copy()

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    refined = _largest_component(refined)

    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(refined)
    if contours:
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return filled


def extract_boundary(leaf_mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(leaf_mask)
    if contours:
        cv2.drawContours(boundary, [max(contours, key=cv2.contourArea)], -1, 255, thickness)
    return boundary


def _healthy_reference(lab: np.ndarray, hsv: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    leaf = leaf_mask > 0
    h, s, v = cv2.split(hsv)
    _, a, _ = cv2.split(lab)
    green = leaf & (h >= 30) & (h <= 95) & (s >= 45) & (v >= 35) & (a <= 145)
    if int(green.sum()) < max(100, int(leaf.sum() * 0.03)):
        values = a[leaf]
        green = leaf & (a <= np.percentile(values, 45))
    pixels = lab[green]
    if len(pixels) == 0:
        pixels = lab[leaf]
    return np.median(pixels, axis=0).astype(np.float32)


def _kmeans_regions(lab: np.ndarray, leaf_mask: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    interior = cv2.erode(
        leaf_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ) > 0
    if int(interior.sum()) < 100:
        interior = leaf_mask > 0
    coords = np.column_stack(np.where(interior))
    pixels = lab[interior].astype(np.float32)
    if len(coords) < 30:
        return np.full(leaf_mask.shape, -1, np.int8), np.empty((0, 3), np.float32)
    model = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    labels = model.fit_predict(pixels)
    image_labels = np.full(leaf_mask.shape, -1, np.int8)
    image_labels[coords[:, 0], coords[:, 1]] = labels
    return image_labels, model.cluster_centers_


def estimate_lesions(
    rgb: np.ndarray, leaf_mask: np.ndarray, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    leaf = leaf_mask > 0
    empty = np.zeros_like(leaf_mask)
    if int(leaf.sum()) < 100:
        return empty, empty.copy(), empty.copy()

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    l, a, b = [x.astype(np.float32) for x in cv2.split(lab)]
    h, s, v = [x.astype(np.float32) for x in cv2.split(hsv)]
    ref_l, ref_a, ref_b = _healthy_reference(lab, hsv, leaf_mask)

    labels, centers = _kmeans_regions(lab, leaf_mask, random_state)
    if centers.size == 0:
        return empty, empty.copy(), empty.copy()

    distance = np.sqrt(
        ((centers[:, 0] - ref_l) / 18.0) ** 2
        + ((centers[:, 1] - ref_a) / 10.0) ** 2
        + ((centers[:, 2] - ref_b) / 12.0) ** 2
    )
    meaningful = np.zeros_like(leaf, dtype=bool)
    for i, d in enumerate(distance):
        if d >= 1.30:
            meaningful |= labels == i

    a_delta, b_delta, l_delta = a - ref_a, b - ref_b, l - ref_l
    brown_red = (a_delta >= 10) & (b_delta >= -3) & (s >= 38) & (v >= 25)
    yellow = (b_delta >= 14) & (a_delta >= -6) & (l >= ref_l - 8) & (s >= 35)
    dark_abnormal = (l_delta <= -22) & (a_delta >= 4) & (s >= 38)

    necrotic = leaf & meaningful & (brown_red | dark_abnormal)
    chlorotic = leaf & meaningful & yellow & ~necrotic

    boundary_band = cv2.dilate(
        extract_boundary(leaf_mask, thickness=3),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    necrotic[boundary_band] = False
    chlorotic[boundary_band] = False

    leaf_area = max(1, int(leaf.sum()))
    min_area = max(30, int(leaf_area * 0.00025))
    necrotic_u8 = _remove_small_components(necrotic.astype(np.uint8) * 255, min_area)
    chlorotic_u8 = _remove_small_components(chlorotic.astype(np.uint8) * 255, min_area)
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    necrotic_u8 = cv2.morphologyEx(necrotic_u8, cv2.MORPH_OPEN, small)
    chlorotic_u8 = cv2.morphologyEx(chlorotic_u8, cv2.MORPH_OPEN, small)
    necrotic_u8 = cv2.bitwise_and(necrotic_u8, leaf_mask)
    chlorotic_u8 = cv2.bitwise_and(chlorotic_u8, leaf_mask)
    chlorotic_u8[necrotic_u8 > 0] = 0
    lesion = cv2.bitwise_or(necrotic_u8, chlorotic_u8)
    return lesion, chlorotic_u8, necrotic_u8


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
