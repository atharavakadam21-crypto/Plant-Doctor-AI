"""CLI preview for leaf segmentation and severity heuristic.

Usage:
    python preprocessing/run_segmentation.py --image path/to/leaf.jpg

Outputs a PNG containing the RGB image, leaf mask, lesion mask, and overlay
under results/segmentation/. It also prints the severity index and components.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from segment_leaf import segment_and_score


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    m = mask > 0
    out[m] = 0.55 * out[m] + 0.45 * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/segmentation"))
    args = parser.parse_args()

    result = segment_and_score(args.image)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    leaf_overlay = overlay_mask(result.rgb, result.leaf_mask, (0, 255, 0))
    lesion_overlay = overlay_mask(result.rgb, result.lesion_mask, (255, 0, 0))

    canvas = np.concatenate(
        [
            result.rgb,
            np.repeat(result.leaf_mask[..., None], 3, axis=2),
            np.repeat(result.lesion_mask[..., None], 3, axis=2),
            lesion_overlay,
        ],
        axis=1,
    )

    out_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    output_path = args.output_dir / f"{args.image.stem}_segmentation.png"
    cv2.imwrite(str(output_path), out_bgr)

    leaf_area = int(np.count_nonzero(result.leaf_mask))
    chlorotic_area = int(np.count_nonzero(result.chlorotic_mask))
    necrotic_area = int(np.count_nonzero(result.necrotic_mask))

    print(f"Input: {args.image}")
    print(f"Leaf area (pixels): {leaf_area}")
    print(f"Chlorotic proxy area (pixels): {chlorotic_area}")
    print(f"Necrotic proxy area (pixels): {necrotic_area}")
    print(f"Severity index (%): {result.severity_percent:.2f}")
    print("Severity thresholds (project-defined):")
    if result.severity_percent <= 10:
        print("  Low")
    elif result.severity_percent <= 30:
        print("  Moderate")
    elif result.severity_percent <= 60:
        print("  High")
    else:
        print("  Severe")
    print(f"Preview: {output_path}")
    print("Note: lesion masks and severity are heuristic proxies, not expert-validated ground truth.")


if __name__ == "__main__":
    main()
