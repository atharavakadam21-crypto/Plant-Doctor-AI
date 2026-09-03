"""Multi-class sanity validation for the heuristic segmentation pipeline.

Runs on one test image per selected class and reports leaf geometry quality,
lesion coverage, and severity. This is an engineering sanity check, not a
validated agricultural ground-truth benchmark.

Usage:
    python -m preprocessing.validate_segmentation --dataset dataset/test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.segment_leaf import segment_and_score
from preprocessing.segmentation_quality import summarize

DEFAULT_CLASSES = [
    "Tomato__healthy",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
    "Potato__healthy",
    "Potato__Early_blight",
    "Potato__Late_blight",
    "Pepper_bell__healthy",
    "Pepper_bell__Bacterial_spot",
]


def first_image(class_dir: Path) -> Path:
    files = sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise FileNotFoundError(f"No supported images found in {class_dir}")
    return files[0]


def overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = mask > 0
    out[m] = 0.55 * out[m] + 0.45 * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset/test"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/segmentation/validation")
    )
    args = parser.parse_args()

    rows = []
    panels = []

    for class_name in DEFAULT_CLASSES:
        class_dir = args.dataset / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        path = first_image(class_dir)
        result = segment_and_score(path)
        quality = summarize(result.leaf_mask, result.lesion_mask, result.severity_percent)

        rows.append(
            {
                "class_name": class_name,
                "image": str(path),
                "leaf_coverage_percent": round(quality.leaf_coverage_percent, 2),
                "lesion_coverage_percent_of_leaf": round(quality.lesion_coverage_percent_of_leaf, 2),
                "severity_percent": round(quality.severity_percent, 2),
                "chlorotic_pixels": int(np.count_nonzero(result.chlorotic_mask)),
                "necrotic_pixels": int(np.count_nonzero(result.necrotic_mask)),
                "lesion_components": quality.lesion_components,
                "border_contact_percent": round(quality.border_contact_percent, 2),
                "solidity": round(quality.solidity, 3),
                "extent": round(quality.extent, 3),
                "segmentation_quality_score": round(quality.quality_score, 2),
                "segmentation_quality": quality.quality,
                "usable_for_severity": quality.usable_for_severity,
            }
        )

        boundary_overlay = overlay(result.rgb, result.boundary_mask, (0, 255, 0))
        lesion_overlay = overlay(result.rgb, result.lesion_mask, (255, 0, 0))
        panels.append((class_name, result.rgb, boundary_overlay, result.lesion_mask, lesion_overlay, quality.quality))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "segmentation_sanity_results.csv"
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(len(panels), 4, figsize=(16, 4 * len(panels)))
    if len(panels) == 1:
        axes = np.asarray([axes])

    for row_idx, (class_name, rgb, boundary, lesion, lesion_overlay, quality_label) in enumerate(panels):
        display = [rgb, boundary, lesion, lesion_overlay]
        titles = [
            f"{class_name}\nRGB",
            f"Leaf boundary\nquality={quality_label}",
            "Lesion proxy",
            "Lesion overlay",
        ]
        for col_idx, (img, title) in enumerate(zip(display, titles)):
            ax = axes[row_idx, col_idx]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    fig.tight_layout()
    figure_path = args.output_dir / "segmentation_sanity_gallery.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Segmentation sanity validation completed.")
    print(df.to_string(index=False))
    print(f"CSV: {csv_path}")
    print(f"Gallery: {figure_path}")
    print(
        "Note: lesion masks and severity are heuristic proxies; no expert-annotated "
        "lesion ground truth is assumed. Segmentation quality is a geometry-based flag."
    )


if __name__ == "__main__":
    main()
