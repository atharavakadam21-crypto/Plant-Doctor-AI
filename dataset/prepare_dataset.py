"""Filter PlantVillage to eight classes and create leakage-aware splits.

Usage:
    python dataset/prepare_dataset.py --source dataset/raw/PlantVillage --output dataset

The classifier always uses original RGB images. This script only prepares
the dataset; it does not preprocess classifier inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Canonical project labels -> folder-name variants seen in PlantVillage mirrors.
TARGET_CLASS_ALIASES = {
    "Tomato__healthy": ["Tomato_healthy", "Tomato___healthy"],
    "Tomato__Early_blight": ["Tomato_Early_blight", "Tomato___Early_blight"],
    "Tomato__Late_blight": ["Tomato_Late_blight", "Tomato___Late_blight"],
    "Potato__healthy": ["Potato___healthy", "Potato_healthy"],
    "Potato__Early_blight": ["Potato___Early_blight", "Potato_Early_blight"],
    "Potato__Late_blight": ["Potato___Late_blight", "Potato_Late_blight"],
    "Pepper_bell__healthy": ["Pepper__bell___healthy", "Pepper_bell_healthy"],
    "Pepper_bell__Bacterial_spot": [
        "Pepper__bell___Bacterial_spot",
        "Pepper_bell_Bacterial_spot",
    ],
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path: Path) -> int | None:
    """Compute a compact perceptual hash for conservative duplicate screening."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(image))
    block = dct[:8, :8]
    median = np.median(block[1:, :].reshape(-1))
    bits = (block > median).astype(np.uint8).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def source_key_from_name(path: Path) -> str:
    """Normalize only explicit augmentation/copy suffixes in filenames."""
    stem = path.stem.lower()
    return re.sub(
        r"([_-](flip|flipped|rotated?|rotation|aug|augment|copy|variant)[_-]?\d*)+$",
        "",
        stem,
    )


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_class(records: list[dict], near_duplicate_threshold: int = 4) -> list[str]:
    """Create conservative source groups within a class.

    Exact duplicates and normalized filename variants are always grouped.
    A perceptual-hash pass is also used to catch very close duplicate/variant
    images. This is a leakage-control heuristic; PlantVillage folder names do
    not provide a guaranteed original-source identifier, so this limitation is
    recorded in dataset_audit.json and must be disclosed in the report.
    """
    uf = UnionFind(len(records))
    by_exact = defaultdict(list)
    by_name = defaultdict(list)

    for i, rec in enumerate(records):
        by_exact[rec["sha256"]].append(i)
        by_name[rec["name_key"]].append(i)

    for bucket in list(by_exact.values()) + list(by_name.values()):
        for i in bucket[1:]:
            uf.union(bucket[0], i)

    # Only compare perceptual hashes inside candidate buckets derived from a
    # coarse visual signature. This avoids the quadratic all-pairs comparison.
    phash_buckets = defaultdict(list)
    for i, rec in enumerate(records):
        if rec["phash"] is not None:
            phash_buckets[rec["phash"] >> 12].append(i)

    for candidates in phash_buckets.values():
        for a in range(len(candidates)):
            i = candidates[a]
            for b in range(a + 1, len(candidates)):
                j = candidates[b]
                if hamming_distance(records[i]["phash"], records[j]["phash"]) <= near_duplicate_threshold:
                    uf.union(i, j)

    root_to_group: dict[int, str] = {}
    group_ids: list[str] = []
    class_name = records[0]["class_name"] if records else "empty"
    for i in range(len(records)):
        root = uf.find(i)
        if root not in root_to_group:
            root_to_group[root] = f"{class_name}_group_{len(root_to_group):05d}"
        group_ids.append(root_to_group[root])
    return group_ids


def grouped_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Split 70/20/10 approximately while keeping source groups intact."""
    from sklearn.model_selection import GroupShuffleSplit

    split_frames = []
    for class_name, class_df in df.groupby("class_name", sort=True):
        groups = class_df["source_group"].to_numpy()
        indices = np.arange(len(class_df))

        train_gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_idx, holdout_idx = next(train_gss.split(indices, groups=groups))

        train = class_df.iloc[train_idx].copy()
        holdout = class_df.iloc[holdout_idx].copy()

        val_gss = GroupShuffleSplit(n_splits=1, test_size=1 / 3, random_state=seed + 1)
        val_rel, test_rel = next(
            val_gss.split(np.arange(len(holdout)), groups=holdout["source_group"].to_numpy())
        )

        val = holdout.iloc[val_rel].copy()
        test = holdout.iloc[test_rel].copy()

        train["split"] = "train"
        val["split"] = "validation"
        test["split"] = "test"
        split_frames.extend([train, val, test])

    return pd.concat(split_frames, ignore_index=True)


def find_class_dirs(source: Path) -> dict[str, list[Path]]:
    """Find all alias-matching class directories recursively."""
    matches: dict[str, list[Path]] = {canonical: [] for canonical in TARGET_CLASS_ALIASES}
    alias_to_canonical = {
        alias: canonical
        for canonical, aliases in TARGET_CLASS_ALIASES.items()
        for alias in aliases
    }
    for directory in source.rglob("*"):
        if directory.is_dir() and directory.name in alias_to_canonical:
            matches[alias_to_canonical[directory.name]].append(directory)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating hard links.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source folder not found: {args.source}")

    class_dirs = find_class_dirs(args.source)
    missing = [canonical for canonical, paths in class_dirs.items() if not paths]
    if missing:
        raise RuntimeError(
            "Missing required PlantVillage class folders: " + ", ".join(missing)
        )

    records: list[dict] = []
    for class_name, directories in class_dirs.items():
        files: list[Path] = []
        for directory in directories:
            files.extend(
                p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
            )
        files = sorted(set(files))

        if not files:
            raise RuntimeError(f"No images found for {class_name}")

        class_records = []
        for path in files:
            class_records.append(
                {
                    "source_path": str(path.resolve()),
                    "class_name": class_name,
                    "original_folder": path.parent.name,
                    "filename": path.name,
                    "sha256": sha256_file(path),
                    "name_key": source_key_from_name(path),
                    "phash": phash(path),
                }
            )

        group_ids = cluster_class(class_records)
        for rec, group_id in zip(class_records, group_ids):
            rec["source_group"] = group_id
            records.append(rec)

    df = pd.DataFrame(records)
    df = grouped_split(df, args.seed)

    output = args.output.resolve()
    for split in ("train", "validation", "test"):
        split_root = output / split
        if split_root.exists():
            shutil.rmtree(split_root)
        split_root.mkdir(parents=True, exist_ok=True)

    for row in df.itertuples(index=False):
        src = Path(row.source_path)
        dst_dir = output / row.split / row.class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / row.filename
        if dst.exists():
            stem, suffix = dst.stem, dst.suffix
            dst = dst_dir / f"{stem}_{hashlib.md5(str(src).encode()).hexdigest()[:8]}{suffix}"

        if args.copy:
            shutil.copy2(src, dst)
        else:
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)

    manifest_path = output / "split_manifest.csv"
    df.to_csv(manifest_path, index=False)

    group_overlap = int(
        df.groupby("source_group")["split"].nunique().max() if not df.empty else 0
    )
    if group_overlap != 1:
        raise RuntimeError("Leakage audit failed: a source group spans multiple splits.")

    class_counts = (
        df.groupby(["split", "class_name"]).size().reset_index(name="count")
    )
    summary = {
        "seed": args.seed,
        "target_classes": list(TARGET_CLASS_ALIASES),
        "split_counts": class_counts.to_dict(orient="records"),
        "total_images": int(len(df)),
        "total_source_groups": int(df["source_group"].nunique()),
        "max_splits_per_source_group": group_overlap,
        "copy_mode": bool(args.copy),
        "leakage_control": {
            "exact_hash": True,
            "normalized_filename_grouping": True,
            "perceptual_hash_screening": True,
            "note": "PlantVillage folder structures do not expose a guaranteed original-source ID. Source groups are therefore a conservative automated leakage-control heuristic, not proof of true source identity."
        },
    }
    with (output / "dataset_audit.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Dataset preparation completed.")
    print(f"Manifest: {manifest_path}")
    print(f"Audit: {output / 'dataset_audit.json'}")
    print(f"Images: {summary['total_images']}")
    print(f"Source groups: {summary['total_source_groups']}")
    print("Leakage audit: PASSED")


if __name__ == "__main__":
    main()
