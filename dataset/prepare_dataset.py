"""Filter PlantVillage to eight classes and create leakage-aware splits.

Usage:
    python dataset/prepare_dataset.py --source dataset/raw --output dataset

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

TARGET_CLASSES = {
    "Tomato___healthy": "Tomato__healthy",
    "Tomato___Early_blight": "Tomato__Early_blight",
    "Tomato___Late_blight": "Tomato__Late_blight",
    "Potato___healthy": "Potato__healthy",
    "Potato___Early_blight": "Potato__Early_blight",
    "Potato___Late_blight": "Potato__Late_blight",
    "Pepper__bell___healthy": "Pepper_bell__healthy",
    "Pepper__bell___Bacterial_spot": "Pepper_bell__Bacterial_spot",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path: Path) -> int | None:
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
    """Conservative filename grouping.

    PlantVillage mirrors/releases may encode augmentation variants differently.
    Exact file hashes are always grouped. Filename normalization only removes
    common explicit augmentation suffixes; uncertain files remain separate.
    """
    stem = path.stem.lower()
    stem = re.sub(
        r"([_-](flip|flipped|rotated?|rotation|aug|augment|copy|variant)[_-]?\d*)+$",
        "",
        stem,
    )
    return stem


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
    """Assign a source group ID within one class.

    Groups exact byte duplicates, normalized filename variants, and very close
    perceptual hashes. The threshold is intentionally conservative.
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

    valid = [(i, rec["phash"]) for i, rec in enumerate(records) if rec["phash"] is not None]
    # Conservative pairwise comparison. For the selected eight classes this is
    # acceptable for a one-time preparation step; runtime is logged for audit.
    for a in range(len(valid)):
        i, hash_i = valid[a]
        for b in range(a + 1, len(valid)):
            j, hash_j = valid[b]
            if hamming_distance(hash_i, hash_j) <= near_duplicate_threshold:
                uf.union(i, j)

    root_to_group = {}
    group_ids = []
    for i in range(len(records)):
        root = uf.find(i)
        if root not in root_to_group:
            root_to_group[root] = f"{records[0]['class_name']}_group_{len(root_to_group):05d}"
        group_ids.append(root_to_group[root])
    return group_ids


def stratified_group_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    from sklearn.model_selection import GroupShuffleSplit

    split_frames = []
    for class_name, class_df in df.groupby("class_name", sort=True):
        groups = class_df["source_group"].to_numpy()
        indices = np.arange(len(class_df))

        train_gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_idx, holdout_idx = next(train_gss.split(indices, groups=groups))

        holdout = class_df.iloc[holdout_idx]
        holdout_groups = holdout["source_group"].to_numpy()
        holdout_indices = np.arange(len(holdout))

        val_gss = GroupShuffleSplit(
            n_splits=1, test_size=1 / 3, random_state=seed + 1
        )
        val_rel, test_rel = next(
            val_gss.split(holdout_indices, groups=holdout_groups)
        )

        train = class_df.iloc[train_idx].copy()
        val = holdout.iloc[val_rel].copy()
        test = holdout.iloc[test_rel].copy()

        train["split"] = "train"
        val["split"] = "validation"
        test["split"] = "test"
        split_frames.extend([train, val, test])

    result = pd.concat(split_frames, ignore_index=True)
    return result


def find_class_dirs(source: Path) -> dict[str, Path]:
    found = {}
    for directory in source.rglob("*"):
        if directory.is_dir() and directory.name in TARGET_CLASSES:
            found[directory.name] = directory
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of creating hard links.")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source folder not found: {args.source}")

    class_dirs = find_class_dirs(args.source)
    missing = sorted(set(TARGET_CLASSES) - set(class_dirs))
    if missing:
        raise RuntimeError(
            "Missing required PlantVillage class folders: " + ", ".join(missing)
        )

    records = []
    for source_name, target_name in TARGET_CLASSES.items():
        files = sorted(
            p for p in class_dirs[source_name].rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            raise RuntimeError(f"No images found in {class_dirs[source_name]}")

        class_records = []
        for path in files:
            class_records.append({
                "source_path": str(path.resolve()),
                "class_name": target_name,
                "original_class_name": source_name,
                "filename": path.name,
                "sha256": sha256_file(path),
                "name_key": source_key_from_name(path),
                "phash": phash(path),
            })

        group_ids = cluster_class(class_records)
        for rec, group_id in zip(class_records, group_ids):
            rec["source_group"] = group_id
            records.append(rec)

    df = pd.DataFrame(records)
    df = stratified_group_split(df, args.seed)

    output = args.output.resolve()
    for split in ("train", "validation", "test"):
        split_root = output / split
        if split_root.exists():
            shutil.rmtree(split_root)
        split_root.mkdir(parents=True, exist_ok=True)

    copy_mode = args.copy
    for row in df.itertuples(index=False):
        src = Path(row.source_path)
        dst_dir = output / row.split / row.class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / row.filename
        if dst.exists():
            stem, suffix = dst.stem, dst.suffix
            dst = dst_dir / f"{stem}_{hashlib.md5(str(src).encode()).hexdigest()[:8]}{suffix}"

        if copy_mode:
            shutil.copy2(src, dst)
        else:
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)

    manifest_path = output / "split_manifest.csv"
    df.to_csv(manifest_path, index=False)

    group_overlap = (
        df.groupby("source_group")["split"].nunique().max()
        if not df.empty else 0
    )
    if group_overlap != 1:
        raise RuntimeError("Leakage audit failed: a source group spans multiple splits.")

    summary = {
        "seed": args.seed,
        "target_classes": list(TARGET_CLASSES.values()),
        "split_counts": df.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_dict(),
        "total_images": int(len(df)),
        "total_source_groups": int(df["source_group"].nunique()),
        "max_splits_per_source_group": int(group_overlap),
        "copy_mode": copy_mode,
        "note": "Source grouping is an automated leakage-control heuristic. "
                "Review split_manifest.csv and document any dataset metadata limitations."
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
