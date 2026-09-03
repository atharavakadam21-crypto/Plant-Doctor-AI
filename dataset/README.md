# Dataset Stage

This project uses exactly eight PlantVillage classes:

- Tomato: healthy, early_blight, late_blight
- Potato: healthy, early_blight, late_blight
- Pepper (bell): healthy, bacterial_spot

Place the unmodified downloaded PlantVillage dataset under `dataset/raw/`.

Do not place downloaded images directly into train/validation/test.

Run:

```powershell
python dataset/prepare_dataset.py --source dataset/raw --output dataset
```

The script filters the eight classes, detects exact duplicate images, groups likely augmented/duplicate images conservatively, and writes leakage-aware train/validation/test splits.

Important: the generated split is audited before training. The script does not report model metrics.
