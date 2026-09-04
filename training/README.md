# Training experiments

## Experiment A — EfficientNet-B0 baseline
Dataset A uses the original RGB training images only. Validation and test are deterministic.

## Experiment B — Traditional augmentation
Dataset B uses the same EfficientNet-B0 architecture and training protocol as Experiment A, but enables the conservative RGB training augmentation defined in `preprocessing/transforms.py`.

The A/B comparison changes the training augmentation only. Test and validation data are unchanged.

Experimental metrics are measured from actual runs; no target performance is assumed in advance.
