"""Central configuration and environment verification for Plant Doctor AI."""
from pathlib import Path
import platform
import random
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 8
# Keep this identical to preprocessing.dataset_loader.CLASS_NAMES so checkpoint
# indices, evaluation reports, and deployment labels remain stable.
CLASS_NAMES = [
    "Tomato__healthy",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
    "Potato__healthy",
    "Potato__Early_blight",
    "Potato__Late_blight",
    "Pepper_bell__healthy",
    "Pepper_bell__Bacterial_spot",
]
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_environment() -> None:
    print("=" * 60)
    print("Plant Doctor AI - Environment Verification")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: CPU mode (Google Colab GPU will be used for training)")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Number of classes: {NUM_CLASSES}")
    print("Classes:")
    for index, class_name in enumerate(CLASS_NAMES):
        print(f"  {index}: {class_name}")
    print("Environment verification completed successfully.")


if __name__ == "__main__":
    set_seed()
    verify_environment()
