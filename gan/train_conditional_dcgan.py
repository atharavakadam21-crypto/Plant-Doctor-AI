"""Train the conditional DCGAN on the PlantVillage training split only."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, utils

from gan.conditional_dcgan import ConditionalDiscriminator, ConditionalGenerator, initialize_weights
from preprocessing.dataset_loader import CLASS_NAMES, CLASS_TO_INDEX

IMAGE_SIZE = 128


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CanonicalImageFolder(datasets.ImageFolder):
    """ImageFolder remapped to the project's canonical class indices."""

    def __init__(self, root: str | Path, transform=None):
        super().__init__(root=root, transform=transform)
        found = set(self.classes)
        expected = set(CLASS_NAMES)
        if found != expected:
            raise RuntimeError(f"Dataset classes mismatch. Found={sorted(found)}")
        self.classes = list(CLASS_NAMES)
        self.class_to_idx = dict(CLASS_TO_INDEX)
        remapped = []
        for path, _ in self.samples:
            cls = Path(path).parent.name
            remapped.append((path, CLASS_TO_INDEX[cls]))
        self.samples = remapped
        self.imgs = remapped
        self.targets = [target for _, target in remapped]


def build_loader(dataset_root: str, batch_size: int, workers: int, seed: int):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CanonicalImageFolder(Path(dataset_root) / "train", transform=transform)

    counts = np.bincount(dataset.targets, minlength=len(CLASS_NAMES)).astype(np.float64)
    weights = 1.0 / counts
    sample_weights = torch.tensor([weights[t] for t in dataset.targets], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return dataset, loader, counts.tolist()


def save_grid(generator, device, fixed_noise, fixed_labels, path: Path, nrow: int = 8):
    generator.eval()
    with torch.inference_mode():
        images = generator(fixed_noise.to(device), fixed_labels.to(device)).cpu()
    images = (images + 1.0) / 2.0
    path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(images, path, nrow=nrow)
    generator.train()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    dataset, loader, counts = build_loader(args.dataset_root, args.batch_size, args.num_workers, args.seed)

    generator = ConditionalGenerator(
        latent_dim=args.latent_dim,
        num_classes=len(CLASS_NAMES),
        base_channels=args.base_channels,
    ).to(device)
    discriminator = ConditionalDiscriminator(
        num_classes=len(CLASS_NAMES),
        image_size=IMAGE_SIZE,
        base_channels=args.base_channels,
    ).to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)

    g_opt = Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_opt = Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    fixed_labels = torch.tensor(
        [i for i in range(len(CLASS_NAMES)) for _ in range(args.samples_per_class)],
        dtype=torch.long,
    )
    fixed_noise = torch.randn(len(fixed_labels), args.latent_dim)

    history = []

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training samples: {len(dataset)}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE} RGB")
    print(f"Class counts: {dict(zip(CLASS_NAMES, counts))}")

    for epoch in range(1, args.epochs + 1):
        g_running = 0.0
        d_running = 0.0
        steps = 0

        for real_images, labels in loader:
            real_images = real_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch = labels.size(0)

            # Discriminator step.
            d_opt.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images, labels)
            real_targets = torch.ones(batch, device=device)
            d_real_loss = criterion(real_logits, real_targets)

            noise = torch.randn(batch, args.latent_dim, device=device)
            fake_images = generator(noise, labels)
            fake_logits = discriminator(fake_images.detach(), labels)
            fake_targets = torch.zeros(batch, device=device)
            d_fake_loss = criterion(fake_logits, fake_targets)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_opt.step()

            # Generator step.
            g_opt.zero_grad(set_to_none=True)
            fake_logits = discriminator(fake_images, labels)
            g_loss = criterion(fake_logits, real_targets)
            g_loss.backward()
            g_opt.step()

            g_running += g_loss.item()
            d_running += d_loss.item()
            steps += 1

        mean_g = g_running / max(steps, 1)
        mean_d = d_running / max(steps, 1)
        history.append({"epoch": epoch, "generator_loss": mean_g, "discriminator_loss": mean_d})

        save_grid(
            generator,
            device,
            fixed_noise,
            fixed_labels,
            sample_dir / f"epoch_{epoch:03d}.png",
            nrow=args.samples_per_class,
        )

        torch.save({
            "model_state_dict": generator.state_dict(),
            "latent_dim": args.latent_dim,
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "epoch": epoch,
            "seed": args.seed,
        }, output_dir / "generator_latest.pt")
        torch.save({
            "model_state_dict": discriminator.state_dict(),
            "num_classes": len(CLASS_NAMES),
            "image_size": IMAGE_SIZE,
            "epoch": epoch,
        }, output_dir / "discriminator_latest.pt")

        print(f"Epoch {epoch:03d}: D_loss={mean_d:.4f} G_loss={mean_g:.4f}")

    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps({
        "model": "Conditional DCGAN",
        "dataset_root": str(args.dataset_root),
        "image_size": IMAGE_SIZE,
        "class_names": CLASS_NAMES,
        "class_counts": dict(zip(CLASS_NAMES, counts)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "base_channels": args.base_channels,
        "lr": args.lr,
        "seed": args.seed,
    }, indent=2), encoding="utf-8")
    print(f"Saved GAN outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
