"""Dataset loading: file scan, stratified split, transforms, DataLoaders."""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Statistics of the ImageNet training set. The pretrained backbone learned its
# weights on inputs normalized with these values, so we must use the same ones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Folders that hold the same dish under a different name. Images from the key
# folder are labeled as the value class, so the dish gets a single label.
CLASS_ALIASES: dict[str, str] = {"mucver": "kabak_mucver"}

Sample = tuple[Path, int]  # (image path, class index)


def scan_samples(
    data_dir: Path, aliases: dict[str, str] | None = None
) -> tuple[list[Sample], list[str]]:
    """Collect (path, label) pairs from a <class>/<image> folder layout.

    Class indices follow the alphabetical order of class names, the same
    convention torchvision's ImageFolder uses. Folders listed in `aliases`
    are merged into their target class instead of becoming a class.
    """
    aliases = aliases or {}
    folders = sorted(
        p.name for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not folders:
        raise FileNotFoundError(f"No class folders found in {data_dir}")

    class_names = sorted({aliases.get(name, name) for name in folders})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    samples: list[Sample] = []
    for folder in folders:
        label = class_to_idx[aliases.get(folder, folder)]
        for path in sorted((data_dir / folder).iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))
    return samples, class_names


def stratified_split(
    samples: list[Sample], val_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Split samples so every class keeps the same train/val ratio."""
    # Local RNG: reproducible, and no side effects on the global state.
    rng = random.Random(seed)
    by_class: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    train: list[Sample] = []
    val: list[Sample] = []
    for label in sorted(by_class):
        items = by_class[label][:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_fraction))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def build_transforms(
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, val_transform)."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose([
        # Random crop covering 70-100% of the image, resized to image_size.
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        # No hue jitter: color is a real signal for food (tea vs coffee).
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform


class FoodDataset(Dataset):
    """Loads one image per index and applies a transform."""

    def __init__(
        self, samples: list[Sample], transform: transforms.Compose
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")  # Guarantees 3 channels.
        return self.transform(image), label


def build_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    image_size: int = 224,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train/val DataLoaders and return them with the class names."""
    samples, class_names = scan_samples(data_dir, CLASS_ALIASES)
    train_samples, val_samples = stratified_split(samples, val_fraction, seed)
    train_transform, val_transform = build_transforms(image_size)

    use_cuda = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,  # Faster host-to-GPU copies; no-op on CPU.
        "persistent_workers": num_workers > 0,  # Keep workers across epochs.
    }
    train_loader = DataLoader(
        FoodDataset(train_samples, train_transform), shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        FoodDataset(val_samples, val_transform), shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, class_names


def main() -> None:
    """Smoke test: build the loaders and print the shape of one batch."""
    parser = argparse.ArgumentParser(description="Inspect one batch.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    train_loader, val_loader, class_names = build_dataloaders(
        args.data_dir.expanduser(), num_workers=args.num_workers
    )
    print(f"classes: {len(class_names)}")
    print(f"train images: {len(train_loader.dataset)} "
          f"-> {len(train_loader)} batches")
    print(f"val images:   {len(val_loader.dataset)} "
          f"-> {len(val_loader)} batches")

    images, labels = next(iter(train_loader))
    print(f"images: shape={tuple(images.shape)} dtype={images.dtype} "
          f"min={images.min():.2f} max={images.max():.2f}")
    print(f"labels: shape={tuple(labels.shape)} dtype={labels.dtype} "
          f"distinct classes in batch={labels.unique().numel()}")


if __name__ == "__main__":  # Required on macOS when num_workers > 0.
    main()
