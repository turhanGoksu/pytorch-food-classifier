"""Sanity-check the image dataset before training.

Verifies that every class folder is non-empty and that every image can be
decoded. Only needs Pillow, so it runs even where torch is not installed.

Usage:
    python scripts/check_data.py --data-dir ~/Desktop/turkish-food
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_class_dirs(data_dir: Path) -> list[Path]:
    """Return visible sub-directories of data_dir, sorted by name."""
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def list_images(class_dir: Path) -> list[Path]:
    """Return image files directly inside class_dir (no recursion)."""
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def check_dataset(data_dir: Path) -> bool:
    """Print a per-class report and return True if no problems were found."""
    class_dirs = list_class_dirs(data_dir)
    if not class_dirs:
        print(f"ERROR: no class folders found in {data_dir}")
        return False

    problems: list[str] = []
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    total = 0

    for class_dir in class_dirs:
        images = list_images(class_dir)
        print(f"{class_dir.name:<24} {len(images):>4} images")
        if not images:
            problems.append(f"empty class folder: {class_dir.name}")
        for path in images:
            try:
                with Image.open(path) as img:
                    img.load()  # Force a full decode, not just the header.
                    sizes[img.size] += 1
                    modes[img.mode] += 1
            except Exception as exc:  # noqa: BLE001 - report any decode error
                problems.append(f"unreadable image: {path} ({exc})")
        total += len(images)

    print(f"\nclasses: {len(class_dirs)} | images: {total}")
    print(f"most common sizes: {sizes.most_common(3)}")
    print(f"distinct sizes: {len(sizes)} | color modes: {dict(modes)}")

    for problem in problems:
        print(f"PROBLEM: {problem}")
    return not problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    ok = check_dataset(args.data_dir.expanduser())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
