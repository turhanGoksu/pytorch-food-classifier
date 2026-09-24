"""Entry point: train a food classifier with a hand-written PyTorch loop.

Usage:
    python train.py --data-dir ~/Desktop/turkish-food --epochs 15
    python train.py --data-dir /kaggle/input/<dataset> \
        --output-dir /kaggle/working/outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from src.checkpoint import load_checkpoint, save_checkpoint
from src.dataset import build_dataloaders
from src.engine import build_optimizer, evaluate, get_device, train_one_epoch
from src.model import UNFREEZABLE_LAYERS, build_model
from src.plotting import History, plot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder with one sub-folder per class.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainable-layers", nargs="*", default=[],
                        choices=UNFREEZABLE_LAYERS,
                        help="Backbone stages to fine-tune, e.g. layer4.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)  # Same head init and shuffle order each run.
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    train_loader, val_loader, class_names = build_dataloaders(
        args.data_dir.expanduser(), batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
    )
    model = build_model(len(class_names), tuple(args.trainable_layers))
    model = model.to(device)  # Move weights before creating the optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, lr=args.lr,
                                weight_decay=args.weight_decay)

    print(f"device={device} classes={len(class_names)} "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    history: History = {"train_loss": [], "train_acc": [],
                        "val_loss": [], "val_acc": []}
    checkpoint_path = output_dir / "best.pt"
    best_val_loss, best_epoch = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        # Val loss, not val acc: with 199 val images accuracy moves in 0.5%
        # jumps, while loss also rewards being confidently right.
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss, best_epoch = val_loss, epoch
            save_checkpoint(checkpoint_path, model, class_names,
                            tuple(args.trainable_layers), epoch,
                            val_loss, val_acc)
        print(f"epoch {epoch:>3}/{args.epochs} | "
              f"train loss {train_loss:.3f} acc {train_acc:.1%} | "
              f"val loss {val_loss:.3f} acc {val_acc:.1%}"
              f"{'  <- saved best' if is_best else ''}")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, output_dir / "curves.png", best_epoch=best_epoch)
    print(f"saved history.json and curves.png to {output_dir}")

    # Round-trip check: a freshly rebuilt model must reproduce the best score.
    best_model, checkpoint = load_checkpoint(checkpoint_path, device)
    reload_loss, reload_acc = evaluate(best_model, val_loader, criterion,
                                       device)
    print(f"best checkpoint: epoch {checkpoint['epoch']} | saved val loss "
          f"{checkpoint['val_loss']:.3f} acc {checkpoint['val_acc']:.1%} | "
          f"reloaded val loss {reload_loss:.3f} acc {reload_acc:.1%}")


if __name__ == "__main__":  # Required for DataLoader workers on macOS.
    main()
