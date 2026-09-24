"""Save and load model checkpoints as plain state_dicts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.model import build_model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: list[str],
    trainable_layers: tuple[str, ...],
    epoch: int,
    val_loss: float,
    val_acc: float,
) -> None:
    """Save everything needed to rebuild the model and read its output."""
    torch.save({
        # All parameters AND buffers (BatchNorm running_mean/var).
        "model_state": model.state_dict(),
        # Maps output index -> dish name; without it predictions are just ints.
        "class_names": class_names,
        # Needed to rebuild the same architecture before loading weights.
        "trainable_layers": list(trainable_layers),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }, path)


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model from a checkpoint, ready for inference.

    Returns:
        (model in eval mode on `device`, the raw checkpoint dict).
    """
    # map_location: a checkpoint saved on a CUDA GPU can be opened on a Mac.
    # weights_only: refuse arbitrary pickled objects, which could run code.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(
        num_classes=len(checkpoint["class_names"]),
        trainable_layers=tuple(checkpoint["trainable_layers"]),
        pretrained=False,  # Weights come from the checkpoint, skip download.
    )
    model.load_state_dict(checkpoint["model_state"])  # strict=True by default.
    model.to(device)
    model.eval()
    return model, checkpoint
