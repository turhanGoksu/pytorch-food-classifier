"""Training engine: optimizer setup and the manually written epoch loops."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def build_optimizer(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-2
) -> torch.optim.Optimizer:
    """Create AdamW over the trainable parameters only.

    Frozen parameters (requires_grad=False) are left out, so the optimizer
    keeps no state for them and the intent is explicit.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=lr,
                             weight_decay=weight_decay)


def get_device() -> torch.device:
    """Pick CUDA (Kaggle), then Apple MPS (local Mac), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one pass over the training data and update the weights.

    Returns:
        (mean loss per sample, accuracy) over the epoch.
    """
    model.train()  # Training-mode behavior for BatchNorm/Dropout.
    total_loss, total_correct, total_seen = 0.0, 0, 0

    for images, labels in loader:
        # non_blocking lets the copy overlap with compute when pin_memory=True.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()             # 1. Clear gradients from last step.
        logits = model(images)            # 2. Forward pass: [B, num_classes].
        loss = criterion(logits, labels)  # 3. Mean loss over the batch.
        loss.backward()                   # 4. Autograd adds dLoss/dW to .grad.
        optimizer.step()                  # 5. W <- W - update(W.grad).

        # .item() returns a Python float, so the graph is not kept alive.
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()  # No graph, no saved activations: less memory, faster.
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Measure loss and accuracy without changing the model.

    Returns:
        (mean loss per sample, accuracy) over the loader.
    """
    model.eval()  # BatchNorm uses running stats and stops updating them.
    total_loss, total_correct, total_seen = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)            # Forward only.
        loss = criterion(logits, labels)  # No backward(), no step().

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen
