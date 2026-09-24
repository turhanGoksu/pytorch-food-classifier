"""Model setup: pretrained ResNet18 with a new classification head."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

UNFREEZABLE_LAYERS = ("layer1", "layer2", "layer3", "layer4")


def build_model(
    num_classes: int,
    trainable_layers: tuple[str, ...] = (),
    pretrained: bool = True,
) -> nn.Module:
    """Load ImageNet-pretrained ResNet18 and replace its final layer.

    Args:
        num_classes: Number of output classes for the new head.
        trainable_layers: Backbone stages to keep trainable, e.g. ("layer4",).
            Empty means the whole backbone is frozen (feature extraction).
        pretrained: Download ImageNet weights. Use False when the weights
            will be overwritten by a checkpoint anyway.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # 1) Freeze every parameter that exists right now (the whole backbone).
    for param in model.parameters():
        param.requires_grad = False

    # 2) Optionally switch selected stages back on for partial fine-tuning.
    for name in trainable_layers:
        if name not in UNFREEZABLE_LAYERS:
            raise ValueError(f"Unknown layer {name!r}; "
                             f"choose from {UNFREEZABLE_LAYERS}")
        for param in getattr(model, name).parameters():
            param.requires_grad = True

    # 3) Replace the head AFTER freezing: new parameters are created with
    #    requires_grad=True, so the head is always trainable.
    in_features = model.fc.in_features  # 512 for ResNet18
    model.fc = nn.Linear(in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def main() -> None:
    """Smoke test: one forward/backward pass on random data."""
    model = build_model(num_classes=47)
    trainable, total = count_parameters(model)
    print(f"trainable params: {trainable:,} / {total:,}")

    images = torch.randn(4, 3, 224, 224)  # Fake batch, shape [B, C, H, W].
    logits = model(images)
    print(f"logits shape: {tuple(logits.shape)}")

    logits.sum().backward()  # Any scalar works for checking gradient flow.
    conv_grad = model.conv1.weight.grad
    fc_grad = model.fc.weight.grad
    print(f"conv1.weight.grad: {conv_grad}")
    print(f"fc.weight.grad shape: "
          f"{None if fc_grad is None else tuple(fc_grad.shape)}")


if __name__ == "__main__":
    main()
