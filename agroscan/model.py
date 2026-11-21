"""Побудова моделі (transfer learning) і завантаження чекпоінта."""
import torch
from torch import nn
from torchvision import models

ARCHS = ("efficientnet_b0", "mobilenet_v3_large")


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Невідома архітектура: {arch}")
    return m


def set_backbone_trainable(model: nn.Module, trainable: bool):
    """Етап 1: основа (features) заморожена, вчиться лише голова (classifier)."""
    for p in model.features.parameters():
        p.requires_grad = trainable


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt
