# src/models/backbone.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights


class EfficientNetB3Backbone(nn.Module):
    """
    TorchVision EfficientNet-B3 classification pretrained backbone.
    Semantic segmentation pretrained weights are NOT used.

    Standard EfficientNet-B3 feature mapping:
        c2 = feature[2], stride /4
        c3 = feature[3], stride /8
        c4 = feature[5], stride /16
        c5 = feature[8], stride /32
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b3(weights=weights)

        self.features = net.features

        self.out_channels = {
            "c2": 32,
            "c3": 48,
            "c4": 136,
            "c5": 1536,
        }

    def forward(self, x):
        c2 = c3 = c4 = c5 = None

        for i, layer in enumerate(self.features):
            x = layer(x)

            if i == 2:
                c2 = x      # /4
            elif i == 3:
                c3 = x      # /8
            elif i == 5:
                c4 = x      # /16
            elif i == 8:
                c5 = x      # /32

        return {
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5,
        }


def build_backbone(cfg):
    name = cfg.model.backbone.lower()
    pretrained = getattr(cfg.model, "pretrained", True)

    if name in ["efficientnet_b3", "efficientnet-b3", "efficientnet"]:
        return EfficientNetB3Backbone(pretrained=pretrained)

    raise ValueError(
        f"Unsupported backbone: {cfg.model.backbone}. "
        "This project is currently configured to use EfficientNet-B3 only."
    )