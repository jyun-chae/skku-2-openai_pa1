"""
Backbone models for semantic segmentation.

This module provides backbone architectures that extract multi-scale features
from input images. Currently supports EfficientNet-B3 with ImageNet pretraining.
"""

import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights


class EfficientNetB3Backbone(nn.Module):
    """
    TorchVision EfficientNet-B3 classification pretrained backbone.

    Uses ImageNet classification weights (not segmentation-specific).
    Extracts multi-scale features at different strides for FPN-style fusion.

    Feature map outputs:
        c2: stride /4, 32 channels
        c3: stride /8, 48 channels
        c4: stride /16, 136 channels
        c5: stride /32, 1536 channels
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b3(weights=weights)

        self.features = net.features

        # Output channels for each feature level
        self.out_channels = {
            "c2": 32,
            "c3": 48,
            "c4": 136,
            "c5": 1536,
        }

    def forward(self, x):
        """Extract multi-scale features from input tensor.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            dict: Feature maps at different scales
                - "c2": [B, 32, H/4, W/4]
                - "c3": [B, 48, H/8, W/8]
                - "c4": [B, 136, H/16, W/16]
                - "c5": [B, 1536, H/32, W/32]
        """
        c2 = c3 = c4 = c5 = None

        # Extract features at specific layers
        for i, layer in enumerate(self.features):
            x = layer(x)

            if i == 2:
                c2 = x      # stride /4
            elif i == 3:
                c3 = x      # stride /8
            elif i == 5:
                c4 = x      # stride /16
            elif i == 8:
                c5 = x      # stride /32

        return {
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5,
        }


def build_backbone(cfg):
    name = cfg.model.backbone.lower()
    pretrained = getattr(cfg.model, "pretrained", True)

    if name == "efficientnet_b3":
        return EfficientNetB3Backbone(pretrained=pretrained)

    raise ValueError(
        f"Unsupported backbone: {cfg.model.backbone}. "
        "This project is currently configured to use EfficientNet-B3 only."
    )