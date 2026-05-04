# src/models/seg_model.py

"""
Semantic segmentation models.

This module implements FPN-like segmentation architecture using EfficientNet-B3 backbone
with depthwise separable convolutions for efficient feature fusion and segmentation head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone


class CBR(nn.Module):
    """Conv-BatchNorm-ReLU block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        dilation=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DWSeparableCBR(nn.Module):
    """
    Depthwise separable Conv-BatchNorm-ReLU block.

    Uses depthwise 3x3 conv followed by pointwise 1x1 conv.
    Significantly reduces FLOPs/Params compared to regular 3x3 conv.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        dilation=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            # Depthwise spatial convolution
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            # Pointwise channel mixing/compression
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HighLevelCompress(nn.Module):
    """
    Compression module for high-level features with many channels.

    Compresses c4/c5 features (with many channels) to FPN channels efficiently.
    Uses depthwise separable conv to preserve spatial context before compression.
    Particularly useful for EfficientNet-B3 c5 with 1536 channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dilation=1,
    ):
        super().__init__()

        self.block = DWSeparableCBR(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

    def forward(self, x):
        return self.block(x)


class FPNLikeSegModel(nn.Module):
    """
    FPN-like semantic segmentation model with EfficientNet-B3 backbone.

    Features multi-scale feature fusion from c2-c5 levels:
        c2: stride /4,  32 channels
        c3: stride /8,  48 channels
        c4: stride /16, 136 channels
        c5: stride /32, 1536 channels

    Architecture changes for efficiency:
        - c4/c5 lateral connections use depthwise separable compression
        - Smooth/head layers use dilation for better segmentation context

    Output is bilinearly upsampled back to input image size.
    """

    def __init__(self, cfg):
        super().__init__()

        self.num_classes = cfg.model.num_classes

        self.backbone = build_backbone(cfg)
        ch = self.backbone.out_channels

        fpn_channels = getattr(cfg.model, "fpn_channels", 64)
        dropout = getattr(cfg.model, "dropout", 0.1)

        # Low/mid-level features: use simple 1x1 projection (channels are small)
        self.lateral_c2 = nn.Conv2d(ch["c2"], fpn_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(ch["c3"], fpn_channels, kernel_size=1)

        # High-level features: use depthwise separable compression
        # c4 at /16 stride, dilation=1 is stable
        self.lateral_c4 = HighLevelCompress(
            in_channels=ch["c4"],
            out_channels=fpn_channels,
            dilation=1,
        )

        # c5 at /32 stride, small spatial size so use dilation=2 for more context
        self.lateral_c5 = HighLevelCompress(
            in_channels=ch["c5"],
            out_channels=fpn_channels,
            dilation=2,
        )

        # p2 at /4 resolution, use depthwise separable to keep it lightweight
        self.smooth_c2 = DWSeparableCBR(
            fpn_channels,
            fpn_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
        )

        # p3/p4 smoothing for top-down fusion refinement
        self.smooth_c3 = DWSeparableCBR(
            fpn_channels,
            fpn_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
        )

        self.smooth_c4 = DWSeparableCBR(
            fpn_channels,
            fpn_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
        )

        # Segmentation head with dilation=2 to expand receptive field
        self.head = nn.Sequential(
            DWSeparableCBR(
                fpn_channels,
                fpn_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(fpn_channels, self.num_classes, kernel_size=1),
        )

    def _upsample_add(self, high, low):
        high = F.interpolate(
            high,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return high + low

    def forward(self, x):
        input_size = x.shape[-2:]

        feats = self.backbone(x)

        c2 = feats["c2"]    # stride /4
        c3 = feats["c3"]    # stride /8
        c4 = feats["c4"]    # stride /16
        c5 = feats["c5"]    # stride /32

        # Top-down feature fusion (FPN-style)
        p5 = self.lateral_c5(c5)
        p4 = self._upsample_add(p5, self.lateral_c4(c4))
        p4 = self.smooth_c4(p4)

        p3 = self._upsample_add(p4, self.lateral_c3(c3))
        p3 = self.smooth_c3(p3)

        p2 = self._upsample_add(p3, self.lateral_c2(c2))
        p2 = self.smooth_c2(p2)

        # Segmentation head
        logits = self.head(p2)

        # Upsample to original input size
        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits