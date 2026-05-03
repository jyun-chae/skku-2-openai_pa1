# src/models/seg_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone


class CBR(nn.Module):
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
    Depthwise separable conv:
        depthwise 3x3 conv + pointwise 1x1 conv

    일반 3x3 Conv보다 FLOPs/Params가 훨씬 적음.
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
            # depthwise spatial conv
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

            # pointwise channel mixing / compression
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
    c4/c5처럼 channel이 큰 high-level feature를 가볍게 fpn_channels로 압축.

    특히 EfficientNet-B3 c5는 1536 channels라서,
    단순 1x1 Conv만 쓰는 것보다 depthwise spatial conv를 먼저 거친 뒤
    pointwise로 압축하는 방식으로 context를 조금 반영하면서 압축함.
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
    EfficientNet-B3 backbone + c2~c5 FPN-like neck.

    Used features:
        c2: stride /4,  channels 32
        c3: stride /8,  channels 48
        c4: stride /16, channels 136
        c5: stride /32, channels 1536

    Changes:
        - c4/c5 lateral compression uses depthwise separable conv
        - smooth/head uses light dilation for segmentation context

    Output is resized back to input image size.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.num_classes = getattr(
            cfg.data,
            "num_classes",
            getattr(cfg.model, "num_classes", 21),
        )

        self.backbone = build_backbone(cfg)
        ch = self.backbone.out_channels

        fpn_channels = getattr(cfg.model, "fpn_channels", 64)
        dropout = getattr(cfg.model, "dropout", 0.1)

        # Low/mid-level features: channel이 작아서 일반 1x1 projection 유지
        self.lateral_c2 = nn.Conv2d(ch["c2"], fpn_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(ch["c3"], fpn_channels, kernel_size=1)

        # High-level features: c4/c5는 depthwise separable compression 사용
        # c4는 /16이라 dilation=1 정도가 안정적
        self.lateral_c4 = HighLevelCompress(
            in_channels=ch["c4"],
            out_channels=fpn_channels,
            dilation=1,
        )

        # c5는 /32라 공간 크기가 작으므로 dilation=2로 context를 조금 넓힘
        self.lateral_c5 = HighLevelCompress(
            in_channels=ch["c5"],
            out_channels=fpn_channels,
            dilation=2,
        )

        # p2는 /4 해상도라 너무 무겁지 않게 depthwise separable 사용
        self.smooth_c2 = DWSeparableCBR(
            fpn_channels,
            fpn_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
        )

        # p3/p4는 top-down fusion 중간 정리에 사용
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

        # segmentation head에는 dilation=2를 넣어 receptive field를 조금 확장
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

        c2 = feats["c2"]    # /4
        c3 = feats["c3"]    # /8
        c4 = feats["c4"]    # /16
        c5 = feats["c5"]    # /32

        p5 = self.lateral_c5(c5)
        p4 = self._upsample_add(p5, self.lateral_c4(c4))
        p4 = self.smooth_c4(p4)

        p3 = self._upsample_add(p4, self.lateral_c3(c3))
        p3 = self.smooth_c3(p3)

        p2 = self._upsample_add(p3, self.lateral_c2(c2))
        p2 = self.smooth_c2(p2)

        logits = self.head(p2)

        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits