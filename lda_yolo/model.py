from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .modules import ConvModule, DDAM, DCFM, IC2f, SGConv, SPPF


class DetectionHead(nn.Module):
    def __init__(self, channels: tuple[int, int, int], num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        output_channels = num_classes + 5
        self.heads = nn.ModuleList()
        for channel in channels:
            hidden = max(32, channel // 2)
            self.heads.append(
                nn.Sequential(
                    ConvModule(channel, hidden, 3),
                    nn.Conv2d(hidden, output_channels, 1),
                )
            )
        self._initialize_biases()

    def _initialize_biases(self) -> None:
        for head in self.heads:
            prediction = head[-1]
            nn.init.constant_(prediction.bias, 0.0)
            with torch.no_grad():
                prediction.bias[4] = -4.5
                prediction.bias[5:] = -4.5

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> list[Tensor]:
        return [head(feature) for head, feature in zip(self.heads, features)]


@dataclass(frozen=True)
class ModelInfo:
    num_classes: int
    base_channels: int
    strides: tuple[int, int, int] = (4, 8, 16)


class LDAYOLO(nn.Module):
    """Three-scale LDA-YOLO with the paper's SGConv, IC2f, DCFM and DDAM blocks."""

    def __init__(self, num_classes: int, base_channels: int = 24) -> None:
        super().__init__()
        if base_channels % 4:
            raise ValueError("base_channels must be divisible by 4")
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.info = ModelInfo(num_classes=num_classes, base_channels=base_channels)

        self.stem_1 = ConvModule(3, c1, 3, stride=2)
        self.stem_2 = ConvModule(c1, c2, 3, stride=2)
        self.sg_2 = SGConv(c2)
        self.down_3 = ConvModule(c2, c3, 3, stride=2)
        self.sg_3 = SGConv(c3)
        self.down_4 = ConvModule(c3, c4, 3, stride=2)
        self.sg_4 = SGConv(c4)
        self.sppf = SPPF(c4, c4)

        self.top_3 = IC2f(c4 + c3, c3)
        self.top_2 = IC2f(c3 + c2, c2)

        self.ddam = DDAM(c2)
        self.detect_2 = DCFM(c2)

        self.down_neck_3 = ConvModule(c2, c3, 3, stride=2)
        self.bottom_3 = IC2f(c3 + c3, c3)
        self.detect_3 = DCFM(c3)

        self.down_neck_4 = ConvModule(c3, c4, 3, stride=2)
        self.bottom_4 = IC2f(c4 + c4, c4)
        self.detect_4 = DCFM(c4)

        self.head = DetectionHead((c2, c3, c4), num_classes)

    @property
    def strides(self) -> tuple[int, int, int]:
        return self.info.strides

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem_1(x)
        p2 = self.sg_2(self.stem_2(x))
        p3 = self.sg_3(self.down_3(p2))
        p4 = self.sppf(self.sg_4(self.down_4(p3)))

        n3 = self.top_3(torch.cat((F.interpolate(p4, size=p3.shape[-2:], mode="nearest"), p3), dim=1))
        n2 = self.top_2(torch.cat((F.interpolate(n3, size=p2.shape[-2:], mode="nearest"), p2), dim=1))

        out2 = self.detect_2(self.ddam(n2))
        n3_bottom = self.bottom_3(torch.cat((self.down_neck_3(n2), n3), dim=1))
        out3 = self.detect_3(n3_bottom)
        n4_bottom = self.bottom_4(torch.cat((self.down_neck_4(n3_bottom), p4), dim=1))
        out4 = self.detect_4(n4_bottom)
        return out2, out3, out4

    def forward(self, x: Tensor) -> list[Tensor]:
        return self.head(self.forward_features(x))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(num_classes: int, base_channels: int = 24) -> LDAYOLO:
    return LDAYOLO(num_classes=num_classes, base_channels=base_channels)

