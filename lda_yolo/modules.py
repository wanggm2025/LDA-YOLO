from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def autopad(kernel_size: int | Tuple[int, int], dilation: int = 1):
    if isinstance(kernel_size, tuple):
        return tuple(dilation * (k - 1) // 2 for k in kernel_size)
    return dilation * (kernel_size - 1) // 2


class ConvModule(nn.Module):
    """Conv2d + BatchNorm + SiLU used by the paper's backbone and neck."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size,
            stride,
            autopad(kernel_size, dilation),
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseConv(ConvModule):
    def __init__(
        self,
        channels: int,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int = 1,
        dilation: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__(
            channels,
            channels,
            kernel_size,
            stride,
            groups=channels,
            dilation=dilation,
            act=act,
        )


class GhostConv(nn.Module):
    """Cheap feature expansion from GhostNet, used as the paper's 1x1 Ghost operation."""

    def __init__(self, c1: int, c2: int, kernel_size: int = 1, ratio: int = 2) -> None:
        super().__init__()
        primary = math.ceil(c2 / ratio)
        cheap = primary * (ratio - 1)
        self.primary = ConvModule(c1, primary, kernel_size)
        self.cheap = ConvModule(primary, cheap, 3, groups=primary)
        self.out_channels = c2

    def forward(self, x: Tensor) -> Tensor:
        primary = self.primary(x)
        return torch.cat((primary, self.cheap(primary)), dim=1)[:, : self.out_channels]


class ECA(nn.Module):
    """Efficient Channel Attention with an adaptive odd 1-D kernel."""

    def __init__(self, channels: int, gamma: float = 2.0, bias: float = 1.0) -> None:
        super().__init__()
        kernel = int(abs((math.log2(max(channels, 2)) + bias) / gamma))
        kernel = kernel if kernel % 2 else kernel + 1
        kernel = max(kernel, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel, padding=kernel // 2, bias=False)

    def weights(self, x: Tensor) -> Tensor:
        descriptor = self.pool(x).squeeze(-1).transpose(1, 2)
        return self.conv(descriptor).transpose(1, 2).unsqueeze(-1).sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weights(x)


def channel_shuffle(x: Tensor, groups: int = 4) -> Tensor:
    batch, channels, height, width = x.shape
    if channels % groups:
        groups = 2 if channels % 2 == 0 else 1
    if groups == 1:
        return x
    x = x.reshape(batch, groups, channels // groups, height, width)
    return x.transpose(1, 2).contiguous().reshape(batch, channels, height, width)


class SGConv(nn.Module):
    """Selective Gated Convolution from equations (1)-(8)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels % 4:
            raise ValueError(f"SGConv channels must be divisible by 4, got {channels}")
        c1, c2, c3 = channels // 2, channels // 4, channels // 4
        self.splits = (c1, c2, c3)
        self.local = nn.Sequential(
            DepthwiseConv(c1, 3, act=False),
            GhostConv(c1, c1, 1),
            ECA(c1),
        )
        self.context = nn.Sequential(
            GhostConv(c2, c2, 1),
            DepthwiseConv(c2, 5, act=False),
            ECA(c2),
        )
        self.directional = nn.Sequential(
            GhostConv(c3, c3, 1),
            DepthwiseConv(c3, (1, 3), act=False),
            DepthwiseConv(c3, (3, 1), act=False),
            ECA(c3),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1, x2, x3 = torch.split(x, self.splits, dim=1)
        enhanced = torch.cat(
            (self.local(x1), self.context(x2), self.directional(x3)), dim=1
        )
        return channel_shuffle(enhanced + x, groups=4)


class IC2f(nn.Module):
    """Improved C2f dual-branch attention from equations (9)-(18)."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.project = ConvModule(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.global_gate = nn.Sequential(nn.Conv2d(c2, c2, 1, bias=True), nn.ReLU(inplace=True))
        self.spatial_1 = DepthwiseConv(c2, 1, act=False)
        self.spatial_3 = DepthwiseConv(c2, 3, act=False)
        self.eca_global = ECA(c2)
        self.eca_spatial = ECA(c2)
        self.out = nn.Sequential(nn.Conv2d(c2, c2, 1, bias=False), nn.BatchNorm2d(c2))

    def forward(self, x: Tensor) -> Tensor:
        base = self.project(x)
        gap = F.adaptive_avg_pool2d(base, 1)
        gmp = F.adaptive_max_pool2d(base, 1)
        global_mask = self.global_gate(gap + gmp)
        spatial_mask = torch.sigmoid(self.spatial_1(base) + self.spatial_3(base))
        branch_global = self.eca_global(base * global_mask)
        branch_spatial = self.eca_spatial(base * spatial_mask)
        scores = torch.stack(
            (
                branch_global.mean(dim=(1, 2, 3)),
                branch_spatial.mean(dim=(1, 2, 3)),
            ),
            dim=1,
        ).softmax(dim=1)
        fused = (
            branch_global * scores[:, 0, None, None, None]
            + branch_spatial * scores[:, 1, None, None, None]
        )
        return channel_shuffle(self.out(fused) + base, groups=4)


class AxisAttention(nn.Module):
    """Linear-cost self-attention along height and width independently."""

    def __init__(self, channels: int, attention_dim: int | None = None) -> None:
        super().__init__()
        dim = attention_dim or max(8, channels // 4)
        self.q = nn.Linear(channels, dim, bias=False)
        self.k = nn.Linear(channels, dim, bias=False)
        self.v = nn.Linear(channels, dim, bias=False)
        self.proj = nn.Linear(dim, channels, bias=False)
        self.scale = dim**-0.5

    def _attention(self, sequence: Tensor) -> Tensor:
        q, k, v = self.q(sequence), self.k(sequence), self.v(sequence)
        weights = (q @ k.transpose(-1, -2) * self.scale).softmax(dim=-1)
        return self.proj(weights @ v)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        height_sequence = x.mean(dim=3).transpose(1, 2)
        width_sequence = x.mean(dim=2).transpose(1, 2)
        height_mask = self._attention(height_sequence).sigmoid().transpose(1, 2).unsqueeze(3)
        width_mask = self._attention(width_sequence).sigmoid().transpose(1, 2).unsqueeze(2)
        return height_mask, width_mask


class DCFM(nn.Module):
    """Dynamic Convolutional Attention Fusion from equations (19)-(31)."""

    def __init__(self, channels: int, residual: bool = True) -> None:
        super().__init__()
        self.attention_input = ConvModule(channels, channels, 1)
        self.axis_attention = AxisAttention(channels)
        self.conv_input = ConvModule(channels, channels, 1)
        self.dilated = DepthwiseConv(channels, 3, dilation=2)
        self.asymmetric = nn.Sequential(
            DepthwiseConv(channels, (1, 3)),
            DepthwiseConv(channels, (3, 1)),
        )
        self.eca_dilated = ECA(channels)
        self.eca_asymmetric = ECA(channels)
        self.out = ConvModule(channels, channels, 1, act=False)
        self.residual = residual

    def forward(self, x: Tensor) -> Tensor:
        feature = self.attention_input(x)
        mask_h, mask_w = self.axis_attention(feature)
        attention_output = mask_h * feature + mask_w * feature

        local = self.conv_input(x)
        dilated = self.eca_dilated(self.dilated(local))
        asymmetric = self.eca_asymmetric(self.asymmetric(local))
        weights = torch.stack(
            (
                dilated.mean(dim=(1, 2, 3)),
                asymmetric.mean(dim=(1, 2, 3)),
            ),
            dim=1,
        ).softmax(dim=1)
        convolution_output = (
            dilated * weights[:, 0, None, None, None]
            + asymmetric * weights[:, 1, None, None, None]
        )
        output = self.out(attention_output + convolution_output)
        return output + x if self.residual else output


def haar_dwt(x: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], tuple[int, int]]:
    """Parameter-free 2-D Haar DWT. Returns subbands and the original H/W."""

    height, width = x.shape[-2:]
    x = F.pad(x, (0, width % 2, 0, height % 2), mode="replicate")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (-a - b + c + d) * 0.5
    hl = (-a + b - c + d) * 0.5
    hh = (a - b - c + d) * 0.5
    return (ll, lh, hl, hh), (height, width)


def haar_idwt(
    subbands: tuple[Tensor, Tensor, Tensor, Tensor], original_size: tuple[int, int]
) -> Tensor:
    ll, lh, hl, hh = subbands
    a = (ll - lh - hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll + lh + hl + hh) * 0.5
    output = torch.empty(
        (*ll.shape[:-2], ll.shape[-2] * 2, ll.shape[-1] * 2),
        dtype=ll.dtype,
        device=ll.device,
    )
    output[..., 0::2, 0::2] = a
    output[..., 0::2, 1::2] = b
    output[..., 1::2, 0::2] = c
    output[..., 1::2, 1::2] = d
    height, width = original_size
    return output[..., :height, :width]


class DDAM(nn.Module):
    """Dual-Domain Attention Module from equations (32)-(42)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_local = DepthwiseConv(channels, 3)
        self.spatial_dilated = DepthwiseConv(channels, 3, dilation=2)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.ReLU(inplace=True)
        )
        self.high_frequency_eca = ECA(channels * 3)
        self.spatial_eca = ECA(channels)
        self.frequency_eca = ECA(channels)

    def forward(self, x: Tensor) -> Tensor:
        spatial = self.spatial_local(x) + self.spatial_dilated(x)
        spatial_descriptor = torch.cat(
            (spatial.mean(dim=1, keepdim=True), spatial.amax(dim=1, keepdim=True)),
            dim=1,
        )
        spatial_feature = spatial * self.spatial_gate(spatial_descriptor)

        (ll, lh, hl, hh), original_size = haar_dwt(x)
        high = torch.cat((lh, hl, hh), dim=1)
        gates = self.high_frequency_eca.weights(high)
        gate_lh, gate_hl, gate_hh = gates.chunk(3, dim=1)
        frequency_feature = haar_idwt(
            (ll, lh * gate_lh, hl * gate_hl, hh * gate_hh), original_size
        )

        spatial_feature = self.spatial_eca(spatial_feature)
        frequency_feature = self.frequency_eca(frequency_feature)
        scores = torch.stack(
            (
                spatial_feature.abs().mean(dim=(1, 2, 3)),
                frequency_feature.abs().mean(dim=(1, 2, 3)),
            ),
            dim=1,
        ).softmax(dim=1)
        fused = (
            spatial_feature * scores[:, 0, None, None, None]
            + frequency_feature * scores[:, 1, None, None, None]
        )
        return x + fused


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, kernel_size: int = 5) -> None:
        super().__init__()
        hidden = c1 // 2
        self.cv1 = ConvModule(c1, hidden, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.cv2 = ConvModule(hidden * 4, c2, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))

