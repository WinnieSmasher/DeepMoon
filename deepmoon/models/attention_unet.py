"""Attention U-Net 模型定义。"""

from __future__ import annotations

import torch
from torch import nn

from deepmoon.models.layers import ConvBlock, UpBlock


class AttentionUNet(nn.Module):
    """带注意力跳连的编码器-解码器分割网络。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        bottleneck_channels = base_channels * 16

        self.enc1 = ConvBlock(in_channels, channels[0], dropout=0.0)
        self.enc2 = ConvBlock(channels[0], channels[1], dropout=0.0)
        self.enc3 = ConvBlock(channels[1], channels[2], dropout=0.0)
        self.enc4 = ConvBlock(channels[2], channels[3], dropout=dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(channels[3], bottleneck_channels, dropout=dropout)

        self.dec4 = UpBlock(bottleneck_channels, channels[3], channels[3], dropout=dropout)
        self.dec3 = UpBlock(channels[3], channels[2], channels[2], dropout=dropout)
        self.dec2 = UpBlock(channels[2], channels[1], channels[1], dropout=dropout)
        self.dec1 = UpBlock(channels[1], channels[0], channels[0], dropout=dropout)
        self.output = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """提取四级跳连特征与瓶颈特征。"""

        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        skip4 = self.enc4(self.pool(skip3))
        bottleneck = self.bottleneck(self.pool(skip4))
        return skip1, skip2, skip3, skip4, bottleneck

    def decode(
        self,
        skip1: torch.Tensor,
        skip2: torch.Tensor,
        skip3: torch.Tensor,
        skip4: torch.Tensor,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        """逐级上采样并融合编码阶段的跳连特征。"""

        x = self.dec4(bottleneck, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.output(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(*self.encode(x))
