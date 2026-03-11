"""模型共享层：卷积块、注意力门控与 Transformer 块。"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """两层卷积 + BN + ReLU。"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention U-Net 的 skip 连接门控。"""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attention = self.psi(self.w_g(gate) + self.w_x(skip))
        return skip * attention


class LearnedPositionEncoding2D(nn.Module):
    """可学习二维位置编码，尺寸变化时自动插值。"""

    def __init__(self, embed_dim: int, height: int, width: int) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, embed_dim, height, width))
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == self.position.shape[-2:]:
            return x + self.position
        resized = nn.functional.interpolate(
            self.position,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return x + resized


class TransformerBlock(nn.Module):
    """ViT 风格 Transformer 编码块。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + residual
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder2D(nn.Module):
    """将二维特征图映射到 Transformer，再映射回二维特征图。"""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        grid_size: int = 16,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.position = LearnedPositionEncoding2D(embed_dim, grid_size, grid_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.position(x)
        batch, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        return self.proj_out(x)


class UpBlock(nn.Module):
    """带注意力门控的上采样块。"""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = (
            AttentionGate(out_channels, skip_channels, inter_channels=max(out_channels // 2, 1))
            if use_attention
            else None
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.attention is not None:
            skip = self.attention(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
