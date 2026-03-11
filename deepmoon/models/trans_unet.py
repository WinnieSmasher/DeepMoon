"""在 Attention U-Net 瓶颈中引入 Transformer 的模型定义。"""

from __future__ import annotations

import torch

from deepmoon.models.attention_unet import AttentionUNet
from deepmoon.models.layers import TransformerEncoder2D


class TransUNet(AttentionUNet):
    """使用 Transformer 增强全局上下文建模的 U-Net 变体。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 48,
        dropout: float = 0.1,
        transformer_dim: int = 384,
        transformer_layers: int = 6,
        transformer_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        image_size: int = 256,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            dropout=dropout,
        )
        bottleneck_channels = base_channels * 16
        grid_size = max(image_size // 16, 1)
        self.transformer = TransformerEncoder2D(
            in_channels=bottleneck_channels,
            embed_dim=transformer_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
            dropout=dropout,
            grid_size=grid_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """先编码，再对瓶颈特征做全局建模，最后解码输出。"""

        skip1, skip2, skip3, skip4, bottleneck = self.encode(x)
        bottleneck = self.transformer(bottleneck).contiguous()
        return self.decode(skip1, skip2, skip3, skip4, bottleneck)
