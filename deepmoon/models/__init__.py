"""模型构建入口。"""

from __future__ import annotations

from typing import Any

from deepmoon.models.attention_unet import AttentionUNet
from deepmoon.models.trans_unet import TransUNet

__all__ = ["AttentionUNet", "TransUNet", "build_model"]


def build_model(config: Any):
    """根据配置选择并构建模型实例。"""

    model_cfg = config.model if hasattr(config, "model") else config
    common_kwargs = dict(
        in_channels=int(model_cfg.in_channels),
        out_channels=int(model_cfg.out_channels),
        base_channels=int(model_cfg.base_channels),
        dropout=float(model_cfg.dropout),
    )
    architecture = str(model_cfg.arch).lower()
    if architecture == "attention_unet":
        return AttentionUNet(**common_kwargs)
    if architecture == "trans_unet":
        data_cfg = getattr(config, "data", None)
        return TransUNet(
            **common_kwargs,
            transformer_dim=int(model_cfg.transformer_dim),
            transformer_layers=int(model_cfg.transformer_layers),
            transformer_heads=int(model_cfg.transformer_heads),
            transformer_mlp_ratio=float(model_cfg.transformer_mlp_ratio),
            image_size=int(getattr(data_cfg, "image_size", 256)),
        )
    raise ValueError(f"不支持的模型架构: {model_cfg.arch}")
