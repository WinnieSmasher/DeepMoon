from __future__ import annotations

import torch

from deepmoon.models import AttentionUNet, TransUNet
from deepmoon.training.trainer import Trainer


def test_attention_unet_io_shape() -> None:
    model = AttentionUNet(in_channels=1, out_channels=1, base_channels=48, dropout=0.1)
    batch = torch.randn(2, 1, 256, 256)
    output = model(batch)
    assert output.shape == (2, 1, 256, 256)


def test_trans_unet_io_shape_and_parameter_budget() -> None:
    model = TransUNet(
        in_channels=1,
        out_channels=1,
        base_channels=48,
        dropout=0.1,
        transformer_dim=384,
        transformer_layers=6,
        transformer_heads=8,
        transformer_mlp_ratio=4.0,
        image_size=256,
    )
    batch = torch.randn(2, 1, 256, 256)
    output = model(batch)
    params = sum(parameter.numel() for parameter in model.parameters())
    assert output.shape == (2, 1, 256, 256)
    assert 24_000_000 <= params <= 32_000_000


def test_trans_unet_supports_backward() -> None:
    model = TransUNet(image_size=256)
    batch = torch.randn(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    output = model(batch)
    loss = (output - target).square().mean()
    loss.backward()
    assert model.output.weight.grad is not None


def test_trainer_auto_device_prefers_mps_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert Trainer._resolve_device("auto").type == "mps"
