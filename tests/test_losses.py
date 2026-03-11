from __future__ import annotations

import torch

from deepmoon.training.losses import BCEDiceLoss


def test_bce_dice_loss_prefers_better_prediction() -> None:
    loss_fn = BCEDiceLoss(bce_weight=0.5)
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    good_logits = torch.tensor([[[[6.0, -6.0], [-6.0, 6.0]]]])
    bad_logits = torch.zeros_like(good_logits)
    assert loss_fn(good_logits, targets) < loss_fn(bad_logits, targets)
