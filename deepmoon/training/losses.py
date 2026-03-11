from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

class BCEDiceLoss(nn.Module):
    #BCEWithLogits + Dice Loss

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = torch.sum(probs * targets, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss
