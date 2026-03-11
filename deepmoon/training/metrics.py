from __future__ import annotations
import torch

def _binarize(predictions: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if predictions.dtype.is_floating_point:
        needs_sigmoid = predictions.min() < 0 or predictions.max() > 1
        predictions = torch.sigmoid(predictions) if needs_sigmoid else predictions
    return (predictions >= threshold).float()


def dice_coefficient(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> float:
    preds = _binarize(predictions, threshold=threshold)
    targets = (targets >= threshold).float()
    dims = tuple(range(1, preds.ndim))
    intersection = torch.sum(preds * targets, dim=dims)
    denominator = torch.sum(preds, dim=dims) + torch.sum(targets, dim=dims)
    score = (2.0 * intersection + smooth) / (denominator + smooth)
    return float(score.mean().item())


def iou_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> float:
    preds = _binarize(predictions, threshold=threshold)
    targets = (targets >= threshold).float()
    dims = tuple(range(1, preds.ndim))
    intersection = torch.sum(preds * targets, dim=dims)
    union = torch.sum((preds + targets) > 0, dim=dims)
    score = (intersection + smooth) / (union + smooth)
    return float(score.mean().item())


def precision_recall_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-8,
) -> tuple[float, float, float]:
    preds = _binarize(predictions, threshold=threshold)
    targets = (targets >= threshold).float()

    true_positive = torch.sum(preds * targets)
    false_positive = torch.sum(preds * (1.0 - targets))
    false_negative = torch.sum((1.0 - preds) * targets)

    precision = (true_positive + smooth) / (
        true_positive + false_positive + smooth
    )
    recall = (true_positive + smooth) / (
        true_positive + false_negative + smooth
    )
    f1 = (2.0 * precision * recall) / (precision + recall + smooth)
    return float(precision.item()), float(recall.item()), float(f1.item())
