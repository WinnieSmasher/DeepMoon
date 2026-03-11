"""训练循环、检查点保存与指标汇总。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from deepmoon.training.losses import BCEDiceLoss
from deepmoon.training.metrics import (
    dice_coefficient,
    iou_score,
    precision_recall_f1,
)


@dataclass
class TrainerState:
    """训练过程中需要跨 epoch 持久化的状态。"""

    epoch: int = 0
    best_val_loss: float = float("inf")
    patience_counter: int = 0


class Trainer:
    """封装单卡训练、验证与最佳模型保存逻辑。"""

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config
        self.model = model
        training_cfg = config.training
        self.device = self._resolve_device(device or training_cfg.device)
        self.model.to(self.device)

        # 当前项目使用 BCE + Dice 的组合损失，与像素级分割目标相匹配。
        self.criterion = BCEDiceLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(training_cfg.learning_rate),
            weight_decay=float(training_cfg.weight_decay),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(int(training_cfg.epochs), 1),
        )
        self.use_amp = bool(training_cfg.amp) and self.device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.state = TrainerState()
        self.checkpoint_dir = Path(training_cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / str(
            training_cfg.checkpoint_name
        )

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _autocast(self):
        """根据设备和 AMP 配置返回自动混合精度上下文。"""

        return torch.autocast(device_type=self.device.type, enabled=self.use_amp)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        training: bool,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """执行单个 batch 的前向、反向与损失计算。"""

        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)
        with self._autocast():
            logits = self.model(images)
            loss = self.criterion(logits, targets)

        if training:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return float(loss.detach().item()), logits.detach(), targets.detach()

    def run_epoch(self, loader, training: bool) -> dict[str, float]:
        """运行一个 epoch，并汇总分割指标。"""

        self.model.train(mode=training)
        losses: list[float] = []
        dice_scores: list[float] = []
        iou_scores: list[float] = []
        precision_scores: list[float] = []
        recall_scores: list[float] = []
        f1_scores: list[float] = []

        iterator = tqdm(loader, desc="train" if training else "eval", leave=False)
        with torch.set_grad_enabled(training):
            for batch in iterator:
                loss, logits, targets = self._step(batch, training=training)
                losses.append(loss)
                dice_scores.append(dice_coefficient(logits, targets))
                iou_scores.append(iou_score(logits, targets))
                precision, recall, f1 = precision_recall_f1(logits, targets)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                iterator.set_postfix(loss=f"{loss:.4f}")

        return {
            "loss": sum(losses) / max(len(losses), 1),
            "dice": sum(dice_scores) / max(len(dice_scores), 1),
            "iou": sum(iou_scores) / max(len(iou_scores), 1),
            "precision": sum(precision_scores) / max(len(precision_scores), 1),
            "recall": sum(recall_scores) / max(len(recall_scores), 1),
            "f1": sum(f1_scores) / max(len(f1_scores), 1),
        }

    def save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        """保存当前最佳模型对应的训练状态。"""

        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict() if hasattr(self.config, "to_dict") else self.config,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self, path: str | Path | None = None) -> dict[str, Any]:
        """加载检查点，并恢复优化器与调度器状态。"""

        checkpoint = torch.load(path or self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        return checkpoint

    def fit(self, train_loader, val_loader) -> list[dict[str, float]]:
        """执行完整训练流程，按验证集损失保存最优模型。"""

        training_cfg = self.config.training
        history: list[dict[str, float]] = []

        for epoch in range(1, int(training_cfg.epochs) + 1):
            train_metrics = self.run_epoch(train_loader, training=True)
            val_metrics = self.run_epoch(val_loader, training=False)
            self.scheduler.step()

            epoch_metrics = {
                "epoch": float(epoch),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(epoch_metrics)

            if val_metrics["loss"] < self.state.best_val_loss:
                self.state.best_val_loss = val_metrics["loss"]
                self.state.patience_counter = 0
                self.save_checkpoint(epoch, epoch_metrics)
            else:
                self.state.patience_counter += 1

            # 早停依据是验证损失连续若干轮未改善。
            if self.state.patience_counter >= int(training_cfg.early_stopping_patience):
                break

        return history


def train_model(model: nn.Module, loaders, config: Any) -> list[dict[str, float]]:
    """训练入口包装，供脚本层直接调用。"""

    trainer = Trainer(model=model, config=config)
    return trainer.fit(loaders["train"], loaders["val"])
