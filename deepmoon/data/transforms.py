"""训练阶段的数据增强组件。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F


TensorPair = tuple[torch.Tensor, torch.Tensor]


class Compose:
    """按顺序组合多种图像/掩码变换。"""

    def __init__(self, transforms: Iterable[Callable[[torch.Tensor, torch.Tensor], TensorPair]]):
        self.transforms = list(transforms)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> TensorPair:
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


@dataclass
class RandomFlip:
    """随机执行水平或垂直翻转。"""

    horizontal_prob: float = 0.5
    vertical_prob: float = 0.5
    rng: random.Random | None = None

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> TensorPair:
        rng = self.rng or random
        if rng.random() < self.horizontal_prob:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        if rng.random() < self.vertical_prob:
            image = torch.flip(image, dims=(-2,))
            mask = torch.flip(mask, dims=(-2,))
        return image, mask


@dataclass
class RandomShift:
    """通过补边后裁剪实现随机平移。"""

    max_pixels: int = 15
    fill_value: float = 0.0
    rng: random.Random | None = None

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> TensorPair:
        rng = self.rng or random
        shift_x = rng.randint(-self.max_pixels, self.max_pixels)
        shift_y = rng.randint(-self.max_pixels, self.max_pixels)
        if shift_x == 0 and shift_y == 0:
            return image, mask

        padding = [self.max_pixels, self.max_pixels, self.max_pixels, self.max_pixels]
        image_padded = F.pad(image, padding, mode="constant", value=self.fill_value)
        mask_padded = F.pad(mask, padding, mode="constant", value=0.0)

        height, width = image.shape[-2:]
        y_start = self.max_pixels - shift_y
        x_start = self.max_pixels - shift_x
        image = image_padded[..., y_start : y_start + height, x_start : x_start + width]
        mask = mask_padded[..., y_start : y_start + height, x_start : x_start + width]
        return image, mask


@dataclass
class RandomRotate90:
    """以 90 度为步长做离散旋转。"""

    probability: float = 0.5
    rng: random.Random | None = None

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> TensorPair:
        rng = self.rng or random
        if rng.random() >= self.probability:
            return image, mask
        k = rng.randint(0, 3)
        return torch.rot90(image, k=k, dims=(-2, -1)), torch.rot90(mask, k=k, dims=(-2, -1))


def build_train_transforms() -> Compose:
    """构造训练集默认增强流水线。"""

    return Compose([RandomFlip(), RandomShift(), RandomRotate90()])
