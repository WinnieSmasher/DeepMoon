from __future__ import annotations

import numpy as np
import torch


def normalize_non_zero(image: np.ndarray, low: float = 0.1, high: float = 1.0) -> np.ndarray:
    #将非零像素缩放到指定区间。
    result = image.astype(np.float32, copy=True)
    if result.max() > 1.0:
        result = result / 255.0

    positive_mask = result > 0
    if not np.any(positive_mask):
        return result

    positive_values = result[positive_mask]
    min_value = float(positive_values.min())
    max_value = float(positive_values.max())
    if np.isclose(max_value, min_value):
        result[positive_mask] = high
        return result

    result[positive_mask] = low + (positive_values - min_value) * (high - low) / (
        max_value - min_value
    )
    return result


def preprocess_batch(data: np.ndarray, low: float = 0.1, high: float = 1.0) -> np.ndarray:
    #批量归一化输入图像。
    images = np.asarray(data, dtype=np.float32)
    processed = np.empty_like(images)
    for index, image in enumerate(images):
        processed[index] = normalize_non_zero(image, low=low, high=high)
    return processed


def as_tensor(image: np.ndarray) -> torch.Tensor:
    #转换为[C, H, W]张量。
    if image.ndim == 2:
        image = image[None, ...]
    return torch.as_tensor(image, dtype=torch.float32)


def get_id(index: int, zeropad: int = 5) -> str:
    #生成与原版一致的样本编号。
    return f"img_{index:0{zeropad}d}"
