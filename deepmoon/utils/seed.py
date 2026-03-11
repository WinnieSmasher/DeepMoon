"""随机种子与确定性实验工具。"""

from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np
import torch


def get_experiment_seed(config: Any) -> int | None:
    """从配置中读取实验随机种子。"""

    experiment_cfg = getattr(config, "experiment", None)
    if experiment_cfg is None:
        return None
    seed = getattr(experiment_cfg, "seed", None)
    if seed is None:
        return None
    return int(seed)


def is_deterministic_experiment(config: Any) -> bool:
    """判断当前实验是否要求确定性执行。"""

    experiment_cfg = getattr(config, "experiment", None)
    if experiment_cfg is None:
        return False
    return bool(getattr(experiment_cfg, "deterministic", False))


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """统一设置 Python、NumPy 与 PyTorch 的随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def build_dataloader_generator(seed: int | None) -> torch.Generator | None:
    """为 DataLoader 构造可复现的随机数生成器。"""

    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def build_worker_init_fn(seed: int | None) -> Callable[[int], None] | None:
    """为 DataLoader worker 派生独立但可复现的随机种子。"""

    if seed is None:
        return None

    def _init_fn(worker_id: int) -> None:
        worker_seed = int(seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_fn
