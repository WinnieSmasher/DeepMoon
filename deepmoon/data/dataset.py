"""HDF5 数据集读取、合成样本构建与 DataLoader 组装。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from deepmoon.data.transforms import Compose, build_train_transforms
from deepmoon.utils.processing import as_tensor, get_id, normalize_non_zero
from deepmoon.utils.seed import build_dataloader_generator, build_worker_init_fn, get_experiment_seed


DEFAULT_LONGLAT_BOUNDS = np.array([-180.0, 180.0, -60.0, 60.0], dtype=np.float32)
_METADATA_GROUP_CANDIDATES = (
    "longlat_bounds",
    "pix_bounds",
    "cll_xy",
    "pix_distortion_coefficient",
)
_IMAGE_ID_PATTERN = re.compile(r"(\d+)$")


def _image_id_sort_key(identifier: str) -> tuple[int, str]:
    match = _IMAGE_ID_PATTERN.search(identifier)
    if match is None:
        return (-1, identifier)
    return (int(match.group(1)), identifier)


def _sorted_image_ids(identifiers: Iterable[str]) -> list[str]:
    return sorted(identifiers, key=_image_id_sort_key)


def resolve_hdf5_image_ids(handle: h5py.File, count: int) -> list[str]:
    """从元数据分组中解析与样本顺序一致的图像 ID。"""

    for group_name in _METADATA_GROUP_CANDIDATES:
        if group_name not in handle:
            continue
        group = handle[group_name]
        keys = _sorted_image_ids(group.keys())
        if len(keys) >= count:
            return keys[:count]
    return [get_id(index) for index in range(count)]


class MoonCraterDataset(Dataset):
    """同时支持 HDF5 文件与内存数组的月球陨石坑数据集。"""

    def __init__(
        self,
        hdf5_path: str | Path | None = None,
        images: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        transforms: Compose | None = None,
        input_key: str = "input_images",
        target_key: str = "target_masks",
        normalize_low: float = 0.1,
        normalize_high: float = 1.0,
    ) -> None:
        if hdf5_path is None and (images is None or masks is None):
            raise ValueError("必须提供 hdf5_path 或 images/masks")

        self.hdf5_path = Path(hdf5_path) if hdf5_path is not None else None
        self.input_key = input_key
        self.target_key = target_key
        self.transforms = transforms
        self.normalize_low = normalize_low
        self.normalize_high = normalize_high
        self._handle: h5py.File | None = None
        self._image_store = None
        self._mask_store = None
        self.images: np.ndarray | None = None
        self.masks: np.ndarray | None = None

        if self.hdf5_path is not None:
            # 初始化阶段只校验结构，不长期持有句柄，避免多进程 DataLoader 出现句柄继承问题。
            with h5py.File(self.hdf5_path, "r") as handle:
                if input_key not in handle:
                    raise KeyError(f"HDF5 中缺少输入数据集: {input_key}")
                if target_key not in handle:
                    raise KeyError(f"HDF5 中缺少目标数据集: {target_key}")
                image_store = handle[input_key]
                mask_store = handle[target_key]
                if image_store.shape[0] != mask_store.shape[0]:
                    raise ValueError("图像与标签数量不一致")
                self.length = int(image_store.shape[0])
                self.image_ids = resolve_hdf5_image_ids(handle, self.length)
        else:
            self.images = np.asarray(images, dtype=np.float32)
            self.masks = np.asarray(masks, dtype=np.float32)
            if self.images.shape[0] != self.masks.shape[0]:
                raise ValueError("图像与标签数量不一致")
            self.length = int(self.images.shape[0])
            self.image_ids = [get_id(index) for index in range(self.length)]

    def __getstate__(self) -> dict[str, Any]:
        """在多进程序列化时丢弃已打开的 HDF5 句柄。"""

        state = self.__dict__.copy()
        state["_handle"] = None
        state["_image_store"] = None
        state["_mask_store"] = None
        return state

    def _ensure_handle(self) -> None:
        """按需延迟打开 HDF5 文件。"""

        if self.hdf5_path is None or self._handle is not None:
            return
        self._handle = h5py.File(self.hdf5_path, "r")
        self._image_store = self._handle[self.input_key]
        self._mask_store = self._handle[self.target_key]

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._image_store = None
        self._mask_store = None

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self.length

    def get_image_id(self, index: int) -> str:
        """返回给定样本下标对应的图像 ID。"""

        return self.image_ids[index]

    def _get_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """按下标读取单张图像及其目标掩码。"""

        if self.hdf5_path is None:
            image = self.images[index]
            mask = self.masks[index]
            return np.asarray(image, dtype=np.float32), np.asarray(mask, dtype=np.float32)

        self._ensure_handle()
        image = self._image_store[index]
        mask = self._mask_store[index]
        return np.asarray(image, dtype=np.float32), np.asarray(mask, dtype=np.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """读取样本、归一化并转换为模型输入张量。"""

        image, mask = self._get_pair(index)
        image = normalize_non_zero(
            image,
            low=self.normalize_low,
            high=self.normalize_high,
        )
        # 历史数据中掩码可能以 0/255 存储，这里统一缩放到 0/1。
        if mask.max() > 1.0:
            mask = mask / 255.0

        image_tensor = as_tensor(image)
        mask_tensor = as_tensor(mask.astype(np.float32, copy=False))
        if self.transforms is not None:
            image_tensor, mask_tensor = self.transforms(image_tensor, mask_tensor)
        return image_tensor.contiguous(), mask_tensor.contiguous()


def _draw_circle(
    image_size: int,
    radius: int,
    center_x: int,
    center_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """绘制一个简化的圆形陨石坑样本。"""

    yy, xx = np.ogrid[:image_size, :image_size]
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    mask = (distance <= radius).astype(np.float32)
    image = np.clip(
        mask * 220.0 + (distance <= radius + 2).astype(np.float32) * 35.0,
        0,
        255,
    )
    return image, mask


def create_synthetic_split(
    num_samples: int,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """构造一个轻量的合成数据切分，用于测试或缺省回退。"""

    images = np.zeros((num_samples, image_size, image_size), dtype=np.float32)
    masks = np.zeros((num_samples, image_size, image_size), dtype=np.float32)
    for index in range(num_samples):
        radius = 12 + (index % 10)
        center_x = image_size // 2 + (index % 3 - 1) * 8
        center_y = image_size // 2 + ((index // 3) % 3 - 1) * 8
        image, mask = _draw_circle(image_size, radius, center_x, center_y)
        images[index] = image
        masks[index] = mask
    return images, masks


def write_synthetic_hdf5(
    path: str | Path,
    num_samples: int,
    image_size: int,
    input_key: str = "input_images",
    target_key: str = "target_masks",
) -> Path:
    """将合成样本写入标准 HDF5 结构。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images, masks = create_synthetic_split(num_samples=num_samples, image_size=image_size)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(input_key, data=images)
        handle.create_dataset(target_key, data=masks)
        longlat_group = handle.create_group("longlat_bounds")
        distortion_group = handle.create_group("pix_distortion_coefficient")
        for index in range(num_samples):
            identifier = get_id(index)
            longlat_group.create_dataset(identifier, data=DEFAULT_LONGLAT_BOUNDS)
            distortion_group.create_dataset(
                identifier,
                data=np.asarray([1.0], dtype=np.float32),
            )
    return path


def ensure_hdf5_dataset(
    path: str | Path,
    num_samples: int,
    image_size: int,
    input_key: str = "input_images",
    target_key: str = "target_masks",
    allow_synthetic: bool = False,
) -> Path:
    """确保路径上存在可读取的 HDF5 数据文件。"""

    path = Path(path)
    if path.exists():
        return path
    if not allow_synthetic:
        raise FileNotFoundError(f"未找到数据文件: {path}")
    return write_synthetic_hdf5(
        path=path,
        num_samples=num_samples,
        image_size=image_size,
        input_key=input_key,
        target_key=target_key,
    )


def _create_dataset_for_split(
    path: str | Path,
    num_samples: int,
    image_size: int,
    transforms: Compose | None,
    input_key: str,
    target_key: str,
    normalize_low: float,
    normalize_high: float,
    allow_synthetic: bool,
) -> MoonCraterDataset:
    """根据切分路径或合成回退策略创建数据集对象。"""

    path = Path(path)
    if path.exists():
        return MoonCraterDataset(
            hdf5_path=path,
            transforms=transforms,
            input_key=input_key,
            target_key=target_key,
            normalize_low=normalize_low,
            normalize_high=normalize_high,
        )
    if not allow_synthetic:
        raise FileNotFoundError(f"未找到数据文件: {path}")

    images, masks = create_synthetic_split(
        num_samples=num_samples,
        image_size=image_size,
    )
    return MoonCraterDataset(
        images=images,
        masks=masks,
        transforms=transforms,
        normalize_low=normalize_low,
        normalize_high=normalize_high,
    )


def get_dataloaders(config: Any) -> dict[str, DataLoader]:
    """基于统一配置构造训练、验证与测试 DataLoader。"""

    train_transforms = build_train_transforms()
    eval_transforms = None
    data_cfg = config.data
    experiment_seed = get_experiment_seed(config)

    train_dataset = _create_dataset_for_split(
        path=data_cfg.train_path,
        num_samples=int(data_cfg.synthetic_samples.train),
        image_size=int(data_cfg.image_size),
        transforms=train_transforms,
        input_key=data_cfg.input_key,
        target_key=data_cfg.target_key,
        normalize_low=float(data_cfg.normalize_low),
        normalize_high=float(data_cfg.normalize_high),
        allow_synthetic=bool(data_cfg.use_synthetic_if_missing),
    )
    val_dataset = _create_dataset_for_split(
        path=data_cfg.val_path,
        num_samples=int(data_cfg.synthetic_samples.val),
        image_size=int(data_cfg.image_size),
        transforms=eval_transforms,
        input_key=data_cfg.input_key,
        target_key=data_cfg.target_key,
        normalize_low=float(data_cfg.normalize_low),
        normalize_high=float(data_cfg.normalize_high),
        allow_synthetic=bool(data_cfg.use_synthetic_if_missing),
    )
    test_dataset = _create_dataset_for_split(
        path=data_cfg.test_path,
        num_samples=int(data_cfg.synthetic_samples.test),
        image_size=int(data_cfg.image_size),
        transforms=eval_transforms,
        input_key=data_cfg.input_key,
        target_key=data_cfg.target_key,
        normalize_low=float(data_cfg.normalize_low),
        normalize_high=float(data_cfg.normalize_high),
        allow_synthetic=bool(data_cfg.use_synthetic_if_missing),
    )

    batch_size = int(data_cfg.batch_size)
    num_workers = int(data_cfg.num_workers)
    generator = build_dataloader_generator(experiment_seed)
    worker_init_fn = build_worker_init_fn(experiment_seed)

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        ),
    }
