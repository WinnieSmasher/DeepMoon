"""预测结果转陨石坑列表的后处理流程。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from deepmoon.data.dataset import ensure_hdf5_dataset, resolve_hdf5_image_ids
from deepmoon.models import build_model
from deepmoon.postprocessing.coordinate_transform import km2pix
from deepmoon.postprocessing.template_match import template_match_t
from deepmoon.utils.processing import preprocess_batch


def add_unique_craters(
    craters: np.ndarray,
    craters_unique: np.ndarray,
    thresh_longlat2: float,
    thresh_rad: float,
) -> np.ndarray:
    """将新检测结果并入全局唯一陨石坑集合。"""

    if len(craters_unique) == 0:
        return np.asarray(craters, dtype=np.float32)

    k2d = 180.0 / (np.pi * 1737.4)
    longitudes, latitudes, radii = craters_unique.T
    for crater in craters:
        longitude, latitude, radius = crater.T
        latitude_mid = (latitude + latitudes) / 2.0
        min_radius = np.minimum(radius, radii)
        distance = (
            (
                (longitudes - longitude)
                / (min_radius * k2d / np.cos(np.pi * latitude_mid / 180.0))
            )
            ** 2
            + ((latitudes - latitude) / (min_radius * k2d)) ** 2
        )
        radius_distance = np.abs(radii - radius) / min_radius
        duplicate_mask = (
            radius_distance < thresh_rad
        ) & (distance < thresh_longlat2)
        if not np.any(duplicate_mask):
            craters_unique = np.vstack((craters_unique, crater))
    return craters_unique


def estimate_longlatdiamkm(dim, llbd, distcoeff, coords: np.ndarray) -> np.ndarray:
    """将像素坐标系下的圆参数换算为经纬度与公里尺度。"""

    long_pix, lat_pix, radii_pix = coords.T
    km_per_pix = 1.0 / km2pix(dim[1], llbd[3] - llbd[2], dc=distcoeff)
    radii_km = radii_pix * km_per_pix
    deg_per_pix = km_per_pix * 180.0 / (np.pi * 1737.4)
    long_central = 0.5 * (llbd[0] + llbd[1])
    lat_central = 0.5 * (llbd[2] + llbd[3])

    lat_deg_firstest = lat_central - deg_per_pix * (lat_pix - dim[1] / 2.0)
    latdiff = np.abs(lat_central - lat_deg_firstest)
    latdiff[latdiff < 1e-7] = 1e-7
    lat_deg = lat_central - (
        deg_per_pix
        * (lat_pix - dim[1] / 2.0)
        * (np.pi * latdiff / 180.0)
        / np.sin(np.pi * latdiff / 180.0)
    )
    long_deg = long_central + (
        deg_per_pix
        * (long_pix - dim[0] / 2.0)
        / np.cos(np.pi * lat_deg / 180.0)
    )
    return np.column_stack((long_deg, lat_deg, radii_km))


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


def _load_model_for_inference(
    model_path: str | Path,
    config: Any,
    device: torch.device,
) -> torch.nn.Module:
    """加载推理模型并切换到评估模式。"""

    model = build_model(config)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _prepare_inference_data_path(config: Any, data_path: str | Path) -> Path:
    """确保推理阶段可获得合法的 HDF5 数据文件。"""

    data_cfg = config.data
    return ensure_hdf5_dataset(
        path=data_path,
        num_samples=int(data_cfg.synthetic_samples.test),
        image_size=int(data_cfg.image_size),
        input_key=data_cfg.input_key,
        target_key=data_cfg.target_key,
        allow_synthetic=bool(data_cfg.use_synthetic_if_missing),
    )


def _load_prediction_ids(prepared_data_path: str | Path) -> list[str]:
    """读取预测文件对应的样本标识。"""

    with h5py.File(prepared_data_path, "r") as handle:
        return resolve_hdf5_image_ids(handle, int(handle["input_images"].shape[0]))


def get_model_predictions(
    config: Any,
    model_path: str | Path,
    data_path: str | Path,
    prediction_path: str | Path,
    return_predictions: bool = True,
) -> np.ndarray | None:
    """运行模型推理，并将概率图缓存为 HDF5 文件。"""

    prediction_path = Path(prediction_path)
    if prediction_path.exists():
        if not return_predictions:
            return None
        with h5py.File(prediction_path, "r") as handle:
            return handle["predictions"][...]

    prepared_data_path = _prepare_inference_data_path(config, data_path)
    batch_size = int(getattr(config.data, "batch_size", 1))
    batch_size = max(batch_size, 1)
    device = _resolve_device(config.training.device)
    if model_path is None:
        raise ValueError("prediction_path 不存在时必须提供 model_path 才能生成预测结果")
    model = _load_model_for_inference(model_path, config, device)
    with h5py.File(prepared_data_path, "r") as handle:
        image_store = handle[config.data.input_key]
        total = int(image_store.shape[0])
        image_size = int(config.data.image_size)

        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(prediction_path, "w") as output_handle:
            prediction_store = output_handle.create_dataset(
                "predictions",
                shape=(total, image_size, image_size),
                dtype=np.float32,
            )
            with torch.no_grad():
                for start in range(0, total, batch_size):
                    stop = min(start + batch_size, total)
                    # 预测阶段沿用训练时的归一化逻辑，保证输入分布一致。
                    batch = image_store[start:stop].astype(np.float32)
                    processed = preprocess_batch(
                        batch,
                        low=float(config.data.normalize_low),
                        high=float(config.data.normalize_high),
                    )
                    batch_tensor = torch.from_numpy(processed[:, None, :, :]).to(device)
                    predictions = torch.sigmoid(model(batch_tensor)).cpu().numpy()[:, 0]
                    prediction_store[start:stop] = predictions.astype(np.float32, copy=False)

    if not return_predictions:
        return None
    with h5py.File(prediction_path, "r") as handle:
        return handle["predictions"][...]


def extract_unique_craters(
    config: Any,
    data_path: str | Path,
    prediction_path: str | Path,
    result_path: str | Path,
    model_path: str | Path | None = None,
) -> np.ndarray:
    """从概率图中提取唯一陨石坑，并保存为 `npy` 文件。"""

    prepared_data_path = _prepare_inference_data_path(config, data_path)
    if not Path(prediction_path).exists():
        get_model_predictions(
            config,
            model_path=model_path,
            data_path=prepared_data_path,
            prediction_path=prediction_path,
            return_predictions=False,
        )

    with h5py.File(prepared_data_path, "r") as data_handle, h5py.File(
        prediction_path,
        "r",
    ) as prediction_handle:
        llbd_key = "longlat_bounds"
        distcoeff_key = "pix_distortion_coefficient"
        prediction_store = prediction_handle["predictions"]
        identifiers = resolve_hdf5_image_ids(data_handle, int(prediction_store.shape[0]))
        dim = (float(config.data.image_size), float(config.data.image_size))
        unique = np.empty((0, 3), dtype=np.float32)
        for index, identifier in enumerate(identifiers):
            # 模板匹配先在单张预测图内去重，再映射到全局经纬度坐标系。
            coords = template_match_t(
                prediction_store[index],
                minrad=int(config.postprocessing.minrad),
                maxrad=int(config.postprocessing.maxrad),
                longlat_thresh2=float(config.postprocessing.llt2),
                rad_thresh=float(config.postprocessing.rt2),
                template_thresh=float(config.postprocessing.template_thresh),
                target_thresh=float(config.postprocessing.target_thresh),
            )
            if len(coords) == 0:
                continue
            llbd = data_handle[llbd_key][identifier][...]
            distcoeff = data_handle[distcoeff_key][identifier][0]
            new_unique = estimate_longlatdiamkm(dim, llbd, distcoeff, coords)
            unique = add_unique_craters(
                new_unique,
                unique,
                thresh_longlat2=float(config.postprocessing.llt2),
                thresh_rad=float(config.postprocessing.rt2),
            )

    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(result_path, unique)
    return unique
