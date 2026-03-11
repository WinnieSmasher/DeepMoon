from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from deepmoon.config import ConfigNode


@pytest.fixture()
def sample_hdf5(tmp_path: Path) -> Path:
    path = tmp_path / "sample.hdf5"
    images = np.zeros((4, 32, 32), dtype=np.float32)
    masks = np.zeros((4, 32, 32), dtype=np.float32)
    images[:, 8:24, 8:24] = 200
    masks[:, 10:22, 10:22] = 1
    with h5py.File(path, "w") as handle:
        handle.create_dataset("input_images", data=images)
        handle.create_dataset("target_masks", data=masks)
        longlat_group = handle.create_group("longlat_bounds")
        distortion_group = handle.create_group("pix_distortion_coefficient")
        for index in range(len(images)):
            identifier = f"img_{index:05d}"
            longlat_group.create_dataset(identifier, data=np.array([-1.0, 1.0, -1.0, 1.0]))
            distortion_group.create_dataset(identifier, data=np.asarray([1.0], dtype=np.float32))
    return path


@pytest.fixture()
def offset_id_hdf5(tmp_path: Path) -> Path:
    path = tmp_path / "offset_ids.hdf5"
    images = np.zeros((2, 32, 32), dtype=np.float32)
    masks = np.zeros((2, 32, 32), dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("input_images", data=images)
        handle.create_dataset("target_masks", data=masks)
        longlat_group = handle.create_group("longlat_bounds")
        distortion_group = handle.create_group("pix_distortion_coefficient")
        for index, identifier in enumerate(["img_00010", "img_00011"]):
            longlat_group.create_dataset(identifier, data=np.array([-180.0, 180.0, -60.0, 60.0]))
            distortion_group.create_dataset(identifier, data=np.asarray([1.0 + index], dtype=np.float32))
    return path


@pytest.fixture()
def sample_config(tmp_path: Path) -> ConfigNode:
    checkpoint_dir = tmp_path / "checkpoints"
    return ConfigNode(
        {
            "experiment": ConfigNode({"seed": 1234, "deterministic": True}),
            "data": ConfigNode(
                {
                    "train_path": str(tmp_path / "missing_train.hdf5"),
                    "val_path": str(tmp_path / "missing_val.hdf5"),
                    "test_path": str(tmp_path / "missing_test.hdf5"),
                    "input_key": "input_images",
                    "target_key": "target_masks",
                    "image_size": 256,
                    "batch_size": 2,
                    "num_workers": 0,
                    "normalize_low": 0.1,
                    "normalize_high": 1.0,
                    "use_synthetic_if_missing": True,
                    "synthetic_samples": ConfigNode({"train": 4, "val": 2, "test": 2}),
                }
            ),
            "model": ConfigNode(
                {
                    "arch": "trans_unet",
                    "in_channels": 1,
                    "out_channels": 1,
                    "base_channels": 48,
                    "dropout": 0.1,
                    "transformer_dim": 384,
                    "transformer_layers": 6,
                    "transformer_heads": 8,
                    "transformer_mlp_ratio": 4.0,
                }
            ),
            "training": ConfigNode(
                {
                    "epochs": 1,
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "amp": False,
                    "early_stopping_patience": 2,
                    "checkpoint_dir": str(checkpoint_dir),
                    "checkpoint_name": "best.pt",
                    "device": "cpu",
                    "save_best_only": True,
                }
            ),
            "postprocessing": ConfigNode(
                {
                    "llt2": 1.8,
                    "rt2": 1.0,
                    "minrad": 5,
                    "maxrad": 40,
                    "template_thresh": 0.5,
                    "target_thresh": 0.1,
                }
            ),
            "prediction": ConfigNode(
                {
                    "model_path": str(checkpoint_dir / "best.pt"),
                    "prediction_path": str(tmp_path / "preds.hdf5"),
                    "result_path": str(tmp_path / "result.npy"),
                }
            ),
        }
    )
