from __future__ import annotations

import h5py
import torch

from deepmoon.data.dataset import (
    MoonCraterDataset,
    ensure_hdf5_dataset,
    get_dataloaders,
)


def test_dataset_reads_hdf5_and_normalizes(sample_hdf5) -> None:
    dataset = MoonCraterDataset(hdf5_path=sample_hdf5)
    image, mask = dataset[0]
    assert image.shape == (1, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert torch.isclose(image.max(), torch.tensor(1.0))
    positive_values = image[image > 0]
    assert torch.all(positive_values >= 0.1)
    assert torch.all(positive_values <= 1.0)


def test_dataset_uses_lazy_hdf5_loading(sample_hdf5) -> None:
    dataset = MoonCraterDataset(hdf5_path=sample_hdf5)
    assert dataset.images is None
    assert dataset.masks is None
    assert dataset.image_ids == ["img_00000", "img_00001", "img_00002", "img_00003"]


def test_get_dataloaders_supports_synthetic_fallback(sample_config) -> None:
    loaders = get_dataloaders(sample_config)
    batch_images, batch_masks = next(iter(loaders["train"]))
    assert batch_images.shape == (2, 1, 256, 256)
    assert batch_masks.shape == (2, 1, 256, 256)


def test_ensure_hdf5_dataset_writes_inference_metadata(tmp_path) -> None:
    dataset_path = ensure_hdf5_dataset(
        path=tmp_path / "synthetic_test.hdf5",
        num_samples=3,
        image_size=64,
        allow_synthetic=True,
    )
    with h5py.File(dataset_path, "r") as handle:
        assert handle["input_images"].shape == (3, 64, 64)
        assert handle["target_masks"].shape == (3, 64, 64)
        assert "img_00000" in handle["longlat_bounds"]
        assert "img_00000" in handle["pix_distortion_coefficient"]
