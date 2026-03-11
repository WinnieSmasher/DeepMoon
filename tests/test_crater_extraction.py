from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from deepmoon.postprocessing.coordinate_transform import coord2pix, km2pix, pix2coord
from deepmoon.postprocessing.crater_extraction import estimate_longlatdiamkm, extract_unique_craters
from deepmoon.postprocessing.template_match import template_match_t2c


def test_template_match_alignment() -> None:
    pred = np.zeros((64, 64), dtype=np.float32)
    cv2.circle(pred, (32, 32), 10, 1.0, 2)
    pixcsv = np.array([[32.0, 32.0, 10.0]], dtype=np.float32)
    n_match, _, _, _, err_long, err_lat, err_radius, _ = template_match_t2c(
        pred,
        pixcsv,
        minrad=8,
        maxrad=12,
    )
    assert n_match == 1
    assert err_long < 0.1
    assert err_lat < 0.1
    assert err_radius < 0.1


def test_estimate_longlatdiamkm_keeps_center_and_radius_scale() -> None:
    llbd = np.array([-10.0, 10.0, -10.0, 10.0], dtype=np.float32)
    dc = np.array([1.0], dtype=np.float32)
    coords = np.array(
        [
            [128.0, 128.0, 5.0],
            [96.0, 160.0, 8.0],
        ],
        dtype=np.float32,
    )

    estimated = estimate_longlatdiamkm((256, 256), llbd, dc, coords)

    expected_km_per_pix = 1.0 / km2pix(256, llbd[3] - llbd[2], dc=float(dc[0]))
    assert np.isclose(estimated[0, 0], 0.0, atol=1e-6)
    assert np.isclose(estimated[0, 1], 0.0, atol=1e-6)
    assert np.isclose(estimated[0, 2], 5.0 * expected_km_per_pix, rtol=1e-6)
    assert estimated[1, 0] < 0.0
    assert estimated[1, 1] < 0.0
    assert np.isclose(estimated[1, 2], 8.0 * expected_km_per_pix, rtol=1e-6)


def test_extract_unique_craters_supports_nonzero_image_ids(offset_id_hdf5, sample_config, tmp_path: Path) -> None:
    prediction_path = tmp_path / "preds.hdf5"
    result_path = tmp_path / "result.npy"
    prediction = np.zeros((2, 32, 32), dtype=np.float32)
    prediction[:, 14:18, 6:28] = 1.0
    prediction[:, 6:28, 14:18] = 1.0
    with h5py.File(prediction_path, "w") as handle:
        handle.create_dataset("predictions", data=prediction)

    config = sample_config.copy()
    config.data.test_path = str(offset_id_hdf5)
    config.data.image_size = 32
    config.data.use_synthetic_if_missing = False
    config.postprocessing.minrad = 5
    config.postprocessing.maxrad = 10
    config.postprocessing.template_thresh = 0.2
    config.postprocessing.target_thresh = 0.1

    unique = extract_unique_craters(
        config=config,
        data_path=offset_id_hdf5,
        prediction_path=prediction_path,
        result_path=result_path,
    )
    assert result_path.exists()
    assert unique.ndim == 2
    assert unique.shape[1] == 3
    assert len(unique) >= 1


@pytest.mark.parametrize("origin", ["lower", "upper"])
def test_coord_pix_roundtrip(origin: str) -> None:
    cdim = [-10.0, 15.0, -20.0, 5.0]
    imgdim = np.array([128, 96])
    cx = np.array([cdim[1], 2.0])
    cy = np.array([cdim[3], -2.0])
    x, y = coord2pix(cx, cy, cdim, imgdim, origin=origin)
    cx_roundtrip, cy_roundtrip = pix2coord(x, y, cdim, imgdim, origin=origin)
    assert np.all(np.isclose(np.r_[cx_roundtrip, cy_roundtrip], np.r_[cx, cy], rtol=1e-7, atol=1e-10))


@pytest.mark.parametrize(
    "imgheight, latextent, dc",
    [
        (1500.0, 180.0, 0.5),
        (312.0, 17.1, 0.7),
        (1138.0, 15.3, 0.931),
        (6500.0, 34.5, 0.878),
    ],
)
def test_km2pix_matches_closed_form(imgheight: float, latextent: float, dc: float) -> None:
    expected = (180.0 / (np.pi * 1737.4)) * (imgheight * dc / latextent)
    assert np.isclose(km2pix(imgheight, latextent, dc=dc, a=1737.4), expected, rtol=1e-10, atol=0.0)
