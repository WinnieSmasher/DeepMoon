from __future__ import annotations
import numpy as np
import cv2
from skimage.feature import match_template

DEFAULT_MINRAD = 5
DEFAULT_MAXRAD = 40
DEFAULT_LONGLAT_THRESH2 = 1.8
DEFAULT_RAD_THRESH = 1.0
DEFAULT_TEMPLATE_THRESH = 0.5
DEFAULT_TARGET_THRESH = 0.1

def template_match_t(
    target: np.ndarray,
    minrad: int = DEFAULT_MINRAD,
    maxrad: int = DEFAULT_MAXRAD,
    longlat_thresh2: float = DEFAULT_LONGLAT_THRESH2,
    rad_thresh: float = DEFAULT_RAD_THRESH,
    template_thresh: float = DEFAULT_TEMPLATE_THRESH,
    target_thresh: float = DEFAULT_TARGET_THRESH,
) -> np.ndarray:

    thresholded = np.asarray(target, dtype=np.float32).copy()
    thresholded[thresholded >= target_thresh] = 1
    thresholded[thresholded < target_thresh] = 0

    ring_width = 2
    coords: list[list[int]] = []
    corr: list[float] = []
    for radius in np.arange(minrad, maxrad + 1, dtype=int):
        size = 2 * (radius + ring_width + 1)
        template = np.zeros((size, size), dtype=np.float32)
        cv2.circle(template, (radius + ring_width + 1, radius + ring_width + 1), radius, 1, ring_width)
        result = match_template(thresholded, template, pad_input=True)
        indices = np.where(result > template_thresh)
        coords_r = np.asarray(list(zip(*indices)))
        corr_r = np.asarray(result[indices])
        if len(coords_r) == 0:
            continue
        for coord in coords_r:
            coords.append([int(coord[1]), int(coord[0]), int(radius)])
        corr.extend(np.abs(corr_r).tolist())

    if not coords:
        return np.empty((0, 3), dtype=np.float32)

    coords_array = np.asarray(coords, dtype=np.float32)
    corr_array = np.asarray(corr, dtype=np.float32)
    index = 0
    while index < len(coords_array):
        longitudes, latitudes, radii = coords_array.T
        longitude, latitude, radius = coords_array[index]
        min_radius = np.minimum(radius, radii)
        distance = ((longitudes - longitude) ** 2 + (latitudes - latitude) ** 2) / (min_radius**2)
        radius_distance = np.abs(radii - radius) / min_radius
        duplicate_mask = (radius_distance < rad_thresh) & (distance < longlat_thresh2)
        duplicate_indices = np.where(duplicate_mask)[0]
        if len(duplicate_indices) > 1:
            duplicate_coords = coords_array[duplicate_indices]
            duplicate_corr = corr_array[duplicate_indices]
            coords_array[index] = duplicate_coords[int(np.argmax(duplicate_corr))]
            duplicate_mask[index] = False
            coords_array = coords_array[~duplicate_mask]
            corr_array = corr_array[~duplicate_mask]
        index += 1

    return coords_array


def template_match_t2c(
    target: np.ndarray,
    csv_coords: np.ndarray,
    minrad: int = DEFAULT_MINRAD,
    maxrad: int = DEFAULT_MAXRAD,
    longlat_thresh2: float = DEFAULT_LONGLAT_THRESH2,
    rad_thresh: float = DEFAULT_RAD_THRESH,
    template_thresh: float = DEFAULT_TEMPLATE_THRESH,
    target_thresh: float = DEFAULT_TARGET_THRESH,
    rmv_oor_csvs: int = 0,
) -> tuple[int, int, int, int, float, float, float, float]:

    templ_coords = template_match_t(
        target,
        minrad=minrad,
        maxrad=maxrad,
        longlat_thresh2=longlat_thresh2,
        rad_thresh=rad_thresh,
        template_thresh=template_thresh,
        target_thresh=target_thresh,
    )
    maxr = int(templ_coords[:, 2].max()) if len(templ_coords) > 0 else 0

    n_match = 0
    frac_dupes = 0.0
    err_lo = 0.0
    err_la = 0.0
    err_r = 0.0
    n_csv = len(csv_coords)
    n_detect = len(templ_coords)
    csv_coords = np.asarray(csv_coords, dtype=np.float32)

    for longitude, latitude, radius in templ_coords:
        if len(csv_coords) == 0:
            break
        longitudes, latitudes, radii = csv_coords.T
        min_radius = np.minimum(radius, radii)
        distance = ((longitudes - longitude) ** 2 + (latitudes - latitude) ** 2) / (min_radius**2)
        radius_distance = np.abs(radii - radius) / min_radius
        match_mask = (radius_distance < rad_thresh) & (distance < longlat_thresh2)
        match_indices = np.where(match_mask)[0]
        if len(match_indices) >= 1:
            gt_longitude, gt_latitude, gt_radius = csv_coords[match_indices[0]]
            mean_radius = (gt_radius + radius) / 2.0
            err_lo += abs(gt_longitude - longitude) / mean_radius
            err_la += abs(gt_latitude - latitude) / mean_radius
            err_r += abs(gt_radius - radius) / mean_radius
            if len(match_indices) > 1:
                frac_dupes += (len(match_indices) - 1) / float(max(len(templ_coords), 1))
        n_match += min(1, len(match_indices))
        csv_coords = csv_coords[~match_mask]

    if rmv_oor_csvs == 1 and len(csv_coords) > 0:
        upper = 15
        lower = DEFAULT_MINRAD
        out_of_range = ((csv_coords[:, 2] > upper) | (csv_coords[:, 2] < lower)).sum()
        if out_of_range < n_csv:
            n_csv -= int(out_of_range)

    if n_match >= 1:
        err_lo /= n_match
        err_la /= n_match
        err_r /= n_match

    return n_match, n_csv, n_detect, maxr, err_lo, err_la, err_r, frac_dupes
