from __future__ import annotations
import numpy as np

def coord2pix(cx, cy, cdim, imgdim, origin: str = "upper"):
    #经纬转像素
    x = imgdim[0] * (cx - cdim[0]) / (cdim[1] - cdim[0])
    if origin == "lower":
        y = imgdim[1] * (cy - cdim[2]) / (cdim[3] - cdim[2])
    else:
        y = imgdim[1] * (cdim[3] - cy) / (cdim[3] - cdim[2])
    return x, y


def pix2coord(x, y, cdim, imgdim, origin: str = "upper"):
    #限速转经纬
    cx = (x / imgdim[0]) * (cdim[1] - cdim[0]) + cdim[0]
    if origin == "lower":
        cy = (y / imgdim[1]) * (cdim[3] - cdim[2]) + cdim[2]
    else:
        cy = cdim[3] - (y / imgdim[1]) * (cdim[3] - cdim[2])
    return cx, cy


def km2pix(imgheight: float, latextent: float, dc: float = 1.0, a: float = 1737.4) -> float:
    return (180.0 / np.pi) * imgheight * dc / latextent / a
