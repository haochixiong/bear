"""原图尺寸的像素表：每个像素记下它是第几种颜色，每种颜色只换算一次。

一张照片有几千万个像素，但不同的颜色一般只有十几万到一两百万种。只和颜色有关的判断（色相、色度、
是不是对立色、是不是发灰候选……）按颜色算一次，再按编号查回每个像素；只和颜色有关的统计
（占比、平均、中位数、分位数）按“颜色 × 这种颜色的像素数”加权。结果和逐个像素算完全一样，
不降采样、每个像素都算进去，只是快很多。和位置有关的判断（边缘、是否孤立、在哪一块）仍然逐个像素做。
"""

from __future__ import annotations

import numpy as np

from colordoctor.core import color_spaces as cs

_BIG = 400_000  # 像素达到这么多时用 24 位颜色表（更快），否则用 np.unique


class PixelTable:
    def __init__(self, rgb: np.ndarray):
        px = np.asarray(rgb)[..., :3]
        if px.dtype != np.uint8:
            px = np.clip(np.rint(px), 0, 255).astype(np.uint8)
        self.shape = px.shape[:2]
        flat = px.reshape(-1, 3)
        packed = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
        if packed.size >= _BIG:
            present = np.bincount(packed, minlength=1 << 24).astype(bool)
            uniq = np.flatnonzero(present).astype(np.uint32)
            lut = np.zeros(1 << 24, dtype=np.int32)
            lut[uniq] = np.arange(uniq.size, dtype=np.int32)
            index = lut[packed]
        else:
            uniq, index = np.unique(packed, return_inverse=True)
            uniq = uniq.astype(np.uint32)
            index = index.astype(np.int32)
        self.index = index.reshape(self.shape)
        self.colours = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=-1).astype(np.uint8)
        self.n = int(uniq.size)
        self.lab = np.asarray(cs.srgb8_to_lab(self.colours), dtype=np.float64)
        lch = cs.lab_to_lch(self.lab)
        self.L, self.C, self.h = lch[:, 0], lch[:, 1], lch[:, 2]
        self.clipped = cs.clipped(self.colours)

    def count(self, mask: np.ndarray | None = None) -> np.ndarray:
        """每种颜色有多少个像素（只数 mask 为 True 的）。"""
        idx = self.index if mask is None else self.index[mask]
        return np.bincount(idx.ravel(), minlength=self.n)

    def pix(self, values: np.ndarray) -> np.ndarray:
        """按颜色给的值 -> 每个像素的值（与原图同尺寸）。"""
        return np.asarray(values)[self.index]


def wmedian(values, weights) -> float:
    return float(cs.weighted_percentile(values, weights, 50))
