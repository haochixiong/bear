"""取样：单点（1×1 / 3×3 / 5×5 / 任意半径）、矩形裁切、确定性网格降采样。"""

from __future__ import annotations

import math

import numpy as np

from colordoctor.core import color_spaces as cs
from colordoctor.core.color_models import ColorContext, ColorValues, PointSample

DEFAULT_RADIUS = 2  # 5×5
MAX_RADIUS = 50


class SampleError(ValueError):
    pass


def _check_image(pixels: np.ndarray) -> None:
    if pixels is None or pixels.ndim != 3 or pixels.shape[2] < 3 or pixels.shape[0] == 0 or pixels.shape[1] == 0:
        raise SampleError("没有可分析的像素（图像为空）")


def point_patch(pixels: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    _check_image(pixels)
    h, w = pixels.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise SampleError(f"取样点 ({x}, {y}) 超出图像范围 {w}×{h}")
    r = int(max(0, min(radius, MAX_RADIUS)))
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    return pixels[y0:y1, x0:x1, :3].reshape(-1, 3)


def color_values(rgb_mean) -> ColorValues:
    """由平均 RGB（0–255 浮点）计算各色彩空间数值。

    与 Photoshop 吸管“取样大小 N×N 平均”一致：先对 8 位数值求平均再四舍五入。
    """
    rgb8 = cs.float_to_srgb8(np.asarray(rgb_mean, dtype=float) / 255.0)
    lab = cs.srgb8_to_lab(rgb8)
    lch = cs.lab_to_lch(lab)
    ok = cs.srgb8_to_oklab(rgb8)

    def t(a, nd=4):
        return tuple(round(float(v), nd) for v in np.ravel(a))

    return ColorValues(
        rgb=tuple(int(v) for v in rgb8),
        hex=cs.to_hex(rgb8),
        hsb=t(cs.srgb8_to_hsb(rgb8)),
        hsl=t(cs.srgb8_to_hsl(rgb8)),
        lab=t(lab),
        lab_d65=t(cs.srgb8_to_lab(rgb8, white="D65")),
        lch=t(lch),
        oklab=t(ok, 5),
        oklch=t(cs.oklab_to_oklch(ok), 5),
        luminance=round(float(cs.relative_luminance(rgb8)), 7),
        hue_name=cs.hue_name(float(lch[2]), float(lch[1])),
    )


def sample_point(pixels: np.ndarray, x: int, y: int, radius: int, context: ColorContext) -> PointSample:
    patch = point_patch(pixels, x, y, radius)
    return PointSample(
        x=int(x),
        y=int(y),
        radius=int(max(0, min(radius, MAX_RADIUS))),  # 与实际取样一致（超过上限会被裁到 MAX_RADIUS）
        pixel_count=int(patch.shape[0]),
        values=color_values(patch.astype(np.float64).mean(axis=0)),
        context=context,
    )


def clamp_rect(rect, width: int, height: int):
    """把 (x, y, w, h) 裁切到图像范围内；没有交集返回 None。"""
    x, y, w, h = (int(round(v)) for v in rect)
    if w <= 0 or h <= 0:
        return None
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def crop(pixels: np.ndarray, rect) -> np.ndarray:
    _check_image(pixels)
    r = clamp_rect(rect, pixels.shape[1], pixels.shape[0])
    if r is None:
        raise SampleError("选区不在图像范围内")
    x, y, w, h = r
    return pixels[y : y + h, x : x + w, :3]


def grid_step(n_pixels: int, max_samples: int) -> int:
    if n_pixels <= max_samples:
        return 1
    return int(math.ceil(math.sqrt(n_pixels / max_samples)))


def grid_downsample(pixels: np.ndarray, max_samples: int) -> tuple[np.ndarray, int]:
    """确定性分层（等间隔网格）降采样，保留二维结构，返回 (子图, 步长)。"""
    _check_image(pixels)
    h, w = pixels.shape[:2]
    step = grid_step(h * w, max_samples)
    if step == 1:
        return pixels[..., :3], 1
    off = step // 2
    sub = pixels[off::step, off::step, :3]
    while sub.shape[0] * sub.shape[1] > max_samples:  # 极端长宽比时保证上限
        step += 1
        off = step // 2
        sub = pixels[off::step, off::step, :3]
    return sub, step
