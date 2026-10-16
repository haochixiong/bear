"""取样：单点（1×1 / 3×3 / 5×5 / 任意半径）、矩形裁切、确定性网格降采样。"""

from __future__ import annotations

import contextvars
import math
from contextlib import contextmanager

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


# 分析按原图尺寸做：每个像素都检查（不降采样），只有超过这个上限的超大图才按格平均缩小
FULL_RES_MAX = 160_000_000
PREVIEW_MAX = 12_000_000  # “先出结果”：先按这么多像素（按格平均）算一遍，原图精查在后面
FAST_MAX = 24_000_000  # 「快速」分析精度：超过这么多像素的图按格平均缩小后分析

# 当前这次分析最多看多少像素、属于哪种精度（"full" / "fast" / "preview"）。用 contextvar：只影响这一次分析
_LIMIT: contextvars.ContextVar[tuple[int, str]] = contextvars.ContextVar("colordoctor_resolution", default=(FULL_RES_MAX, "full"))


@contextmanager
def resolution(max_pixels: int, label: str):
    """在这段代码里，分析最多看 max_pixels 个像素（超过的按格平均缩小）。label：精度名称，写进结果。"""
    token = _LIMIT.set((min(int(max_pixels), FULL_RES_MAX), label))
    try:
        yield
    finally:
        _LIMIT.reset(token)


def resolution_limit() -> tuple[int, str]:
    return _LIMIT.get()


def grid_step(n_pixels: int, max_samples: int) -> int:
    if n_pixels <= max_samples:
        return 1
    return int(math.ceil(math.sqrt(n_pixels / max_samples)))


def grid_downsample(pixels: np.ndarray, max_samples: int) -> tuple[np.ndarray, int]:
    """确定性网格降采样，保留二维结构，返回 (子图, 步长)。

    每个样本是原图 step×step 一格的平均（和 Photoshop 缩小图像一样），第 i 个样本代表 [i*step, (i+1)*step)。
    以前只取每格中间的一个像素：同一张图在 45 MP 和 2048 像素（插件传来的就是 PS 缩小过的）时，
    噪点、零星杂色像素的统计差很多，桌面版和插件的结论也就不一样。
    """
    _check_image(pixels)
    max_samples = min(max_samples, _LIMIT.get()[0])
    h, w = pixels.shape[:2]
    step = grid_step(h * w, max_samples)
    if step == 1:
        return pixels[..., :3], 1
    while (h // step) * (w // step) > max_samples:  # 极端长宽比时保证上限
        step += 1
    if h // step == 0 or w // step == 0:  # 一条线这样的极端长宽比：退回每格取一个像素
        return pixels[min(step // 2, h - 1)::step, min(step // 2, w - 1)::step, :3], step
    from colordoctor.core.pixel_table import cached

    # 同一次分析里各项都要同一张缩小图：只缩一次（也让它们共用同一份颜色编号）
    return cached(("block_mean", step), pixels, lambda: block_mean(pixels, step)), step


def block_mean(pixels: np.ndarray, step: int) -> np.ndarray:
    """按 step×step 一格求平均（不够一格的边缘舍去）。8 位图用 Pillow 的 reduce（快，不复制整幅图）。"""
    h, w = pixels.shape[:2]
    H, W = h // step, w // step
    if pixels.dtype == np.uint8 and pixels.shape[2] in (3, 4):
        from PIL import Image

        arr = pixels if pixels.flags.c_contiguous else np.ascontiguousarray(pixels)
        img = Image.fromarray(arr, "RGB" if arr.shape[2] == 3 else "RGBA")
        return np.array(img.reduce(step, box=(0, 0, W * step, H * step)))[..., :3]
    crop = pixels[: H * step, : W * step, :3].astype(np.float32)
    return crop.reshape(H, step, W, step, 3).mean(axis=(1, 3)).astype(pixels.dtype)
