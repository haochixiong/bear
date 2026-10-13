"""分析前的检查：当前图层能不能分析。

插件会先按图层类型拦截（调整图层、空白图层），这里再按像素内容兜底，
并把半透明图层裁到不透明范围、把透明处填成中性灰（中性灰不计入颜色统计）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# PS 的图层类型（UXP constants.LayerKind 的取值）中没有像素、不能直接分析的类型
NO_PIXEL_KINDS = {
    "brightnessContrast", "levels", "curves", "exposure", "vibrance", "hueSaturation", "colorBalance",
    "blackAndWhite", "photoFilter", "channelMixer", "colorLookup", "invert", "posterize", "threshold",
    "gradientMap", "selectiveColor", "solidColor", "gradientFill", "pattern", "light", "clarity",
    "dehaze", "grain", "adjustment", "inversion", "solidFill", "patternFill",
}
KIND_NAMES = {"group": "图层组", "text": "文字图层", "smartObject": "智能对象", "video": "视频图层", "threedimensional": "3D 图层"}

MIN_OPAQUE = 0.01  # 不透明像素少于 1%：视为空白
ALPHA_MIN = 8


class BlockedError(ValueError):
    """当前图层不能分析；message 是给用户看的中文原因。"""


@dataclass
class Prepared:
    rgb: np.ndarray  # (h, w, 3) uint8
    offset: tuple[int, int]  # 裁剪后的左上角（在原图中的位置）
    opaque_fraction: float
    notes: tuple[str, ...]


def check_layer_kind(kind: str | None, name: str = "") -> None:
    if not kind:
        return
    label = f"「{name}」" if name else "当前图层"
    if kind in NO_PIXEL_KINDS:
        raise BlockedError(f"{label}是调整图层或填充图层，本身没有像素。请在 PS 里选中像素图层，或把分析对象改为「合并可见图层」。")


def prepare(pixels: np.ndarray) -> Prepared:
    """pixels：(h, w, 3|4) uint8。返回可分析的 RGB；不能分析时抛 BlockedError。"""
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4) or pixels.shape[0] < 2 or pixels.shape[1] < 2:
        raise BlockedError("没有收到有效的像素（图层为空或尺寸为 0）。")
    notes: list[str] = []
    offset = (0, 0)
    rgb = pixels[..., :3]
    frac = 1.0
    if pixels.shape[2] == 4:
        opaque = pixels[..., 3] >= ALPHA_MIN
        frac = float(opaque.mean())
        if frac < MIN_OPAQUE:
            raise BlockedError("当前图层是空白图层（几乎没有不透明像素），无法分析。")
        if frac < 0.999:
            ys, xs = np.nonzero(opaque)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            offset = (int(x0), int(y0))
            rgb = rgb[y0:y1, x0:x1].copy()
            rgb[~opaque[y0:y1, x0:x1]] = 128  # 中性灰：不计入颜色
            notes.append(f"图层有 {1 - frac:.0%} 是透明的，只分析不透明部分")
    flat = rgb.reshape(-1, 3)
    sample = flat[:: max(1, flat.shape[0] // 200_000)]
    if float(sample.astype(np.float32).std(axis=0).max()) < 0.5:
        raise BlockedError("当前图层是一整块纯色，没有可分析的颜色变化。")
    return Prepared(np.ascontiguousarray(rgb), offset, frac, tuple(notes))
