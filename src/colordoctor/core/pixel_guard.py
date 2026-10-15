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
    rgb: np.ndarray  # (h, w, 3) uint8；透明处已填中性灰
    offset: tuple[int, int]  # 裁剪后的左上角（在原图中的位置）
    opaque_fraction: float
    notes: tuple[str, ...]
    valid: np.ndarray | None = None  # (h, w) bool：不透明像素；全不透明时为 None

    def filled_for_statistics(self) -> np.ndarray:
        """给只看整体分布的分析（比例、鲜艳度、影调、仿色）用：透明处用本图层的不透明像素随机重采样填满，
        这样颜色和明度的分布与只统计不透明像素相同，不会多出一块“灰色”。"""
        if self.valid is None:
            return self.rgb
        out = self.rgb.copy()
        rng = np.random.default_rng(0)
        # 从不透明像素里抽一部分（最多 100 万）当填充的来源：分布一样，不必把几千万个像素整个复制一遍
        idx = np.flatnonzero(self.valid)
        if idx.size > 1_000_000:
            idx = rng.choice(idx, 1_000_000, replace=False)
        src = self.rgb.reshape(-1, 3)[idx]
        holes = ~self.valid
        out[holes] = src[rng.integers(0, src.shape[0], int(holes.sum()))]
        return out


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
    valid = None
    rgb = pixels[..., :3]
    frac = 1.0
    if pixels.shape[2] == 4:
        opaque = pixels[..., 3] >= ALPHA_MIN
        frac = float(opaque.mean())
        if frac < MIN_OPAQUE:
            raise BlockedError("当前图层是空白图层（几乎没有不透明像素），无法分析。")
        if frac < 0.999:
            # 外接矩形用行 / 列的 any 求（不用 nonzero 列出几千万个坐标）
            rows, cols = np.flatnonzero(opaque.any(axis=1)), np.flatnonzero(opaque.any(axis=0))
            y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
            offset = (int(x0), int(y0))
            rgb = rgb[y0:y1, x0:x1].copy()
            valid = opaque[y0:y1, x0:x1].copy()
            rgb[~valid] = 128  # 中性灰；各分析通过 valid 蒙版排除这些像素
            notes.append(f"图层有 {1 - frac:.0%} 是透明的，只分析不透明部分")
    flat = rgb.reshape(-1, 3)
    # 看每一个像素（以前只抽查 20 万个：一整块纯色里零星几个杂色点会被漏掉、整张图被拒绝分析）
    if int((flat.max(axis=0).astype(np.int16) - flat.min(axis=0)).max()) <= 1:
        raise BlockedError("当前图层是一整块纯色，没有可分析的颜色变化。")
    return Prepared(np.ascontiguousarray(rgb), offset, frac, tuple(notes), valid)
