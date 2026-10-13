"""多显示器 / 高 DPI 坐标换算（纯函数，可单元测试）。

Qt 使用“逻辑坐标”（受 Windows 缩放比例影响），截图得到的是“设备像素”。
每块屏幕：设备像素 = (全局逻辑坐标 - 屏幕左上角逻辑坐标) × devicePixelRatio。
上游 RNV 的屏幕取色直接把逻辑坐标当作像素坐标，缩放 ≠ 100% 时会取错像素，
多显示器时也只能截取一块屏幕；这里统一处理。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenInfo:
    name: str
    x: int  # 逻辑坐标
    y: int
    width: int
    height: int
    dpr: float = 1.0

    @property
    def device_size(self) -> tuple[int, int]:
        return (int(round(self.width * self.dpr)), int(round(self.height * self.dpr)))

    def contains(self, gx: float, gy: float) -> bool:
        return self.x <= gx < self.x + self.width and self.y <= gy < self.y + self.height


def screen_index_at(screens: list[ScreenInfo], gx: float, gy: float) -> int | None:
    for i, s in enumerate(screens):
        if s.contains(gx, gy):
            return i
    return None


def logical_to_device(s: ScreenInfo, gx: float, gy: float) -> tuple[int, int]:
    w, h = s.device_size
    px = int(math.floor((gx - s.x) * s.dpr))
    py = int(math.floor((gy - s.y) * s.dpr))
    return (min(max(px, 0), w - 1), min(max(py, 0), h - 1))


def device_to_logical(s: ScreenInfo, px: float, py: float) -> tuple[float, float]:
    return (s.x + px / s.dpr, s.y + py / s.dpr)


def rect_device(s: ScreenInfo, x: float, y: float, w: float, h: float) -> tuple[int, int, int, int] | None:
    """把屏幕内的设备像素矩形裁切到截图范围；无交集返回 None。"""
    W, H = s.device_size
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    x0, y0 = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
    x1, y1 = min(W, int(math.ceil(x + w))), min(H, int(math.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def virtual_bounds(screens: list[ScreenInfo]) -> tuple[int, int, int, int]:
    if not screens:
        return (0, 0, 0, 0)
    x0 = min(s.x for s in screens)
    y0 = min(s.y for s in screens)
    x1 = max(s.x + s.width for s in screens)
    y1 = max(s.y + s.height for s in screens)
    return (x0, y0, x1 - x0, y1 - y0)


def view_to_image(
    view_x: float, view_y: float, view_w: float, view_h: float, img_w: int, img_h: int
) -> tuple[float, float, float]:
    """图像等比缩放居中显示在视图里：返回 (缩放, 偏移x, 偏移y)，image = (view - offset) / scale。"""
    if img_w <= 0 or img_h <= 0 or view_w <= 0 or view_h <= 0:
        return (1.0, 0.0, 0.0)
    scale = min(view_w / img_w, view_h / img_h)
    ox = view_x + (view_w - img_w * scale) / 2
    oy = view_y + (view_h - img_h * scale) / 2
    return (scale, ox, oy)
