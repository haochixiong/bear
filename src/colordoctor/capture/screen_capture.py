"""屏幕截取（Qt）：冻结所有显示器、实时读取光标附近的小块像素。"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QCursor, QGuiApplication, QImage

from colordoctor.capture.geometry import ScreenInfo, logical_to_device
from colordoctor.capture.windows_display import monitor_icc_bytes


@dataclass
class ScreenCapture:
    info: ScreenInfo
    pixels: np.ndarray  # (H, W, 3) uint8，显示器 RGB（未做色彩管理换算）
    icc: bytes | None = None
    qscreen: object = field(default=None, repr=False)


def qimage_to_array(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    if w == 0 or h == 0:
        return np.zeros((0, 0, 3), np.uint8)
    bpl = img.bytesPerLine()
    ptr = img.constBits()
    ptr.setsize(bpl * h)
    arr = np.frombuffer(ptr, np.uint8).reshape(h, bpl)[:, : w * 3].reshape(h, w, 3)
    return arr.copy()


def array_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr[..., :3], dtype=np.uint8)
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return img.copy()  # 脱离 numpy 内存


def screen_info(qs) -> ScreenInfo:
    g = qs.geometry()
    return ScreenInfo(qs.name(), g.x(), g.y(), g.width(), g.height(), float(qs.devicePixelRatio()))


_icc_cache: dict[str, bytes | None] = {}


def icc_for(name: str, refresh: bool = False, info: ScreenInfo | None = None) -> bytes | None:
    """显示器 ICC。name 是 Qt 屏幕名；info 用来在 Qt 返回型号名时按位置找到 GDI 设备名（Windows）/ 显示器（macOS）。"""
    if sys.platform == "darwin":
        return _icc_for_mac(name, refresh, info)
    from colordoctor.capture.windows_display import gdi_device_name

    center = None
    if info is not None:  # Qt 6：每块屏幕的左上角保持物理坐标，尺寸按缩放比例换算
        center = (int(info.x + info.width * info.dpr / 2), int(info.y + info.height * info.dpr / 2))
    device = gdi_device_name(name, center) or name
    key = (device, center)  # 两块同型号显示器不会共用缓存
    if refresh or key not in _icc_cache:
        _icc_cache[key] = monitor_icc_bytes(device)
    return _icc_cache[key]


def _icc_for_mac(name: str, refresh: bool, info: ScreenInfo | None) -> bytes | None:
    from colordoctor.capture.mac_display import icc_at

    # macOS：Qt 的全局坐标就是“点”，和 CoreGraphics 一致：用屏幕中心点找显示器
    center = (info.x + info.width / 2, info.y + info.height / 2) if info is not None else (0.0, 0.0)
    key = ("mac", name, center)
    if refresh or key not in _icc_cache:
        _icc_cache[key] = icc_at(*center)
    return _icc_cache[key]


def list_screens() -> list[ScreenInfo]:
    return [screen_info(s) for s in QGuiApplication.screens()]


def freeze_all_screens() -> list[ScreenCapture]:
    out = []
    for qs in QGuiApplication.screens():
        pm = qs.grabWindow(0)
        arr = qimage_to_array(pm.toImage())
        info = screen_info(qs)
        out.append(ScreenCapture(info=info, pixels=arr, icc=icc_for(info.name, refresh=True, info=info), qscreen=qs))
    return out


def grab_around_cursor(radius_px: int) -> tuple[np.ndarray, ScreenInfo, tuple[int, int]] | None:
    """抓取光标附近 (2r+1)² 设备像素区域。返回 (像素, 屏幕, 光标全局逻辑坐标)。"""
    pos = QCursor.pos()
    qs = QGuiApplication.screenAt(pos)
    if qs is None:
        return None
    info = screen_info(qs)
    dpr = info.dpr
    r_log = int(np.ceil((radius_px + 1) / dpr))
    lx, ly = pos.x() - info.x, pos.y() - info.y
    pm = qs.grabWindow(0, lx - r_log, ly - r_log, 2 * r_log + 1, 2 * r_log + 1)
    arr = qimage_to_array(pm.toImage())
    if arr.size == 0:
        return None
    # 光标在抓取图中的设备像素位置
    cx = int(min(arr.shape[1] - 1, max(0, round(r_log * dpr))))
    cy = int(min(arr.shape[0] - 1, max(0, round(r_log * dpr))))
    x0, x1 = max(0, cx - radius_px), min(arr.shape[1], cx + radius_px + 1)
    y0, y1 = max(0, cy - radius_px), min(arr.shape[0], cy + radius_px + 1)
    return arr[y0:y1, x0:x1], info, (pos.x(), pos.y())


def cursor_device_position(info: ScreenInfo, pos: QPoint) -> tuple[int, int]:
    return logical_to_device(info, pos.x(), pos.y())
