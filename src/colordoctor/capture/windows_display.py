"""读取 Windows 每块显示器当前使用的 ICC 配置文件（ctypes，无额外依赖）。

Photoshop 使用的就是这里的显示器配置文件（Windows 颜色管理 → 设备 → 配置文件）。
非 Windows 平台返回 None，调用方会按 sRGB 处理并提示。
"""

from __future__ import annotations

import sys
from pathlib import Path


def monitor_icc_path(device_name: str) -> str | None:
    """device_name：Qt 的 QScreen.name()，在 Windows 上形如 '\\\\.\\DISPLAY1'。"""
    if not sys.platform.startswith("win"):
        return None
    try:  # pragma: no cover - Windows only
        import ctypes
        from ctypes import wintypes

        gdi32 = ctypes.windll.gdi32
        gdi32.CreateDCW.restype = wintypes.HDC
        gdi32.CreateDCW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p]
        gdi32.GetICMProfileW.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR]
        gdi32.DeleteDC.argtypes = [wintypes.HDC]
        hdc = gdi32.CreateDCW("DISPLAY", device_name, None, None)
        if not hdc:
            return None
        try:
            size = wintypes.DWORD(0)
            gdi32.GetICMProfileW(hdc, ctypes.byref(size), None)
            if size.value == 0:
                return None
            buf = ctypes.create_unicode_buffer(size.value)
            if not gdi32.GetICMProfileW(hdc, ctypes.byref(size), buf):
                return None
            return buf.value or None
        finally:
            gdi32.DeleteDC(hdc)
    except Exception:
        return None


def monitor_icc_bytes(device_name: str) -> bytes | None:
    path = monitor_icc_path(device_name)
    if not path:
        return None
    try:
        return Path(path).read_bytes()
    except OSError:
        return None
