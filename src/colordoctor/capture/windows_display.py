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


def gdi_device_name(qt_name: str, center_device_xy: tuple[int, int] | None = None) -> str | None:
    """把 Qt 的屏幕名换成 GDI 设备名（'\\\\.\\DISPLAYn'）。

    较新的 Qt 6 在 Windows 上 QScreen.name() 返回显示器的型号名（如 “DELL U2720Q”），
    不能直接交给 CreateDCW；这时用屏幕中心点（物理像素）找到对应的 HMONITOR 再取设备名。
    """
    if qt_name.startswith("\\\\.\\"):
        return qt_name
    if not sys.platform.startswith("win") or center_device_xy is None:
        return None
    try:  # pragma: no cover - Windows only
        import ctypes
        from ctypes import wintypes

        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD), ("szDevice", wintypes.WCHAR * 32)]

        user32 = ctypes.windll.user32
        user32.MonitorFromPoint.restype = wintypes.HMONITOR
        user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
        user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]
        hmon = user32.MonitorFromPoint(wintypes.POINT(int(center_device_xy[0]), int(center_device_xy[1])), 2)  # NEAREST
        if not hmon:
            return None
        mi = MONITORINFOEXW()
        mi.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if not user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            return None
        return mi.szDevice or None
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
