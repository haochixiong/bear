"""读取 macOS 每块显示器当前使用的 ICC 配置文件（ctypes 调 CoreGraphics，无额外依赖）。

Photoshop 在 Mac 上用的就是这里的显示器配置文件（系统设置 → 显示器 → 颜色描述文件）。
非 macOS 或读取失败返回 None，调用方会按 sRGB 处理并提示。

坐标：Qt 在 macOS 上的全局坐标是“点”（不乘 Retina 倍数），原点在主显示器左上角，
和 CoreGraphics 的全局显示坐标一致，所以直接用屏幕中心点找显示器。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

_CG = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
_CF = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"


class _CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def _libs():
    cg = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreGraphics") or _CG)
    cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation") or _CF)
    cg.CGGetDisplaysWithPoint.argtypes = [_CGPoint, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                                          ctypes.POINTER(ctypes.c_uint32)]
    cg.CGGetDisplaysWithPoint.restype = ctypes.c_int32
    cg.CGMainDisplayID.restype = ctypes.c_uint32
    cg.CGDisplayCopyColorSpace.argtypes = [ctypes.c_uint32]
    cg.CGDisplayCopyColorSpace.restype = ctypes.c_void_p
    cg.CGColorSpaceCopyICCData.argtypes = [ctypes.c_void_p]  # macOS 10.12+
    cg.CGColorSpaceCopyICCData.restype = ctypes.c_void_p
    cf.CFDataGetLength.argtypes = [ctypes.c_void_p]
    cf.CFDataGetLength.restype = ctypes.c_long
    cf.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
    cf.CFDataGetBytePtr.restype = ctypes.c_void_p
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFRelease.restype = None
    return cg, cf


def display_at(x: float, y: float) -> int | None:
    """全局坐标（点）所在显示器的 CGDirectDisplayID；找不到时用主显示器。"""
    if sys.platform != "darwin":
        return None
    try:  # pragma: no cover - macOS only
        cg, _ = _libs()
        ids = (ctypes.c_uint32 * 8)()
        count = ctypes.c_uint32(0)
        err = cg.CGGetDisplaysWithPoint(_CGPoint(float(x), float(y)), 8, ids, ctypes.byref(count))
        if err == 0 and count.value > 0:
            return int(ids[0])
        return int(cg.CGMainDisplayID())
    except Exception:
        return None


def display_icc_bytes(display_id: int | None) -> bytes | None:
    """显示器颜色空间的 ICC 数据；读不到返回 None。"""
    if sys.platform != "darwin" or display_id is None:
        return None
    try:  # pragma: no cover - macOS only
        cg, cf = _libs()
        space = cg.CGDisplayCopyColorSpace(ctypes.c_uint32(display_id))
        if not space:
            return None
        try:
            data = cg.CGColorSpaceCopyICCData(space)
            if not data:
                return None
            try:
                n = int(cf.CFDataGetLength(data))
                ptr = cf.CFDataGetBytePtr(data)
                if n <= 128 or not ptr:  # ICC 头就有 128 字节
                    return None
                return ctypes.string_at(ptr, n)
            finally:
                cf.CFRelease(data)
        finally:
            cf.CFRelease(space)
    except Exception:
        return None


def icc_at(x: float, y: float) -> bytes | None:
    return display_icc_bytes(display_at(x, y))
