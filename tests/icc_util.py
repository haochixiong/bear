"""测试用：生成最小的矩阵/TRC 型 ICC v2 RGB 配置文件。"""

from __future__ import annotations

import struct

import numpy as np

from colordoctor.core import color_spaces as cs

D65_XY = (0.3127, 0.3290)

PRIMARIES = {
    "adobe_rgb": ((0.6400, 0.3300), (0.2100, 0.7100), (0.1500, 0.0600)),
    "display_p3": ((0.6800, 0.3200), (0.2650, 0.6900), (0.1500, 0.0600)),
    "srgb": ((0.6400, 0.3300), (0.3000, 0.6000), (0.1500, 0.0600)),
}


def rgb_to_xyz_matrix(primaries, white=D65_XY) -> np.ndarray:
    def xyz(xy):
        x, y = xy
        return np.array([x / y, 1.0, (1 - x - y) / y])

    P = np.stack([xyz(p) for p in primaries], axis=1)
    S = np.linalg.solve(P, xyz(white))
    return P * S


def _s15(v: float) -> bytes:
    return struct.pack(">i", int(round(v * 65536)))


def _xyz_tag(v) -> bytes:
    return b"XYZ " + b"\0" * 4 + b"".join(_s15(c) for c in v)


def _curv_gamma(g: float) -> bytes:
    return b"curv" + b"\0" * 4 + struct.pack(">I", 1) + struct.pack(">H", int(round(g * 256))) + b"\0\0"


def _para_srgb() -> bytes:
    params = [2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045]
    return b"para" + b"\0" * 4 + struct.pack(">HH", 3, 0) + b"".join(_s15(p) for p in params)


def _desc(text: str) -> bytes:
    t = text.encode("ascii") + b"\0"
    return b"desc" + b"\0" * 4 + struct.pack(">I", len(t)) + t + b"\0" * 8 + b"\0" * 3 + b"\0" * 67


def _text(text: str) -> bytes:
    return b"text" + b"\0" * 4 + text.encode("ascii") + b"\0"


def make_rgb_profile(name: str, primaries, trc: str = "srgb", gamma: float = 2.2) -> bytes:
    """trc: "srgb"（分段 sRGB 曲线）或 "gamma"（纯伽马）。"""
    m = rgb_to_xyz_matrix(primaries)
    m50 = cs._M_BRADFORD_D65_TO_D50 @ m  # 列向量即 D50 适应后的原色
    curve = _para_srgb() if trc == "srgb" else _curv_gamma(gamma)
    tags = [
        (b"desc", _desc(name)),
        (b"cprt", _text("test")),
        (b"wtpt", _xyz_tag(cs.WHITE_D50)),
        (b"rXYZ", _xyz_tag(m50[:, 0])),
        (b"gXYZ", _xyz_tag(m50[:, 1])),
        (b"bXYZ", _xyz_tag(m50[:, 2])),
        (b"rTRC", curve),
        (b"gTRC", curve),
        (b"bTRC", curve),
    ]
    offset = 128 + 4 + 12 * len(tags)
    body = b""
    table = b""
    for sig, data in tags:
        while len(data) % 4:
            data += b"\0"
        table += sig + struct.pack(">II", offset + len(body), len(data))
        body += data
    size = offset + len(body)
    header = (
        struct.pack(">I", size)
        + b"none"
        + bytes([2, 0x10, 0, 0])
        + b"mntrRGB XYZ "
        + b"\0" * 12
        + b"acsp"
        + b"MSFT"
        + b"\0" * 4
        + b"\0" * 4
        + b"\0" * 4
        + b"\0" * 8
        + b"\0" * 4
        + b"".join(_s15(c) for c in cs.WHITE_D50)
        + b"\0" * 4
        + b"\0" * 16
        + b"\0" * 28
    )
    assert len(header) == 128
    return header + struct.pack(">I", len(tags)) + table + body


def srgb8_to_profile_rgb8(rgb8: np.ndarray, primaries, trc: str = "srgb", gamma: float = 2.2) -> np.ndarray:
    """数学上把 sRGB 颜色编码到另一 RGB 空间（用于验证 ICC 转换结果）。"""
    xyz = cs.srgb8_to_linear(rgb8) @ cs._M_SRGB_TO_XYZ.T
    lin = xyz @ np.linalg.inv(rgb_to_xyz_matrix(primaries)).T
    lin = np.clip(lin, 0, 1)
    if trc == "srgb":
        c = cs.linear_to_srgb_float(lin)
    else:
        c = lin ** (1 / gamma)
    return cs.float_to_srgb8(c)


def profile_rgb8_to_srgb8(rgb8: np.ndarray, primaries, trc: str = "srgb", gamma: float = 2.2) -> np.ndarray:
    c = np.asarray(rgb8, dtype=float) / 255
    if trc == "srgb":
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    else:
        lin = c**gamma
    xyz = lin @ rgb_to_xyz_matrix(primaries).T
    return cs.float_to_srgb8(cs.linear_to_srgb_float(xyz @ cs._M_XYZ_TO_SRGB.T))
