"""色彩空间换算（全部 NumPy 向量化）。

约定：
- 输入的 RGB 一律是 **sRGB 8 位数值**（0–255），形状 (..., 3)。
- CIE Lab 默认使用 **D50 白点 + Bradford 色适应**，与 Photoshop 信息面板
  （以及 ICC 配置文件连接空间）一致；需要 CIE 标准 D65 Lab 时传 white="D65"。
- 色相角一律为 0–360 度。
"""

from __future__ import annotations

import numpy as np

# IEC 61966-2-1 sRGB -> XYZ (D65)
_M_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_M_XYZ_TO_SRGB = np.linalg.inv(_M_SRGB_TO_XYZ)

# Bradford D65 -> D50 (Lindbloom)
_M_BRADFORD_D65_TO_D50 = np.array(
    [
        [1.0478112, 0.0228866, -0.0501270],
        [0.0295424, 0.9904844, -0.0170491],
        [-0.0092345, 0.0150436, 0.7521316],
    ]
)
_M_BRADFORD_D50_TO_D65 = np.linalg.inv(_M_BRADFORD_D65_TO_D50)

WHITE_D65 = _M_SRGB_TO_XYZ @ np.ones(3)
WHITE_D50 = _M_BRADFORD_D65_TO_D50 @ WHITE_D65

_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def _as_float(rgb) -> np.ndarray:
    return np.asarray(rgb, dtype=np.float64)


def srgb8_to_linear(rgb) -> np.ndarray:
    c = _as_float(rgb) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb_float(lin) -> np.ndarray:
    """线性 RGB -> sRGB 0–1 浮点（不裁切）。"""
    lin = np.asarray(lin, dtype=np.float64)
    a = np.abs(lin)
    v = np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1 / 2.4) - 0.055)
    return np.sign(lin) * v


def float_to_srgb8(c) -> np.ndarray:
    """sRGB 0–1 浮点 -> 0–255 整数（四舍五入并裁切）。"""
    return np.clip(np.rint(np.asarray(c, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def srgb8_to_xyz(rgb, white: str = "D50") -> np.ndarray:
    xyz = srgb8_to_linear(rgb) @ _M_SRGB_TO_XYZ.T
    if white == "D50":
        xyz = xyz @ _M_BRADFORD_D65_TO_D50.T
    elif white != "D65":
        raise ValueError(f"unknown white point: {white}")
    return xyz


def xyz_to_lab(xyz, white: str = "D50") -> np.ndarray:
    wp = WHITE_D50 if white == "D50" else WHITE_D65
    r = np.asarray(xyz, dtype=np.float64) / wp
    f = np.where(r > _EPS, np.cbrt(r), (_KAPPA * r + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab, white: str = "D50") -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    wp = WHITE_D50 if white == "D50" else WHITE_D65
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    xr = np.where(fx**3 > _EPS, fx**3, (116.0 * fx - 16.0) / _KAPPA)
    yr = np.where(lab[..., 0] > _KAPPA * _EPS, fy**3, lab[..., 0] / _KAPPA)
    zr = np.where(fz**3 > _EPS, fz**3, (116.0 * fz - 16.0) / _KAPPA)
    return np.stack([xr, yr, zr], axis=-1) * wp


def srgb8_to_lab(rgb, white: str = "D50") -> np.ndarray:
    return xyz_to_lab(srgb8_to_xyz(rgb, white), white)


def lab_to_srgb_float(lab, white: str = "D50") -> np.ndarray:
    xyz = lab_to_xyz(lab, white)
    if white == "D50":
        xyz = xyz @ _M_BRADFORD_D50_TO_D65.T
    return linear_to_srgb_float(xyz @ _M_XYZ_TO_SRGB.T)


def lab_to_srgb8(lab, white: str = "D50") -> np.ndarray:
    return float_to_srgb8(lab_to_srgb_float(lab, white))


def lab_in_srgb_gamut(lab, white: str = "D50", tol: float = 0.5 / 255) -> np.ndarray:
    c = lab_to_srgb_float(lab, white)
    return np.all((c >= -tol) & (c <= 1 + tol), axis=-1)


def lab_to_lch(lab) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    C = np.hypot(lab[..., 1], lab[..., 2])
    h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0
    return np.stack([lab[..., 0], C, h], axis=-1)


def lch_to_lab(lch) -> np.ndarray:
    lch = np.asarray(lch, dtype=np.float64)
    hr = np.radians(lch[..., 2])
    return np.stack([lch[..., 0], lch[..., 1] * np.cos(hr), lch[..., 1] * np.sin(hr)], axis=-1)


# ---------------------------------------------------------------- OKLab
_M_OK1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M_OK2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)


def srgb8_to_oklab(rgb) -> np.ndarray:
    lms = srgb8_to_linear(rgb) @ _M_OK1.T
    return np.cbrt(lms) @ _M_OK2.T


def oklab_to_oklch(ok) -> np.ndarray:
    return lab_to_lch(ok)


# ---------------------------------------------------------------- HSB / HSL
def _hue_from_rgb(c: np.ndarray, mx: np.ndarray, mn: np.ndarray) -> np.ndarray:
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    d = mx - mn
    safe = np.where(d == 0, 1.0, d)
    h = np.where(
        mx == r,
        ((g - b) / safe) % 6.0,
        np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    )
    return np.where(d == 0, 0.0, h * 60.0) % 360.0


def srgb8_to_hsb(rgb) -> np.ndarray:
    """HSB（= HSV，Photoshop 拾色器的 H/S/B）。H 0–360，S/B 0–100。"""
    c = _as_float(rgb) / 255.0
    mx = c.max(axis=-1)
    mn = c.min(axis=-1)
    h = _hue_from_rgb(c, mx, mn)
    s = np.where(mx == 0, 0.0, (mx - mn) / np.where(mx == 0, 1.0, mx))
    return np.stack([h, s * 100.0, mx * 100.0], axis=-1)


def srgb8_to_hsl(rgb) -> np.ndarray:
    """HSL：H 0–360，S/L 0–100（顺序为 H, S, L）。"""
    c = _as_float(rgb) / 255.0
    mx = c.max(axis=-1)
    mn = c.min(axis=-1)
    h = _hue_from_rgb(c, mx, mn)
    light = (mx + mn) / 2.0
    d = mx - mn
    denom = np.where(light <= 0.5, mx + mn, 2.0 - mx - mn)
    s = np.where(d == 0, 0.0, d / np.where(denom == 0, 1.0, denom))
    return np.stack([h, s * 100.0, light * 100.0], axis=-1)


def _hsx_sector(h: np.ndarray, chroma: np.ndarray, m: np.ndarray) -> np.ndarray:
    hp = (h % 360.0) / 60.0
    x = chroma * (1 - np.abs(hp % 2 - 1))
    z = np.zeros_like(hp)
    i = np.floor(hp).astype(int) % 6
    r = np.choose(i, [chroma, x, z, z, x, chroma])
    g = np.choose(i, [x, chroma, chroma, x, z, z])
    b = np.choose(i, [z, z, x, chroma, chroma, x])
    return np.stack([r + m, g + m, b + m], axis=-1)


def hsb_to_srgb8(hsb) -> np.ndarray:
    hsb = np.asarray(hsb, dtype=np.float64)
    v = hsb[..., 2] / 100.0
    chroma = v * hsb[..., 1] / 100.0
    return float_to_srgb8(_hsx_sector(hsb[..., 0], chroma, v - chroma))


def hsl_to_srgb8(hsl) -> np.ndarray:
    hsl = np.asarray(hsl, dtype=np.float64)
    light = hsl[..., 2] / 100.0
    chroma = (1 - np.abs(2 * light - 1)) * hsl[..., 1] / 100.0
    return float_to_srgb8(_hsx_sector(hsl[..., 0], chroma, light - chroma / 2))


def relative_luminance(rgb) -> np.ndarray:
    """WCAG/sRGB 相对亮度 Y（0–1）。"""
    return srgb8_to_linear(rgb) @ _M_SRGB_TO_XYZ[1]


def to_hex(rgb) -> str:
    r, g, b = (int(v) for v in np.asarray(rgb).ravel()[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------- ΔE00
def delta_e_2000(lab1, lab2, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0):
    """CIEDE2000 色差（Sharma 2005 实现）。可广播。"""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25.0**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = np.where((a1p == 0) & (b1 == 0), 0.0, np.degrees(np.arctan2(b1, a1p)) % 360)
    h2p = np.where((a2p == 0) & (b2 == 0), 0.0, np.degrees(np.arctan2(b2, a2p)) % 360)
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0.0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(
        C1p * C2p == 0,
        hsum,
        np.where(
            np.abs(h1p - h2p) <= 180,
            hsum / 2,
            np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2),
        ),
    )
    T = (
        1
        - 0.17 * np.cos(np.radians(hbp - 30))
        + 0.24 * np.cos(np.radians(2 * hbp))
        + 0.32 * np.cos(np.radians(3 * hbp + 6))
        - 0.20 * np.cos(np.radians(4 * hbp - 63))
    )
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp**7 / (Cbp**7 + 25.0**7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )


# ---------------------------------------------------------------- 色相工具
def hue_diff(h_from, h_to):
    """从 h_from 到 h_to 的有符号最短角度差（-180, 180]。"""
    d = (np.asarray(h_to, dtype=np.float64) - np.asarray(h_from, dtype=np.float64)) % 360.0
    d = np.where(d > 180.0, d - 360.0, d)
    return d if np.ndim(d) else float(d)


# Lab(D50) 色相分区（度）。分界取 sRGB 原色/间色色相角的大致中点，
# 测试中验证每个原色都落在自己的分区里。
HUE_SECTORS: tuple[tuple[str, float, float], ...] = (
    ("红", 15.0, 50.0),
    ("橙", 50.0, 80.0),
    ("黄", 80.0, 115.0),
    ("绿", 115.0, 170.0),
    ("青", 170.0, 245.0),
    ("蓝", 245.0, 315.0),
    ("紫/品红", 315.0, 375.0),  # 跨越 0 度
)
NEUTRAL_CHROMA = 4.0


def hue_name(h: float, chroma: float | None = None) -> str:
    if chroma is not None and chroma < NEUTRAL_CHROMA:
        return "中性灰"
    h = float(h) % 360.0
    for name, lo, hi in HUE_SECTORS:
        if lo <= h < hi or lo <= h + 360.0 < hi:
            return name
    return "紫/品红"


def opposite_hue(h: float) -> float:
    return (float(h) + 180.0) % 360.0
