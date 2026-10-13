"""Camera Raw / Lightroom 混色器（HSL）8 个通道的近似模型。

每个像素按 HSB 色相落入相邻两个通道（三角形权重），权重再乘以饱和度，
得到每个通道“管着”画面中多少颜色、这些颜色的平均色相/饱和度/亮度。
滑块数值换算是**估算**（Adobe 未公开算法）：
- 色相滑块 ±100 ≈ 色相移动 ±30°（正值 = 色相角增大：红→橙→黄→绿→青→蓝→紫→洋红）
- 饱和度滑块 -100 = 完全去色，+100 ≈ 饱和度 ×2
- 明亮度滑块 ±100 ≈ L* 变化 ±25
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colordoctor.core import color_spaces as cs

# (名称, HSB 色相中心)
CHANNELS: tuple[tuple[str, float], ...] = (
    ("红色", 0.0),
    ("橙色", 30.0),
    ("黄色", 60.0),
    ("绿色", 120.0),
    ("浅绿色", 180.0),
    ("蓝色", 240.0),
    ("紫色", 270.0),
    ("洋红", 300.0),
)
NAMES = tuple(n for n, _ in CHANNELS)
HUE_PER_UNIT = 0.3
LUM_PER_UNIT = 0.25


@dataclass(frozen=True)
class ChannelStat:
    name: str
    center: float
    share: float  # 占全图“有彩颜色量”的比例
    mean_hue: float | None  # HSB 色相
    mean_sat: float  # HSB 饱和度 0–100
    mean_bri: float  # HSB 亮度 0–100
    mean_lab: tuple[float, float, float] | None


def memberships(hue_deg: np.ndarray) -> np.ndarray:
    """(N,) HSB 色相 -> (N, 8) 通道权重（相邻两通道三角形插值，和为 1）。"""
    h = np.asarray(hue_deg, dtype=float) % 360.0
    centers = np.array([c for _, c in CHANNELS])
    nxt = np.roll(centers, -1)
    nxt[-1] += 360.0
    w = np.zeros((h.size, len(CHANNELS)))
    for i in range(len(CHANNELS)):
        c0, c1 = centers[i], nxt[i]
        hh = np.where(h < c0, h + 360.0, h)
        inside = (hh >= c0) & (hh < c1)
        t = (hh - c0) / (c1 - c0)
        w[inside, i] += 1 - t[inside]
        w[inside, (i + 1) % len(CHANNELS)] += t[inside]
    return w


def channel_stats(rgb_pixels: np.ndarray, min_sat: float = 8.0, primary: bool = False) -> list[ChannelStat]:
    """primary=True：平均色相/饱和度/亮度只统计“主要属于”该通道的像素（隶属度最大），
    否则相邻颜色（例如偏黄的绿）会把平均色相拉偏，算出方向相反的滑块。占比 share 不受影响。"""
    px = np.asarray(rgb_pixels).reshape(-1, 3)
    hsb = cs.srgb8_to_hsb(px)
    sel = hsb[:, 1] >= min_sat
    if not sel.any():
        return [ChannelStat(n, c, 0.0, None, 0.0, 0.0, None) for n, c in CHANNELS]
    hsb = hsb[sel]
    lab = cs.srgb8_to_lab(px[sel])
    m = memberships(hsb[:, 0]) * (hsb[:, 1:2] / 100.0)
    total = m.sum()
    top = m.argmax(axis=1)
    out = []
    for i, (name, center) in enumerate(CHANNELS):
        w = m[:, i]
        tw = w.sum()
        if primary:
            w = np.where(top == i, w, 0.0)
        if tw <= 1e-9 or w.sum() <= 1e-9:
            out.append(ChannelStat(name, center, round(float(tw / total), 4), None, 0.0, 0.0, None))
            continue
        share, tw = tw / total, w.sum()
        hr = np.radians(hsb[:, 0])
        mh = float(np.degrees(np.arctan2((w * np.sin(hr)).sum(), (w * np.cos(hr)).sum())) % 360)
        out.append(
            ChannelStat(
                name=name,
                center=center,
                share=round(float(share), 4),
                mean_hue=round(mh, 2),
                mean_sat=round(float((w * hsb[:, 1]).sum() / tw), 2),
                mean_bri=round(float((w * hsb[:, 2]).sum() / tw), 2),
                mean_lab=tuple(round(float(v), 2) for v in (w[:, None] * lab).sum(axis=0) / tw),
            )
        )
    return out


SAT_PER_UNIT = 0.01  # 饱和度 +100 ≈ ×2
CALIBRATION_FILE = "acr_calibration.json"
_calibration: dict | None = None


def calibration() -> dict:
    """用户在自己的 ACR 上实测的换算表（见 acr_calibration.py）；没有时为空，使用经验值。"""
    global _calibration
    if _calibration is None:
        import json

        from colordoctor.core.skin_presets import user_config_dir

        try:
            data = json.loads((user_config_dir() / CALIBRATION_FILE).read_text(encoding="utf-8"))
            _calibration = data.get("channels", {}) if isinstance(data, dict) else {}
        except (OSError, ValueError):
            _calibration = {}
    return _calibration


def set_calibration(channels: dict | None) -> None:
    """测试或校准完成后直接替换内存中的换算表（None = 下次重新读文件）。"""
    global _calibration
    _calibration = channels


def _per_unit(channel: str | None, key: str, default: float) -> float:
    """实测的“每格多少”。太小（不到经验值的 10%，说明这个滑块几乎推不动该通道的色块，多半是测量噪声）时不用，
    否则任何调整都会被换算成 ±100。"""
    v = calibration().get(channel or "", {}).get(key) if channel else None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) and abs(v) >= 0.1 * abs(default) else default


def _slider(delta: float, channel: str | None, key: str, default: float) -> int:
    u = _per_unit(channel, key, default)
    # 正负方向分别实测过时，按要拉的方向取对应的系数（方向与总体系数不一致时不用）
    side = _per_unit(channel, f"{key}_{'pos' if delta / u >= 0 else 'neg'}", u)
    if side * u <= 0:
        side = u
    return int(np.clip(round(delta / side), -100, 100))


def is_calibrated(channel: str | None = None) -> bool:
    cal = calibration()
    return bool(cal.get(channel)) if channel else bool(cal)


def hue_slider(from_hue: float, to_hue: float, channel: str | None = None) -> int:
    return _slider(float(cs.hue_diff(from_hue, to_hue)), channel, "hue_per_unit", HUE_PER_UNIT)


def sat_slider(from_sat: float, to_sat: float, channel: str | None = None) -> int:
    if from_sat <= 1e-6:
        return 0
    return _slider(to_sat / from_sat - 1.0, channel, "sat_per_unit", SAT_PER_UNIT)


def lum_slider(dL: float, channel: str | None = None) -> int:
    return _slider(dL, channel, "lum_per_unit", LUM_PER_UNIT)


SLIDER_CAP = 40  # 单条建议的上限：更大的数多半是模型外推，先调到这里、重新分析再看


def cap_slider(v: int) -> tuple[int, str]:
    """把滑块建议限制在 ±SLIDER_CAP 内；超出时返回说明文字。"""
    v = int(v)
    if abs(v) <= SLIDER_CAP:
        return v, ""
    c = int(np.sign(v)) * SLIDER_CAP
    return c, f"（估算需要 {v:+d}，先调到 {c:+d}，重新分析后再看）"


def slider_strength(v: float) -> str:
    """强度标签由滑块数值决定，避免“小幅”配上 ±100。"""
    m = abs(float(v))
    return "轻微" if m < 15 else ("小幅" if m < 30 else "中等")


def lab_hue_of(lab) -> float:
    return float(np.degrees(np.arctan2(lab[2], lab[1])) % 360)


def lab_hue_to_hsb_hue(lab_hue: float, L: float = 62.0, C: float = 35.0) -> float:
    """把 Lab 色相角换成 HSB 色相（ACR 滑块和颜色分级色轮用的是后者）。"""
    rgb = cs.lab_to_srgb8(cs.lch_to_lab(np.array([L, C, lab_hue])))
    return float(cs.srgb8_to_hsb(rgb)[0])


def channel_of_lab_hue(lab_hue: float) -> str:
    w = memberships(np.array([lab_hue_to_hsb_hue(lab_hue)]))[0]
    return NAMES[int(np.argmax(w))]


def tint_to_grading(L: float, a: float, b: float) -> tuple[int, int]:
    """把一个色调偏移（Lab a/b）换成 Camera Raw 颜色分级色轮的（色相, 饱和度）估算值。"""
    C = float(np.hypot(a, b))
    if C < 0.5:
        return 0, 0
    hue = lab_hue_to_hsb_hue(float(np.degrees(np.arctan2(b, a)) % 360), L=max(20.0, min(85.0, L)), C=max(20.0, C))
    return int(round(hue)) % 360, int(np.clip(round(C * 2.5), 0, 100))
