"""在用户自己的 ACR 上实测“滑块 1 格 = 多少”。

Adobe 没有公布混色器滑块与色相角度、饱和度、明度之间的换算，所以：
1. chart() 生成一张色卡：8 个混色器通道 × 3 个明度，加 3 个中性灰块；
2. 插件把色卡放进 PS，对每个(通道, 滑块, 数值)单独应用一次 Camera Raw，把结果像素送回；
3. fit() 比较前后色块的 HSB 色相 / Lab 色度 / 明度，得出每个通道的换算系数；
4. 保存到用户配置目录的 acr_calibration.json，acr_channels 的滑块换算随即改用实测值。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from colordoctor.core import acr_channels as ac
from colordoctor.core import color_spaces as cs

LIGHTNESS = (45.0, 62.0, 78.0)
CHROMA = 30.0
PATCH = 48
GAP = 8
SLIDERS = ("hue", "sat", "lum")
DEFAULT_VALUES = (-100, -50, 50, 100)


@dataclass(frozen=True)
class Patch:
    channel: str | None  # None = 中性灰
    L: float
    rect: tuple[int, int, int, int]  # x, y, w, h
    lab: tuple[float, float, float]


def _lab_hue_for_hsb(hsb_target: float, L: float) -> float:
    hues = np.arange(0, 360, 0.5)
    hsb = np.array([ac.lab_hue_to_hsb_hue(h, L=L, C=CHROMA) for h in hues])
    d = np.abs(((hsb - hsb_target + 180) % 360) - 180)
    return float(hues[int(np.argmin(d))])


def chart() -> tuple[np.ndarray, list[Patch]]:
    cols = len(ac.CHANNELS)
    rows = len(LIGHTNESS) + 1
    w = GAP + cols * (PATCH + GAP)
    h = GAP + rows * (PATCH + GAP)
    img = np.full((h, w, 3), 128, np.uint8)
    patches: list[Patch] = []
    for r, L in enumerate(LIGHTNESS):
        for c, (name, center) in enumerate(ac.CHANNELS):
            lab = cs.lch_to_lab(np.array([L, CHROMA, _lab_hue_for_hsb(center, L)]))
            x, y = GAP + c * (PATCH + GAP), GAP + r * (PATCH + GAP)
            img[y:y + PATCH, x:x + PATCH] = cs.lab_to_srgb8(lab)
            real = cs.srgb8_to_lab(img[y + PATCH // 2, x + PATCH // 2])
            patches.append(Patch(name, L, (x, y, PATCH, PATCH), tuple(round(float(v), 2) for v in real)))
    for i, L in enumerate((30.0, 55.0, 80.0)):
        x, y = GAP + i * (PATCH + GAP), GAP + len(LIGHTNESS) * (PATCH + GAP)
        img[y:y + PATCH, x:x + PATCH] = cs.lab_to_srgb8(np.array([L, 0.0, 0.0]))
        patches.append(Patch(None, L, (x, y, PATCH, PATCH), (L, 0.0, 0.0)))
    return img, patches


def measure(img: np.ndarray, patches: list[Patch]) -> np.ndarray:
    """每个色块中心 60% 区域的平均 Lab，(n, 3)。"""
    out = []
    for p in patches:
        x, y, w, h = p.rect
        mx, my = int(w * 0.2), int(h * 0.2)
        out.append(cs.srgb8_to_lab(img[y + my:y + h - my, x + mx:x + w - mx, :3]).reshape(-1, 3).mean(axis=0))
    return np.array(out)


def _hsb_hue(lab: np.ndarray) -> float:
    return float(cs.srgb8_to_hsb(cs.lab_to_srgb8(lab))[0])


def fit(base: np.ndarray, samples: list[tuple[str, str, int, np.ndarray]], patches: list[Patch] | None = None) -> dict:
    """base：未调整的色卡图；samples：[(通道名, 'hue'|'sat'|'lum', 滑块值, 调整后的色卡图)]。

    返回 {通道: {"hue_per_unit": 度/格, "sat_per_unit": 色度倍率变化/格, "lum_per_unit": L*/格, "n": 样本数}}。
    只用该通道自己的色块；滑块值为 0 的样本忽略。
    """
    if patches is None:
        _, patches = chart()
    b = measure(base, patches)
    acc: dict[str, dict[str, list[float]]] = {}
    for channel, slider, value, img in samples:
        if value == 0 or slider not in SLIDERS:
            continue
        a = measure(img, patches)
        idx = [i for i, p in enumerate(patches) if p.channel == channel]
        if not idx:
            raise ValueError(f"色卡里没有通道：{channel}")
        vals = []
        for i in idx:
            if slider == "hue":
                d = ((_hsb_hue(a[i]) - _hsb_hue(b[i]) + 180) % 360) - 180
                vals.append(d / value)
            elif slider == "sat":
                c0, c1 = np.hypot(*b[i][1:]), np.hypot(*a[i][1:])
                vals.append((c1 / max(c0, 1e-6) - 1.0) / value)
            else:
                vals.append((a[i][0] - b[i][0]) / value)
        acc.setdefault(channel, {}).setdefault(slider, []).append(float(np.median(vals)))
    out = {}
    for ch, d in acc.items():
        entry = {f"{k}_per_unit": round(float(np.mean(v)), 5) for k, v in d.items()}
        entry["n"] = sum(len(v) for v in d.values())
        out[ch] = entry
    return out


def store(channels: dict, acr_version: str = "") -> str:
    from colordoctor.core.skin_presets import user_config_dir

    d = user_config_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / ac.CALIBRATION_FILE
    path.write_text(json.dumps({"acr_version": acr_version, "channels": channels}, ensure_ascii=False, indent=1), encoding="utf-8")
    ac.set_calibration(channels)
    return str(path)
