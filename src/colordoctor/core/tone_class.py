"""影调分类（九种影调）：按直方图描述画面现在是什么影调，不评判对错。

摄影里的影调 = 基调 × 调长：
  基调（高 / 中 / 低）：画面的主要亮度落在直方图的哪一段——亮部、中间调还是暗部占得最多。
  调长（长 / 中 / 短）：从最暗到最亮跨了多大范围——跨度大是长调（反差大），跨度小是短调（柔和、平）。
直方图按 PS 的“明度”算（0.30R + 0.59G + 0.11B，0–255），分成 10 格（每格约 25.5 级）：
  第 0–3 格为暗部、4–6 格为中间调、7–9 格为亮部，哪一段像素最多就是什么基调；
  去掉两端各 2% 的零星像素后量跨度：≥ 170 级（约 2/3 个直方图）为长调、90–170 为中调、< 90 为短调。
例如：日系高调人像多是“高短调 / 高中调”，暗调情绪人像多是“低长调 / 低中调”，胶片、莫兰迪多是“中短调”。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KEYS = ("高", "中", "低")
LENGTHS = ("长", "中", "短")
TONE_NAMES = tuple(f"{k}{n}调" for k in KEYS for n in LENGTHS)  # 高长调 … 低短调

# 每种影调的一句话说明（显示在九宫格里）
TONE_DESC = {
    "高长调": "明亮为主、带少量深色，清爽有力度",
    "高中调": "明亮、层次适中，干净通透",
    "高短调": "明亮柔和、反差小，轻盈朦胧（日系 / 高调人像）",
    "中长调": "亮暗都有、层次最丰富，最常见",
    "中中调": "中间调为主、反差适中，自然平和",
    "中短调": "中间调为主、反差小，柔和偏灰（胶片 / 莫兰迪）",
    "低长调": "暗为主、有亮点，强烈有戏剧性（暗调情绪 / 伦勃朗光）",
    "低中调": "暗为主、反差适中，沉稳含蓄",
    "低短调": "暗而平、几乎没有亮部，压抑神秘",
}

ZONES = 10
EDGE = 0.02  # 两端各去掉 2% 的零星像素（高光点、死黑）再量跨度
LONG_SPAN = 170.0  # 跨度（明度级）≥ 此值为长调
SHORT_SPAN = 90.0  # 跨度 < 此值为短调
NEAR = 12.0  # 跨度离分界不到这么多级时提示“接近 …”
LUMA = np.array([0.30, 0.59, 0.11])  # PS“明度”直方图的权重


def luminance(rgb: np.ndarray) -> np.ndarray:
    """sRGB 8 位像素 -> PS 明度（0–255，浮点）。"""
    return np.asarray(rgb, dtype=float)[..., :3] @ LUMA


@dataclass(frozen=True)
class ToneClass:
    name: str  # 例如 "低长调"
    key: str  # 高 / 中 / 低
    length: str  # 长 / 中 / 短
    shares: tuple[float, float, float]  # 暗部 / 中间调 / 亮部 各占多少
    lo: float  # 去掉 2% 后最暗处（明度 0–255）
    hi: float  # 去掉 2% 后最亮处
    span: float  # 跨度（明度级，0–255）
    histogram: tuple[float, ...]  # 32 段直方图（占比），用于显示
    near: str = ""  # 处在边界上时，相邻的那种影调（例如“接近低中调”）

    @property
    def desc(self) -> str:
        return TONE_DESC[self.name]


def classify(rgb: np.ndarray, valid: np.ndarray | None = None) -> ToneClass:
    """画面（或 valid 范围内）的影调。"""
    y = luminance(rgb)
    if valid is not None:
        y = y[valid]
    y = y.ravel()
    if y.size == 0:
        raise ValueError("没有像素")
    zones = np.minimum((y // (256 / ZONES)).astype(int), ZONES - 1)
    counts = np.bincount(zones, minlength=ZONES) / y.size
    shares = (float(counts[:4].sum()), float(counts[4:7].sum()), float(counts[7:].sum()))  # 暗 / 中 / 亮
    key = ("低", "中", "高")[int(np.argmax(shares))]
    lo, hi = (float(v) for v in np.percentile(y, [EDGE * 100, 100 - EDGE * 100]))
    span = hi - lo
    length = "长" if span >= LONG_SPAN else ("中" if span >= SHORT_SPAN else "短")
    # 边界提示：主导的一段只比第二多一点，或跨度离分界很近
    order = sorted(range(3), key=lambda i: -shares[i])
    near = ""
    if shares[order[0]] - shares[order[1]] < 0.08:
        near = ("低", "中", "高")[order[1]] + length + "调"
    elif abs(span - LONG_SPAN) < NEAR:
        near = key + ("中" if length == "长" else "长") + "调"
    elif abs(span - SHORT_SPAN) < NEAR:
        near = key + ("短" if length == "中" else "中") + "调"
    hist = np.histogram(y, bins=32, range=(0, 256))[0] / y.size
    name = f"{key}{length}调"
    return ToneClass(name, key, length, tuple(round(s, 4) for s in shares), round(lo, 1), round(hi, 1), round(span, 1),
                     tuple(round(float(v), 4) for v in hist), near if near != name else "")


# ---------------------------------------------------------------- 目标影调的“标准直方图”（分位点）
# 每种影调的代表性分位点（明度 0–255）：P2 / P25 / P50 / P75 / P98。往目标影调调整时按分位点做直方图匹配。
_CENTER = {"高": 200.0, "中": 128.0, "低": 55.0}
_SPAN = {"长": 215.0, "中": 140.0, "短": 75.0}


def target_quantiles(name: str) -> tuple[float, float, float, float, float]:
    key, length = name[0], name[1]
    c, s = _CENTER[key], _SPAN[length]
    if key == "低":  # 暗部为主：从接近黑开始往上延伸
        p2 = 8.0 if length == "长" else (14.0 if length == "中" else 20.0)
        p98 = min(250.0, p2 + s)
        p50 = c
    elif key == "高":  # 亮部为主：从接近白往下延伸
        p98 = 248.0 if length == "长" else (240.0 if length == "中" else 232.0)
        p2 = max(5.0, p98 - s)
        p50 = c
    else:
        p2, p98, p50 = c - s / 2, c + s / 2, c
    p25 = p2 + (p50 - p2) * 0.6
    p75 = p50 + (p98 - p50) * 0.4
    return (round(p2, 1), round(p25, 1), round(p50, 1), round(p75, 1), round(p98, 1))


def quantiles(rgb: np.ndarray, valid: np.ndarray | None = None) -> tuple[float, float, float, float, float]:
    y = luminance(rgb)
    y = (y[valid] if valid is not None else y).ravel()
    return tuple(round(float(v), 1) for v in np.percentile(y, [2, 25, 50, 75, 98]))


MAX_MOVE = 45.0  # 一次曲线每个点最多移动多少级（0–255），其余提示重新分析或回 RAW 调曝光


def match_curve(current: tuple, target: tuple) -> tuple[list[list[int]], bool]:
    """按分位点做直方图匹配：把当前的 P2/P25/P50/P75/P98 映射到目标的分位点，得到 RGB 复合曲线的点。

    返回 (曲线点（含 0,0 和 255,255），是否因为一次移动太多而被限幅)。保证单调。
    """
    pts = [[0, 0]]
    capped = False
    last_x, last_y = 0, 0
    for x, y in zip(current, target, strict=True):
        if y - x > MAX_MOVE or x - y > MAX_MOVE:
            capped = True
            y = x + np.clip(y - x, -MAX_MOVE, MAX_MOVE)
        xi, yi = int(round(x)), int(round(y))
        if xi <= last_x + 3 or xi >= 252:  # 太近的点合并，避免曲线打折
            continue
        yi = max(yi, last_y + 1)
        pts.append([xi, min(yi, 254)])
        last_x, last_y = xi, pts[-1][1]
    pts.append([255, 255])
    return pts, capped
