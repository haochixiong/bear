"""Photoshop 调整的数值模型，以及“把当前颜色移向目标颜色”的参数求解。

精度说明（会在界面上同样标注）：
- **曲线**：对取样平均色是精确的（每个通道一个“输入→输出”点）。
- **色彩平衡**：采用 GIMP 公开的色彩平衡算法（其设计目标即模仿 Photoshop），
  与 Photoshop 的实际结果存在差异，数值为估算。
- **色相/饱和度**：HSL 模型 + Photoshop 默认的“红色/黄色”范围及过渡带，估算。
- **可选颜色**：按通道“最大值-中间值”/“中间值-最小值”的颜色量近似，估算。

所有估算都应在 Photoshop 中应用后重新取样确认（本工具会显示剩余差异）。
函数输入输出均为 sRGB 0–1 浮点数组，形状 (N, 3)。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from colordoctor.core import color_spaces as cs


# ------------------------------------------------------------------ HSL 工具
def _rgb_to_hsl(x: np.ndarray):
    mx = x.max(axis=-1)
    mn = x.min(axis=-1)
    light = (mx + mn) / 2
    d = mx - mn
    denom = np.where(light <= 0.5, mx + mn, 2 - mx - mn)
    s = np.where(d == 0, 0.0, d / np.where(denom == 0, 1, denom))
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    safe = np.where(d == 0, 1, d)
    h = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4))
    h = np.where(d == 0, 0.0, h / 6.0) % 1.0
    return h, s, light


def _hsl_to_rgb(h, s, light):
    q = np.where(light < 0.5, light * (1 + s), light + s - light * s)
    p = 2 * light - q

    def f(t):
        t = t % 1.0
        return np.where(
            t < 1 / 6, p + (q - p) * 6 * t, np.where(t < 0.5, q, np.where(t < 2 / 3, p + (q - p) * (2 / 3 - t) * 6, p))
        )

    return np.stack([f(h + 1 / 3), f(h), f(h - 1 / 3)], axis=-1)


# ------------------------------------------------------------------ 色彩平衡
def _cb_weights(lightness):
    """GIMP 色彩平衡：阴影 / 中间调 / 高光 三个色调对某个 HSL 明度的作用权重。"""
    a, b = 0.25, 0.333
    sh = np.clip((lightness - b) / -a + 0.5, 0, 1)
    md = np.clip((lightness - b) / a + 0.5, 0, 1) * np.clip((lightness + b - 1) / -a + 0.5, 0, 1)
    hl = np.clip((lightness + b - 1) / a + 0.5, 0, 1)
    return sh, md, hl


def _cb_map(value, lightness, shadows, midtones, highlights):
    sh, md, hl = _cb_weights(lightness)
    return np.clip(value + (shadows * sh + midtones * md + highlights * hl) * 0.7, 0, 1)


def color_balance(
    x: np.ndarray,
    shadows=(0, 0, 0),
    midtones=(0, 0, 0),
    highlights=(0, 0, 0),
    preserve_luminosity: bool = True,
) -> np.ndarray:
    """每组三个值依次为：青-红、洋红-绿、黄-蓝（-100..100，正值为红/绿/蓝）。"""
    x = np.asarray(x, dtype=float)
    _, _, light = _rgb_to_hsl(x)
    s = np.asarray(shadows, float) / 100
    m = np.asarray(midtones, float) / 100
    hgh = np.asarray(highlights, float) / 100
    out = np.stack([_cb_map(x[..., i], light, s[i], m[i], hgh[i]) for i in range(3)], axis=-1)
    if preserve_luminosity:
        h2, s2, _ = _rgb_to_hsl(out)
        out = _hsl_to_rgb(h2, s2, light)
    return out


# ------------------------------------------------------------------ 色相/饱和度
# Photoshop 默认范围（HSB 色相，度）：(过渡起点, 核心起点, 核心终点, 过渡终点)
PS_RANGES = {
    "reds": (315.0, 345.0, 15.0, 45.0),
    "yellows": (15.0, 45.0, 75.0, 105.0),
    "greens": (75.0, 105.0, 135.0, 165.0),
    "cyans": (135.0, 165.0, 195.0, 225.0),
    "blues": (195.0, 225.0, 255.0, 285.0),
    "magentas": (255.0, 285.0, 315.0, 345.0),
}
RANGE_NAMES = {"reds": "红色", "yellows": "黄色", "greens": "绿色", "cyans": "青色", "blues": "蓝色", "magentas": "洋红"}
SKIN_RANGES = ("reds", "yellows")


def _range_weight(hue_deg: np.ndarray, rng: str) -> np.ndarray:
    if rng == "master":
        return np.ones_like(hue_deg)
    a, b, c, d = PS_RANGES[rng]
    h = (hue_deg - a) % 360.0
    b_, c_, d_ = (b - a) % 360, (c - a) % 360, (d - a) % 360
    return np.where(h < b_, h / b_, np.where(h <= c_, 1.0, np.where(h < d_, (d_ - h) / (d_ - c_), 0.0)))


def hue_saturation(x: np.ndarray, hue: float = 0, saturation: float = 0, lightness: float = 0, color_range: str = "master"):
    """hue: -180..180；saturation/lightness: -100..100。"""
    x = np.asarray(x, dtype=float)
    h, s, light = _rgb_to_hsl(x)
    h2 = (h + hue / 360.0) % 1.0
    s2 = np.clip(s * (1 + saturation / 100.0), 0, 1)
    v = lightness / 100.0
    l2 = light * (1 + v) if v < 0 else light + v * (1 - light)
    out = _hsl_to_rgb(h2, s2, l2)
    chroma = x.max(axis=-1) - x.min(axis=-1)
    w = _range_weight(h * 360.0, color_range) * (chroma > 1e-9 if color_range != "master" else 1.0)
    return x + (out - x) * w[..., None]


# ------------------------------------------------------------------ 可选颜色
_SEL_CLASS = {  # 颜色类别 -> (判定条件, 颜色量)
    "reds": lambda r, g, b, mx, md, mn: np.where(r == mx, mx - md, 0.0),
    "yellows": lambda r, g, b, mx, md, mn: np.where(b == mn, md - mn, 0.0),
    "greens": lambda r, g, b, mx, md, mn: np.where(g == mx, mx - md, 0.0),
    "cyans": lambda r, g, b, mx, md, mn: np.where(r == mn, md - mn, 0.0),
    "blues": lambda r, g, b, mx, md, mn: np.where(b == mx, mx - md, 0.0),
    "magentas": lambda r, g, b, mx, md, mn: np.where(g == mn, md - mn, 0.0),
}


def selective_color(
    x: np.ndarray, color: str, cyan: float = 0, magenta: float = 0, yellow: float = 0, black: float = 0, relative: bool = True
):
    """近似模型：青/洋红/黄分别作用于 R/G/B 通道，作用量与该颜色类别的“颜色量”成正比。"""
    x = np.asarray(x, dtype=float)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    srt = np.sort(x, axis=-1)
    mn, md, mx = srt[..., 0], srt[..., 1], srt[..., 2]
    amount = _SEL_CLASS[color](r, g, b, mx, md, mn)
    out = x.copy()
    for i, s in enumerate((cyan, magenta, yellow)):
        s = s / 100.0
        ink = 1 - x[..., i]
        delta = -s * amount * (ink if relative else 1.0)
        out[..., i] = x[..., i] + delta
    k = black / 100.0
    if k:
        out = out - k * amount[..., None] * (out if relative else 1.0)
    return np.clip(out, 0, 1)


# ------------------------------------------------------------------ 曲线
CHANNELS = ("红", "绿", "蓝")


def curves_points(mean_rgb8, target_rgb8) -> dict[str, tuple[int, int]]:
    return {CHANNELS[i]: (int(mean_rgb8[i]), int(target_rgb8[i])) for i in range(3)}


def apply_curves(x: np.ndarray, points: dict[str, tuple[int, int]]) -> np.ndarray:
    """每通道一个控制点（加端点 0、255）的单调分段三次插值（PCHIP）。"""
    x = np.asarray(x, dtype=float)
    out = x.copy()
    for i, ch in enumerate(CHANNELS):
        if ch not in points:
            continue
        xi, yi = points[ch]
        xs = np.array([0.0, min(max(xi, 1), 254) / 255.0, 1.0])
        ys = np.array([0.0, min(max(yi, 1), 254) / 255.0, 1.0])
        out[..., i] = _pchip(xs, ys, x[..., i])
    return out


def _pchip(xs, ys, x):
    h = np.diff(xs)
    delta = np.diff(ys) / h
    d = np.zeros_like(ys)
    for k in range(1, len(xs) - 1):
        if delta[k - 1] * delta[k] > 0:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    d[0] = delta[0]
    d[-1] = delta[-1]
    idx = np.clip(np.searchsorted(xs, x) - 1, 0, len(h) - 1)
    t = (x - xs[idx]) / h[idx]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * ys[idx] + h10 * h[idx] * d[idx] + h01 * ys[idx + 1] + h11 * h[idx] * d[idx + 1]


# ------------------------------------------------------------------ 求解
@dataclass(frozen=True)
class Solution:
    tool: str
    params: tuple[tuple[str, float], ...]
    start_de: float
    residual_de: float
    result_lab: tuple[float, float, float]
    extra: dict


TOOL_SPECS = {
    # 名称: (参数名列表, 每个参数范围, 正则化权重)
    # 色彩平衡的代价面很平：三个滑块互相抵消也能凑出差不多的结果，正则化要足够强
    "color_balance": (("青-红", "洋红-绿", "黄-蓝"), (-60, 60), 0.02),
    "hue_saturation": (("色相", "饱和度", "明度"), (-40, 40), 0.006),  # 明度另限 ±15
    "selective_color": (("青色", "洋红", "黄色", "黑色"), (-50, 50), 0.01),
}
CB_TONES = {"shadows": "阴影", "midtones": "中间调", "highlights": "高光"}
CB_SPARSE_TOL = 0.4  # 少用一个滑块、剩余色差只多这么多（ΔE00）时，用更简单的组合


def cb_tone(x: np.ndarray) -> str:
    """色彩平衡选哪个色调：按 GIMP 模型里三个色调对这些像素的平均权重取最大（浅肤色多半是高光）。"""
    _, _, light = _rgb_to_hsl(np.asarray(x, dtype=float))
    w = [float(v.mean()) for v in _cb_weights(light)]
    return list(CB_TONES)[int(np.argmax(w))]


def _cb_signs(cur_lab: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    """色彩平衡三个滑块允许的方向（0 = 不限）：任何一个滑块都不能和需要的 Δa*/Δb* 方向相反。"""
    da, db = float(target[1] - cur_lab[1]), float(target[2] - cur_lab[2])
    sa = int(np.sign(da)) if abs(da) >= 1 else 0
    sb = int(np.sign(db)) if abs(db) >= 1 else 0
    return sa, -sa, -sb  # 青-红：+红 = a* 增；洋红-绿：+绿 = a* 减；黄-蓝：+蓝 = b* 减


def _mean_lab(x: np.ndarray) -> np.ndarray:
    return cs.srgb8_to_lab(np.clip(x, 0, 1) * 255.0).mean(axis=0)


def _range_for(x: np.ndarray, candidates=SKIN_RANGES) -> str:
    """在候选的 PS 颜色范围里选覆盖这些像素最多的一个（肤色只在红色 / 黄色之间选）。"""
    h, _, _ = _rgb_to_hsl(x)
    weights = {r: float(_range_weight(h * 360.0, r).mean()) for r in candidates}
    return max(candidates, key=lambda r: weights[r])


def _apply(tool: str, x: np.ndarray, p: np.ndarray, rng: str) -> np.ndarray:
    if tool == "color_balance":  # rng = 色调（阴影 / 中间调 / 高光）
        return color_balance(x, **{rng: tuple(p)}, preserve_luminosity=True)
    if tool == "hue_saturation":
        return hue_saturation(x, hue=p[0], saturation=p[1], lightness=p[2], color_range=rng)
    if tool == "selective_color":
        return selective_color(x, rng, cyan=p[0], magenta=p[1], yellow=p[2], black=p[3], relative=True)
    raise ValueError(tool)


def solve(pixels: np.ndarray, target_lab, tool: str, max_pixels: int = 3000, ranges=SKIN_RANGES) -> Solution:
    """在模型下用确定性坐标搜索寻找整数参数，使平均色尽量接近 target_lab（ΔE00）。

    色彩平衡：自动选作用最强的色调；滑块方向不与需要的 Δa*/Δb* 相反；
    一两个滑块就能达到差不多效果时不用三个（三个滑块互相抵消的组合在 PS 里不可信）。
    """
    x = np.asarray(pixels, dtype=float).reshape(-1, 3)
    if x.shape[0] > max_pixels:
        step = int(np.ceil(x.shape[0] / max_pixels))
        x = x[step // 2 :: step]
    target = np.asarray(target_lab, dtype=float)
    names, (lo, hi), reg = TOOL_SPECS[tool]
    cb = tool == "color_balance"
    rng = cb_tone(x) if cb else _range_for(x, ranges)
    start_lab = _mean_lab(x)
    signs = _cb_signs(start_lab, target) if cb else (0,) * len(names)

    def cost(p):
        lab = _mean_lab(_apply(tool, x, p, rng))
        return float(cs.delta_e_2000(lab, target)) + reg * float(np.abs(p).sum()), lab

    def search(active):
        p = np.zeros(len(names))
        best, best_lab = cost(p)
        for step in (16, 8, 4, 2, 1):
            improved = True
            while improved:
                improved = False
                for i in active:
                    for sgn in (1, -1):
                        q = p.copy()
                        q[i] = np.clip(q[i] + sgn * step, lo, hi)
                        if tool == "hue_saturation" and i == 2:
                            q[i] = np.clip(q[i], -15, 15)
                        if signs[i] and q[i] * signs[i] < 0:
                            continue
                        c, lab = cost(q)
                        if c < best - 1e-9:
                            p, best, best_lab = q, c, lab
                            improved = True
        return p, best, best_lab

    p, best, best_lab = search(range(len(names)))
    if cb:
        full = float(cs.delta_e_2000(best_lab, target))
        for k in (1, 2):
            ok = [r for r in (search(s) for s in combinations(range(len(names)), k))
                  if float(cs.delta_e_2000(r[2], target)) <= full + CB_SPARSE_TOL]
            if ok:
                p, best, best_lab = min(ok, key=lambda r: r[1])
                break
    residual = float(cs.delta_e_2000(best_lab, target))
    prefix = f"{CB_TONES[rng]} " if cb else ""
    return Solution(
        tool=tool,
        params=tuple((prefix + n, float(v)) for n, v in zip(names, p)),
        start_de=round(float(cs.delta_e_2000(start_lab, target)), 3),
        residual_de=round(residual, 3),
        result_lab=tuple(round(float(v), 3) for v in best_lab),
        extra={"range": "master", "tone": rng} if cb else {"range": rng},
    )
