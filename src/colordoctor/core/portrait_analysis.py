"""人像肤色分析（用户手动框选皮肤，不做自动人脸识别）。

判断依据以 Lab 色相角 h° 为主：各种深浅的肤色，色相角大多落在同一个红-黄区间，
而明度 L* 和色度 C* 随肤色深浅变化很大，所以“偏红/偏黄/偏绿”用色相判断，
“发灰”用与明度无关的饱和度 s = C*/L* 判断，避免对深色或浅色肤色产生偏见。
"""

from __future__ import annotations

import math

import numpy as np

from colordoctor.core import color_spaces as cs
from colordoctor.core import contamination as ct
from colordoctor.core.color_models import Comparison, QuickRuleResult, SkinPreset, SkinReport
from colordoctor.core.region_statistics import weighted_circular_hue
from colordoctor.core.sampling import SampleError, grid_downsample

# 相对“自然肤色”的色相区间（D50 Lab），超出即给出倾向
NATURAL_HUE = (40.0, 68.0)
MAGENTA_HUE = 25.0
GREEN_HUE = 80.0
GREEN_A = 2.0
SKIN_HUE_RANGE = (0.0, 110.0)
MIXED_SHARE = 0.75
NATURAL_SATURATION = (18.0 / 74.0, 30.0 / 62.0)


def ita_angle(L: float, b: float) -> float:
    """Individual Typology Angle（Chardon 1991）：ITA = atan((L*-50)/b*)。"""
    return math.degrees(math.atan2(L - 50.0, b))


def ita_label(ita: float) -> str:
    if ita > 55:
        return "很浅"
    if ita > 41:
        return "浅"
    if ita > 28:
        return "中等"
    if ita > 10:
        return "小麦/棕褐"
    if ita > -30:
        return "棕"
    return "深"


def tendency_of(a: float, b: float, hue_range: tuple[float, float] = NATURAL_HUE) -> tuple[str, float]:
    """相对自然肤色（或所选风格的色相区间）的总体倾向与超出量（度，≥ 0：到区间最近一端的色相差）。

    色相按顺时针连续处理：0° 两侧不再跳变（h = 359° 与 h = 1° 的超出量相近）。
    a* < 0 时皮肤已经没有红色成分：b* ≥ 0 视为偏绿，b* < 0 视为偏青/偏蓝。
    """
    if math.hypot(a, b) < 4.0:
        return "正常", 0.0  # 几乎没有颜色（灰 / 黑白）：色相只是噪声，发灰另有判断
    h = math.degrees(math.atan2(b, a)) % 360.0
    lo, hi = hue_range
    if a < 0:
        return ("偏绿" if b >= 0 else "偏青/偏蓝"), float(abs(((h - hi + 180) % 360) - 180) if b >= 0 else abs(((h - lo + 180) % 360) - 180))
    if h > 180:  # a ≥ 0、b < 0：红偏紫
        return "偏洋红", float(lo + (360 - h))
    if h > GREEN_HUE or a <= GREEN_A:
        return "偏绿", float(max(0.0, h - hi))
    if h > hi:
        return "偏黄", float(h - hi)
    if h < min(MAGENTA_HUE, lo):
        return "偏洋红", float(lo - h)
    if h < lo:
        return "偏红", float(lo - h)
    return "正常", 0.0


def _expected_saturation(preset: SkinPreset | None, L: float) -> tuple[float, float]:
    """肤色饱和度（C*/L*）的目标范围。

    选了具体风格时按该风格；没选或选“自然通用”时，按明度找所有覆盖这个明度的内置风格，取最宽的范围——
    浅肤色天然 C*/L* 较低（日系、韩系、cos 都在 0.14–0.22），不能因此被判为“发灰”。
    """
    if preset is not None and preset.key != "natural":
        return (preset.chroma[0] / preset.L[1], preset.chroma[1] / preset.L[0])
    from colordoctor.core.skin_presets import builtin_presets

    presets = builtin_presets()
    lo_L = min(p.L[0] for p in presets) - 3
    hi_L = max(p.L[1] for p in presets) + 3
    L = min(max(L, lo_L), hi_L)  # 超出所有风格明度范围的极亮 / 极暗皮肤：按最接近的风格判断，避免下限突然跳变
    cover = [p for p in presets if p.L[0] - 3 <= L <= p.L[1] + 3]
    if not cover:
        return NATURAL_SATURATION
    lo = min(p.chroma[0] / p.L[1] for p in cover)
    hi = max(p.chroma[1] / p.L[0] for p in cover)
    return (min(lo, NATURAL_SATURATION[0]), max(hi, NATURAL_SATURATION[1]))


def _band_mean(lab, sel):
    return lab[sel].mean(axis=0) if sel.any() else None


def analyze_skin(
    region: np.ndarray, preset: SkinPreset | None = None, max_samples: int = 262_144, valid: np.ndarray | None = None
) -> SkinReport:
    if region is None or region.ndim != 3 or region.shape[0] == 0 or region.shape[1] == 0:
        raise SampleError("选区为空")
    sub, step = grid_downsample(region, max_samples)
    vsub = ct.downsample_mask(valid, step, sub.shape[:2]).reshape(-1)
    lab = cs.srgb8_to_lab(sub).reshape(-1, 3)[vsub]
    if lab.shape[0] == 0:
        raise SampleError("蒙版内没有皮肤像素")
    lch = cs.lab_to_lch(lab)
    L, C, h = lch[:, 0], lch[:, 1], lch[:, 2]
    skin = (C >= 3.0) & (h >= SKIN_HUE_RANGE[0]) & (h <= SKIN_HUE_RANGE[1]) & (L >= 10) & (L <= 98)
    share = float(skin.mean())
    mixed = share < MIXED_SHARE
    if skin.sum() < 1:
        skin = np.ones_like(skin)

    sl = lab[skin]
    mean = sl.mean(axis=0)
    mean_lch = cs.lab_to_lch(mean)
    lab65 = cs.srgb8_to_lab(cs.lab_to_srgb8(mean), white="D65")
    ita = ita_angle(float(lab65[0]), float(lab65[2]))
    custom = preset is not None and preset.key != "natural"
    hue_range = tuple(preset.hue) if custom else NATURAL_HUE
    tendency, amount = tendency_of(float(mean[1]), float(mean[2]), hue_range)

    Ls = sl[:, 0]
    p15, p40, p60, p85 = np.percentile(Ls, [15, 40, 60, 85])
    hi = _band_mean(sl, Ls >= p85)
    lo = _band_mean(sl, Ls <= p15)
    mid = _band_mean(sl, (Ls >= p40) & (Ls <= p60))
    if mid is None:
        mid = mean

    def shift(band):
        if band is None:
            return (0.0, 0.0, 0.0), 0.0
        d = tuple(round(float(v), 3) for v in band - mid)
        hb = cs.lab_to_lch(band)
        hm = cs.lab_to_lch(mid)
        if hb[1] < 4 or hm[1] < 4:
            return d, 0.0
        return d, round(float(cs.hue_diff(hm[2], hb[2])), 3)

    hl_shift, hl_hue = shift(hi)
    sh_shift, sh_hue = shift(lo)

    exp_sat = _expected_saturation(preset, float(mean[0]))
    contam = ct.analyze(region, mode="skin", max_samples=max_samples, valid=valid)
    return SkinReport(
        mean_lab=tuple(round(float(v), 3) for v in mean),
        mean_lch=tuple(round(float(v), 3) for v in mean_lch),
        ita=round(ita, 2),
        depth_label=ita_label(ita),
        tendency=tendency,
        tendency_amount=round(amount, 2),
        highlight_shift=hl_shift,
        shadow_shift=sh_shift,
        highlight_hue_shift=hl_hue,
        shadow_hue_shift=sh_hue,
        saturation_ab=round(float(mean_lch[1] / max(mean_lch[0], 1e-6)), 4),
        expected_saturation=(round(exp_sat[0], 4), round(exp_sat[1], 4)),
        mixed_region=bool(mixed),
        anomaly_fractions=dict(contam.anomaly_fractions),
        hue_range=hue_range,
        hue_ref=f"「{preset.name}」" if custom else "自然肤色",
    )


def quick_rule_check(lab_pixels: np.ndarray, rule: tuple) -> QuickRuleResult:
    """你的经验规则：a*、b* 都在区间内，且 |a* - b*| 不超过上限。"""

    (alo, ahi), (blo, bhi), md = rule
    px = np.asarray(lab_pixels, dtype=float).reshape(-1, 3)
    a, b = float(px[:, 1].mean()), float(px[:, 2].mean())
    a_ok, b_ok, d_ok = alo <= a <= ahi, blo <= b <= bhi, abs(a - b) <= md
    ok = (px[:, 1] >= alo) & (px[:, 1] <= ahi) & (px[:, 2] >= blo) & (px[:, 2] <= bhi) & (np.abs(px[:, 1] - px[:, 2]) <= md)
    tips = []
    if a < alo:
        tips.append(f"a* {a:.1f} 偏低（缺红、没血色或偏绿）：色调往洋红、红色饱和度略加")
    if a > ahi:
        tips.append(f"a* {a:.1f} 偏高（太红）：色调往绿、红色/橙色色相往黄")
    if b < blo:
        tips.append(f"b* {b:.1f} 偏低（缺黄、偏冷发青）：色温往黄、橙色色相往黄")
    if b > bhi:
        tips.append(f"b* {b:.1f} 偏高（太黄）：色温往蓝、黄色/橙色饱和度降低或色相往红")
    if not d_ok:
        tips.append(f"a* 与 b* 相差 {abs(a - b):.1f}（上限 {md:g}）：{'偏黄' if b > a else '偏红/偏粉'}")
    verdict = "正常肤色" if (a_ok and b_ok and d_ok) else "不符合经验范围"
    return QuickRuleResult(
        a=round(a, 2), b=round(b, 2), a_range=(alo, ahi), b_range=(blo, bhi), max_diff=md,
        a_ok=a_ok, b_ok=b_ok, diff_ok=d_ok, pixel_pass_share=round(float(ok.mean()), 4),
        verdict=verdict, tips=tuple(tips),
    )


def compare_lab(from_lab, to_lab, opposite_change: float = 0.0) -> Comparison:
    """from -> to 的差异（to 减 from）。"""
    f = np.asarray(from_lab, dtype=float)
    t = np.asarray(to_lab, dtype=float)
    lf, lt = cs.lab_to_lch(f), cs.lab_to_lch(t)
    dE = float(cs.delta_e_2000(f, t))
    dL = float(t[0] - f[0])
    dC = float(lt[1] - lf[1])
    da, db = float(t[1] - f[1]), float(t[2] - f[2])
    dH2 = max(0.0, da * da + db * db - dC * dC)
    dh = float(cs.hue_diff(lf[2], lt[2])) if lf[1] >= 1 and lt[1] >= 1 else 0.0
    dH = math.copysign(math.sqrt(dH2), dh) if dh else 0.0
    th = math.radians(lf[2])
    along = da * math.cos(th) + db * math.sin(th)
    across = -da * math.sin(th) + db * math.cos(th)
    r = lambda v: round(float(v), 3)  # noqa: E731
    return Comparison(
        ref_lab=tuple(r(v) for v in f),
        target_lab=tuple(r(v) for v in t),
        delta_e00=r(dE),
        dL=r(dL),
        dC=r(dC),
        dH=r(dH),
        dh=r(dh),
        da=r(da),
        db=r(db),
        along_ref_hue=r(along),
        across_ref_hue=r(across),
        opposite_component_change=r(opposite_change),
    )


def compare_to_preset(report: SkinReport, preset: SkinPreset) -> Comparison:
    """当前肤色 -> 目标风格中心值。"""
    return compare_lab(report.mean_lab, preset.target_lab)


def preset_fit(report: SkinReport, preset: SkinPreset) -> dict[str, str]:
    """逐项判断当前肤色是否落在预设区间内。"""
    L, a, b = report.mean_lab
    _, C, h = report.mean_lch
    out = {}
    for key, val, (lo, hi), unit in (
        ("明度 L*", L, preset.L, ""),
        ("a*", a, preset.a, ""),
        ("b*", b, preset.b, ""),
        ("色度 C*", C, preset.chroma, ""),
        ("色相角 h°", h, preset.hue, "°"),
    ):
        if key.startswith("色相"):
            # 色相是环形的。区间可能写成 (-5, 10)，也可能写成 lo > hi 的 (350, 20)，两种都表示跨过 0°
            span = (hi - lo) % 360 or 360.0
            center = (lo + span / 2.0) % 360
            d = float(cs.hue_diff(center, val % 360))
            shown = val % 360
            if abs(d) <= span / 2.0:
                out[key] = f"在范围内（{shown:.1f}°）"
            else:
                out[key] = f"{'偏低' if d < 0 else '偏高'}（{shown:.1f}°，目标 {lo:g}–{hi:g}°）"
            continue
        if val < lo:
            out[key] = f"偏低（{val:.1f}{unit}，目标 {lo:g}–{hi:g}{unit}）"
        elif val > hi:
            out[key] = f"偏高（{val:.1f}{unit}，目标 {lo:g}–{hi:g}{unit}）"
        else:
            out[key] = f"在范围内（{val:.1f}{unit}）"
    return out


def mean_hue_of(region_lab: np.ndarray) -> float | None:
    lch = cs.lab_to_lch(region_lab.reshape(-1, 3))
    sel = lch[:, 1] >= 5
    mean, _, _ = weighted_circular_hue(lch[sel, 2], lch[sel, 1])
    return mean
