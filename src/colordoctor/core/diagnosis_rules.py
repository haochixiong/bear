"""可解释的“脏色风险”诊断规则。

每个风险分项都给出 0–100 分、严重程度、数值证据和适用性；总分为
“最高分项 + 其余适用分项之和 × 0.2”（上限 100），界面始终同时展示各分项。
所有阈值集中在本文件顶部，便于用真实样片校准。
"""

from __future__ import annotations

import numpy as np

from colordoctor.core import color_spaces as cs
from colordoctor.core.color_models import (
    ColorContext,
    Comparison,
    ContaminationResult,
    RegionStats,
    RiskComponent,
    SkinReport,
    severity_from_score,
)

# ---- 阈值（经验值，可校准）
OPP_CURVE = ([0, 0.005, 0.01, 0.03, 0.05, 0.08, 0.15, 0.30], [0, 15, 25, 45, 58, 72, 85, 95])
OPP_VISIBLE_CHROMA = 15.0
SPREAD_CURVE = ([10, 20, 30, 45], [0, 35, 60, 85])
GRAY_CURVE = ([0, 0.03, 0.10, 0.20, 0.40], [0, 15, 40, 60, 80])
SAT_DEFICIT_CURVE = ([0, 0.10, 0.25, 0.50], [0, 25, 50, 75])
CAST_CURVE = ([6, 10, 20, 30], [0, 30, 60, 80])
SKIN_CAST_CURVE = ([0, 4, 10, 20], [0, 25, 50, 75])
DELTA_C_CURVE = ([0, 2, 5, 10], [0, 20, 50, 80])
OUTLIER_CURVE = ([0, 0.01, 0.03, 0.05], [0, 10, 25, 40])
LIGHTNESS_RANGE_MIN = 12.0

NAMES = {
    "opposite": "对立色混入风险",
    "hue_dispersion": "色相离散风险",
    "chroma_compression": "色度压缩（发灰）风险",
    "local_cast": "局部偏色风险",
    "lightness_compression": "明度压缩风险",
    "outlier": "异常色比例",
    "skin_cast": "肤色整体偏色",
    "uniformity": "肤色色相/饱和度不均匀",
}


def _interp(x, curve):
    return float(np.interp(x, curve[0], curve[1]))


def _rc(key, score, evidence, applicable=True, note=None) -> RiskComponent:
    score = float(np.clip(score, 0, 100))
    return RiskComponent(
        key=key,
        name=NAMES[key],
        score=round(score, 1),
        severity=severity_from_score(score if applicable else 0),
        evidence=tuple(evidence),
        applicable=applicable,
        note=note,
    )


# ------------------------------------------------------------------ 分项
def opposite_component(ct: ContaminationResult, skin: bool = False) -> RiskComponent:
    if ct.base_hue is None and ct.opposite_fraction == 0:
        return _rc("opposite", 0, ["区域基本为无彩色（中性灰），不评估对立色"], note="无彩色区域")
    if ct.bicolor:
        return _rc(
            "opposite",
            0,
            [f"主色“{ct.base_hue_name}”只占有彩像素的 {ct.base_share:.1%}，另一种对立色占比同样很大"],
            note="两种颜色都占较大比例，判断为配色或不同物体，不算污染",
        )
    f = ct.opposite_fraction
    if f <= 0:
        return _rc(
            "opposite",
            0,
            [
                f"主色“{ct.base_hue_name}”（h {ct.base_hue:.0f}°，C* {ct.base_chroma:.1f}）"
                f"占有彩像素 {ct.base_share:.1%}，未发现对立色像素（阈值 C* ≥ {ct.opposite_threshold:.1f}）"
            ],
        )
    cf = float(np.clip(ct.opposite_chroma / OPP_VISIBLE_CHROMA, 0.4, 1.0))
    score = _interp(f, OPP_CURVE) * cf
    if skin and ct.anomaly_fractions:
        names = "、".join(f"{n} {v:.1%}" for n, v in ct.anomaly_fractions if v >= 0.0005)
        what = f"异常色像素：{names}"
    else:
        what = f"“{ct.opposite_hue_name}”像素占 {f:.1%}"
    ev = [
        f"主色“{ct.base_hue_name}”：h {ct.base_hue:.0f}°，C* {ct.base_chroma:.1f}，占有彩像素 {ct.base_share:.1%}",
        f"{what}（平均 C* {ct.opposite_chroma:.1f}，h {ct.opposite_hue:.0f}°）",
        f"两组色相相差 {abs(cs.hue_diff(ct.base_hue, ct.opposite_hue)):.0f}°",
        f"分布：{'散点混杂' if ct.interspersion >= 2 else '成片/集中'}（混杂度 {ct.interspersion:.1f}）",
        f"判定阈值：与主色相差 ≥ {60 if skin else 90}° 且 C* ≥ {ct.opposite_threshold:.1f}（随噪声自适应）",
    ]
    note = None
    if ct.noise_like:
        score = min(score, 30.0)
        note = "异常像素零散、色度接近阈值，更像噪点，结论降级"
    elif cf < 1:
        note = f"混入的颜色本身色度较低（C* {ct.opposite_chroma:.1f}），视觉上表现为“发脏/发灰”而不是明显杂色"
    return _rc("opposite", score, ev, note=note)


def hue_dispersion_component(ct: ContaminationResult) -> RiskComponent:
    if ct.base_hue is None or ct.chromatic_fraction < 0.3 or ct.base_chroma < 8 or ct.hue_spread is None:
        return _rc(
            "hue_dispersion",
            0,
            [f"有彩像素 {ct.chromatic_fraction:.0%}，主色色度 C* {ct.base_chroma:.1f}"],
            applicable=False,
            note="色度太低时色相本身不稳定，不评估色相离散",
        )
    s = ct.hue_spread
    return _rc("hue_dispersion", _interp(s, SPREAD_CURVE), [f"去除对立色后，色相圆形标准差 {s:.1f}°（≤10° 视为集中）"])


def chroma_component_region(ct: ContaminationResult) -> RiskComponent:
    g = ct.graying_fraction
    if g <= 0:
        return _rc(
            "chroma_compression",
            0,
            ["同色区域内部没有发现明显更灰的位置"],
            note="整片统一的低饱和不算问题（只比较同色区域内部的差异）",
        )
    score = _interp(g, GRAY_CURVE) * float(np.clip(ct.graying_depth / 0.5, 0.5, 1.0))
    return _rc(
        "chroma_compression",
        score,
        [
            f"局部发灰像素占同色区域 {g:.1%}",
            f"这些位置的色度比同明度的干净处低约 {ct.graying_depth:.0%}",
        ],
    )


def chroma_component_skin(ct: ContaminationResult, skin: SkinReport) -> RiskComponent:
    base = chroma_component_region(ct)
    lo, hi = skin.expected_saturation
    s = skin.saturation_ab
    ev = list(base.evidence)
    score = base.score
    ev.append(f"肤色饱和度 s = C*/L* = {s:.2f}（目标范围 {lo:.2f}–{hi:.2f}）")
    if s < lo:
        deficit = (lo - s) / lo
        score = max(score, _interp(deficit, SAT_DEFICIT_CURVE))
        ev.append(f"低于目标下限 {deficit:.0%}：肤色发灰、不够透亮")
    return _rc("chroma_compression", score, ev)


def chroma_component_compare(c: Comparison) -> RiskComponent:
    if c.dC >= 0:
        return _rc("chroma_compression", 0, [f"问题区色度 C* 变化 {c.dC:+.1f}（未降低）"])
    return _rc(
        "chroma_compression",
        _interp(-c.dC, DELTA_C_CURVE),
        [f"问题区色度 C* 比参考区低 {abs(c.dC):.1f}（ΔC*）"],
    )


def local_cast_skin(skin: SkinReport) -> RiskComponent:
    hs, ss = skin.highlight_hue_shift, skin.shadow_hue_shift
    x = max(abs(hs), abs(ss))
    ev = [
        f"高光相对中间调：色相 {hs:+.1f}°，Δa* {skin.highlight_shift[1]:+.1f}，Δb* {skin.highlight_shift[2]:+.1f}",
        f"阴影相对中间调：色相 {ss:+.1f}°，Δa* {skin.shadow_shift[1]:+.1f}，Δb* {skin.shadow_shift[2]:+.1f}",
    ]
    return _rc("local_cast", _interp(x, CAST_CURVE), ev)


def local_cast_bands(bands: dict) -> RiskComponent:
    mid = bands.get("mid")
    if mid is None:
        return _rc("local_cast", 0, ["像素太少，无法分段比较"], applicable=False)
    m_lch = cs.lab_to_lch(mid)
    ev, worst = [], 0.0
    for key, label in (("highlight", "高光"), ("shadow", "阴影")):
        band = bands.get(key)
        if band is None:
            continue
        b_lch = cs.lab_to_lch(band)
        if b_lch[1] < 8 or m_lch[1] < 8:
            continue
        d = float(cs.hue_diff(m_lch[2], b_lch[2]))
        worst = max(worst, abs(d))
        ev.append(f"{label}相对中间调色相 {d:+.1f}°（{label}偏{cs.hue_name(b_lch[2])}）")
    if not ev:
        return _rc("local_cast", 0, ["色度太低，无法比较分段色相"], applicable=False)
    return _rc("local_cast", _interp(worst, CAST_CURVE), ev)


def local_cast_compare(c: Comparison) -> RiskComponent:
    x = abs(c.dh) if np.hypot(c.da, c.db) >= 2 else 0.0
    return _rc(
        "local_cast",
        _interp(x * 1.0, CAST_CURVE) if abs(c.across_ref_hue) >= 2 else 0.0,
        [f"色相角差 Δh {c.dh:+.1f}°，垂直于参考色相方向的偏移 {c.across_ref_hue:+.1f}"],
    )


def lightness_component(stats: RegionStats, chroma_score: float) -> RiskComponent:
    rng = stats.L_p90 - stats.L_p10
    ev = [f"明度 L* 的 P10/P50/P90 = {stats.L_p10:.1f} / {stats.L_p50:.1f} / {stats.L_p90:.1f}（范围 {rng:.1f}）"]
    if rng < LIGHTNESS_RANGE_MIN and chroma_score >= 20:
        return _rc("lightness_compression", 25, ev + ["明度与色度同时被压缩，容易显得发闷"])
    return _rc(
        "lightness_compression", 0, ev, note="明度分布集中或偏暗本身不代表问题（暗调≠脏），只作辅助证据"
    )


def outlier_component(stats: RegionStats, ct: ContaminationResult) -> RiskComponent:
    if not stats.clusters:
        return _rc("outlier", 0, ["无聚类数据"], applicable=False)
    main = np.array(stats.clusters[0].lab)
    small = [c for c in stats.clusters[1:] if c.share < 0.05 and float(cs.delta_e_2000(main, np.array(c.lab))) > 15]
    f = sum(c.share for c in small)
    f_extra = max(0.0, f - ct.opposite_fraction)
    ev = [f"与主色差异很大（ΔE00>15）的小簇共占 {f:.1%}"] + [
        f"{c.hue_name} {c.share:.1%}（Lab {c.lab[0]:.0f}, {c.lab[1]:.0f}, {c.lab[2]:.0f}）" for c in small[:4]
    ]
    return _rc("outlier", _interp(f_extra, OUTLIER_CURVE), ev, note="已扣除对立色分项中计入的像素")


def skin_cast_component(skin: SkinReport) -> RiskComponent:
    if skin.tendency == "正常":
        return _rc("skin_cast", 0, [f"肤色色相角 h {skin.mean_lch[2]:.1f}°，在自然肤色区间内"])
    return _rc(
        "skin_cast",
        _interp(skin.tendency_amount, SKIN_CAST_CURVE),
        [
            f"肤色平均 Lab = ({skin.mean_lab[0]:.1f}, {skin.mean_lab[1]:.1f}, {skin.mean_lab[2]:.1f})，h {skin.mean_lch[2]:.1f}°",
            f"总体{skin.tendency}，超出自然肤色色相区间约 {skin.tendency_amount:.1f}°",
        ],
        note="倾向是相对“自然肤色”而言；如果是刻意的风格（例如粉白、暖金），请对照风格预设判断",
    )


def uniformity_component(uni) -> RiskComponent:
    if uni is None or uni.skin_pixels < 20:
        return _rc("uniformity", 0, ["皮肤像素太少，无法评估均匀度"], applicable=False)
    score = 100.0 - uni.score
    return _rc(
        "uniformity",
        score,
        [
            f"均匀度 {uni.score:.0f}/100；各小块色相标准差 {uni.hue_spread:.1f}°，饱和度变异 {uni.sat_cv:.0%}",
            f"色相偏离 ≥6° 的皮肤 {uni.uneven_hue_share:.0%}，饱和度偏离 ≥22% 的皮肤 {uni.uneven_sat_share:.0%}",
        ],
    )


# ------------------------------------------------------------------ 汇总
def overall(components: list[RiskComponent]) -> float:
    scores = sorted((c.score for c in components if c.applicable), reverse=True)
    if not scores:
        return 0.0
    return round(min(100.0, scores[0] + 0.2 * sum(scores[1:])), 1)


def confidence(
    stats: RegionStats,
    context: ColorContext,
    ct: ContaminationResult | None = None,
    skin: SkinReport | None = None,
) -> tuple[float, list[str]]:
    c = 1.0
    reasons = []
    n = stats.sampled_pixels
    if n < 25:
        c *= 0.5
        reasons.append(f"选区太小（{n} 像素）")
    elif n < 400:
        c *= 0.75
        reasons.append(f"选区较小（{n} 像素）")
    base_c = ct.base_chroma if ct is not None else stats.C_p50
    if base_c < 5:
        c *= 0.6
        reasons.append("颜色接近中性，色相不可靠")
    elif base_c < 8:
        c *= 0.8
        reasons.append("色度较低，色相判断偏不稳定")
    if stats.clipped_fraction > 0.2:
        c *= 0.75
        reasons.append(f"{stats.clipped_fraction:.0%} 像素被裁切（纯黑/纯白/溢色）")
    if context.assumed_srgb:
        c *= 0.85
        reasons.append("未找到 ICC，按 sRGB 假定")
    if context.source == "screen":
        if not context.managed_source:
            c *= 0.9
            reasons.append("屏幕取样（未做色彩管理的软件）")
        else:
            c *= 0.95
            reasons.append("屏幕取样（受显示器配置文件精度和缩放影响）")
    if skin is not None and skin.mixed_region:
        c *= 0.65
        reasons.append("选区混入了非皮肤区域")
    if ct is not None and ct.noise_like:
        c *= 0.85
        reasons.append("异常像素更像噪点")
    return float(np.clip(c, 0.15, 0.97)), reasons
