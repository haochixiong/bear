"""仿色分析：参考作品是什么色彩？我的图差多少？适不适合照着调？

逐项比较（全部可解释）：
1. 影调：L* 的 P5/P25/P50/P75/P95 → 给出把我的影调拉向参考的 RGB 曲线点；
2. 整体偏色：接近中性像素的平均 a*/b*（白平衡倾向）→ 色温/色调方向；
3. 阴影/高光色调：→ 颜色分级（估算）与曲线点（对阴影/高光平均色精确）；
4. 色彩丰富度；
5. 混色器 8 个通道：参考的色相/饱和度/亮度 vs 我的 → 滑块方向和估算值；
6. 主色：参考的主要颜色在我的图里有没有对应；
7. 肤色（可选）：两边都给了皮肤平均 Lab 时比较。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colordoctor.core import acr_channels as ac
from colordoctor.core import color_spaces as cs
from colordoctor.core import palette as pl
from colordoctor.core.color_models import Adjustment

PCTS = (5, 25, 50, 75, 95)
NEUTRAL_C = 12.0


@dataclass(frozen=True)
class ChannelDiff:
    name: str
    ref_share: float
    my_share: float
    hue_slider: int
    sat_slider: int
    lum_slider: int


@dataclass(frozen=True)
class MatchReport:
    similarity: float
    tone_ref: tuple[float, ...]
    tone_mine: tuple[float, ...]
    curve_points: tuple[tuple[int, int], ...]  # RGB 复合曲线 (输入, 输出)
    cast_ref: tuple[float, float] | None  # 中性区域平均 (a*, b*)；中性像素太少时为 None（无法判断）
    cast_mine: tuple[float, float] | None
    colorfulness_ref: float
    colorfulness_mine: float
    channels: tuple[ChannelDiff, ...]
    palette_ref: tuple[pl.PaletteColor, ...]
    palette_match: tuple[tuple[str, float, str, float], ...]  # (参考主色, 占比, 我的对应色, ΔE00)
    skin_delta: tuple[float, float, float, float] | None  # (ΔE00, ΔL, Δa, Δb)：参考 - 我的
    fit: tuple[str, ...]  # 适不适合
    gaps: tuple[str, ...]
    adjustments: tuple[Adjustment, ...]


def _neutral_cast(lab: np.ndarray, sel: np.ndarray | None = None) -> tuple[float, float] | None:
    """接近中性的像素的平均色调。太少（< 20 个）时返回 None：画面里没有可参考的中性色，
    不能当成“没有偏色”，否则会按错误的起点给出色温 / 颜色分级建议。"""
    C = np.hypot(lab[:, 1], lab[:, 2])
    m = C < NEUTRAL_C
    if sel is not None:
        m &= sel
    if m.sum() < 20:
        return None
    return (float(lab[m, 1].mean()), float(lab[m, 2].mean()))


def _gray(L: float) -> int:
    return int(cs.lab_to_srgb8(np.array([L, 0.0, 0.0]))[0])


def match_reference(mine: np.ndarray, ref: np.ndarray, skin_mine=None, skin_ref=None) -> MatchReport:
    s_m, _, lab_m, lch_m = pl._prep(mine)
    s_r, _, lab_r, lch_r = pl._prep(ref)
    lm, lr = lab_m.reshape(-1, 3), lab_r.reshape(-1, 3)
    tm = tuple(float(v) for v in np.percentile(lm[:, 0], PCTS))
    tr = tuple(float(v) for v in np.percentile(lr[:, 0], PCTS))
    pts: list[tuple[int, int]] = []
    for a, b in zip(tm, tr, strict=True):
        x, y = _gray(a), _gray(b)
        if abs(x - y) < 2:
            continue
        # 曲线点之间至少相隔 12 级、输出单调不降，否则曲线过陡会出现断层（PS 也可能合并太近的点）
        if pts and (x - pts[-1][0] < 12 or y < pts[-1][1]):
            continue
        pts.append((x, y))
    curve = tuple(pts)
    cast_m, cast_r = _neutral_cast(lm), _neutral_cast(lr)
    cfo_m, cfo_r = pl.colorfulness(s_m, lch_m), pl.colorfulness(s_r, lch_r)
    cf_m, cf_r = cfo_m.value, cfo_r.value
    # 参考是黑白作品、我的明显是彩色：不按通道逐个降饱和，直接建议转黑白（色调/颜色分级仍照常比较，适用于棕调等单色调）
    ref_bw = pl.is_monochrome(cfo_r) and not pl.is_monochrome(cfo_m) and cf_m >= pl.COLORFULNESS_LEVELS[0][0]

    # 通道
    ch_m = {c.name: c for c in ac.channel_stats(s_m)}
    ch_r = {c.name: c for c in ac.channel_stats(s_r)}
    diffs = []
    for name in () if ref_bw else ac.NAMES:
        a, b = ch_m[name], ch_r[name]
        if b.share < 0.03 and a.share < 0.03:
            continue
        if b.share < 0.01 <= a.share:  # 参考里几乎没有这种颜色 -> 降饱和（单条建议上限，先降到这里再对比）
            diffs.append(ChannelDiff(name, b.share, a.share, 0, -ac.SLIDER_CAP, 0))
            continue
        if a.mean_hue is None or b.mean_hue is None:
            continue
        hs = ac.hue_slider(a.mean_hue, b.mean_hue, channel=a.name)
        ss = ac.sat_slider(a.mean_sat, b.mean_sat, channel=a.name)
        ls = ac.lum_slider((b.mean_lab[0] - a.mean_lab[0]) if (a.mean_lab and b.mean_lab) else 0.0, channel=a.name)
        diffs.append(ChannelDiff(name, b.share, a.share, hs, ss, ls))

    # 主色对应
    pal_r, _ = pl.ratio_631(lab_r)
    pal_m, _ = pl.ratio_631(lab_m)
    matches = []
    pal_de = 0.0
    for p in pal_r[:5]:
        best = min(pal_m, key=lambda q: float(cs.delta_e_2000(np.array(p.lab), np.array(q.lab))))
        de = float(cs.delta_e_2000(np.array(p.lab), np.array(best.lab)))
        matches.append((p.family, p.share, best.family, round(de, 1)))
        pal_de += p.share * de
    pal_de /= max(1e-6, sum(p.share for p in pal_r[:5]))

    skin_delta = None
    if skin_mine is not None and skin_ref is not None:
        sm, sr = np.array(skin_mine, float), np.array(skin_ref, float)
        skin_delta = (round(float(cs.delta_e_2000(sm, sr)), 2), *(round(float(v), 2) for v in sr - sm))

    tone_d = float(np.mean(np.abs(np.array(tr) - np.array(tm))))
    both_cast = cast_r is not None and cast_m is not None
    cast_d = float(np.hypot(cast_r[0] - cast_m[0], cast_r[1] - cast_m[1])) if both_cast else 0.0
    sim = 100 - 2.0 * pal_de - 1.2 * tone_d - 0.4 * abs(cf_r - cf_m) - 1.5 * cast_d
    if skin_delta is not None:
        sim -= 1.5 * skin_delta[0]
    sim = float(np.clip(sim, 0, 100))

    gaps = [
        f"影调（L* 中位）参考 {tr[2]:.0f} / 我的 {tm[2]:.0f}；暗部 P5 {tr[0]:.0f} / {tm[0]:.0f}；亮部 P95 {tr[4]:.0f} / {tm[4]:.0f}",
        (f"中性区域偏色：参考 a* {cast_r[0]:+.1f}、b* {cast_r[1]:+.1f}；我的 a* {cast_m[0]:+.1f}、b* {cast_m[1]:+.1f}" if both_cast
         else "中性区域偏色：有一张图几乎没有接近中性的颜色，无法比较白平衡"),
        f"色彩丰富度：参考 {cf_r:.0f} / 我的 {cf_m:.0f}",
        f"主色平均差异 ΔE00 {pal_de:.1f}",
    ]
    if skin_delta is not None:
        gaps.append(f"肤色差异 ΔE00 {skin_delta[0]:.1f}（参考 - 我的：ΔL* {skin_delta[1]:+.1f}，Δa* {skin_delta[2]:+.1f}，Δb* {skin_delta[3]:+.1f}）")

    fit = []
    if abs(tr[2] - tm[2]) > 20:
        fit.append(f"影调差异很大（中位明度相差 {abs(tr[2] - tm[2]):.0f}）：参考是{'高调' if tr[2] > tm[2] else '低调'}，"
                   "只靠调色很难还原，需要从曝光/光线入手，否则容易发灰或死黑")
    if ref_bw:
        fit.append("参考是黑白作品：把你的图转成黑白后，主要比较影调（曲线）即可")
    for fam, share, mine_fam, de in () if ref_bw else matches:
        if share >= 0.15 and de > 20:
            fit.append(f"参考的主色“{fam}”（占 {share:.0%}）在你的画面里没有接近的颜色（最近的是“{mine_fam}”，ΔE00 {de:.0f}）："
                       "只能靠大幅偏色制造，效果可能不自然")
    if cf_r - cf_m > 25:
        fit.append("参考明显更鲜艳：强行提高饱和度容易出现色块和溢色，注意肤色")
    if skin_delta is not None and skin_delta[0] > 10:
        fit.append("肤色差异很大：建议先单独把肤色调到参考附近，再统一环境色")
    if not fit:
        fit.append("参考与你的画面在影调和主色上比较接近，适合照着调")

    adjs: list[Adjustment] = []
    if curve:
        adjs.append(Adjustment("曲线", "RGB 复合曲线（把影调拉向参考）：" + "；".join(f"输入 {a} → 输出 {b}" for a, b in curve), "小幅", False,
                               tuple((f"输入 {a}", float(b)) for a, b in curve)))
    if ref_bw:
        adjs.insert(0, Adjustment("Camera Raw", "参考是黑白作品：先把你的图转黑白（Camera Raw 处理方式选「黑白」，或饱和度 -100），"
                                  "再用曲线对齐影调；各颜色转成多亮的灰用「黑白混合」调", None, False))
    da, db = (cast_r[0] - cast_m[0], cast_r[1] - cast_m[1]) if both_cast else (0.0, 0.0)
    if abs(db) >= 1.5:
        adjs.append(Adjustment("Camera Raw", f"色温往{'黄' if db > 0 else '蓝'}（中性区域 b* 需要 {db:+.1f}）", "小幅"))
    if abs(da) >= 1.5:
        adjs.append(Adjustment("Camera Raw", f"色调往{'洋红' if da > 0 else '绿'}（中性区域 a* 需要 {da:+.1f}）", "小幅"))
    for label, sel_fn in (("阴影", lambda L, p: L <= p[0]), ("高光", lambda L, p: L >= p[1])):
        pm = np.percentile(lm[:, 0], [15, 85])
        pr = np.percentile(lr[:, 0], [15, 85])
        tm_ = _neutral_cast(lm, sel_fn(lm[:, 0], pm))
        tr_ = _neutral_cast(lr, sel_fn(lr[:, 0], pr))
        if tm_ is None or tr_ is None:
            continue  # 这一段没有可比较的中性色
        if np.hypot(tr_[0] - tm_[0], tr_[1] - tm_[1]) >= 2:
            gh, gs = ac.tint_to_grading(60.0, tr_[0] - tm_[0], tr_[1] - tm_[1])  # 要加的是差值
            adjs.append(Adjustment("Camera Raw 颜色分级", f"{label}：色相 {gh}，饱和度 {gs}（参考{label}色调 a* {tr_[0]:+.1f}、b* {tr_[1]:+.1f}，我的 a* {tm_[0]:+.1f}、b* {tm_[1]:+.1f}）", "小幅",
                                   True, (("色相", gh), ("饱和度", gs))))
    for d in diffs:
        parts = []
        (hs, hn), (ss, sn), (ls, ln) = (ac.cap_slider(v) for v in (d.hue_slider, d.sat_slider, d.lum_slider))
        if abs(hs) >= 8:
            parts.append(f"色相 {hs:+d}{hn}")
        if abs(ss) >= 10:
            parts.append(f"饱和度 {ss:+d}{sn}")
        if abs(ls) >= 10:
            parts.append(f"明亮度 {ls:+d}{ln}")
        if parts:
            adjs.append(Adjustment("Camera Raw 混色器", f"{d.name}（参考占 {d.ref_share:.0%}，我的占 {d.my_share:.0%}）：" + "，".join(parts),
                                   ac.slider_strength(max(abs(hs), abs(ss), abs(ls))), True,
                                   (("色相", hs), ("饱和度", ss), ("明亮度", ls))))
    adjs.append(Adjustment("提示", "建议顺序：先曲线定影调 → 再色温/色调 → 混色器 → 颜色分级；每一步后重新对比", None, False))
    return MatchReport(
        similarity=round(sim, 1),
        tone_ref=tuple(round(v, 1) for v in tr),
        tone_mine=tuple(round(v, 1) for v in tm),
        curve_points=curve,
        cast_ref=None if cast_r is None else tuple(round(v, 2) for v in cast_r),
        cast_mine=None if cast_m is None else tuple(round(v, 2) for v in cast_m),
        colorfulness_ref=round(cf_r, 1),
        colorfulness_mine=round(cf_m, 1),
        channels=tuple(diffs),
        palette_ref=tuple(pal_r[:5]),
        palette_match=tuple(matches),
        skin_delta=skin_delta,
        fit=tuple(fit),
        gaps=tuple(gaps),
        adjustments=tuple(adjs),
    )
