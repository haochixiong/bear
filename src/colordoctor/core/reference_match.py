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
from colordoctor.core import native_layers as nl
from colordoctor.core import palette as pl
from colordoctor.core import ps_models as pm_mod
from colordoctor.core import tone_class as tcl
from colordoctor.core.color_models import Adjustment
from colordoctor.core.perception import de_words
from colordoctor.core.pixel_table import together

PCTS = (5, 25, 50, 75, 95)
NEUTRAL_C = 12.0
MIN_MY_SHARE = 0.01  # 我的图里某个混色器通道的颜色不到 1%：不给这个通道的色相 / 饱和度 / 明亮度建议


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


def _neutral(lab: np.ndarray, sel: np.ndarray | None = None) -> np.ndarray:
    """接近中性的颜色：色度上限随明度收紧（很暗的深蓝、深绿不算中性的黑）；sel 一般已去掉被裁切的纯白 / 死黑。"""
    m = np.hypot(lab[:, 1], lab[:, 2]) < cs.neutral_chroma_limit(lab[:, 0], NEUTRAL_C)
    return m & sel if sel is not None else m


def _neutral_lab(lab: np.ndarray, sel: np.ndarray | None = None, weights=None):
    """接近中性的像素的平均 Lab（含明度）；太少（< 20 个像素）时为 None。
    lab 是“不同的颜色”时 weights 是每种颜色的像素数，结果和逐个像素算一样。"""
    m = _neutral(lab, sel)
    w = np.ones(len(lab)) if weights is None else np.asarray(weights, dtype=float)
    n = w[m].sum()
    if n < 20:
        return None
    return (lab[m] * w[m][:, None]).sum(axis=0) / n


def _neutral_cast(lab: np.ndarray, sel: np.ndarray | None = None, weights=None) -> tuple[float, float] | None:
    """接近中性的像素的平均色调。太少时返回 None：画面里没有可参考的中性色，
    不能当成“没有偏色”，否则会按错误的起点给出色温 / 颜色分级建议。"""
    m = _neutral_lab(lab, sel, weights)
    return None if m is None else (float(m[1]), float(m[2]))


def _huesat_rows(ch_m: dict, ch_r: dict) -> list[tuple]:
    """PS 色相/饱和度用的混色器行 (通道, HSB 中心, 色相度数, 饱和度, 明亮度, 我的占比)，取舍规则与 Camera Raw 混色器建议相同。"""
    rows = []
    for name in ac.NAMES:
        a, b = ch_m[name], ch_r[name]
        if b.share < 0.03 and a.share < 0.03:
            continue
        if b.share < 0.01 <= a.share:
            rows.append((name, a.mean_hue if a.mean_hue is not None else a.center, 0.0, -ac.SLIDER_CAP, 0, a.share))
            continue
        if a.mean_hue is None or b.mean_hue is None or a.share < MIN_MY_SHARE:
            continue
        hs = ac.cap_slider(ac.hue_slider(a.mean_hue, b.mean_hue, channel=name))[0]
        ss = ac.cap_slider(ac.sat_slider(a.mean_sat, b.mean_sat, channel=name))[0]
        ls = ac.cap_slider(ac.lum_slider((b.mean_lab[0] - a.mean_lab[0]) if (a.mean_lab and b.mean_lab) else 0.0, channel=name))[0]
        if abs(hs) >= 8 or abs(ss) >= 10 or abs(ls) >= 10:
            rows.append((name, a.mean_hue, float(cs.hue_diff(a.mean_hue, b.mean_hue)) if abs(hs) >= 8 else 0.0,
                         ss if abs(ss) >= 10 else 0, ls if abs(ls) >= 10 else 0, a.share))
    return rows


def _gray(L: float) -> int:
    return int(cs.lab_to_srgb8(np.array([L, 0.0, 0.0]))[0])


def match_reference(mine: np.ndarray, ref: np.ndarray, skin_mine=None, skin_ref=None) -> MatchReport:
    # 原图尺寸、每个像素都算：只看颜色分布，按“不同的颜色 × 像素数”加权，和逐个像素算完全一样
    (s_m, w_m, lm, lch_m), (s_r, w_r, lr, lch_r) = together(lambda: pl._colours(mine), lambda: pl._colours(ref))
    tm = tuple(float(v) for v in cs.weighted_percentile(lm[:, 0], w_m, np.array(PCTS)))
    tr = tuple(float(v) for v in cs.weighted_percentile(lr[:, 0], w_r, np.array(PCTS)))
    pts: list[tuple[int, int]] = []
    for a, b in zip(tm, tr, strict=True):
        x, y = _gray(a), _gray(b)
        if abs(x - y) < 2:
            y = x  # 这一段已经和参考一致：保留为不动的锚点，否则曲线会从旁边的点直接插值过来，把它也带着动
        # 曲线点之间至少相隔 12 级、输出单调不降，否则曲线过陡会出现断层（PS 也可能合并太近的点）
        if pts and (x - pts[-1][0] < 12 or y < pts[-1][1]):
            continue
        pts.append((x, y))
    if any(x != y for x, y in pts):
        # PS 的曲线是平滑样条：在会冲过头 / 往回走的段中间加辅助点，实际曲线才和这些点一致
        safe = tcl.spline_safe(nl.curve_points(pts))  # 端点 0 / 255 只留一个（点落在 0 或 255 上时不重复）
        curve = tuple((x, y) for x, y in safe if (x, y) not in ((0, 0), (255, 255)))
    else:
        curve = ()  # 全部一致：不给曲线
    # 建议顺序是先用曲线定影调、再调颜色：白平衡、阴影 / 高光色调、混色器都在“曲线之后”的图上和参考比。
    # 否则混色器的明亮度会把曲线已经提亮 / 压暗的部分再调一次，色彩平衡也会按曲线之前的明度去求
    if curve:
        lut = np.rint(tcl.spline_lut(nl.curve_points(curve))).astype(np.uint8)  # 按 PS 的平滑曲线算
        s_c = lut[s_m]  # 每种颜色过曲线以后的颜色（像素数不变）
        lc = np.asarray(cs.srgb8_to_lab(s_c), dtype=np.float64)
    else:
        s_c, lc = s_m, lm
    # 被裁切的纯白 / 死黑“中性”只是因为被裁切：不参与白平衡和阴影 / 高光色调的估计
    ok_m, ok_r = ~cs.clipped(s_c), ~cs.clipped(s_r)
    cast_m, cast_r = _neutral_cast(lc, ok_m, w_m), _neutral_cast(lr, ok_r, w_r)
    cfo_m, cfo_r = pl.colorfulness(s_m, lch_m, w_m), pl.colorfulness(s_r, lch_r, w_r)
    cf_m, cf_r = cfo_m.value, cfo_r.value
    # 参考是黑白作品、我的明显是彩色：不按通道逐个降饱和，直接建议转黑白（色调/颜色分级仍照常比较，适用于棕调等单色调）
    ref_bw = pl.is_monochrome(cfo_r) and not pl.is_monochrome(cfo_m) and cf_m >= pl.COLORFULNESS_LEVELS[0][0]

    # 通道
    ch_m = {c.name: c for c in ac.channel_stats(s_c, weights=w_m)}
    ch_r = {c.name: c for c in ac.channel_stats(s_r, weights=w_r)}
    diffs = []
    for name in () if ref_bw else ac.NAMES:
        a, b = ch_m[name], ch_r[name]
        if b.share < 0.03 and a.share < 0.03:
            continue
        if b.share < 0.01 <= a.share:  # 参考里几乎没有这种颜色 -> 降饱和（单条建议上限，先降到这里再对比）
            diffs.append(ChannelDiff(name, b.share, a.share, 0, -ac.SLIDER_CAP, 0))
            continue
        if a.mean_hue is None or b.mean_hue is None or a.share < MIN_MY_SHARE:
            continue  # 我的图里几乎没有这种颜色：没有可调的，也不能拿零星几个像素的色相去推整段范围
        hs = ac.hue_slider(a.mean_hue, b.mean_hue, channel=a.name)
        ss = ac.sat_slider(a.mean_sat, b.mean_sat, channel=a.name)
        ls = ac.lum_slider((b.mean_lab[0] - a.mean_lab[0]) if (a.mean_lab and b.mean_lab) else 0.0, channel=a.name)
        diffs.append(ChannelDiff(name, b.share, a.share, hs, ss, ls))

    # 主色对应
    (pal_r, _), (pal_m, _) = together(lambda: pl.ratio_631(lr, weights=w_r), lambda: pl.ratio_631(lm, weights=w_m))
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
        f"主色平均差异 ΔE00 {pal_de:.1f}（{de_words(pal_de)}）",
    ]
    if skin_delta is not None:
        gaps.append(f"肤色差异 ΔE00 {skin_delta[0]:.1f}（{de_words(skin_delta[0])}；参考 - 我的：ΔL* {skin_delta[1]:+.1f}，"
                    f"Δa* {skin_delta[2]:+.1f}，Δb* {skin_delta[3]:+.1f}）")

    fit = []
    if abs(tr[2] - tm[2]) > 20:
        fit.append(f"影调差异很大（中位明度相差 {abs(tr[2] - tm[2]):.0f}）：参考是{'高调' if tr[2] > tm[2] else '低调'}，"
                   "只靠调色很难还原，需要从曝光/光线入手，否则容易发灰或死黑")
    if ref_bw:
        fit.append("参考是黑白作品：把你的图转成黑白后，主要比较影调（曲线）即可")
    for fam, share, mine_fam, de in () if ref_bw else matches:
        if share >= 0.15 and de > 20:
            fit.append(f"参考的主色“{fam}”（占 {share:.0%}）在你的画面里没有接近的颜色（最近的是“{mine_fam}”，ΔE00 {de:.0f}，{de_words(de)}）："
                       "只能靠大幅偏色制造，效果可能不自然")
    if cf_r - cf_m > 25:
        fit.append("参考明显更鲜艳：强行提高饱和度容易出现色块和溢色，注意肤色")
    if skin_delta is not None and skin_delta[0] > 10:
        fit.append("肤色差异很大：建议先单独把肤色调到参考附近，再统一环境色")
    if not fit:
        fit.append("参考与你的画面在影调和主色上比较接近，适合照着调")

    adjs: list[Adjustment] = []
    if curve:
        adjs.append(Adjustment("曲线", "RGB 复合曲线（把影调拉向参考）：" + "；".join(f"输入 {a} → 输出 {b}" + ("（已一致，固定不动）" if a == b else "") for a, b in curve), "小幅", False,
                               tuple((f"输入 {a}", float(b)) for a, b in curve),
                               layer=nl.layer("curves", nl.curves({"composite": list(curve)}), "global", "仿色影调 · 曲线", "ref_curve")))
    if ref_bw:
        adjs.insert(0, Adjustment("Camera Raw", "参考是黑白作品：先把你的图转黑白（Camera Raw 处理方式选「黑白」，或饱和度 -100），"
                                  "再用曲线对齐影调；各颜色转成多亮的灰用「黑白混合」调", None, False))
        adjs.insert(1, nl.adjustment("hueSaturation", nl.huesat(0, None, sat=-100, master=True),
                                     "PS 做法：色相/饱和度 全图 饱和度 -100（转黑白；想控制各颜色转成多亮的灰，改用 PS「黑白」调整图层）",
                                     "global", "转黑白 · 色相/饱和度", strength=None))
    da, db = (cast_r[0] - cast_m[0], cast_r[1] - cast_m[1]) if both_cast else (0.0, 0.0)
    # PS 色彩平衡：中间调在“我的”中性色的实际平均色上求（白平衡）
    n_m = _neutral_lab(lc, ok_m, w_m)
    balance = {"midtones": nl.balance_for_shift(da, db, "midtones", lab=n_m)
               if max(abs(da), abs(db)) >= 1.5 and n_m is not None else [0, 0, 0]}
    if abs(db) >= 1.5:
        adjs.append(Adjustment("Camera Raw", f"色温往{'黄' if db > 0 else '蓝'}（中性区域 b* 需要 {db:+.1f}）", "小幅"))
    if abs(da) >= 1.5:
        adjs.append(Adjustment("Camera Raw", f"色调往{'洋红' if da > 0 else '绿'}（中性区域 a* 需要 {da:+.1f}）", "小幅"))
    # 阴影 / 高光：与“整图色彩”相同的分段（暗调画面最亮处是中灰、高调画面最暗处是浅灰，都不算），已去掉被裁切的像素
    bands_m, bands_r = pl.tone_bands(lc[:, 0], s_c, w_m), pl.tone_bands(lr[:, 0], s_r, w_r)
    for label, key in (("阴影", "shadow"), ("高光", "highlight")):
        if bands_m[key] is None or bands_r[key] is None:
            continue  # 有一张图几乎没有这一段
        tm_ = _neutral_cast(lc, bands_m[key], w_m)
        tr_ = _neutral_cast(lr, bands_r[key], w_r)
        if tm_ is None or tr_ is None:
            continue  # 这一段没有可比较的中性色
        if np.hypot(tr_[0] - tm_[0], tr_[1] - tm_[1]) >= 2:
            gh, gs = ac.tint_to_grading(60.0, tr_[0] - tm_[0], tr_[1] - tm_[1])  # 要加的是差值
            # PS 色彩平衡：阴影/高光只补中间调那一层在这一段上没修到的部分（按模型算中间调那一层对这一段实际改了多少）
            band = _neutral_lab(lc, bands_m[key], w_m)
            done = nl.balance_effect(band, "midtones", balance["midtones"]) if any(balance["midtones"]) else np.zeros(2)
            ra, rb = tr_[0] - tm_[0] - float(done[0]), tr_[1] - tm_[1] - float(done[1])
            tone = "shadows" if label == "阴影" else "highlights"
            if np.hypot(ra, rb) >= 2:
                lv = nl.balance_for_shift(ra, rb, tone, lab=band)
                if any(lv):
                    balance[tone] = lv
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
    has_balance = any(any(v) for v in balance.values())
    if has_balance:
        p = nl.balance(**balance)
        after_curve = "；按做完影调曲线以后的颜色算，放在曲线上方" if curve else ""
        adjs.append(nl.adjustment("colorBalance", p, f"PS 做法（代替上面的色温/色调和颜色分级{after_curve}），勾选“保留明度”：{nl.balance_text(p)}",
                                  "global", "仿色色调 · 色彩平衡"))
    # PS 色相/饱和度：按“色彩平衡那一层之后”的颜色和参考比（否则整体偏色会被色彩平衡和混色器各修一次）
    ch_after = ch_m
    if has_balance:
        x = s_c.astype(float) / 255.0
        after = np.clip(np.rint(pm_mod.color_balance(x, **{k: v for k, v in balance.items()}) * 255.0), 0, 255).astype(np.uint8)
        ch_after = {c.name: c for c in ac.channel_stats(after, weights=w_m)}
    rows = [] if ref_bw else _huesat_rows(ch_after, ch_r)
    for params, words in nl.mixer_rows_to_huesat(rows):
        done = "、".join(n for n, on in (("影调曲线", bool(curve)), ("色彩平衡", has_balance)) if on)
        note = f"；数值是做完{done}以后还差的部分，放在{'它' if '、' not in done else '它们'}上方" if done else ""
        adjs.append(nl.adjustment("hueSaturation", params, f"PS 做法（代替上面的 Camera Raw 混色器{note}）：{words}",
                                  "global", "仿色颜色 · 色相/饱和度"))
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
