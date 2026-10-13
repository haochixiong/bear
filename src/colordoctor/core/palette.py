"""整图色彩分析：色彩丰富度（缺色）、6:3:1 比例、某种颜色是否统一、与风格模板的差距。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources

import numpy as np

from colordoctor.core import acr_channels as ac
from colordoctor.core import clustering
from colordoctor.core import color_spaces as cs
from colordoctor.core.color_models import Adjustment
from colordoctor.core.contamination import _components
from colordoctor.core.region_statistics import weighted_circular_hue
from colordoctor.core.sampling import SampleError, grid_downsample

MAX_SAMPLES = 200_000
CHROMA_MIN = 8.0
ACCENT_L_GAP = 20.0  # 点缀色与主色、辅色的明度差至少这么大，才算“靠明度跳出来”

# Hasler & Süsstrunk (2003) 色彩丰富度分级
COLORFULNESS_LEVELS = (
    (15, "不鲜艳（接近灰）"),
    (33, "略有色彩"),
    (45, "中等"),
    (59, "较鲜艳"),
    (82, "很鲜艳"),
    (1e9, "极鲜艳"),
)


# ================================================================ 数据结构
@dataclass(frozen=True)
class Colorfulness:
    value: float  # Hasler-Süsstrunk M
    label: str
    mean_chroma: float  # 有彩像素平均 C*
    p90_chroma: float
    chromatic_share: float


@dataclass(frozen=True)
class PaletteColor:
    family: str
    share: float
    lab: tuple[float, float, float]
    rgb: tuple[int, int, int]
    chroma: float


@dataclass(frozen=True)
class RatioResult:
    roles: tuple[PaletteColor, ...]  # 主色、辅色、点缀色
    other_share: float
    score: float
    verdict: str
    tips: tuple[str, ...]


@dataclass(frozen=True)
class UnityPatch:
    bbox: tuple[int, int, int, int]
    hue_shift: float
    note: str


@dataclass(frozen=True)
class FamilyUnity:
    family: str
    share: float  # 该颜色占全图像素的比例
    mean_lab: tuple[float, float, float]
    target_lab: tuple[float, float, float] | None
    hue_spread: float  # 色相标准差（度）
    sat_cv: float
    lightness_std: float
    delta_e_to_target: float | None
    score: float  # 统一度 0–100
    channels: tuple[tuple[str, float, float], ...]  # (通道, 占该颜色的比例, 平均 HSB 色相)
    adjustments: tuple[Adjustment, ...]
    patches: tuple[UnityPatch, ...] = ()


@dataclass(frozen=True)
class SectorMatch:
    name: str
    hue_range: tuple[float, float]
    share: float  # 占有彩颜色量
    mean_hue: float | None
    hue_offset: float | None  # 与区段中心的差（度）


@dataclass(frozen=True)
class StyleMatch:
    key: str
    name: str
    score: float
    on_target: float
    off_target: tuple[tuple[str, float], ...]  # (色相名, 占比)
    sectors: tuple[SectorMatch, ...]
    colorfulness_ok: bool
    colorfulness_range: tuple[float, float]
    shadow_tint: tuple[float, float] | None  # (色相, 色度)
    highlight_tint: tuple[float, float] | None
    shadow_ok: bool | None
    highlight_ok: bool | None
    black_point: float
    gaps: tuple[str, ...]  # “差多少”的文字说明
    adjustments: tuple[Adjustment, ...]
    recipe: dict = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    contrast: float | None = None  # P95 − P5 明度差
    category: str = ""


@dataclass(frozen=True)
class PaletteReport:
    colorfulness: Colorfulness
    palette: tuple[PaletteColor, ...]
    ratio: RatioResult
    channels: tuple[ac.ChannelStat, ...]
    tone: tuple[float, float, float, float, float]  # L* 的 P5/P25/P50/P75/P95
    unity: FamilyUnity | None = None
    style: StyleMatch | None = None


# ================================================================ 基础
def _prep(image: np.ndarray, max_samples: int = MAX_SAMPLES):
    if image is None or image.ndim != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise SampleError("图像为空")
    sub, step = grid_downsample(image, max_samples)
    lab = cs.srgb8_to_lab(sub)
    return sub, step, lab, cs.lab_to_lch(lab)


def colorfulness(rgb: np.ndarray, lch: np.ndarray | None = None) -> Colorfulness:
    px = np.asarray(rgb, dtype=float).reshape(-1, 3)
    rg = px[:, 0] - px[:, 1]
    yb = 0.5 * (px[:, 0] + px[:, 1]) - px[:, 2]
    M = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    label = next(name for lim, name in COLORFULNESS_LEVELS if M < lim)
    if lch is None:
        lch = cs.lab_to_lch(cs.srgb8_to_lab(px))
    C = lch.reshape(-1, 3)[:, 1]
    chrom = C >= CHROMA_MIN
    return Colorfulness(
        value=round(M, 2),
        label=label,
        mean_chroma=round(float(C[chrom].mean()) if chrom.any() else 0.0, 2),
        p90_chroma=round(float(np.percentile(C, 90)), 2),
        chromatic_share=round(float(chrom.mean()), 4),
    )


MONO_CHROMATIC_SHARE = 0.01  # 有彩像素不到 1%：黑白（或几乎无色的单色调）作品


def is_monochrome(cf: Colorfulness) -> bool:
    return cf.chromatic_share < MONO_CHROMATIC_SHARE and cf.p90_chroma < CHROMA_MIN


def family_of(L: float, C: float, h: float) -> str:
    if C < CHROMA_MIN:
        if L < 30:
            return "黑/深灰"
        if L > 78:
            return "白/浅灰"
        return "灰"
    return cs.hue_name(h)


# ================================================================ 6:3:1
def ratio_631(lab: np.ndarray) -> tuple[tuple[PaletteColor, ...], RatioResult]:
    pts = lab.reshape(-1, 3)
    clusters = clustering.dominant_clusters(pts, max_k=8, merge_de=8.0)
    fam: dict[str, list] = {}
    for center, share in clusters:
        L, C, h = cs.lab_to_lch(center)
        fam.setdefault(family_of(L, C, h), []).append((center, share))
    palette = []
    for name, items in fam.items():
        share = sum(s for _, s in items)
        center = sum(c * s for c, s in items) / share
        palette.append(
            PaletteColor(
                family=name,
                share=round(float(share), 4),
                lab=tuple(round(float(v), 2) for v in center),
                rgb=tuple(int(v) for v in cs.lab_to_srgb8(center)),
                chroma=round(float(np.hypot(center[1], center[2])), 2),
            )
        )
    palette.sort(key=lambda p: -p.share)
    roles = tuple(palette[:3])
    shares = [p.share for p in roles] + [0.0] * (3 - len(roles))
    other = max(0.0, 1.0 - sum(shares))
    dev = abs(shares[0] - 0.6) + abs(shares[1] - 0.3) + abs(shares[2] - 0.1) + other
    score = float(np.clip(100 - 100 * dev, 0, 100))
    tips = []
    if shares[0] < 0.5:
        tips.append(f"主色只占 {shares[0]:.0%}（目标约 60%）：主色不够突出，可扩大主色面积或降低其他颜色的饱和度")
    if shares[0] > 0.75:
        tips.append(f"主色占 {shares[0]:.0%}：画面偏单调，可增加一个辅色（约 30%）")
    if len(roles) >= 3 and shares[2] > 0.2:
        tips.append(f"点缀色占 {shares[2]:.0%}（目标约 10%）：点缀色太多，会抢主体")
    if len(roles) >= 3 and any(p.chroma >= CHROMA_MIN for p in roles):
        acc = roles[2]
        l_gap = min(abs(acc.lab[0] - roles[0].lab[0]), abs(acc.lab[0] - roles[1].lab[0]))
        if acc.chroma <= max(roles[0].chroma, roles[1].chroma) and l_gap < ACCENT_L_GAP:
            tips.append(f"点缀色不够“跳”：点缀色的饱和度不高于主色/辅色，明度差也只有 {l_gap:.0f}；"
                        "点缀色的饱和度或明度对比应高于主色和辅色")
    if other > 0.1:
        rest = "、".join(p.family for p in palette[3:6])
        tips.append(f"其余颜色合计 {other:.0%}（{rest}）：颜色偏杂，可在混色器中把它们降饱和或色相并入主色/辅色")
    verdict = "接近 6:3:1" if score >= 75 else ("基本可用" if score >= 55 else "比例失衡")
    return tuple(palette), RatioResult(roles, round(other, 4), round(score, 1), verdict, tuple(tips))


# ================================================================ 颜色统一
FAMILY_SECTORS = {name: (lo, hi) for name, lo, hi in cs.HUE_SECTORS}


def family_unity(
    image: np.ndarray, family: str, target_lab=None, offset=(0, 0), max_samples: int = MAX_SAMPLES
) -> FamilyUnity:
    """检查画面中某一种颜色（例如所有的绿色）是否统一，并给出混色器建议。"""
    sub, step, lab, lch = _prep(image, max_samples)
    L, C, h = lch[..., 0], lch[..., 1], lch[..., 2]
    lo, hi = FAMILY_SECTORS[family]
    lo, hi = lo - 12, hi + 12
    hh = np.where(h < lo, h + 360, h)
    sel = (C >= CHROMA_MIN) & (hh >= lo) & (hh <= hi)
    n = int(sel.sum())
    if n < 20:
        return FamilyUnity(family, round(n / h.size, 4), (0, 0, 0), None, 0, 0, 0, None, 100.0, (), (
            Adjustment("提示", f"画面中几乎没有{family}", None, False),))
    hm, _, spread = weighted_circular_hue(h[sel], C[sel])
    s = C / np.maximum(L, 1)
    sat_cv = float(s[sel].std() / max(1e-6, s[sel].mean()))
    mean = lab[sel].mean(axis=0)
    target = np.array(target_lab, float) if target_lab is not None else None
    de = float(cs.delta_e_2000(mean, target)) if target is not None else None
    score = float(np.clip(100 - 3.5 * max(0.0, spread - 4) - 60 * max(0.0, sat_cv - 0.15), 0, 100))
    if de is not None:
        score = float(np.clip(score - 3 * max(0.0, de - 3), 0, 100))

    # 小块定位：哪里的这种颜色偏离了整体
    ts = int(max(8, round(min(sub.shape[:2]) / 12)))
    th, tw = (sub.shape[0] + ts - 1) // ts, (sub.shape[1] + ts - 1) // ts
    flags = np.zeros((th, tw), bool)
    shifts = np.zeros((th, tw))
    for ty in range(th):
        for tx in range(tw):
            sl = (slice(ty * ts, (ty + 1) * ts), slice(tx * ts, (tx + 1) * ts))
            v = sel[sl]
            if v.sum() < max(6, 0.25 * v.size):
                continue
            t_h, _, _ = weighted_circular_hue(h[sl][v], C[sl][v])
            d = float(cs.hue_diff(hm, t_h))
            shifts[ty, tx] = d
            flags[ty, tx] = abs(d) >= 8
    patches = []
    for comp in _components(flags):
        ys = [c[0] for c in comp]
        xs = [c[1] for c in comp]
        d = float(np.median([shifts[c] for c in comp]))
        patches.append(
            UnityPatch(
                bbox=(int(offset[0] + min(xs) * ts * step), int(offset[1] + min(ys) * ts * step),
                      int((max(xs) - min(xs) + 1) * ts * step), int((max(ys) - min(ys) + 1) * ts * step)),
                hue_shift=round(d, 1),
                note=f"这里的{family}偏{_toward_name(hm, d)}（色相 {d:+.0f}°）",
            )
        )

    # 混色器建议：涉及的通道往统一的目标色相收拢
    px = sub[sel]
    stats = [c for c in ac.channel_stats(px, primary=True) if c.share >= 0.08 and c.mean_hue is not None]
    goal_lab_h = float(cs.lab_to_lch(target)[2]) if target is not None else float(hm)
    goal = ac.lab_hue_to_hsb_hue(goal_lab_h)
    sat_goal = float(np.median([c.mean_sat for c in stats])) if stats else 0
    adjs = []
    for c in stats:
        hs, h_note = ac.cap_slider(ac.hue_slider(c.mean_hue, goal, channel=c.name))
        if hs * cs.hue_diff(ac.lab_hue_of(c.mean_lab), goal_lab_h) <= 0:
            hs, h_note = 0, ""  # 这一通道的颜色要往相反方向走（或已在目标上）：不给色相建议
        ss, s_note = ac.cap_slider(ac.sat_slider(c.mean_sat, sat_goal, channel=c.name) if len(stats) > 1 else 0)
        parts = []
        if abs(hs) >= 5:
            parts.append(f"色相 {hs:+d}{h_note}")
        if abs(ss) >= 8:
            parts.append(f"饱和度 {ss:+d}{s_note}")
        if parts:
            adjs.append(
                Adjustment(
                    "Camera Raw 混色器",
                    f"{c.name}（管着这种{family}的 {c.share:.0%}）：" + "，".join(parts) + f" → 收拢到同一个{family}",
                    ac.slider_strength(max(abs(hs), abs(ss))),
                    True,
                    (("色相", hs if abs(hs) >= 5 else 0), ("饱和度", ss if abs(ss) >= 8 else 0)),
                )
            )
    if target is not None and de is not None and de >= 3:
        cur = cs.lab_to_lch(mean)
        tl = cs.lab_to_lch(target)
        hs, h_note = ac.cap_slider(ac.hue_slider(ac.lab_hue_to_hsb_hue(cur[2]), ac.lab_hue_to_hsb_hue(tl[2])))
        ls, l_note = ac.cap_slider(ac.lum_slider(tl[0] - cur[0]))
        adjs.append(
            Adjustment(
                "Camera Raw 混色器",
                f"整体往目标{family}靠：色相 {hs:+d}{h_note}，明亮度 {ls:+d}{l_note}（当前与目标相差 ΔE00 {de:.1f}）",
                ac.slider_strength(max(abs(hs), abs(ls))),
                True,
            )
        )
    if not adjs:
        adjs.append(Adjustment("提示", f"画面中的{family}已经比较统一", None, False))
    adjs.append(Adjustment("提示", "混色器调整会影响整张照片中同一通道的颜色；只想改局部时用蒙版", None, False))
    chans = tuple((c.name, c.share, c.mean_hue) for c in stats)
    return FamilyUnity(
        family=family,
        share=round(n / h.size, 4),
        mean_lab=tuple(round(float(v), 2) for v in mean),
        target_lab=None if target is None else tuple(float(v) for v in target),
        hue_spread=round(float(spread or 0.0), 2),
        sat_cv=round(sat_cv, 3),
        lightness_std=round(float(L[sel].std()), 2),
        delta_e_to_target=None if de is None else round(de, 2),
        score=round(score, 1),
        channels=chans,
        adjustments=tuple(adjs),
        patches=tuple(patches[:8]),
    )


def _toward_name(h: float, d: float) -> str:
    """色相从 h 往 d 的方向偏：说成“偏青 / 偏黄”这样的相邻色，而不是“绿偏绿”。"""
    base, step = cs.hue_name(h), (1.0 if d >= 0 else -1.0)
    return next((n for n in (cs.hue_name(h + step * k) for k in range(1, 181)) if n != base), base)


# ================================================================ 风格模板
def load_styles() -> list[dict]:
    text = resources.files("colordoctor.data").joinpath("grading_styles.json").read_text(encoding="utf-8")
    return json.loads(text)["styles"]


def get_style(key: str) -> dict:
    for s in load_styles():
        if s["key"] == key:
            return s
    raise KeyError(key)


def _in_range(h: np.ndarray, lo: float, hi: float) -> np.ndarray:
    h = np.asarray(h) % 360
    return (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)


def _center(lo: float, hi: float) -> float:
    return ((lo + ((hi - lo) % 360) / 2.0)) % 360


# “接近中性”的像素才能代表调色加的色调：优先用 C* < 8 的；没有足够多时才放宽到 C* < 15
# （任何风格要的色调都不超过 15；淡蓝的天空 C* 约 20，是有颜色的物体，不能当成高光色调）
TINT_NEUTRAL_C = 8.0
TINT_MAX_C = 15.0
GRADING_SAT_MAX = 30  # 颜色分级饱和度建议上限（常用做法 10–25）
DESAT_SAT_FLOOR = 10.0  # 通道平均饱和度（HSB）已低于此值：不再建议降饱和


def _tint(lab: np.ndarray, sel: np.ndarray, min_share: float = 0.0, band=None) -> tuple[float, float, np.ndarray] | None:
    if sel.sum() < 10 or (band is not None and sel.sum() < min_share * max(1, band.sum())):
        return None
    m = lab[sel].mean(axis=0)
    return float(np.degrees(np.arctan2(m[2], m[1])) % 360), float(np.hypot(m[1], m[2])), m


def _band_tint(lab, C, band):
    return (_tint(lab, band & (C < TINT_NEUTRAL_C), min_share=0.05, band=band)
            or _tint(lab, band & (C < TINT_MAX_C), min_share=0.05, band=band))


def _tint_ok(t, spec) -> bool | None:
    if t is None or spec is None:
        return None
    h, c, _ = t
    lo, hi = spec["chroma"]
    hue_ok = spec["hue"] == [0, 360] or c < 1.5 or bool(_in_range(np.array([h]), *spec["hue"])[0])
    return bool(hue_ok and lo - 0.5 <= c <= hi + 0.5)


def style_match(image: np.ndarray, style: dict, max_samples: int = MAX_SAMPLES) -> StyleMatch:
    sub, _, lab, lch = _prep(image, max_samples)
    L, C, h = lch[..., 0].ravel(), lch[..., 1].ravel(), lch[..., 2].ravel()
    lab2 = lab.reshape(-1, 3)
    chrom = C >= CHROMA_MIN
    w = np.where(chrom, C, 0.0)
    tot = w.sum() or 1.0
    sectors = []
    on = np.zeros_like(chrom)
    for sec in style.get("sectors", []):
        lo, hi = sec["hue"]
        inside = chrom & _in_range(h, lo - 5, hi + 5)
        on |= inside
        share = float(w[inside].sum() / tot)
        mh, _, _ = weighted_circular_hue(h[inside], C[inside]) if inside.any() else (None, 0, None)
        sectors.append(
            SectorMatch(sec["name"], (lo, hi), round(share, 4), None if mh is None else round(mh, 1),
                        None if mh is None else round(float(cs.hue_diff(_center(lo, hi), mh)), 1))
        )
    has_sectors = bool(style.get("sectors"))
    on_target = float(w[on].sum() / tot) if has_sectors else 1.0
    off = []
    if has_sectors:
        names = np.array([cs.hue_name(v) for v in h[chrom & ~on]]) if (chrom & ~on).any() else np.array([])
        ww = w[chrom & ~on]
        for name in np.unique(names):
            share = float(ww[names == name].sum() / tot)
            if share >= 0.02:
                off.append((str(name), round(share, 4)))
        off.sort(key=lambda t: -t[1])
    cf = colorfulness(sub, lch)
    cr = tuple(style.get("colorfulness", [0, 200]))
    cf_ok = cr[0] <= cf.value <= cr[1]
    p5, p15, p50, p85, p95 = np.percentile(L, [5, 15, 50, 85, 95])
    contrast = float(p95 - p5)
    # 调色加在阴影/高光上的色调，主要体现在“接近中性”的像素上（灰墙、白衣、暗部），
    # 本身就很鲜艳的物体（绿叶、蓝天）不能代表色调。
    sh = _band_tint(lab2, C, L <= p15)
    hl = _band_tint(lab2, C, L >= p85)
    sh_ok = _tint_ok(sh, style.get("shadow_tint"))
    hl_ok = _tint_ok(hl, style.get("highlight_tint"))

    gaps = []
    if has_sectors:
        gaps.append(f"目标色区（{'、'.join(s['name'] for s in style['sectors'])}）占有彩颜色 {on_target:.0%}（理想 ≥ 85%）")
        for name, share in off[:4]:
            gaps.append(f"目标之外：{name} 占 {share:.0%}")
        for s in sectors:
            if s.hue_offset is not None and abs(s.hue_offset) >= 10 and s.share >= 0.05:
                gaps.append(f"“{s.name}”平均色相与目标中心差 {s.hue_offset:+.0f}°")
    gaps.append(f"色彩丰富度 {cf.value:.0f}（{cf.label}），目标 {cr[0]:g}–{cr[1]:g}{'' if cf_ok else '，需要' + ('降低' if cf.value > cr[1] else '提高')}")
    for label, t, key in (("阴影", sh, "shadow_tint"), ("高光", hl, "highlight_tint")):
        if t is None and style.get(key):
            gaps.append(f"{label}里接近中性的区域太少（有颜色的物体如天空、绿叶不算），无法判断{label}色调")
    for label, t, ok, key in (("阴影", sh, sh_ok, "shadow_tint"), ("高光", hl, hl_ok, "highlight_tint")):
        if t is not None and ok is False:
            now = f"色相 {t[0]:.0f}°、色度 {t[1]:.1f}" if t[1] >= 1.5 else f"接近中性（色度 {t[1]:.1f}）"
            gaps.append(f"{label}色调：{now}，目标色相 {style[key]['hue']}、色度 {style[key]['chroma']}")
    bp_max = style.get("black_point_max")
    if bp_max is not None and p5 > bp_max:
        gaps.append(f"黑位不够深：P5 明度 {p5:.0f}（目标 ≤ {bp_max}）")
    if style.get("min_median_L") and p50 < style["min_median_L"]:
        gaps.append(f"整体偏暗：中位明度 {p50:.0f}（目标 ≥ {style['min_median_L']}）")
    bp_min = style.get("black_point_min")
    if bp_min is not None and p5 < bp_min:
        gaps.append(f"黑位太深：P5 明度 {p5:.0f}（这种风格黑场是抬起的，目标 ≥ {bp_min}）")
    ct = style.get("contrast")
    ct_ok = None if ct is None else bool(ct[0] <= contrast <= ct[1])
    if ct_ok is False:
        gaps.append(f"对比度：亮暗差（P95−P5 明度）{contrast:.0f}，目标 {ct[0]}–{ct[1]}，需要{'降低' if contrast > ct[1] else '提高'}")

    # 评分
    parts = [(0.5, min(1.0, on_target / 0.85)) if has_sectors else (0.5, 1.0 if cf_ok else 0.5)]
    parts.append((0.2, 1.0 if cf_ok else max(0.0, 1 - min(abs(cf.value - cr[0]), abs(cf.value - cr[1])) / 30)))
    for ok, wgt in ((sh_ok, 0.15), (hl_ok, 0.15)):
        if ok is not None:
            parts.append((wgt, 1.0 if ok else 0.3))
    if bp_max is not None:
        parts.append((0.1, 1.0 if p5 <= bp_max else max(0.0, 1 - (p5 - bp_max) / 20)))
    if bp_min is not None:
        parts.append((0.1, 1.0 if p5 >= bp_min else max(0.0, 1 - (bp_min - p5) / 10)))
    if ct_ok is not None:
        dist = 0.0 if ct_ok else min(abs(contrast - ct[0]), abs(contrast - ct[1]))
        parts.append((0.15, max(0.0, 1 - dist / 25)))
    score = 100 * sum(wt * v for wt, v in parts) / sum(wt for wt, _ in parts)

    adjs = _style_adjustments(sub, style, lab2, L, (p15, p85), sh, hl, sh_ok, hl_ok, cf.value)
    if bp_min is not None and p5 < bp_min:
        gray = int(cs.lab_to_srgb8(np.array([float(bp_min), 0.0, 0.0]))[0])
        adjs.insert(0, Adjustment("曲线", f"RGB 黑场抬起：输入 0 → 输出约 {gray}（P5 明度 {p5:.0f} → ≥ {bp_min}）", "小幅", False))
    if ct_ok is False:
        if contrast > ct[1]:
            text = f"RGB 降低对比：亮暗差 {contrast:.0f} → {ct[1]} 以内（黑场抬、白场略压，或中间调 S 曲线减弱）"
        else:
            text = f"RGB 加对比：亮暗差 {contrast:.0f} → ≥ {ct[0]}（S 形曲线，暗部下压、亮部上提）"
        adjs.insert(0, Adjustment("曲线", text, "小幅", True))
    return StyleMatch(
        key=style["key"],
        name=style["name"],
        score=round(float(score), 1),
        on_target=round(on_target, 4),
        off_target=tuple(off),
        sectors=tuple(sectors),
        colorfulness_ok=cf_ok,
        colorfulness_range=cr,
        shadow_tint=None if sh is None else (round(sh[0], 1), round(sh[1], 2)),
        highlight_tint=None if hl is None else (round(hl[0], 1), round(hl[1], 2)),
        shadow_ok=sh_ok,
        highlight_ok=hl_ok,
        black_point=round(float(p5), 1),
        gaps=tuple(gaps),
        adjustments=tuple(adjs),
        recipe=style.get("recipe", {}),
        notes=tuple(style.get("notes", ())),
        contrast=round(contrast, 1),
        category=style.get("category", ""),
    )


def _style_adjustments(sub, style, lab2, L, pct, sh, hl, sh_ok=None, hl_ok=None, cf_value=None) -> list[Adjustment]:
    out: list[Adjustment] = []
    sectors = {s["name"]: tuple(s["hue"]) for s in style.get("sectors", [])}
    cr = style.get("colorfulness", [0, 200])
    for c in ac.channel_stats(sub, primary=True):
        if c.share < 0.03 or c.mean_hue is None:
            continue
        act = style.get("channel_targets", {}).get(c.name, "keep")
        if act.startswith("toward:"):
            sec = act.split(":", 1)[1]
            lo, hi = sectors[sec]
            cur_h = ac.lab_hue_of(c.mean_lab)
            if _in_range(np.array([cur_h]), lo, hi)[0]:
                continue  # 已经在目标色区里：不动
            # 往最近的边界（略往里）靠，而不是区段中心
            margin = min(5.0, ((hi - lo) % 360) / 4)
            near_lo = abs(cs.hue_diff(cur_h, lo)) <= abs(cs.hue_diff(cur_h, hi))
            goal_h = (lo + margin) % 360 if near_lo else (hi - margin) % 360
            hs = ac.hue_slider(c.mean_hue, ac.lab_hue_to_hsb_hue(goal_h), channel=c.name)
            if hs * cs.hue_diff(cur_h, goal_h) <= 0:
                continue  # 换算后方向与目标相反：不给
            hs, note = ac.cap_slider(hs)
            if abs(hs) >= 8:
                text = f"{c.name}（占画面颜色 {c.share:.0%}）：色相 {hs:+d}，往“{sec}”靠{note}"
                out.append(Adjustment("Camera Raw 混色器", text, ac.slider_strength(hs), True, (("色相", hs),)))
        elif act.startswith("desat"):
            if c.mean_sat <= DESAT_SAT_FLOOR:
                continue  # 这个通道的颜色已经很淡
            if sectors:  # 有目标色区：这是区外的颜色，降到很淡
                ratio = DESAT_SAT_FLOOR / c.mean_sat
            elif cf_value is None or cf_value <= cr[1]:
                continue  # 整体低饱和风格：色彩丰富度已经在目标内，不用再降
            else:
                ratio = cr[1] / cf_value
            amount = int(round((ratio - 1) * 100))
            if act == "desat_light":
                amount = max(amount, -15)
            amount, note = ac.cap_slider(amount)
            if amount > -8:
                continue
            text = f"{c.name}（占画面颜色 {c.share:.0%}）：饱和度 {amount:+d}{note}"
            prm = [("饱和度", amount)]
            if act == "desat_dark":
                lum = -int(np.clip(20 + 100 * c.share, 20, ac.SLIDER_CAP))
                text += f"，明亮度 {lum:+d}"
                prm.append(("明亮度", lum))
            out.append(Adjustment("Camera Raw 混色器", text, ac.slider_strength(amount), True, tuple(prm)))
    # 阴影/高光色调：颜色分级（估算）+ 曲线（对阴影/高光平均色精确）
    for key, label, t, ok in (("shadow_tint", "阴影", sh, sh_ok), ("highlight_tint", "高光", hl, hl_ok)):
        spec = style.get(key)
        if spec is None or t is None or ok is True:
            continue  # 已经达标就不再建议
        lo, hi = spec["hue"]
        cur = t[2]
        if spec["hue"] == [0, 360]:
            # 只限制色度（例如“阴影接近中性”）：只在超标时把现有色调减弱到上限，绝不加新的色相
            cmax = float(spec["chroma"][1])
            if t[1] <= cmax:
                continue
            k = cmax / max(t[1], 1e-6)
            goal = np.array([cur[0], cur[1] * k, cur[2] * k])
        else:
            goal_h = _center(lo, hi)
            goal_c = float(np.mean(spec["chroma"]))
            goal = np.array([cur[0], goal_c * np.cos(np.radians(goal_h)), goal_c * np.sin(np.radians(goal_h))])
        if float(np.hypot(*(goal[1:] - cur[1:]))) < 2:
            continue
        # 颜色分级是在现有色调上“加”一个色调：要加的是差值（目标 − 当前），不是目标本身
        gh, gs = ac.tint_to_grading(goal[0], goal[1] - cur[1], goal[2] - cur[2])
        gs = min(gs, GRADING_SAT_MAX)
        out.append(Adjustment("Camera Raw 颜色分级", f"{label}：色相 {gh}，饱和度 {gs}（目标 Lab 色调 a* {goal[1]:+.1f}、b* {goal[2]:+.1f}，当前 a* {cur[1]:+.1f}、b* {cur[2]:+.1f}）", "小幅", True,
                              (("色相", gh), ("饱和度", gs))))
        rgb_cur = cs.lab_to_srgb8(cur).astype(int)
        rgb_goal = cs.lab_to_srgb8(goal).astype(int)
        pts = "；".join(f"{n}通道 输入 {a} → 输出 {b}" for n, a, b in zip(("红", "绿", "蓝"), rgb_cur, rgb_goal) if a != b)
        if pts:
            out.append(Adjustment("曲线", f"{label}（平均色）：{pts}", "小幅", False))
    for panel, lines in style.get("recipe", {}).items():
        for line in lines:
            out.append(Adjustment(f"常用做法·{panel}", line, None, True))
    return out


# ================================================================ 汇总
def analyze_palette(image: np.ndarray, style_key: str | None = None, family: str | None = None,
                    family_target_lab=None, offset=(0, 0)) -> PaletteReport:
    sub, _, lab, lch = _prep(image)
    cf = colorfulness(sub, lch)
    palette, ratio = ratio_631(lab)
    tone = tuple(round(float(v), 1) for v in np.percentile(lch[..., 0], [5, 25, 50, 75, 95]))
    unity = family_unity(image, family, family_target_lab, offset) if family else None
    style = style_match(image, get_style(style_key)) if style_key else None
    return PaletteReport(cf, palette, ratio, tuple(ac.channel_stats(sub)), tone, unity, style)
