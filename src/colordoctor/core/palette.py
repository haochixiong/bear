"""整图色彩分析：色彩丰富度（缺色）、6:3:1 比例、某种颜色是否统一、与风格模板的差距。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources

import numpy as np

from colordoctor.core import acr_channels as ac
from colordoctor.core import clustering
from colordoctor.core import color_spaces as cs
from colordoctor.core import native_layers as nl
from colordoctor.core.color_models import Adjustment
from colordoctor.core.perception import de_words
from colordoctor.core.contamination import _components
from colordoctor.core.region_statistics import weighted_circular_hue
from colordoctor.core.sampling import SampleError, grid_downsample
from colordoctor.core.tone_class import TONE_STYLES

TONE_STYLE_NAMES = {k: v["name"] for k, v in TONE_STYLES.items()}

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
    tone_key: str = ""  # 画面影调基调："low"（暗调）/ "high"（高调）/ ""（见 tone_key()）


@dataclass(frozen=True)
class PaletteReport:
    colorfulness: Colorfulness
    palette: tuple[PaletteColor, ...]
    ratio: RatioResult
    channels: tuple[ac.ChannelStat, ...]
    tone: tuple[float, float, float, float, float]  # L* 的 P5/P25/P50/P75/P95
    unity: FamilyUnity | None = None
    style: StyleMatch | None = None
    tone_key: str = ""  # 颜色分析按哪种基调处理："low"（暗调）/ "high"（高调）/ ""：选了目标影调时按目标，否则按画面本身
    detected_key: str = ""  # 画面本身的基调（tone_key()）
    tone_class: object = None  # 画面现在的影调（tone_class.ToneClass，九种之一）
    tone_target: str = ""  # 目标影调（选了影调风格或直接选了影调时），例如“高短调”
    tone_label: str = ""  # 目标的显示名（影调风格名，例如“日系”；直接选影调时就是影调名）
    tone_also: tuple[str, ...] = ()  # 也算符合的影调（影调风格的备选影调）
    tone_choice: str = ""  # 用户的选择："s:<影调风格>" / "t:<影调>" / ""（不改影调）


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


# 影调基调：暗调（低调）/ 高调画面的暗、亮是风格本身，不按“偏暗”“对比不足”“缺色”处理
LOW_KEY_MEDIAN_L = 35.0  # 中位明度不高于此值……
HIGH_KEY_MEDIAN_L = 72.0  # 中位明度不低于此值……
KEY_OTHER_END_SHARE = 0.10  # ……且另一端（暗调的亮部 L* ≥ 70 / 高调的暗部 L* ≤ 30）不到 10%
KEY_NAME = {"low": "暗调", "high": "高调"}


def tone_key(L) -> str:
    """画面的影调基调："low" = 暗调（大部分像素很暗、亮部很少），"high" = 高调，"" = 普通。"""
    L = np.asarray(L, dtype=float).ravel()
    if L.size == 0:
        return ""
    p50 = float(np.median(L))
    if p50 <= LOW_KEY_MEDIAN_L and float((L >= 70).mean()) < KEY_OTHER_END_SHARE:
        return "low"
    if p50 >= HIGH_KEY_MEDIAN_L and float((L <= 30).mean()) < KEY_OTHER_END_SHARE:
        return "high"
    return ""


MONO_CHROMATIC_SHARE = 0.01  # 有彩像素不到 1%：黑白（或几乎无色的单色调）作品


def is_monochrome(cf: Colorfulness) -> bool:
    return cf.chromatic_share < MONO_CHROMATIC_SHARE and cf.p90_chroma < CHROMA_MIN


def family_of(L: float, C: float, h: float) -> str:
    """无彩色（黑 / 灰 / 白）要色度低；很暗的地方门槛更低（L* 5 的深蓝 C* 7 是深蓝，不是黑，见 cs.neutral_chroma_limit）。"""
    if C < float(cs.neutral_chroma_limit(L, CHROMA_MIN)):
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
    rows = []  # PS 色相/饱和度：(通道, HSB 中心, 色相度数, 饱和度, 明亮度)
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
            rows.append((c.name, c.mean_hue, float(cs.hue_diff(c.mean_hue, goal)) if abs(hs) >= 5 else 0.0,
                         ss if abs(ss) >= 8 else 0, 0, c.share))
            adjs.append(
                Adjustment(
                    "Camera Raw 混色器",
                    f"{c.name}（管着这种{family}的 {c.share:.0%}）：" + "，".join(parts) + f" → 收拢到同一个{family}",
                    ac.slider_strength(max(abs(hs), abs(ss))),
                    True,
                    (("色相", hs if abs(hs) >= 5 else 0), ("饱和度", ss if abs(ss) >= 8 else 0)),
                )
            )
    toward = False
    if target is not None and de is not None and de >= 3:
        cur = cs.lab_to_lch(mean)
        tl = cs.lab_to_lch(target)
        # 色相按这种颜色自己的明度、色度换成 HSB；目标的色度（浓淡）也要靠过去，否则差距主要在色度时这一层几乎没用
        c0 = ac.lab_hue_to_hsb_hue(cur[2], L=cur[0], C=max(cur[1], 8.0))
        c1 = ac.lab_hue_to_hsb_hue(tl[2], L=cur[0], C=max(cur[1], 8.0))
        hs, h_note = ac.cap_slider(ac.hue_slider(c0, c1))
        s0, s1 = (float(cs.srgb8_to_hsb(cs.lab_to_srgb8(np.asarray(v, float)))[1]) for v in (mean, target))
        ss, s_note = ac.cap_slider(ac.sat_slider(s0, s1))
        ss = ss if abs(ss) >= 5 else 0
        ls, l_note = ac.cap_slider(ac.lum_slider(tl[0] - cur[0]))
        adjs.append(
            Adjustment(
                "Camera Raw 混色器",
                f"整体往目标{family}靠：色相 {hs:+d}{h_note}，" + (f"饱和度 {ss:+d}{s_note}，" if ss else "")
                + f"明亮度 {ls:+d}{l_note}（当前与目标相差 ΔE00 {de:.1f}，{de_words(de)}）",
                ac.slider_strength(max(abs(hs), abs(ss), abs(ls))),
                True,
            )
        )
        for params, words in nl.mixer_rows_to_huesat([(family, c0, float(cs.hue_diff(c0, c1)), ss, ls)]):
            adjs.append(nl.adjustment("hueSaturation", params, f"往目标{family}靠：{words}（PS 做法，与“Camera Raw 混色器”那一条二选一）",
                                      "global", f"{family}往目标靠 · 色相/饱和度"))
            toward = True
    # 有“往目标靠”那一层时，各通道收拢的那一层做的是同一件事（都往目标转），不再另给（否则同一批像素转两次）
    adjs += _unity_native(family, mean, patches[:8], [] if toward else rows)
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


def _unity_native(family: str, mean, patches, rows) -> list[Adjustment]:
    """统一度的 PS 做法：有偏离的小块时逐块推回主色（各自只作用于那一块）；没有时一层色相/饱和度把各通道收拢。"""
    out = []
    L0, a0, b0 = (float(v) for v in mean)
    main_h = float(np.degrees(np.arctan2(b0, a0))) % 360
    C0 = float(np.hypot(a0, b0))
    for i, p in enumerate(patches):
        patch_h = main_h + p.hue_shift
        hue = nl.clip(nl.hsb_shift(patch_h, main_h, L0, C0), -30, 30)  # 把偏离的一块推回主色
        if hue == 0:
            continue
        idx, ramps = nl.custom_range(ac.lab_hue_to_hsb_hue(patch_h % 360, L=L0, C=max(C0, 8.0)))
        rname = nl.PS_RANGES[idx - 1][1]
        out.append(nl.adjustment("hueSaturation", nl.huesat(idx, ramps, hue=hue),
                                 f"{p.note}：{rname}（范围已对准这一块的颜色）色相 {hue:+d}（只作用于这一块）",
                                 "local", f"{family}统一 {i + 1} · 色相/饱和度", f"unity@{i}"))
    if not out:
        for params, words in nl.mixer_rows_to_huesat(rows):
            out.append(nl.adjustment("hueSaturation", params, f"PS 做法（与上面的混色器二选一）：{words}",
                                     "global", f"{family}统一 · 色相/饱和度"))
    return out


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
SHADOW_MAX_L = 45.0  # 阴影：最暗 15% 里 L* 不超过这个值的像素（高调画面最暗的地方是浅灰，不是阴影）
HIGHLIGHT_MIN_L = 60.0  # 高光：最亮 15% 里 L* 不低于这个值的像素（暗调画面最亮的地方是中灰，不是高光）
BAND_MIN_SHARE = 0.01  # 这样的阴影 / 高光不到全图 1%：画面里几乎没有这一段，不判断它的色调
DESAT_SAT_FLOOR = 10.0  # 通道平均饱和度（HSB）已低于此值：不再建议降饱和


def _tint(lab: np.ndarray, sel: np.ndarray, min_share: float = 0.0, band=None) -> tuple[float, float, np.ndarray] | None:
    if sel.sum() < 10 or (band is not None and sel.sum() < min_share * max(1, band.sum())):
        return None
    m = lab[sel].mean(axis=0)
    return float(np.degrees(np.arctan2(m[2], m[1])) % 360), float(np.hypot(m[1], m[2])), m


def _band_tint(lab, C, band, L=None):
    """L 给出时“接近中性”的色度上限随明度收紧（很暗的深蓝、深绿不算中性的黑，见 cs.neutral_chroma_limit）。"""
    lim = (TINT_NEUTRAL_C, TINT_MAX_C) if L is None else (cs.neutral_chroma_limit(L, TINT_NEUTRAL_C),
                                                          cs.neutral_chroma_limit(L, TINT_MAX_C))
    return (_tint(lab, band & (C < lim[0]), min_share=0.05, band=band)
            or _tint(lab, band & (C < lim[1]), min_share=0.05, band=band))


def tone_bands(L: np.ndarray, rgb: np.ndarray | None = None) -> dict[str, np.ndarray | None]:
    """{"shadow": 阴影像素, "highlight": 高光像素}（布尔，与 L 同长）。

    阴影 = 最暗 15% 且 L* ≤ SHADOW_MAX_L，高光 = 最亮 15% 且 L* ≥ HIGHLIGHT_MIN_L：暗调画面最亮的 15% 是中灰、
    高调画面最暗的 15% 是浅灰，都不是真正的高光 / 阴影。被裁切的纯白 / 纯黑（rgb 给出时）不算：
    它们“中性”只是因为被裁切，会把色调估计拉向中性。某一段不到全图 BAND_MIN_SHARE 时为 None（画面里几乎没有）。
    """
    L = np.asarray(L, dtype=float).ravel()
    ok = np.ones(L.shape, bool) if rgb is None else ~cs.clipped(np.asarray(rgb).reshape(-1, 3))
    out: dict[str, np.ndarray | None] = {"shadow": None, "highlight": None}
    if ok.sum() < 10:
        return out
    p15, p85 = np.percentile(L[ok], [15, 85])  # 分位数只在没被裁切的像素里算（大片死白不把“最亮 15%”全占掉）
    for key, sel in (("shadow", (L <= p15) & (L <= SHADOW_MAX_L)), ("highlight", (L >= p85) & (L >= HIGHLIGHT_MIN_L))):
        sel &= ok
        out[key] = sel if sel.sum() >= max(10, BAND_MIN_SHARE * L.size) else None
    return out


def _no_band_text(label: str) -> str:
    rng, clip = ("L* ≤ 45", "黑") if label == "阴影" else ("L* ≥ 60", "白")
    return f"画面里几乎没有真正的{label}（{rng}、且不是被裁切的纯{clip}），无法判断{label}色调"


def _tint_ok(t, spec) -> bool | None:
    if t is None or spec is None:
        return None
    h, c, _ = t
    lo, hi = spec["chroma"]
    hue_ok = spec["hue"] == [0, 360] or c < 1.5 or bool(_in_range(np.array([h]), *spec["hue"])[0])
    return bool(hue_ok and lo - 0.5 <= c <= hi + 0.5)


def style_match(image: np.ndarray, style: dict, max_samples: int = MAX_SAMPLES, key: str | None = None,
                target: str = "") -> StyleMatch:
    """与颜色模板的差距。只看颜色（色区、丰富度、阴影 / 高光色调），明暗和反差归「影调风格」管。

    key：按哪种基调处理（"low" / "high" / ""）；None 时按画面本身判断。target：key 来自用户选的目标影调时，目标影调名。
    """
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
    tk = tone_key(L) if key is None else key
    # 调色加在阴影/高光上的色调，主要体现在“接近中性”的像素上（灰墙、白衣、暗部），
    # 本身就很鲜艳的物体（绿叶、蓝天）不能代表色调。
    bands = tone_bands(L, sub)
    sh = _band_tint(lab2, C, bands["shadow"], L) if bands["shadow"] is not None else None
    hl = _band_tint(lab2, C, bands["highlight"], L) if bands["highlight"] is not None else None
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
    for label, t, band, tkey in (("阴影", sh, "shadow", "shadow_tint"), ("高光", hl, "highlight", "highlight_tint")):
        if t is None and style.get(tkey):
            gaps.append(_no_band_text(label) if bands[band] is None else
                        f"{label}里接近中性的区域太少（有颜色的物体如天空、绿叶不算），无法判断{label}色调")
    for label, t, ok, skey in (("阴影", sh, sh_ok, "shadow_tint"), ("高光", hl, hl_ok, "highlight_tint")):
        if t is not None and ok is False:
            now = f"色相 {t[0]:.0f}°、色度 {t[1]:.1f}" if t[1] >= 1.5 else f"接近中性（色度 {t[1]:.1f}）"
            gaps.append(f"{label}色调：{now}，目标色相 {style[skey]['hue']}、色度 {style[skey]['chroma']}")
    if tk and target:
        gaps.append(f"目标影调是{target}：颜色按{KEY_NAME[tk]}处理（{'暗部的颜色少、浅' if tk == 'low' else '亮、淡'}是正常的）")
    elif tk == "low":
        gaps.append(f"暗调画面（中位明度 {p50:.0f}）：颜色按暗调处理，暗部的颜色少、浅是正常的")
    elif tk == "high":
        gaps.append(f"高调画面（中位明度 {p50:.0f}）：颜色按高调处理，亮、淡是这张图的基调")

    # 评分
    parts = [(0.5, min(1.0, on_target / 0.85)) if has_sectors else (0.5, 1.0 if cf_ok else 0.5)]
    parts.append((0.2, 1.0 if cf_ok else max(0.0, 1 - min(abs(cf.value - cr[0]), abs(cf.value - cr[1])) / 30)))
    for ok, wgt in ((sh_ok, 0.15), (hl_ok, 0.15)):
        if ok is not None:
            parts.append((wgt, 1.0 if ok else 0.3))
    score = 100 * sum(wt * v for wt, v in parts) / sum(wt for wt, _ in parts)

    adjs = _style_adjustments(sub, style, lab2, L, (p15, p85), sh, hl, sh_ok, hl_ok, cf.value)
    pair = [TONE_STYLE_NAMES[k] for k in style.get("tone_pair", ()) if k in TONE_STYLE_NAMES]
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
        notes=tuple(style.get("notes", ())) + ((f"明暗不在颜色模板里：「{style['name']}」常配的影调风格是{'、'.join(pair)}（在「影调风格」里选，也可以配别的）",) if pair else ()),
        contrast=round(contrast, 1),
        category=style.get("category", ""),
        tone_key=tk,
    )


def _style_adjustments(sub, style, lab2, L, pct, sh, hl, sh_ok=None, hl_ok=None, cf_value=None) -> list[Adjustment]:
    out: list[Adjustment] = []
    rows = []  # PS 色相/饱和度：(通道, HSB 中心, 色相度数, 饱和度, 明亮度)
    grading = {}  # 色彩平衡：色调 -> [青-红, 洋红-绿, 黄-蓝]
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
                rows.append((c.name, c.mean_hue, float(cs.hue_diff(c.mean_hue, ac.lab_hue_to_hsb_hue(goal_h))), 0, 0, c.share))
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
            # 颜色模板只管颜色：desat_dark（例如黑金里的冷色）只降饱和，压暗交给影调风格（例如暗调情绪）
            rows.append((c.name, c.mean_hue, 0.0, amount, 0, c.share))
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
        # PS 色彩平衡：在这一段的实际平均色上求滑块（这一段的明度决定色彩平衡的“阴影/高光”管不管得到）
        lv = nl.balance_for_shift(goal[1] - cur[1], goal[2] - cur[2], "shadows" if label == "阴影" else "highlights", lab=cur)
        if any(lv):
            grading["shadows" if label == "阴影" else "highlights"] = lv
        out.append(Adjustment("Camera Raw 颜色分级", f"{label}：色相 {gh}，饱和度 {gs}（目标 Lab 色调 a* {goal[1]:+.1f}、b* {goal[2]:+.1f}，当前 a* {cur[1]:+.1f}、b* {cur[2]:+.1f}）", "小幅", True,
                              (("色相", gh), ("饱和度", gs))))
        rgb_cur = cs.lab_to_srgb8(cur).astype(int)
        rgb_goal = cs.lab_to_srgb8(goal).astype(int)
        pts = "；".join(f"{n}通道 输入 {a} → 输出 {b}" for n, a, b in zip(("红", "绿", "蓝"), rgb_cur, rgb_goal) if a != b)
        if pts:
            alt = ("（可代替色彩平衡，二选一）" if any(lv) else
                   f"（这一段的明度 L* {cur[0]:.0f} 在色彩平衡的“{label}”范围之外，PS 里用这条曲线，配合明亮度蒙版只作用于{label}）")
            out.append(Adjustment("曲线", f"{label}（平均色）：{pts}{alt}", "小幅", False))
    for params, words in nl.mixer_rows_to_huesat(rows):
        out.append(nl.adjustment("hueSaturation", params, f"PS 做法（代替上面的 Camera Raw 混色器）：{words}",
                                 "global", f"「{style['name']}」颜色 · 色相/饱和度"))
    if any(any(v) for v in grading.values()):
        p = nl.balance(**grading)
        out.append(nl.adjustment("colorBalance", p, f"PS 做法（代替上面的颜色分级），勾选“保留明度”：{nl.balance_text(p)}",
                                 "global", f"「{style['name']}」色调 · 色彩平衡"))
    for panel, lines in style.get("recipe", {}).items():
        for line in lines:
            line = colour_only(panel, line)
            if line:
                out.append(Adjustment(f"常用做法·{panel}", line, None, True))
    return out


# 常用做法里讲明暗的部分（对比、黑场、提亮、S 曲线……）属于影调，由「影调风格」管：颜色模板的建议里去掉，
# 只留颜色的部分（某个颜色通道的曲线、饱和度、色温……）
TONE_PANELS = ("曲线", "基本")
_COLOUR_WORDS = re.compile(r"[红绿蓝]通道|饱和|色温|色调|白平衡|黑白")
_TONE_WORDS = re.compile(r"对比|黑场|白场|提亮|压暗|压深|压实|曝光|S 形|S形|暗部|亮部|高光|阴影|柔和|明亮")


def colour_only(panel: str, line: str) -> str:
    if panel not in TONE_PANELS:
        return line
    keep = []
    for part in re.split(r"[；，,;]", line):
        part = part.strip()
        if part and (_COLOUR_WORDS.search(part) or not _TONE_WORDS.search(part)):
            keep.append(part)
    return "；".join(keep)


# ================================================================ 汇总
def analyze_palette(image: np.ndarray, style_key: str | None = None, family: str | None = None,
                    family_target_lab=None, offset=(0, 0), tone_style: str = "", tone_target: str = "") -> PaletteReport:
    """tone_style / tone_target：影调风格（tone_class.TONE_STYLES 的键）/ 直接选的目标影调（九种之一），都可以为空。"""
    from colordoctor.core import tone_class as tcl

    sub, _, lab, lch = _prep(image)
    cf = colorfulness(sub, lch)
    palette, ratio = ratio_631(lab)
    tone = tuple(round(float(v), 1) for v in np.percentile(lch[..., 0], [5, 25, 50, 75, 95]))
    unity = family_unity(image, family, family_target_lab, offset) if family else None
    detected = tone_key(lch[..., 0])
    main, also, label = tcl.resolve_target(tone_style, tone_target)
    cls = tcl.classify(sub)
    target = tcl.pick_goal(cls.name, main, also)  # 影调风格有几种可接受的影调：往最近的那种走
    key = tcl.key_code(target) if target else detected  # 选了目标影调：颜色按目标的基调处理
    style = style_match(image, get_style(style_key), key=key, target=target) if style_key else None
    return PaletteReport(cf, palette, ratio, tuple(ac.channel_stats(sub)), tone, unity, style,
                         tone_key=key, detected_key=detected, tone_class=cls,
                         tone_target=target, tone_label=label, tone_also=also,
                         tone_choice=f"t:{tone_target}" if tone_target else (f"s:{tone_style}" if tone_style else ""))
