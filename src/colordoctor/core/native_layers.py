"""Photoshop 原生调整图层：参数格式、Camera Raw 建议换成 PS 的等效做法、卡片文字。

能直接生成图层的建议（Adjustment）带一个 layer 字典：
  {"kind": 图层类型, "params": 参数, "scope": "skin" | "local" | "global", "name": 图层名}
另有以下划线开头的内部键（不导出到 JSON）："_at" = 局部蒙版位置 / 图层 id 前缀，例如
"opposite"、"opposite@2"（第 2 处问题位置）、"skin@偏黄"（肤色不均匀的这一类小块）、"unity@0"。
卡片上的数字与图层参数完全相同。参数单位与 PS 属性面板一致：
- hueSaturation：{"master": {hue, saturation, lightness}, "ranges": [{localRange 1–6, beginRamp, beginSustain,
  endSustain, endRamp, hue, saturation, lightness}]}，色相是 HSB 度数（±180），饱和度/明度 ±100；
- colorBalance：{"shadows"|"midtones"|"highlights": [青-红, 洋红-绿, 黄-蓝], "preserveLuminosity": true}，±100；
- vibrance：{"vibrance", "saturation"}；
- curves：{"channels": {"composite"|"red"|"green"|"blue": [[输入, 输出], ...]}}，0–255，含端点；
- selectiveColor：{"method": "relative", "colors": {颜色: {"c", "m", "y", "k"}}}，±100，只列出改动的颜色。
Camera Raw 滑块换成 PS 数值是估算（Adobe 未公开算法），建好图层后可以在属性面板里继续改。
"""

from __future__ import annotations

import numpy as np

from colordoctor.core import acr_channels as ac
from colordoctor.core import color_spaces as cs
from colordoctor.core import ps_models as pm
from colordoctor.core.color_models import Adjustment

# PS 色相/饱和度的六个颜色范围：(localRange 编号, 名称, HSB 中心色相, 默认渐变 beginRamp..endRamp)
PS_RANGES = (
    (1, "红色", 0, (315, 345, 15, 45)),
    (2, "黄色", 60, (15, 45, 75, 105)),
    (3, "绿色", 120, (75, 105, 135, 165)),
    (4, "青色", 180, (135, 165, 195, 225)),
    (5, "蓝色", 240, (195, 225, 255, 285)),
    (6, "洋红", 300, (255, 285, 315, 345)),
)
KIND_NAME = {"hueSaturation": "色相/饱和度", "colorBalance": "色彩平衡", "vibrance": "自然饱和度", "curves": "曲线",
             "selectiveColor": "可选颜色"}
LAYER_KINDS = tuple(KIND_NAME)
SC_COLORS = {"reds": "红色", "yellows": "黄色", "greens": "绿色", "cyans": "青色", "blues": "蓝色", "magentas": "洋红",
             "whites": "白色", "neutrals": "中性色", "blacks": "黑色"}
SC_INKS = (("c", "青色"), ("m", "洋红"), ("y", "黄色"), ("k", "黑色"))
CB_TONES = (("shadows", "阴影"), ("midtones", "中间调"), ("highlights", "高光"))
CB_AXES = (("青", "红"), ("洋红", "绿"), ("黄", "蓝"))
CURVE_CH = {"composite": "RGB", "red": "红通道", "green": "绿通道", "blue": "蓝通道"}

PS_HUE_CAP = 30  # Camera Raw 换算来的 PS 色相（度）上限
PS_SAT_CAP = 60
PS_LIGHT_CAP = 30
SC_CAP = 30  # 可选颜色单项上限（%）
SC_PER_LAB = 2.5  # 可选颜色：约 2.5% 对应 1 个 Lab 单位（经验值）
S_CURVE = ((64, 58), (192, 198))  # 小幅 S 形加对比
MERGE_HSB = 12.0  # 两个混色器通道的颜色相距不到这么多度（HSB）时合成一个 PS 范围


def clip(v, lo, hi) -> int:
    return int(np.clip(round(float(v)), lo, hi))


BLENDS = ("normal", "luminosity")


def layer(kind: str, params: dict, scope: str = "global", name: str = "", at: str | None = None,
          blend: str = "normal") -> dict:
    """blend：图层混合模式。"luminosity"（明度）= 只改明暗不改颜色（影调曲线用）。"""
    d = {"kind": kind, "params": params, "scope": scope, "name": name or KIND_NAME[kind]}
    if at:
        d["_at"] = at
    if blend != "normal":
        assert blend in BLENDS, blend
        d["blend"] = blend
    return d


def adjustment(kind: str, params: dict, text: str, scope: str, name: str, at: str | None = None,
               strength: str | None = "小幅", estimated: bool = True, tool: str | None = None,
               blend: str = "normal") -> Adjustment:
    """一条带图层的建议：卡片文字 text 里的数字就是 params 里的数字。"""
    return Adjustment(tool or KIND_NAME[kind], text, strength, estimated, layer=layer(kind, params, scope, name, at, blend))


# ---------------------------------------------------------------- 色相/饱和度
def ps_range_for_lab_hue(lab_hue: float) -> tuple[int, str, tuple[int, int, int, int]]:
    """Lab 色相角 -> PS 色相/饱和度里最接近的颜色范围。"""
    hsb = ac.lab_hue_to_hsb_hue(lab_hue)
    best = min(PS_RANGES, key=lambda r: abs(((hsb - r[2] + 180) % 360) - 180))
    return best[0], best[1], best[3]


def ps_hsb_of_lab(lab) -> float:
    """一个 Lab 颜色在 PS（HSB）色相环上的位置（范围对准它用）。"""
    L, a, b = (float(v) for v in lab[:3])
    return ac.lab_hue_to_hsb_hue(float(np.degrees(np.arctan2(b, a))) % 360, L=min(max(L, 15.0), 90.0), C=max(float(np.hypot(a, b)), 8.0))


def ps_range_by_name(name: str):
    """PS 范围名（红色 / 黄色 ...）-> (编号, 名称, 默认渐变)；不认识时为 None。"""
    r = next((r for r in PS_RANGES if r[1] == name), None)
    return None if r is None else (r[0], r[1], r[3])


def hsb_shift(lab_from: float, lab_to: float, L: float = 62.0, C: float = 30.0) -> float:
    """Lab 色相从 lab_from 变到 lab_to，对应 PS（HSB）色相要转多少度。

    两种色相刻度差别很大（绿色区域约 2 倍、肤色区域约 0.5 倍），不能直接拿 Lab 度数当 PS 滑块值。
    """
    a = ac.lab_hue_to_hsb_hue(lab_from % 360, L=L, C=max(C, 8.0))
    b = ac.lab_hue_to_hsb_hue(lab_to % 360, L=L, C=max(C, 8.0))
    return ((b - a + 180) % 360) - 180


def custom_range(hsb_center: float, half_core: int = 10, half_ramp: int = 30) -> tuple[int, tuple[int, int, int, int]]:
    """以某个 HSB 色相为中心的自定义颜色范围（PS 色相/饱和度里可以拖动范围滑块，效果相同）。"""
    idx = min(PS_RANGES, key=lambda r: abs(((hsb_center - r[2] + 180) % 360) - 180))[0]
    c = int(round(hsb_center))
    return idx, ((c - half_ramp) % 360, (c - half_core) % 360, (c + half_core) % 360, (c + half_ramp) % 360)


def huesat(range_idx: int, ramps, hue=0, sat=0, light=0, master=False) -> dict:
    if master:
        return {"master": {"hue": int(hue), "saturation": int(sat), "lightness": int(light)}, "ranges": []}
    b0, b1, e0, e1 = ramps
    return {"master": {"hue": 0, "saturation": 0, "lightness": 0},
            "ranges": [{"localRange": range_idx, "beginRamp": b0, "beginSustain": b1, "endSustain": e0,
                        "endRamp": e1, "hue": int(hue), "saturation": int(sat), "lightness": int(light)}]}


def huesat_multi(items) -> dict:
    """items：[(HSB 中心, 核心半宽, 过渡半宽, 色相, 饱和度, 明度)] -> 一层色相/饱和度，每个范围占一个编号（最多 6 个）。"""
    used: set[int] = set()
    ranges = []
    for center, core, ramp, hue, sat, light in items[:6]:
        order = sorted(PS_RANGES, key=lambda r: abs(((center - r[2] + 180) % 360) - 180))
        idx = next(r[0] for r in order if r[0] not in used)
        used.add(idx)
        c = int(round(center))
        ranges.append({"localRange": idx, "beginRamp": (c - ramp) % 360, "beginSustain": (c - core) % 360,
                       "endSustain": (c + core) % 360, "endRamp": (c + ramp) % 360,
                       "hue": int(hue), "saturation": int(sat), "lightness": int(light)})
    return {"master": {"hue": 0, "saturation": 0, "lightness": 0}, "ranges": ranges}


def _span(r) -> tuple[int, int]:
    return int(r["beginRamp"]), int((r["endRamp"] - r["beginRamp"]) % 360)


def ranges_overlap(r1: dict, r2: dict) -> bool:
    """两个色相/饱和度范围（含过渡带）在色相环上有没有重叠。"""
    s1, w1 = _span(r1)
    s2, w2 = _span(r2)
    return (s2 - s1) % 360 < w1 or (s1 - s2) % 360 < w2


def huesat_overlap(p1: dict, p2: dict) -> bool:
    """两层色相/饱和度会不会改到同一批颜色（任一层有“全图”数值时一定会）。"""
    if any(p1["master"].values()) or any(p2["master"].values()):
        return True
    return any(ranges_overlap(a, b) for a in p1["ranges"] for b in p2["ranges"])


def _hsl_words(hue=0, sat=0, light=0) -> str:
    parts = [f"{n} {int(v):+d}" for n, v in (("色相", hue), ("饱和度", sat), ("明度", light)) if int(v)]
    return "，".join(parts)


def _acr_gap(name: str) -> float:
    """混色器通道与最近的相邻通道相隔多少度（HSB），决定 PS 范围宽窄。"""
    centers = dict(ac.CHANNELS)
    c = centers.get(name)
    if c is None:
        return 60.0
    return min(abs(((c - o + 180) % 360) - 180) for n, o in ac.CHANNELS if n != name)


def mixer_rows_to_huesat(rows) -> list[tuple[dict, str]]:
    """Camera Raw 混色器 -> PS 色相/饱和度（每个通道一个对准该颜色的自定义范围）。

    rows：[(通道名, HSB 中心, 色相度数, ACR 饱和度滑块, ACR 明亮度滑块[, 权重])]。色相用真实要转的度数（不从滑块反推）；
    饱和度 ±100 与 PS 大致相当（上限 ±60）；明亮度在 PS 里会同时变灰，按一半换算、上限 ±30。
    权重（一般是该通道的颜色占比）用于合并：几乎没有像素的通道不能把同一范围里大片颜色往它的方向拉。
    PS 的明度是往白 / 往黑混合，颜色会按 1 − |明度| 变淡（Camera Raw 的明亮度不会）：饱和度相应补回来，
    否则“饱和度 +40、明度 +20”的结果反而更灰。
    返回 [(参数, 卡片文字)]，超过 6 个范围时分成多层。
    """
    # 两个通道的颜色实际落在几乎同一个色相上时，PS 的一个范围没法把它往两边拉：合成一个范围，数值按权重平均
    merged: list[list] = []
    for row in sorted(rows, key=lambda r: float(r[1]) % 360):
        name, center, hue_deg, sat, lum = row[:5]
        wt = max(float(row[5]), 1e-6) if len(row) > 5 else 1.0
        c = float(center) % 360
        near = next((m for m in merged if abs(((c - m[1] + 180) % 360) - 180) < MERGE_HSB), None)
        if near is None:
            merged.append([[name], c, [hue_deg], [sat], [lum or 0], _acr_gap(name), [wt]])
        else:
            near[0].append(name)
            near[2].append(hue_deg)
            near[3].append(sat)
            near[4].append(lum or 0)
            near[5] = min(near[5], _acr_gap(name))
            near[6].append(wt)
    items, words = [], []
    for names, c, hues, sats, lums, gap, wts in merged:
        h = clip(np.average(hues, weights=wts), -PS_HUE_CAP, PS_HUE_CAP)
        li = clip(np.average(lums, weights=wts) * 0.5, -PS_LIGHT_CAP, PS_LIGHT_CAP)
        k = 1.0 + float(np.average(sats, weights=wts)) / 100.0  # 想要的色度倍数
        s = clip(sat_for_ratio(k / (1.0 - abs(li) / 100.0)), -PS_SAT_CAP, PS_SAT_CAP)
        if not (h or s or li):
            continue
        others = [abs(((c - m[1] + 180) % 360) - 180) for m in merged if m[1] != c]
        gap = min([gap, *others])  # 范围不要盖到相邻那一个
        items.append((c, max(3, int(round(gap / 4))), max(8, int(round(gap * 0.75))), h, s, li))
        words.append(f"{'、'.join(names)}（范围对准色相 {c:.0f}°）{_hsl_words(h, s, li)}")
    out = []
    for i in range(0, len(items), 6):
        out.append((huesat_multi(items[i:i + 6]), "；".join(words[i:i + 6])))
    return out


def huesat_text(params: dict) -> str:
    m = params["master"]
    parts = [f"全图 {_hsl_words(m['hue'], m['saturation'], m['lightness'])}"] if _hsl_words(
        m["hue"], m["saturation"], m["lightness"]) else []
    for r in params["ranges"]:
        c = (r["beginSustain"] + ((r["endSustain"] - r["beginSustain"]) % 360) / 2) % 360
        parts.append(f"范围对准色相 {c:.0f}° {_hsl_words(r['hue'], r['saturation'], r['lightness'])}")
    return "；".join(parts)


# ---------------------------------------------------------------- 色彩平衡
def balance(shadows=(0, 0, 0), midtones=(0, 0, 0), highlights=(0, 0, 0)) -> dict:
    return {"shadows": [int(v) for v in shadows], "midtones": [int(v) for v in midtones],
            "highlights": [int(v) for v in highlights], "preserveLuminosity": True}


def balance_text(params: dict) -> str:
    """“阴影 青-红 +5（往红），黄-蓝 -8（往黄）；高光 ……”"""
    out = []
    for key, tone in CB_TONES:
        parts = [f"{a}-{b} {int(v):+d}（往{b if v > 0 else a}）" for (a, b), v in zip(CB_AXES, params[key], strict=True) if int(v)]
        if parts:
            out.append(f"{tone} " + "，".join(parts))
    return "；".join(out)


_TONE_L = {"shadows": 25.0, "midtones": 50.0, "highlights": 75.0}
CB_MIN_EFFECT = 0.06  # 每格滑块至少改变这么多 a*/b*（Lab）才算“管得到”这个颜色；更弱时该色调不用色彩平衡
CB_MIN_REACH = 0.5  # 限幅后至少能完成需要变化的一半，否则不给色彩平衡数值（改用其他做法）
_CB_STEP = 10.0


def _rgb01(lab) -> np.ndarray:
    return np.clip(cs.lab_to_srgb_float(np.asarray(lab, dtype=float)[:3]), 0.0, 1.0)[None, :]


def balance_effect(lab, tone: str, levels, ref_lab=None) -> np.ndarray:
    """色彩平衡（模型，勾选“保留明度”）作用在颜色 lab 上带来的 (Δa*, Δb*)；给出 ref_lab 时是相对 ref_lab 的变化。"""
    def one(c):
        x = _rgb01(c)
        base = cs.srgb8_to_lab(x * 255.0)[0]
        out = pm.color_balance(x, **{tone: tuple(float(v) for v in levels)})
        return cs.srgb8_to_lab(np.clip(out, 0, 1) * 255.0)[0][1:] - base[1:]
    d = one(lab)
    return d - one(ref_lab) if ref_lab is not None else d


def balance_jacobian(lab, tone: str, ref_lab=None) -> np.ndarray:
    """2×3：三个滑块（青-红、洋红-绿、黄-蓝）每格改变多少 (a*, b*)，在 lab 这个颜色上（中心差分）。"""
    cols = []
    for i in range(3):
        lv = [0.0, 0.0, 0.0]
        lv[i] = _CB_STEP
        up = balance_effect(lab, tone, lv, ref_lab)
        lv[i] = -_CB_STEP
        dn = balance_effect(lab, tone, lv, ref_lab)
        cols.append((up - dn) / (2 * _CB_STEP))
    return np.array(cols).T


def balance_for_shift(da: float, db: float, tone: str = "midtones", cap: int = 40, lab=None, ref_lab=None) -> list[int]:
    """让颜色 lab 的 a*/b* 变化 (da, db) 需要的色彩平衡 [青-红, 洋红-绿, 黄-蓝]（估算）。

    lab：要改的那一段的平均色（默认是该色调典型明度的中性灰）；ref_lab：给出时求的是 lab 相对 ref_lab 的变化
    （例如“把高光拉回中间调的颜色”：中间调也会被同一个滑块带动）。用色彩平衡模型（ps_models，GIMP 算法）
    在这个颜色上求三个滑块的完整雅可比（洋红-绿也会改 b*、黄-蓝也会改 a*），带少量正则的最小二乘 + 两步迭代；
    限幅时整体等比缩小（不改方向）。该色调在这个明度上几乎不起作用、或限幅后完成不到一半时返回全 0。
    """
    need = np.array([float(da), float(db)])
    if np.hypot(*need) < 0.5:
        return [0, 0, 0]
    base = np.array([_TONE_L[tone], 0.0, 0.0]) if lab is None else np.asarray(lab, dtype=float)[:3]
    J = balance_jacobian(base, tone, ref_lab)
    if float(np.linalg.norm(J, axis=0).max()) < CB_MIN_EFFECT:
        return [0, 0, 0]
    A = J.T @ J
    A = A + np.eye(3) * (1e-3 * float(np.trace(A)) + 1e-9)
    p = np.linalg.solve(A, J.T @ need)
    for _ in range(2):  # 模型不是线性的：按实际效果再修正
        r = need - balance_effect(base, tone, p, ref_lab)
        p = p + np.linalg.solve(A, J.T @ r)
    m = float(np.abs(p).max())
    if m > cap:
        p = p * (cap / m)
    lv = [int(round(float(v))) for v in p]
    got = balance_effect(base, tone, lv, ref_lab)
    if float(got @ need) / float(need @ need) < CB_MIN_REACH:
        return [0, 0, 0]
    return lv


# ---------------------------------------------------------------- 可选颜色
def selective(colors: dict) -> dict:
    """{颜色: {c, m, y, k}} -> 可选颜色参数（方法“相对”）；去掉全为 0 的颜色。"""
    out = {}
    for name, inks in colors.items():
        vals = {k: clip(inks.get(k, 0), -100, 100) for k, _ in SC_INKS}
        if any(vals.values()):
            out[name] = vals
    return {"method": "relative", "colors": out}


def selective_text(params: dict) -> str:
    """“颜色选“红色”：青色 +6%，洋红 -8%；颜色选“黄色”：黄色 -12%”"""
    out = []
    for name, inks in params["colors"].items():
        parts = [f"{label} {int(inks[k]):+d}%" for k, label in SC_INKS if int(inks.get(k, 0))]
        out.append(f"颜色选“{SC_COLORS[name]}”：" + "，".join(parts))
    return "；".join(out)


def skin_selective(da: float, db: float) -> dict:
    """肤色整体偏色 -> 可选颜色（红色 / 黄色，方法“相对”）。da/db：需要的 a*/b* 变化（目标 − 当前）。

    修图常用做法：偏黄 → 黄色、红色里减黄（黄色为主）；偏红 → 红色里加青、减洋红；偏绿 → 红色/黄色里加洋红；
    缺黄（偏洋红、偏青）→ 加黄。约 1 个 Lab 单位 ≈ 2.5%（经验值），单项上限 ±30%。
    """
    k = SC_PER_LAB
    reds: dict[str, float] = {"y": db * k * 0.6}
    yellows: dict[str, float] = {"y": db * k}
    if da < 0:
        reds["m"] = da * k
        reds["c"] = -da * k * 0.5
        yellows["m"] = da * k * 0.5
    else:
        reds["m"] = da * k
        yellows["m"] = da * k * 0.8
    cap = lambda d: {key: clip(v, -SC_CAP, SC_CAP) for key, v in d.items()}  # noqa: E731
    return selective({"reds": cap(reds), "yellows": cap(yellows)})


def sat_for_ratio(k: float) -> float:
    """想要的色度倍数 k -> PS 色相/饱和度的“饱和度”。PS 减饱和度时色度 ×(1 + s)，加饱和度时 ×1/(1 − s)：
    想 ×1.4 只要 +29，不是 +40（以前按线性算，加得过头）。"""
    k = float(k)
    return (1.0 - 1.0 / k) * 100.0 if k > 1 else (k - 1.0) * 100.0


def skin_dull_huesat(deficit: float, skin_lab) -> dict:
    """肤色发灰、不够透亮 -> 色相/饱和度：范围对准肤色的饱和度提高。deficit：低于下限的比例（色度要乘 1/(1-deficit)）。

    PS 加饱和度时色度 ×1/(1 − 饱和度/100)（见 sat_for_ratio）；可选颜色“减黑”主要是提亮，几乎不加色度，
    自然饱和度在 PS 里会刻意少动肤色，都不适合。范围中心对准皮肤平均色，±15° 全量、±45° 过渡；上限 +30。
    """
    d = float(np.clip(deficit, 0.0, 0.9))
    sat = clip(sat_for_ratio(1.0 / (1.0 - d)), 5, 30)
    idx, ramps = custom_range(ps_hsb_of_lab(skin_lab), 15, 45)
    return huesat(idx, ramps, sat=sat)


# ---------------------------------------------------------------- 曲线
def gray8(L: float) -> int:
    """L* 对应的 sRGB 灰阶（曲线的输入/输出值）。"""
    return int(cs.lab_to_srgb8(np.array([float(L), 0.0, 0.0]))[0])


def curve_points(points) -> list[list[int]]:
    """给定点 + 端点 (0,0)、(255,255)：给定点优先（例如 255 → 215 覆盖默认端点），按输入排序、同一输入只留一个。"""
    by_x: dict[int, int] = {0: 0, 255: 255}
    for x, y in points:
        by_x[clip(x, 0, 255)] = clip(y, 0, 255)
    return [[x, by_x[x]] for x in sorted(by_x)]


def curves(channels: dict) -> dict:
    return {"channels": {k: curve_points(v) for k, v in channels.items()}}


def curves_text(params: dict) -> str:
    """“RGB：输入 64 → 输出 58、输入 192 → 输出 198”（不写没动的端点）。"""
    out = []
    for ch, pts in params["channels"].items():
        moved = [f"输入 {x} → 输出 {y}" for x, y in pts if not ((x, y) in ((0, 0), (255, 255)))]
        if moved:
            out.append(f"{CURVE_CH.get(ch, ch)}：" + "、".join(moved))
    return "；".join(out)


def contrast_curve(p5: float, p95: float, target: float) -> dict | None:
    """把亮暗差（P95 − P5 明度）调到 target：暗部、亮部各移一半。"""
    shift = (target - (p95 - p5)) / 2.0
    lo_in, hi_in = gray8(p5), gray8(p95)
    lo_out = gray8(float(np.clip(p5 - shift, 0, 100)))
    hi_out = gray8(float(np.clip(p95 + shift, 0, 100)))
    pts = [(x, y) for x, y in ((lo_in, lo_out), (hi_in, hi_out)) if 0 < x < 255 and x != y]
    if not pts or hi_in - lo_in < 24:
        return None
    return curves({"composite": pts})


# ---------------------------------------------------------------- 肤色小块（均匀度）
def skin_patch_layer(kind: str, dh: float, ds: float, lab=(68.0, 14.0, 19.0)):
    """肤色问题块 -> (图层类型, 参数, 说明)。dh：相对整体的 Lab 色相偏移（度，正 = 偏黄）；lab：问题块平均色。"""
    L, a, b = lab
    hp = float(np.degrees(np.arctan2(b, a))) % 360
    C = float(np.hypot(a, b))
    if kind in ("偏黄", "偏洋红"):
        shift = hsb_shift(hp, hp - dh, L, C)  # 把问题块拉回整体肤色
        hue = int(np.sign(shift) or (-1 if kind == "偏黄" else 1)) * clip(abs(shift), 2, 15)
        idx, ramps = custom_range(ac.lab_hue_to_hsb_hue(hp, L=L, C=max(C, 8.0)))
        rname = PS_RANGES[idx - 1][1]
        if kind == "偏黄":
            sat = patch_sat(ds)
            return ("hueSaturation", huesat(idx, ramps, hue=hue, sat=sat),
                    f"{rname}（范围已对准问题色）色相 {hue:+d}（往{'黄' if hue > 0 else '红'}）" + (f"，饱和度 {sat:+d}" if sat else ""))
        return "hueSaturation", huesat(idx, ramps, hue=hue), f"{rname}（范围已对准问题色）色相 {hue:+d}（往{'黄' if hue > 0 else '红'}）"
    if kind in ("偏红橙", "偏青绿"):
        # 色彩平衡（中间调）在这块皮肤的实际颜色上求：色度不变、色相转回整体（以前按经验值给，暗皮肤上会转过头一倍）
        goal = np.radians(hp - dh)
        lv = balance_for_shift(C * np.cos(goal) - a, C * np.sin(goal) - b, "midtones", cap=15, lab=lab)
        if any(lv):
            p = balance(midtones=lv)
            return "colorBalance", p, balance_text(p)
        shift = hsb_shift(hp, hp - dh, L, C)  # 中间调管不到（很暗 / 很亮的皮肤）：改用色相/饱和度转色相
        hue = int(np.sign(shift) or (1 if kind == "偏红橙" else -1)) * clip(abs(shift), 2, 15)
        idx, ramps = custom_range(ac.lab_hue_to_hsb_hue(hp, L=L, C=max(C, 8.0)))
        return ("hueSaturation", huesat(idx, ramps, hue=hue),
                f"{PS_RANGES[idx - 1][1]}（范围已对准问题色）色相 {hue:+d}（往{'黄' if hue > 0 else '红'}）")
    if kind == "缺色发灰":
        # 不用自然饱和度：PS 的自然饱和度会刻意少动肤色，加在皮肤上几乎没效果
        p = skin_dull_huesat(min(abs(ds), 0.5), lab)
        return "hueSaturation", p, f"{huesat_text(p)}（范围对准这块皮肤的颜色，只加饱和度、不改明暗）"
    if kind == "饱和度偏高":
        sat = -clip(abs(ds) * 60, 5, 25)
        return "hueSaturation", huesat(0, None, sat=sat, master=True), f"全图 饱和度 {sat}"
    return None


def patch_sat(ds: float) -> int:
    """偏色小块的饱和度修正：比周围更艳就降、更灰就加（与卡片文字一致），变化很小时不动。"""
    return clip(-ds * 60, -20, 20) if abs(ds) >= 0.05 else 0


def validate(spec: dict) -> None:
    """检查 layer 字典是否符合插件约定（测试和调试用）；不符合时抛 ValueError。"""
    kind, p = spec.get("kind"), spec.get("params")
    if kind not in LAYER_KINDS or not isinstance(p, dict) or spec.get("scope") not in ("skin", "local", "global"):
        raise ValueError(f"layer 格式不对：{spec}")

    def rng(v, lo, hi):
        if not isinstance(v, int) or not lo <= v <= hi:
            raise ValueError(f"{kind} 数值超出范围：{v}")

    if kind == "hueSaturation":
        for k, lo in (("hue", 180), ("saturation", 100), ("lightness", 100)):
            rng(p["master"][k], -lo, lo)
        seen = set()
        for r in p["ranges"]:
            rng(r["localRange"], 1, 6)
            if r["localRange"] in seen:
                raise ValueError("色相/饱和度的颜色范围编号重复")
            seen.add(r["localRange"])
            for k in ("beginRamp", "beginSustain", "endSustain", "endRamp"):
                rng(r[k], 0, 359)
            for k, lo in (("hue", 180), ("saturation", 100), ("lightness", 100)):
                rng(r[k], -lo, lo)
    elif kind == "colorBalance":
        for key, _ in CB_TONES:
            if len(p[key]) != 3:
                raise ValueError("色彩平衡每个色调要 3 个值")
            for v in p[key]:
                rng(v, -100, 100)
    elif kind == "vibrance":
        rng(p["vibrance"], -100, 100)
        rng(p["saturation"], -100, 100)
    elif kind == "curves":
        for ch, pts in p["channels"].items():
            if ch not in CURVE_CH or len(pts) < 2:
                raise ValueError(f"曲线通道不对：{ch}")
            xs = [pt[0] for pt in pts]
            if xs != sorted(set(xs)) or xs[0] != 0 or xs[-1] != 255:
                raise ValueError(f"曲线点不对：{pts}")
            for x, y in pts:
                rng(x, 0, 255)
                rng(y, 0, 255)
    elif kind == "selectiveColor":
        if p.get("method") != "relative" or not p["colors"]:
            raise ValueError("可选颜色格式不对")
        for name, inks in p["colors"].items():
            if name not in SC_COLORS or set(inks) != {"c", "m", "y", "k"}:
                raise ValueError(f"可选颜色的颜色不对：{name}")
            for v in inks.values():
                rng(v, -100, 100)
