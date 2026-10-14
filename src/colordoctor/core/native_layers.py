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
GRADING_TO_BALANCE = 0.6  # 颜色分级饱和度 30 ≈ 色彩平衡最强一项 18
S_CURVE = ((64, 58), (192, 198))  # 小幅 S 形加对比
MERGE_HSB = 12.0  # 两个混色器通道的颜色相距不到这么多度（HSB）时合成一个 PS 范围


def clip(v, lo, hi) -> int:
    return int(np.clip(round(float(v)), lo, hi))


def layer(kind: str, params: dict, scope: str = "global", name: str = "", at: str | None = None) -> dict:
    d = {"kind": kind, "params": params, "scope": scope, "name": name or KIND_NAME[kind]}
    if at:
        d["_at"] = at
    return d


def adjustment(kind: str, params: dict, text: str, scope: str, name: str, at: str | None = None,
               strength: str | None = "小幅", estimated: bool = True, tool: str | None = None) -> Adjustment:
    """一条带图层的建议：卡片文字 text 里的数字就是 params 里的数字。"""
    return Adjustment(tool or KIND_NAME[kind], text, strength, estimated, layer=layer(kind, params, scope, name, at))


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

    rows：[(通道名, HSB 中心, 色相度数, ACR 饱和度滑块, ACR 明亮度滑块)]。色相用真实要转的度数（不从滑块反推）；
    饱和度 ±100 与 PS 大致相当（上限 ±60）；明亮度在 PS 里会同时变灰，按一半换算、上限 ±30。
    返回 [(参数, 卡片文字)]，超过 6 个范围时分成多层。
    """
    # 两个通道的颜色实际落在几乎同一个色相上时，PS 的一个范围没法把它往两边拉：合成一个范围，数值取平均
    merged: list[list] = []
    for name, center, hue_deg, sat, lum in sorted(rows, key=lambda r: float(r[1]) % 360):
        c = float(center) % 360
        near = next((m for m in merged if abs(((c - m[1] + 180) % 360) - 180) < MERGE_HSB), None)
        if near is None:
            merged.append([[name], c, [hue_deg], [sat], [lum or 0], _acr_gap(name)])
        else:
            near[0].append(name)
            near[2].append(hue_deg)
            near[3].append(sat)
            near[4].append(lum or 0)
            near[5] = min(near[5], _acr_gap(name))
    items, words = [], []
    for names, c, hues, sats, lums, gap in merged:
        h = clip(np.mean(hues), -PS_HUE_CAP, PS_HUE_CAP)
        s = clip(np.mean(sats), -PS_SAT_CAP, PS_SAT_CAP)
        li = clip(np.mean(lums) * 0.5, -PS_LIGHT_CAP, PS_LIGHT_CAP)
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


def grading_to_balance(hue: float, sat: float) -> list[int]:
    """Camera Raw 颜色分级（HSB 色相 0–359，饱和度 0–100）-> 色彩平衡的 [青-红, 洋红-绿, 黄-蓝]。

    方向取色相在 R/G/B 三个方向上的分量，强度按饱和度 30 ≈ 最强一项 18 换算（估算）。
    """
    if sat <= 0:
        return [0, 0, 0]
    comp = np.cos(np.radians([hue, hue - 120.0, hue - 240.0]))
    comp = comp / max(1e-6, float(np.abs(comp).max()))
    return [clip(v * sat * GRADING_TO_BALANCE, -100, 100) for v in comp]


_TONE_L = {"shadows": 25.0, "midtones": 50.0, "highlights": 75.0}


def balance_for_shift(da: float, db: float, tone: str = "midtones", cap: int = 40) -> list[int]:
    """让 a*/b* 变化 (da, db) 需要的色彩平衡 [青-红, 洋红-绿, 黄-蓝]：只用洋红-绿（色调）和黄-蓝（色温）两个滑块。

    在该色调最典型明度的中性灰上用色彩平衡模型（ps_models，GIMP 算法）估算每个滑块每格改变多少 a*/b*，
    两个滑块各管各的（不按模型里的相互影响去抵消，那部分和 PS 实际差别较大，会把数值放大），估算值。
    """
    if abs(da) < 0.5 and abs(db) < 0.5:
        return [0, 0, 0]
    gray = cs.lab_to_srgb8(np.array([_TONE_L[tone], 0.0, 0.0]))[None, :] / 255.0
    base = cs.srgb8_to_lab(gray * 255.0)[0]
    per = []
    for i, axis in ((1, 1), (2, 2)):  # 洋红-绿 -> a*，黄-蓝 -> b*
        lv = [0, 0, 0]
        lv[i] = 20
        out = pm.color_balance(gray, **{tone: lv})
        per.append(float(cs.srgb8_to_lab(np.clip(out, 0, 1) * 255.0)[0][axis] - base[axis]) / 20.0)
    mg = da / per[0] if abs(per[0]) > 1e-3 and abs(da) >= 0.5 else 0.0
    yb = db / per[1] if abs(per[1]) > 1e-3 and abs(db) >= 0.5 else 0.0
    return [0, clip(mg, -cap, cap), clip(yb, -cap, cap)]


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


def skin_dull_selective(deficit: float) -> dict:
    """肤色发灰、不够透亮 -> 可选颜色：红色 / 黄色里减黑（红色再少一点青，颜色更干净）。deficit：低于下限的比例。"""
    n = clip(deficit * 80, 5, 25)
    return selective({"reds": {"c": -max(2, n // 2), "k": -n}, "yellows": {"k": -max(3, round(n * 0.6))}})


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
    if kind == "偏红橙":
        cr = -clip(abs(dh) / 2, 2, 15)
        yb = -clip(abs(dh) / 3, 1, 10)  # 负值 = 往黄
        return "colorBalance", balance(midtones=(cr, 0, yb)), f"中间调 青-红 {cr}（往青），黄-蓝 {yb}（往黄）"
    if kind == "偏青绿":
        cr = clip(abs(dh) / 2, 3, 15)
        mg = -clip(cr / 2, 1, 8)
        return "colorBalance", balance(midtones=(cr, mg, 0)), f"中间调 青-红 +{cr}（往红），洋红-绿 {mg}（往洋红）"
    if kind == "缺色发灰":
        vib = clip(abs(ds) * 60, 8, 30)
        return "vibrance", {"vibrance": vib, "saturation": 0}, f"自然饱和度 +{vib}"
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
