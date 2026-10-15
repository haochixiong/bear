"""每一个生成的调整图层，方向和幅度都要对：系统地遍历“场景 × 风格 × 问题类型”。

做法：用独立的 PS 调整模型（色相/饱和度含自定义范围、色彩平衡、曲线、可选颜色“相对”、自然饱和度）把图层按它的蒙版
作用到图上，再量一次卡片说的那个量（肤色色相、某一段的色调、某块皮肤的偏移、杂色的色度、亮暗差……），检查：
1. 往卡片说的目标走（离目标更近）；
2. 不冲过头（越过目标的部分不超过模型误差）；
3. 卡片上的方向词（往红 / 往黄 / 往青 / 提亮 / 压暗 / 降 / 加……）和数值的正负一致。
模型是近似的，容差放得比较宽，只抓“方向反了 / 修过头 / 词和数对不上”。
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from synth import face_scene, flat, highkey_face_scene, lab_color, lch_color, lowkey_face_scene, noisy, sprinkle
from test_layer_plan_native import landscape, teal_orange
from test_native_accuracy import _hsl, _ramp, _rgb, _skin_band, color_balance, curves, morandi, portrait_outdoor

from colordoctor.core import analysis as an
from colordoctor.core import color_spaces as cs
from colordoctor.core import contamination as ct_mod
from colordoctor.core import layer_plan as lp
from colordoctor.core import palette as pl
from colordoctor.core import portrait_analysis as pa
from colordoctor.core import ps_models as pm
from colordoctor.core import reference_match as rm
from colordoctor.core import skin_presets as sp
from colordoctor.core.color_management import synthetic_context
from colordoctor.core.region_statistics import weighted_circular_hue
from colordoctor.core.sampling import grid_downsample

CTX = synthetic_context()
STYLES = [s["key"] for s in pl.load_styles()]


# ------------------------------------------------------------------ PS 模型（sRGB 0–1）
def _hs_one(x, hue, sat, light):
    """PS 色相/饱和度：HSL 里转色相、乘饱和度；明度是往黑 / 往白混合（“加黑 / 加白”，颜色同时变淡），
    不是保持 HSL 饱和度只改 HSL 明度（那样浅色压暗时会变得非常艳）。"""
    h, s, li = _hsl(x)
    y = _rgb((h + hue) % 360, np.clip(s * (1 + sat / 100 if sat <= 0 else 1 / max(1 - sat / 100, 1e-3)), 0, 1), li)
    v = light / 100
    return y * (1 + v) if v < 0 else y + (1 - y) * v


def hue_sat(x, p):
    m = p["master"]
    if any(m.values()):
        x = _hs_one(x, m["hue"], m["saturation"], m["lightness"])
    h, _, _ = _hsl(x)
    colored = (x.max(-1) - x.min(-1)) > 1e-9
    out = x.copy()
    for r in p["ranges"]:
        w = _ramp(h, r["beginRamp"], r["beginSustain"], r["endSustain"], r["endRamp"]) * colored
        out = out + (_hs_one(x, r["hue"], r["saturation"], r["lightness"]) - x) * w[..., None]
    return np.clip(out, 0, 1)


def vibrance(x, p):
    """自然饱和度：越不饱和的颜色加得越多（PS 还会少动肤色，这里不模拟，方向一致即可）。"""
    h, s, li = pm._rgb_to_hsl(x)
    v = p["vibrance"] / 100.0
    s2 = np.clip(s * (1 + v * (1 - s)), 0, 1) if v >= 0 else s * (1 + v)
    s2 = np.clip(s2 * (1 + p["saturation"] / 100.0), 0, 1)
    return pm._hsl_to_rgb(h, s2, li)


def selective(x, p):
    out = x.copy()
    for name, ink in p["colors"].items():
        out = out + (pm.selective_color(x, name, ink["c"], ink["m"], ink["y"], ink["k"], relative=True) - x)
    return np.clip(out, 0, 1)


MODEL = {"hueSaturation": hue_sat, "colorBalance": color_balance, "curves": curves, "vibrance": vibrance,
         "selectiveColor": selective}


def apply(img, plan, ly):
    """按图层蒙版（羽化后的灰度）把调整作用到图上。"""
    x = img.reshape(-1, 3).astype(float) / 255.0
    y = np.clip(MODEL[ly.kind](x, ly.params), 0, 1)
    m = plan.full_mask(ly.id)
    w = 1.0 if m is None else (m.reshape(-1, 1).astype(float) / 255.0)
    return np.clip(np.rint((x + (y - x) * w) * 255), 0, 255).astype(np.uint8).reshape(img.shape)


def lab_of(img):
    return cs.srgb8_to_lab(img)


def mean_lab(img, sel=None):
    lab = lab_of(img).reshape(-1, 3)
    return lab if sel is None else lab[np.asarray(sel).reshape(-1)]


def hue(lab_mean):
    return float(np.degrees(np.arctan2(lab_mean[2], lab_mean[1])) % 360)


def circ_spread(lab):
    C = np.hypot(lab[:, 1], lab[:, 2])
    keep = C >= 5
    _, _, s = weighted_circular_hue(np.degrees(np.arctan2(lab[keep, 2], lab[keep, 1])) % 360, C[keep])
    return float(s or 0.0)


# ------------------------------------------------------------------ 判定
class Check:
    def __init__(self, tag):
        self.tag, self.bad = tag, []

    def fail(self, msg):
        self.bad.append(f"{self.tag}: {msg}")

    def toward(self, what, v0, v1, target, slack=1.0, min_need=None):
        """标量：离目标更近，且越过目标的部分不超过 max(slack, 30% 需要量)。"""
        d0, d1 = target - v0, target - v1
        if abs(d0) < (slack if min_need is None else min_need):
            return
        if abs(d1) >= abs(d0):
            self.fail(f"{what} 没有往目标走：{v0:.2f} → {v1:.2f}（目标 {target:.2f}）")
        elif d0 * d1 < 0 and abs(d1) > max(slack, 0.3 * abs(d0)):
            self.fail(f"{what} 冲过头：{v0:.2f} → {v1:.2f}（目标 {target:.2f}）")

    def toward_vec(self, what, v0, v1, target, slack=1.0, hi=1.35):
        """二维（a*, b*）：沿需要方向的完成比例在 (0, hi]，剩余差距变小。"""
        v0, v1, target = (np.asarray(v, float)[:2] for v in (v0, v1, target))
        need = target - v0
        n2 = float(need @ need)
        if np.sqrt(n2) < slack:
            return
        ratio = float((v1 - v0) @ need) / n2
        if ratio <= 0.05:
            self.fail(f"{what} 没有往目标走：{np.round(v0, 2)} → {np.round(v1, 2)}（目标 {np.round(target, 2)}）")
        elif ratio > hi and np.hypot(*(target - v1)) > slack:
            self.fail(f"{what} 冲过头（完成 {ratio:.0%}）：{np.round(v0, 2)} → {np.round(v1, 2)}（目标 {np.round(target, 2)}）")

    def up(self, what, v0, v1, cap=None, eps=0.0):
        if not v1 > v0 + eps:
            self.fail(f"{what} 应该变大：{v0:.3f} → {v1:.3f}")
        if cap is not None and v1 > cap:
            self.fail(f"{what} 超过上限 {cap:.3f}：{v0:.3f} → {v1:.3f}")

    def down(self, what, v0, v1, floor=None, eps=0.0):
        if not v1 < v0 - eps:
            self.fail(f"{what} 应该变小：{v0:.3f} → {v1:.3f}")
        if floor is not None and v1 < floor:
            self.fail(f"{what} 低于下限 {floor:.3f}：{v0:.3f} → {v1:.3f}")


# ------------------------------------------------------------------ 卡片文字：方向词与数值
CB_RE = re.compile(r"([青洋红黄绿蓝]{1,2})-([青洋红黄绿蓝]{1,2}) ([+-]\d+)（往([青洋红黄绿蓝]{1,2})）")
HUE_RE = re.compile(r"色相 ([+-]\d+)（往(黄|红)）")
SAT_RE = re.compile(r"饱和度略(降|加)（约 ([+-]\d+)）")
CURVE_RE = re.compile(r"输入 (\d+) → 输出 (\d+)")
SEL_RE = re.compile(r"颜色选“([^”]+)”：([^；（]+)")
INK = {"青色": "c", "洋红": "m", "黄色": "y", "黑色": "k"}


def check_text(chk: Check, text: str, layer: dict | None):
    for left, right, v, word in CB_RE.findall(text):
        want = right if int(v) > 0 else left
        if word != want:
            chk.fail(f"色彩平衡方向词不对：“{left}-{right} {v}（往{word}）”应为往{want}")
    for v, word in HUE_RE.findall(text):
        if (int(v) > 0) != (word == "黄"):
            chk.fail(f"色相方向词不对：“色相 {v}（往{word}）”")
    for word, v in SAT_RE.findall(text):
        if (int(v) < 0) != (word == "降"):
            chk.fail(f"饱和度方向词不对：“饱和度略{word}（约 {v}）”")
    if not layer:
        return
    p = layer["params"]
    if layer["kind"] == "curves":
        pts = {tuple(pt) for ch in p["channels"].values() for pt in ch}
        for a, b in CURVE_RE.findall(text):
            if (int(a), int(b)) not in pts and int(a) not in (0, 255):
                chk.fail(f"曲线文字的点 {a}→{b} 不在图层参数里：{p}")
    elif layer["kind"] == "vibrance":
        m = re.search(r"自然饱和度 ([+-]?\d+)", text)
        if m and int(m.group(1)) != p["vibrance"]:
            chk.fail(f"自然饱和度文字 {m.group(1)} 与图层 {p['vibrance']} 不一致")
    elif layer["kind"] == "selectiveColor":
        names = {v: k for k, v in {"reds": "红色", "yellows": "黄色", "greens": "绿色", "cyans": "青色", "blues": "蓝色",
                                   "magentas": "洋红"}.items()}
        for cname, parts in SEL_RE.findall(text):
            inks = p["colors"].get(names.get(cname, ""), {})
            for ink, v in re.findall(r"(青色|洋红|黄色|黑色) ([+-]\d+)%", parts):
                if inks.get(INK[ink]) != int(v):
                    chk.fail(f"可选颜色文字“{cname} {ink} {v}%”与图层 {inks} 不一致")


TONE_WORDS = (("提亮", 1), ("压深", -1), ("压暗", -1), ("抬起", 1))


def check_tone_words(chk: Check, reason: str, params: dict):
    """影调曲线：说“提亮 / 抬起”的要有往上的点，说“压深 / 压暗”的要有往下的点。"""
    pts = params["channels"].get("composite", [])
    said = re.sub(r"不(提亮|压深|压暗|抬起)", "", reason)  # “只压深暗部，不提亮”里的“不提亮”不算
    for word, sign in TONE_WORDS:
        if word in said and not any((y - x) * sign > 0 for x, y in pts):
            chk.fail(f"曲线说“{word}”，但没有对应方向的点：{pts}")


# ------------------------------------------------------------------ 各类问题的检查
def skin_mean(img, valid):
    return mean_lab(img, valid).mean(axis=0)


def check_skin_report(tag, img, rep, preset=None, other_lab=None):
    chk = Check(tag)
    plan = lp.plan_from_report(rep, img.shape)
    valid = rep.valid_mask
    for i, f in enumerate(rep.findings):
        for a in f.adjustments:
            check_text(chk, a.instruction, a.layer)
        for ly in plan.for_finding(i):
            after = apply(img, plan, ly)
            c = Check(f"{tag} [{f.problem}] {ly.name}")
            p = f.problem
            if p.startswith("肤色整体"):
                cur, new = skin_mean(img, valid), skin_mean(after, valid)
                tgt = rep.skin.target_hue
                d0, d1 = cs.hue_diff(hue(cur), tgt), cs.hue_diff(hue(new), tgt)
                c.toward("肤色色相（到目标的角度差）", float(d0), float(d1), 0.0, slack=1.5)
                if abs(new[0] - cur[0]) > 2.5:
                    c.fail(f"只改色相，明度却变了 {new[0] - cur[0]:+.1f}")
            elif p.startswith("肤色发灰"):
                s0 = rep.skin.saturation_ab
                s1 = pa.analyze_skin(after, valid=valid).saturation_ab
                c.up("肤色饱和度 C*/L*", s0, s1, cap=rep.skin.expected_saturation[1])
            elif p.startswith(("肤色高光", "肤色阴影")):
                r0, r1 = _band_rel(img, after, valid, p.startswith("肤色高光"))
                c.toward_vec(f"{p[2:4]}相对中间调的 (a*, b*)", r0, r1, (0.0, 0.0), slack=1.5)
            elif p.startswith("肤色不均匀"):
                check_patch(c, img, after, rep, ly)
            elif p.startswith(("与目标风格", "问题区相对参考区")) or "的肤色差异" in p:
                tgt = (preset.target_for(rep.skin.mean_lab) if p.startswith("与目标风格") else other_lab)
                cur, new = skin_mean(img, valid), skin_mean(after, valid)
                check_toward_lab(c, cur, new, tgt, ly)
            elif "杂色" in p:
                check_contamination(c, img, after, rep, ly)
            elif p.startswith("色相杂乱"):
                c.down("皮肤色相离散", circ_spread(mean_lab(img, valid)), circ_spread(mean_lab(after, valid)))
            else:
                c.fail("没有对应的检查（新的问题类型？）")
            chk.bad += c.bad
    return chk.bad


def _band_rel(before, after, valid, highlight):
    """皮肤最亮 / 最暗 15% 相对中间调（40–60%）的 (a*, b*)：分段按调整前的明度定，前后用同一批像素。"""
    lab0, lab1 = mean_lab(before, valid), mean_lab(after, valid)
    L = lab0[:, 0]
    p15, p40, p60, p85 = np.percentile(L, [15, 40, 60, 85])
    band = L >= p85 if highlight else L <= p15
    mid = (L >= p40) & (L <= p60)
    return (lab0[band, 1:].mean(0) - lab0[mid, 1:].mean(0), lab1[band, 1:].mean(0) - lab1[mid, 1:].mean(0))


def check_toward_lab(c: Check, cur, new, tgt, ly):
    """拉向目标 Lab 的曲线：颜色曲线管 a*/b*，RGB 复合曲线管明度。"""
    ch = ly.params["channels"] if ly.kind == "curves" else {}
    if ly.kind == "curves" and set(ch) == {"composite"}:
        c.toward("明度 L*", cur[0], new[0], tgt[0], slack=1.0)
        return
    c.toward_vec("(a*, b*)", cur[1:], new[1:], tgt[1:], slack=1.0)
    if ly.kind == "curves" and abs(new[0] - cur[0]) > 2.0:
        c.fail(f"颜色曲线把明度也改了 {new[0] - cur[0]:+.1f}（明度由单独的 RGB 复合曲线管）")
    tgt = np.array([cur[0], tgt[1], tgt[2]])  # 只比颜色
    if float(cs.delta_e_2000(new, tgt)) >= float(cs.delta_e_2000(cur, tgt)):
        c.fail(f"离目标的 ΔE00 没有变小：{np.round(cur, 1)} → {np.round(new, 1)}（目标 {np.round(tgt, 1)}）")


def check_patch(c: Check, img, after, rep, ly):
    kind = ly.name.split(" ")[0][2:]  # 图层名“肤色偏黄 · 色相/饱和度”
    uni = rep.uniformity
    ps = [p for p in uni.patches if p.kind == kind]
    if not ps:
        c.fail(f"找不到“{kind}”的小块")
        return
    valid = rep.valid_mask
    box = np.zeros(valid.shape, bool)
    for p in ps:
        for x, y, w, h in p.cells or (p.bbox,):
            box[y:y + h, x:x + w] = True
    inside, rest = box & valid, ~box & valid
    if inside.sum() < 20 or rest.sum() < 20:
        return
    ref = mean_lab(img, rest)
    lab0, lab1 = mean_lab(img, inside), mean_lab(after, inside)
    m0, m1 = lab0.mean(0), lab1.mean(0)
    if kind in ("偏黄", "偏红橙", "偏洋红", "偏青绿"):
        h_ref = weighted_circular_hue(np.degrees(np.arctan2(ref[:, 2], ref[:, 1])) % 360, np.hypot(ref[:, 1], ref[:, 2]))[0]
        c.toward("这块皮肤相对其他皮肤的色相差", float(cs.hue_diff(h_ref, hue(m0))), float(cs.hue_diff(h_ref, hue(m1))), 0.0,
                 slack=1.5)
    else:
        def sat(lab):
            return float(np.median(np.hypot(lab[:, 1], lab[:, 2]) / np.maximum(lab[:, 0], 1)))
        near = ref[np.abs(ref[:, 0] - m0[0]) <= 10]
        c.toward("这块皮肤的饱和度 C*/L*", sat(lab0), sat(lab1), sat(near if len(near) >= 20 else ref), slack=0.01)


def check_contamination(c: Check, img, after, rep, ly):
    flags = lp._contamination_flags(rep, img.shape)
    m = lp.LayerPlan("", img.shape[1], img.shape[0], [ly]).full_mask(ly.id)
    opp = flags == ct_mod.MASK_OPPOSITE
    if m is not None:
        opp &= m > 127
    if opp.sum() < 3:
        return
    C0 = np.hypot(*mean_lab(img, opp)[:, 1:].T).mean()
    C1 = np.hypot(*mean_lab(after, opp)[:, 1:].T).mean()
    c.down("杂色的色度", C0, C1, eps=0.5)
    base = (flags == ct_mod.MASK_NORMAL) & ((m > 127) if m is not None else True)
    if base.sum() >= 20:
        B0 = np.hypot(*mean_lab(img, base)[:, 1:].T).mean()
        B1 = np.hypot(*mean_lab(after, base)[:, 1:].T).mean()
        if B1 < 0.85 * B0:
            c.fail(f"降杂色时把主色的色度也降了 {1 - B1 / B0:.0%}")


def check_region_report(tag, img, rep):
    chk = Check(tag)
    plan = lp.plan_from_report(rep, img.shape)
    flags = lp._contamination_flags(rep, img.shape)
    for i, f in enumerate(rep.findings):
        for a in f.adjustments:
            check_text(chk, a.instruction, a.layer)
        for ly in plan.for_finding(i):
            after = apply(img, plan, ly)
            c = Check(f"{tag} [{f.problem}] {ly.name}")
            m = plan.full_mask(ly.id)
            where = (m > 127) if m is not None else np.ones(img.shape[:2], bool)
            if ly.kind == "hueSaturation" and ("对立色" in f.problem or "脏色" in f.problem):
                check_contamination(c, img, after, rep, ly)
            elif ly.kind == "vibrance":
                gray = (flags == ct_mod.MASK_GRAYING) & where
                if gray.sum() >= 5:
                    C0 = np.hypot(*mean_lab(img, gray)[:, 1:].T).mean()
                    C1 = np.hypot(*mean_lab(after, gray)[:, 1:].T).mean()
                    clean = (flags == ct_mod.MASK_NORMAL)
                    cap = np.percentile(np.hypot(*mean_lab(img, clean)[:, 1:].T), 95) * 1.15 if clean.sum() > 20 else None
                    c.up("发灰处的色度", C0, C1, cap=cap)
            elif ly.kind == "curves":
                c.up("局部明度的标准差（对比）", mean_lab(img, where)[:, 0].std(), mean_lab(after, where)[:, 0].std())
            elif ly.kind == "hueSaturation" and f.problem.startswith("色相杂乱"):
                c.down("色相离散", circ_spread(mean_lab(img, where)), circ_spread(mean_lab(after, where)))
            else:
                c.fail(f"没有对应的检查：{ly.kind}")
            chk.bad += c.bad
    return chk.bad


# ------------------------------------------------------------------ 整图色彩：风格 / 缺色 / 6:3:1 / 统一
def _tint_ab(t):
    return None if t is None else np.array([t[1] * np.cos(np.radians(t[0])), t[1] * np.sin(np.radians(t[0]))])


def _tint_goal(cur_ab, spec):
    if spec["hue"] == [0, 360]:
        c = np.hypot(*cur_ab)
        return cur_ab * min(1.0, spec["chroma"][1] / max(c, 1e-6))
    gh, gc = pl._center(*spec["hue"]), float(np.mean(spec["chroma"]))
    return np.array([gc * np.cos(np.radians(gh)), gc * np.sin(np.radians(gh))])


def _in_sector_dist(h, lo, hi):
    if pl._in_range(np.array([h]), lo, hi)[0]:
        return 0.0
    return float(min(abs(cs.hue_diff(h, lo)), abs(cs.hue_diff(h, hi))))


def _tone(img):
    L = lab_of(img)[..., 0].ravel()
    return np.percentile(L, [5, 50, 95])


def _tint_pixels(img):
    """style_match 量色调用的那批像素（阴影 / 高光里接近中性的）：调整前后在同一批像素上比较，
    否则调完以后“接近中性”的像素换了一批，量到的是别的东西。"""
    sub, _, lab, lch = pl._prep(img, pl.MAX_SAMPLES)
    L, C = lch[..., 0].ravel(), lch[..., 1].ravel()
    lab2 = lab.reshape(-1, 3)
    out = {}
    for tone, band in zip(("shadows", "highlights"), pl.tone_bands(L, sub).values(), strict=True):
        if band is None:
            continue
        for lim in (pl.TINT_NEUTRAL_C, pl.TINT_MAX_C):
            sel = band & (C < cs.neutral_chroma_limit(L, lim))
            if pl._tint(lab2, sel, 0.05, band) is not None:
                out[tone] = sel
                break
    return out


def check_style(c: Check, img, after, style, ly, m0):
    if ly.kind == "colorBalance":
        sel = _tint_pixels(img)
        lab0 = pl._prep(img, pl.MAX_SAMPLES)[2].reshape(-1, 3)
        lab1 = pl._prep(after, pl.MAX_SAMPLES)[2].reshape(-1, 3)
        for tone, key in (("shadows", "shadow_tint"), ("highlights", "highlight_tint")):
            if not any(ly.params[tone]):
                continue
            if tone not in sel:
                c.fail(f"{tone} 没有可量的色调，却给了色彩平衡")
                continue
            a0, a1 = lab0[sel[tone], 1:].mean(0), lab1[sel[tone], 1:].mean(0)
            c.toward_vec(f"{tone} 色调 (a*, b*)", a0, a1, _tint_goal(a0, style[key]), slack=1.0, hi=1.4)
    elif ly.kind == "hueSaturation":
        check_style_huesat(c, img, after, style, ly)
    else:
        c.fail(f"没有对应的检查：{ly.kind}")


def check_style_huesat(c: Check, img, after, style, ly):
    sectors = {s["name"]: tuple(s["hue"]) for s in style.get("sectors", [])}
    centers = {}  # 范围中心（HSB）-> 通道名
    for names, cen in re.findall(r"([^：；（）]+)（范围对准色相 (\d+)°）", ly.reason):
        centers[int(cen)] = [n.strip() for n in names.split("、")]
    x = img.reshape(-1, 3).astype(float) / 255.0
    hsb, _, _ = pm._rgb_to_hsl(x)
    hsb = hsb * 360.0
    lab0, lab1 = lab_of(img).reshape(-1, 3), lab_of(after).reshape(-1, 3)
    for r in ly.params["ranges"]:
        core = ((hsb - r["beginSustain"]) % 360 <= (r["endSustain"] - r["beginSustain"]) % 360) & ((x.max(1) - x.min(1)) > 0.03)
        if core.sum() < 20:
            continue
        cen = int(round((r["beginSustain"] + ((r["endSustain"] - r["beginSustain"]) % 360) / 2) % 360))
        names = centers.get(cen) or next((v for k, v in centers.items() if abs(((k - cen + 180) % 360) - 180) <= 1), "")
        if r["hue"]:
            # 几个通道落在同一个色相上时合成一个范围（数值取平均）：往其中任何一个通道的目标色区走都算对
            acts = [style.get("channel_targets", {}).get(n, "") for n in names]
            secs = [sectors[a.split(":", 1)[1]] for a in acts if a.startswith("toward:")]
            if not secs:
                c.fail(f"范围 {cen}° 转了色相，但对应通道 {names} 都不是 toward")
                continue
            h0, h1 = hue(lab0[core].mean(0)), hue(lab1[core].mean(0))
            moved = [(lo, hi) for lo, hi in secs if _in_sector_dist(h1, lo, hi) < _in_sector_dist(h0, lo, hi)
                     or _in_sector_dist(h0, lo, hi) <= 2]
            if not moved:
                c.fail(f"{'、'.join(names)} 色相没有往目标色区靠：{h0:.0f}° → {h1:.0f}°（目标 {secs}）")
            for lo, hi in moved:
                mid = pl._center(lo, hi)
                if _in_sector_dist(h1, lo, hi) > 2 and np.sign(cs.hue_diff(h0, mid)) != np.sign(cs.hue_diff(h1, mid)):
                    c.fail(f"{'、'.join(names)} 色相冲过了目标色区：{h0:.0f}° → {h1:.0f}°（目标 {lo}–{hi}°）")
        if r["saturation"] < 0:
            c.down(f"范围 {cen}° 的色度", np.hypot(*lab0[core, 1:].T).mean(), np.hypot(*lab1[core, 1:].T).mean())
        if r["lightness"] < 0 and not lab1[core, 0].mean() < lab0[core, 0].mean():
            c.fail(f"范围 {cen}° 明度应该变暗")


def check_palette_report(tag, img, rep, style_key=None, family_target=None):
    chk = Check(tag)
    plan = lp.plan_from_report(rep, img.shape)
    style = pl.get_style(style_key) if style_key else None
    m0 = rep.palette.style
    cf0 = rep.palette.colorfulness.value
    for i, f in enumerate(rep.findings):
        for a in f.adjustments:
            check_text(chk, a.instruction, a.layer)
        for ly in plan.for_finding(i):
            after = apply(img, plan, ly)
            c = Check(f"{tag} [{f.problem}] {ly.name}")
            p = f.problem
            if p.startswith("影调："):
                check_tone_style(c, img, after, rep, ly)
            elif ly.kind == "curves":
                c.fail("明暗只由“影调”那张卡改：其他卡片（颜色、缺色）不应该有曲线图层")
            elif p.startswith("整体缺色"):
                c.up("色彩丰富度", cf0, pl.colorfulness(after).value, cap=style["colorfulness"][1] + 5 if style else None)
            elif p.startswith("颜色过艳"):
                c.down("色彩丰富度", cf0, pl.colorfulness(after).value, floor=style["colorfulness"][0] - 5)
            elif p.startswith("色彩比例"):
                check_ratio_desat(c, img, after, rep, ly)
            elif "统一度" in p:
                check_unity(c, img, after, rep, ly, family_target)
            elif p.startswith("与颜色「"):
                check_style(c, img, after, style, ly, m0)
            else:
                c.fail("没有对应的检查（新的问题类型？）")
            chk.bad += c.bad
    return chk.bad


def check_tone_style(c: Check, img, after, rep, ly):
    """影调卡的曲线：明度分位点整体更接近目标影调，影调（九宫格上）不会离目标更远；文字说的方向和曲线一致。"""
    from colordoctor.core import tone_class as tcl

    if ly.kind != "curves":
        c.fail(f"影调卡只应该有曲线：{ly.kind}")
        return
    goal = rep.palette.tone_target
    tq = np.array(tcl.target_quantiles(goal))
    q0, q1 = np.array(tcl.quantiles(img)), np.array(tcl.quantiles(after))
    c.down("分位点与目标影调的平均差", np.abs(q0 - tq).mean(), np.abs(q1 - tq).mean())
    n0, n1 = tcl.classify(img).name, tcl.classify(after).name
    if tcl.tone_distance(n1, goal) > tcl.tone_distance(n0, goal):
        c.fail(f"影调离目标更远了：{n0} → {n1}（目标 {goal}）")
    check_tone_words(c, ly.reason, ly.params)


def check_ratio_desat(c: Check, img, after, rep, ly):
    lab0, lab1 = lab_of(img).reshape(-1, 3), lab_of(after).reshape(-1, 3)
    C0, C1 = np.hypot(lab0[:, 1], lab0[:, 2]), np.hypot(lab1[:, 1], lab1[:, 2])
    changed = np.abs(C1 - C0) > 0.5
    if changed.sum() >= 5:
        c.down("降饱和那些颜色的色度", C0[changed].mean(), C1[changed].mean())
    for role in rep.palette.ratio.roles:  # 主色、辅色、点缀色不动
        if role.chroma < 8:
            continue
        near = cs.delta_e_2000(lab0, np.array(role.lab)) < 6
        if near.sum() >= 20 and C1[near].mean() < 0.9 * C0[near].mean():
            c.fail(f"{role.family}（主/辅/点缀色）也被降了饱和：{C0[near].mean():.1f} → {C1[near].mean():.1f}")


def check_unity(c: Check, img, after, rep, ly, target):
    u = rep.palette.unity
    lo, hi = pl.FAMILY_SECTORS[u.family]
    lab0, lab1 = lab_of(img).reshape(-1, 3), lab_of(after).reshape(-1, 3)

    def fam(lab):
        C = np.hypot(lab[:, 1], lab[:, 2])
        h = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360
        hh = np.where(h < lo - 12, h + 360, h)
        return (C >= pl.CHROMA_MIN) & (hh >= lo - 12) & (hh <= hi + 12)

    sel = fam(lab0)
    at = next((a.layer.get("_at") for a in rep.findings[ly.finding].adjustments if a.layer and a.layer["name"] == ly.name), None)
    if at and at.startswith("unity@"):
        p = u.patches[int(at.split("@")[1])]
        x, y, w, h = p.bbox
        box = np.zeros(img.shape[:2], bool)
        box[y:y + h, x:x + w] = True
        box = box.reshape(-1) & sel
        rest = sel & ~box
        if box.sum() < 10 or rest.sum() < 10:
            return
        h_ref = hue(lab0[rest].mean(0))
        c.toward("这一块相对整体的色相差", float(cs.hue_diff(h_ref, hue(lab0[box].mean(0)))),
                 float(cs.hue_diff(h_ref, hue(lab1[box].mean(0)))), 0.0, slack=2.0)
    elif "往目标" in ly.name and target is not None:
        de0 = float(cs.delta_e_2000(lab0[sel].mean(0), np.asarray(target)))
        de1 = float(cs.delta_e_2000(lab1[sel].mean(0), np.asarray(target)))
        if not de1 < de0:
            c.fail(f"{u.family}离目标的 ΔE00 没有变小：{de0:.1f} → {de1:.1f}")
    else:
        c.down(f"{u.family}的色相离散", circ_spread(lab0[sel]), circ_spread(lab1[sel]) - 0.3)


# ------------------------------------------------------------------ 仿色 / 对比
def _ref_bands(img):
    """仿色比较用的像素：整体 / 阴影 / 高光里接近中性的（与 reference_match 相同的选法）。"""
    sub, _, lab, _ = pl._prep(img)
    lab = lab.reshape(-1, 3)
    ok = ~cs.clipped(sub.reshape(-1, 3))
    bands = pl.tone_bands(lab[:, 0], sub)
    out = {"midtones": rm._neutral(lab, ok)}
    for tone, key in (("shadows", "shadow"), ("highlights", "highlight")):
        if bands[key] is not None:
            out[tone] = rm._neutral(lab, bands[key])
    return lab, out


def _channel_gap(mine, ref):
    """混色器通道（两边都占 ≥ 3%）的平均色相差、饱和度差（按参考占比加权）。"""
    from colordoctor.core import acr_channels as ac
    a = {c.name: c for c in ac.channel_stats(pl._prep(mine)[0])}
    b = {c.name: c for c in ac.channel_stats(pl._prep(ref)[0])}
    rows = [(a[n], b[n]) for n in a if a[n].share >= 0.03 and b[n].share >= 0.03 and a[n].mean_hue is not None
            and b[n].mean_hue is not None]
    if not rows:
        return 0.0, 0.0
    w = np.array([y.share for _, y in rows])
    dh = np.array([abs(cs.hue_diff(x.mean_hue, y.mean_hue)) for x, y in rows])
    ds = np.array([abs(x.mean_sat - y.mean_sat) for x, y in rows])
    return float(np.average(dh, weights=w)), float(np.average(ds, weights=w))


def check_reference_report(tag, mine, ref, rep):
    chk = Check(tag)
    plan = lp.plan_from_report(rep, mine.shape)
    m0 = rep.reference
    t_ref = np.array(m0.tone_ref)
    for f in rep.findings:
        for a in f.adjustments:
            check_text(chk, a.instruction, a.layer)
    lab_r, sel_r = _ref_bands(ref)
    cb = next((ly for ly in plan.layers if ly.kind == "colorBalance"), None)
    tone = next((ly for ly in plan.layers if ly.kind == "curves"), None)
    curved = apply(mine, plan, tone) if tone is not None else mine  # 颜色都按“曲线之后”的图算（建议顺序：先曲线）
    lab_m, sel_m = _ref_bands(curved)
    for ly in plan.layers:
        c = Check(f"{tag} {ly.name}")
        if ly.kind == "curves":
            m1 = rm.match_reference(apply(mine, plan, ly), ref)
            d0 = np.abs(np.array(m0.tone_mine) - t_ref).mean()
            d1 = np.abs(np.array(m1.tone_mine) - t_ref).mean()
            c.down("影调分位数与参考的平均差", d0, d1)
        elif ly.kind == "colorBalance":
            for tone in ("midtones", "shadows", "highlights"):
                if not any(ly.params[tone]) or tone not in sel_m or tone not in sel_r:
                    continue
                if sel_m[tone].sum() < 20 or sel_r[tone].sum() < 20:
                    continue
                # 中间调（白平衡）单独看；阴影 / 高光是补中间调没修到的部分，看整层
                p = {**ly.params, **{t: [0, 0, 0] for t in ("shadows", "highlights")}} if tone == "midtones" else ly.params
                one = lp.LayerSpec(ly.id, ly.name, ly.kind, p, ly.reason)
                after = apply(curved, lp.LayerPlan("", mine.shape[1], mine.shape[0], [one]), one)
                lab1 = pl._prep(after)[2].reshape(-1, 3)
                c.toward_vec(f"{tone} 中性色 (a*, b*)", lab_m[sel_m[tone], 1:].mean(0), lab1[sel_m[tone], 1:].mean(0),
                             lab_r[sel_r[tone], 1:].mean(0), slack=1.0, hi=1.5)
        elif ly.kind == "hueSaturation":
            base = apply(curved, plan, cb) if cb is not None and "色彩平衡" in ly.reason else curved
            after = apply(base, plan, ly)
            (h0, s0), (h1, s1) = _channel_gap(base, ref), _channel_gap(after, ref)
            if h1 > h0 + 1.0 or s1 > s0 + 1.0:
                c.fail(f"混色器通道离参考更远：色相差 {h0:.1f} → {h1:.1f}，饱和度差 {s0:.1f} → {s1:.1f}")
        chk.bad += c.bad
    after = mine
    for ly in plan.layers:  # 全部图层一起：相似度不降
        after = apply(after, plan, ly)
    sim1 = rm.match_reference(after, ref).similarity
    if sim1 < m0.similarity - 1.0:
        chk.fail(f"全部图层叠加后相似度变低：{m0.similarity:.1f} → {sim1:.1f}")
    return chk.bad


def _skin_means(before, after):
    """an._skin_pixels 选出的同一批皮肤像素（同一个取样网格）在调整前后的平均 Lab。"""
    sub0, _ = grid_downsample(before, 20_000)
    sub1, _ = grid_downsample(after, 20_000)
    px0, px1 = sub0.reshape(-1, 3), sub1.reshape(-1, 3)
    lch = cs.lab_to_lch(cs.srgb8_to_lab(px0))
    sel = (lch[:, 1] >= 3) & (lch[:, 2] <= 110) & (lch[:, 0] >= 10) & (lch[:, 0] <= 98)
    if sel.sum() < 10:
        sel[:] = True
    return cs.srgb8_to_lab(px0[sel]).mean(0), cs.srgb8_to_lab(px1[sel]).mean(0)


def check_compare_report(tag, ref, prob, rep, skin):
    chk = Check(tag)
    plan = lp.plan_from_report(rep, prob.shape)
    ref_lab = (an._mean_lab(an._skin_pixels(ref)) if skin else lab_of(ref).reshape(-1, 3).mean(0))
    for i, f in enumerate(rep.findings):
        for a in f.adjustments:
            check_text(chk, a.instruction, a.layer)
        if not f.problem.startswith("问题区相对参考区"):
            continue
        for ly in plan.for_finding(i):
            after = apply(prob, plan, ly)
            c = Check(f"{tag} {ly.name}")
            if skin:
                cur, new = _skin_means(prob, after)
            else:
                cur, new = lab_of(prob).reshape(-1, 3).mean(0), lab_of(after).reshape(-1, 3).mean(0)
            check_toward_lab(c, cur, new, ref_lab, ly)
            chk.bad += c.bad
    return chk.bad


# ------------------------------------------------------------------ 场景
SKIN_LABS = [(68, 14, 19), (66, 9, 28), (65, 24, 12), (66, 6, 8), (50, 10, 12), (75, 6, 18), (66, 4, 20), (66, 20, 8),
             (66, -2, 18), (55, 16, 28), (40, 9, 20), (80, 6, 22), (34, 11, 14)]
PRESETS = [None, "natural", "japanese", "moody", "french", "deep", "cosplay", "hongkong"]


def _patchy(base, kind):
    img, truth = face_scene(skin_lab=base)
    lab = lab_of(img)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    spot = truth & ((xx - 80) ** 2 + (yy - 120) ** 2 <= 22 ** 2)
    L, a, b = lab[spot].T
    C, h = np.hypot(a, b), np.degrees(np.arctan2(b, a))
    dh, k = {"yellow": (14, 1.0), "red": (-14, 1.0), "magenta": (-32, 1.0), "green": (40, 0.8), "gray": (0, 0.5),
             "sat": (0, 1.6)}[kind]
    h, C = np.radians(h + dh), C * k
    lab[spot, 1], lab[spot, 2] = C * np.cos(h), C * np.sin(h)
    return cs.lab_to_srgb8(lab), truth


def skin_cases():
    for k, lab in enumerate(SKIN_LABS):
        img = face_scene(h=140, w=140, skin_lab=lab)[0]
        for pk in PRESETS if k % 3 == 0 else (None, "natural", PRESETS[2 + k % 6]):  # 每种肤色风格都跑到几种肤色
            yield f"skin{lab} preset={pk}", img, pk, None
    for pk in PRESETS:
        yield f"lowkey preset={pk}", lowkey_face_scene(h=150, w=140)[0], pk, None
        yield f"highkey preset={pk}", highkey_face_scene(h=150, w=140)[0], pk, None
    for args in (((45, 12, 16), "sh", -6, 0), ((50, 14, 19), "sh", 0, 8), ((66, 14, 19), "hl", 5, -6),
                 ((66, 14, 19), "hl", -4, 6), ((40, 10, 15), "sh", 5, -5)):
        yield f"band{args}", _skin_band(*args), None, None
    for kind in ("yellow", "red", "magenta", "green", "gray", "sat"):
        for base in ((66, 14, 19), (45, 12, 16)):
            yield f"patch-{kind}{base}", _patchy(base, kind)[0], None, None
    for col in (lch_color(60, 25, 200), lch_color(55, 30, 300), lch_color(60, 30, 140)):
        img, truth = face_scene()
        dirty = sprinkle(img, col, 0.03, seed=3)
        dirty[~truth] = img[~truth]
        yield f"dirty-skin{col}", dirty, None, None
    yield "face-vs-body", face_scene(skin_lab=(66, 18, 16))[0], None, ("身体", (64.0, 13.0, 20.0))
    img, truth = face_scene(skin_lab=(66, 14, 19))
    lab = lab_of(img)
    rng = np.random.default_rng(5)
    jitter = np.radians(np.where(rng.random(truth.shape) < 0.5, -25.0, 25.0) * (rng.random(truth.shape) < 0.5))
    C, h = np.hypot(lab[..., 1], lab[..., 2]), np.arctan2(lab[..., 2], lab[..., 1]) + jitter
    lab[truth, 1], lab[truth, 2] = (C * np.cos(h))[truth], (C * np.sin(h))[truth]
    yield "skin-hue-spread", cs.lab_to_srgb8(lab), None, None
    yield "face-vs-body-dark", lowkey_face_scene()[0], None, ("身体", (40.0, 11.0, 16.0))


@pytest.mark.parametrize("tag,img,preset,other", list(skin_cases()), ids=lambda v: v if isinstance(v, str) else "")
def test_skin_layers_move_toward_their_targets(tag, img, preset, other):
    rep = an.analyze_skin_region(img, CTX, preset_key=preset, part="脸部" if other else None, other_part=other)
    bad = check_skin_report(tag, img, rep, sp.get_preset(preset) if preset else None, other[1] if other else None)
    assert not bad, "\n".join(bad)


def region_cases():
    for base in ((60, 40, 30), (55, -30, 35), (70, 5, 5), (50, 0, -30), (30, 25, 20)):
        img = flat(lab_color(*base), 150, 150)
        for cont in ((60, -30, 20), (60, 30, -30)):
            yield f"dirty{base}{cont}", sprinkle(noisy(img, 2), lab_color(*cont), 0.02)
        g = noisy(img, 2).copy()
        g[40:90, 40:90] = lab_color(base[0], base[1] * 0.35, base[2] * 0.35)
        yield f"gray{base}", g
    sky = np.clip(np.rint(cs.lab_to_srgb_float(np.stack(np.broadcast_arrays(
        np.linspace(50, 80, 150)[:, None], np.full((1, 150), -8.0), np.full((1, 150), -30.0)), -1)) * 255), 0, 255).astype(np.uint8)
    yield "sky-spread", sprinkle(sky, lab_color(65, 10, -30), 0.08, seed=2)
    for h0 in (55, 140, 250):  # 同一块颜色里色相分散（一半像素往两侧各偏 ±32°）：色相收拢
        yield f"hue-spread{h0}", _hue_spread(h0)


def _hue_spread(h0, H=120, W=160, seed=4):
    rng = np.random.default_rng(seed)
    h = h0 + np.where(rng.random((H, W)) < 0.5, -32.0, 32.0) * (rng.random((H, W)) < 0.5) + rng.normal(0, 2, (H, W))
    lab = np.stack([np.full((H, W), 60.0), 30 * np.cos(np.radians(h)), 30 * np.sin(np.radians(h))], -1)
    return cs.lab_to_srgb8(lab)


def busy_palette(H=200, W=240):
    """主色 45%、辅色 25%，另有 5 种颜色各约 6%：6:3:1 里“其余颜色偏杂”。"""
    lab = np.zeros((H, W, 3))
    lab[:] = (55, -25, 30)
    lab[:, int(W * 0.45):int(W * 0.7)] = (60, 10, -35)
    for i, hh in enumerate((20, 75, 300, 180, 340)):
        x0 = int(W * 0.7) + i * int(W * 0.06)
        lab[:, x0:x0 + int(W * 0.06)] = (58, 35 * np.cos(np.radians(hh)), 35 * np.sin(np.radians(hh)))
    lab[..., 0] += np.linspace(-8, 8, H)[:, None]
    return cs.lab_to_srgb8(lab)


def patchy_grass(H=200, W=240):
    """一片草地，其中一块偏黄绿（色相 +16°）：“绿”统一度里的偏离小块。"""
    hh = np.full((H, W), 135.0)
    hh[60:120, 40:110] = 119.0
    L = 50 + np.linspace(-6, 6, H)[:, None] + np.zeros((1, W))
    return cs.lab_to_srgb8(np.stack([L, 35 * np.cos(np.radians(hh)), 35 * np.sin(np.radians(hh))], -1))


@pytest.mark.parametrize("tag,img", list(region_cases()), ids=lambda v: v if isinstance(v, str) else "")
def test_region_and_scan_layers_move_toward_their_targets(tag, img):
    bad = check_region_report(f"region {tag}", img, an.analyze_region(img, CTX))
    bad += check_region_report(f"scan {tag}", img, an.scan_image(img, CTX))
    assert not bad, "\n".join(bad)


IMAGES = {  # 整图色彩的场景（尺寸取小一些：每张图要对 30 种风格各分析、模拟一遍）
    "landscape": landscape(120, 144), "teal_orange": teal_orange(120, 144), "morandi": morandi(120, 144),
    "portrait": portrait_outdoor(120, 144), "face": face_scene(h=130, w=130)[0],
    "lowkey": lowkey_face_scene(h=143, w=130)[0], "highkey": highkey_face_scene(h=143, w=130)[0],
    "mono": np.repeat(np.clip(np.linspace(10, 245, 144), 0, 255).astype(np.uint8)[None, :, None], 120, 0).repeat(3, 2),
    "busy": busy_palette(120, 144), "grass": patchy_grass(120, 144),
}


FULL = ("landscape", "portrait", "lowkey", "highkey", "face")  # 这几张跑全部风格，其余的跑一部分（每种风格仍至少跑 5 张图）


@pytest.mark.parametrize("name", list(IMAGES))
def test_palette_layers_move_toward_every_style(name):
    img = IMAGES[name]
    bad = []
    k = list(IMAGES).index(name)
    for key in (STYLES if name in FULL else STYLES[k::5]) + [None]:
        bad += check_palette_report(f"{name} style={key}", img, an.analyze_palette_report(img, CTX, style_key=key), key)
    for fam, tl in (("绿", (50, -35, 30)), ("蓝", (60, -5, -35)), ("橙", None), ("橙", (66, 12, 24))):
        rep = an.analyze_palette_report(img, CTX, family=fam, family_target_lab=tl)
        bad += check_palette_report(f"{name} family={fam} target={tl}", img, rep, None, tl)
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("name", list(IMAGES))
def test_tone_style_curves_move_toward_the_target_tone(name):
    """每种影调风格、每种直接选的影调：曲线把明暗分布拉向目标影调。"""
    from colordoctor.core import tone_class as tcl

    img = IMAGES[name]
    bad = []
    for key in tcl.TONE_STYLES:
        bad += check_palette_report(f"{name} tone_style={key}", img, an.analyze_palette_report(img, CTX, tone_style=key))
    for t in tcl.TONE_NAMES:
        bad += check_palette_report(f"{name} tone={t}", img, an.analyze_palette_report(img, CTX, tone_target=t))
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("mine,ref", [("landscape", "teal_orange"), ("landscape", "highkey"), ("portrait", "landscape"),
                                      ("portrait", "teal_orange"), ("teal_orange", "morandi"), ("lowkey", "landscape"),
                                      ("lowkey", "highkey"), ("lowkey", "teal_orange"), ("face", "highkey"), ("face", "portrait"),
                                      ("highkey", "lowkey"), ("morandi", "portrait")])
def test_reference_layers_move_toward_the_reference(mine, ref):
    rep = an.compare_reference(IMAGES[mine], IMAGES[ref], CTX)
    bad = check_reference_report(f"ref {mine}<-{ref}", IMAGES[mine], IMAGES[ref], rep)
    assert not bad, "\n".join(bad)


def compare_cases():
    for a, b in (((68, 14, 19), (66, 9, 28)), ((68, 14, 19), (65, 24, 12)), ((45, 12, 16), (68, 14, 19)),
                 ((70, 12, 16), (60, 16, 26))):
        yield f"skin{a}<-{b}", S_skin(a), S_skin(b), True
    for a, b in (((55, -30, 35), (50, -20, 40)), ((60, 5, -30), (62, -5, -25)), ((40, 30, 20), (45, 20, 25))):
        yield f"color{a}<-{b}", noisy(flat(lab_color(*a), 80, 80), 2), noisy(flat(lab_color(*b), 80, 80), 2, seed=1), False


def S_skin(lab):
    from synth import skin_region
    return skin_region(*lab)


@pytest.mark.parametrize("tag,ref,prob,skin", list(compare_cases()), ids=lambda v: v if isinstance(v, str) else "")
def test_compare_layers_pull_problem_region_to_reference(tag, ref, prob, skin):
    rep = an.compare_regions(ref, prob, CTX, skin=skin)
    bad = check_compare_report(tag, ref, prob, rep, skin)
    assert not bad, "\n".join(bad)


# ------------------------------------------------------------------ 针对性的回归（上面的系统检查发现的问题）
def test_skin_patch_layer_only_touches_the_off_colour_cells():
    """偏色小块的图层只作用于真正偏离的小块；以前作用于整个外接框，框里正常的皮肤被往反方向推（偏黄那块 +8.5° → −4°）。"""
    img = _skin_band((45, 12, 16), "sh", -6, 0)
    rep = an.analyze_skin_region(img, CTX)
    plan = lp.plan_from_report(rep, img.shape)
    i = next(i for i, f in enumerate(rep.findings) if f.problem.startswith("肤色不均匀"))
    (ly,) = plan.for_finding(i)
    p = next(p for p in rep.uniformity.patches if ly.name.startswith(f"肤色{p.kind}"))
    cells = np.zeros(img.shape[:2], bool)
    for x, y, w, h in p.cells:
        cells[y:y + h, x:x + w] = True
    x, y, w, h = p.bbox
    box = np.zeros(img.shape[:2], bool)
    box[y:y + h, x:x + w] = True
    m = plan.full_mask(ly.id) > 127
    outside = box & ~cells & rep.valid_mask
    assert outside.sum() > 100  # 外接框里确实有不偏的皮肤
    assert (m & outside).sum() < 0.25 * outside.sum()


def test_merged_mixer_range_follows_the_channel_that_has_the_pixels():
    """两个通道落在同一个 PS 范围里时按占比合并：几乎没有像素的通道（黄色 0.1%，要转 +61°）不能把整片橙色带着转。"""
    (params, _), = nl_rows([("橙色", 21.6, 0.0, 0, 0, 0.47), ("红色", 19.2, 2.0, 0, 0, 0.21), ("黄色", 30.6, 61.0, 0, 0, 0.001)])
    assert abs(params["ranges"][0]["hue"]) <= 2


def test_ps_lightness_does_not_cancel_the_saturation_it_comes_with():
    """PS 色相/饱和度的“明度”是往白 / 往黑混合，颜色会变淡：饱和度要补回来，否则“饱和度 +40、明度 +20”反而更灰。"""
    (params, _), = nl_rows([("蓝色", 238.0, 0.0, 40, 40, 1.0)])
    r = params["ranges"][0]
    assert r["lightness"] == 20 and r["saturation"] > 40
    x = np.array([[0.2, 0.35, 0.75]])
    s0 = cs.lab_to_lch(cs.srgb8_to_lab(x * 255))[0, 1]
    s1 = cs.lab_to_lch(cs.srgb8_to_lab(hue_sat(x, params) * 255))[0, 1]
    assert s1 > s0  # 色度确实变高（与卡片上的“饱和度 +”一致）


def test_reference_mixer_is_measured_after_the_tone_curve():
    """仿色：影调曲线已经把亮度对齐以后，混色器不再按“曲线之前”的明度差再提亮一次（以前所有颜色 明度 +20）。"""
    land = landscape()
    lab = cs.srgb8_to_lab(land)
    lab[..., 0] -= 18
    darker = cs.lab_to_srgb8(lab)
    rep = an.compare_reference(darker, land, CTX)
    hs = [a.layer for a in rep.reference.adjustments if a.layer and a.layer["kind"] == "hueSaturation"]
    assert rep.reference.curve_points
    assert all(abs(r["lightness"]) <= 6 for h in hs for r in h["params"]["ranges"]), hs
    assert not [a for a in rep.reference.adjustments if a.tool == "Camera Raw 混色器" and "明亮度 +" in a.instruction]


def nl_rows(rows):
    from colordoctor.core import native_layers as nl
    return nl.mixer_rows_to_huesat(rows)
