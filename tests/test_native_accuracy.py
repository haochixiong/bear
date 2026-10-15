"""PS 原生图层的数值是否真的把颜色往目标推、推多少：用一个小的、独立的 PS 调整模型（本文件自带）检查。

模型是近似的（色彩平衡用 GIMP 公开算法，色相/饱和度用 HSL），只用来抓“方向反了 / 修两次 / 过头一倍 / 根本不起作用”
这类问题；容差都放得比较宽。
"""

import numpy as np
import pytest
from synth import face_scene, flat, lab_color, noisy, skin_region
from test_layer_plan_native import landscape

from colordoctor.bridge import server as bs
from colordoctor.core import analysis as an
from colordoctor.core import color_spaces as cs
from colordoctor.core import layer_plan as lp
from colordoctor.core import native_layers as nl
from colordoctor.core import palette as pl
from colordoctor.core.color_management import synthetic_context

CTX = synthetic_context()


# ------------------------------------------------------------------ 自带的 PS 模型（sRGB 0–1）
def _hsl(x):
    mx, mn = x.max(-1), x.min(-1)
    li = (mx + mn) / 2
    d = mx - mn
    s = np.where(d == 0, 0.0, d / np.where(li <= 0.5, np.maximum(mx + mn, 1e-9), np.maximum(2 - mx - mn, 1e-9)))
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    sd = np.where(d == 0, 1, d)
    h = np.where(mx == r, ((g - b) / sd) % 6, np.where(mx == g, (b - r) / sd + 2, (r - g) / sd + 4))
    return np.where(d == 0, 0.0, h * 60.0) % 360, s, li


def _rgb(h, s, li):
    q = np.where(li < 0.5, li * (1 + s), li + s - li * s)
    p = 2 * li - q

    def f(t):
        t = t % 1.0
        return np.where(t < 1 / 6, p + (q - p) * 6 * t, np.where(t < 0.5, q, np.where(t < 2 / 3, p + (q - p) * (2 / 3 - t) * 6, p)))

    hh = h / 360.0
    return np.stack([f(hh + 1 / 3), f(hh), f(hh - 1 / 3)], -1)


def _ramp(h, b0, b1, e0, e1):
    t = (h - b0) % 360
    B1, E0, E1 = (b1 - b0) % 360, (e0 - b0) % 360, (e1 - b0) % 360
    return np.where(t < B1, t / max(B1, 1e-6), np.where(t <= E0, 1.0, np.where(t < E1, (E1 - t) / max(E1 - E0, 1e-6), 0.0)))


def _hs_one(x, hue, sat, light):
    h, s, li = _hsl(x)
    v = light / 100
    return _rgb((h + hue) % 360, np.clip(s * (1 + sat / 100 if sat <= 0 else 1 / max(1 - sat / 100, 1e-3)), 0, 1), li * (1 + v) if v < 0 else li + v * (1 - li))


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


def color_balance(x, p):
    _, _, li = _hsl(x)
    a, b = 0.25, 0.333
    w = (np.clip((li - b) / -a + 0.5, 0, 1),
         np.clip((li - b) / a + 0.5, 0, 1) * np.clip((li + b - 1) / -a + 0.5, 0, 1),
         np.clip((li + b - 1) / a + 0.5, 0, 1))
    lv = [np.asarray(p[t], float) / 100 for t in ("shadows", "midtones", "highlights")]
    out = np.stack([np.clip(x[..., i] + sum(wt * v[i] for wt, v in zip(w, lv)) * 0.7, 0, 1) for i in range(3)], -1)
    h2, s2, _ = _hsl(out)
    return _rgb(h2, s2, li)  # 保留明度


def curves(x, p):
    """PS 曲线：平滑的三次样条穿过各点（不是点之间连直线），用和生成曲线时同一个近似。"""
    from colordoctor.core.tone_class import spline_lut

    def ap(v, pts):
        return np.interp(v * 255, np.arange(256), spline_lut(pts)) / 255

    x = x.copy()
    for i, ch in enumerate(("red", "green", "blue")):
        if ch in p["channels"]:
            x[..., i] = ap(x[..., i], p["channels"][ch])
    if "composite" in p["channels"]:
        x = ap(x, p["channels"]["composite"])
    return x


MODEL = {"hueSaturation": hue_sat, "colorBalance": color_balance, "curves": curves}


def apply_layers(rgb8, layers):
    x = np.asarray(rgb8, float).reshape(-1, 3) / 255.0
    for ly in layers:
        x = MODEL[ly.kind](x, ly.params)
    return np.clip(x, 0, 1) * 255.0


def lab_of(rgb):
    return cs.srgb8_to_lab(np.asarray(rgb, float).reshape(-1, 3))


def lchlab(L, C, h):
    return (L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h)))


def morandi(H=200, W=240):
    L = np.zeros((H, W, 3))
    L[:H // 2] = (70, 4, 6)
    L[H // 2:] = (60, -3, 4)
    L[:, :W // 3] = (65, 6, -4)
    L[..., 0] += np.linspace(-15, 15, H)[:, None]
    return cs.lab_to_srgb8(L)


def portrait_outdoor(H=200, W=240):
    rng = np.random.default_rng(0)
    L = np.zeros((H, W, 3))
    yy, xx = np.linspace(0, 1, H)[:, None], np.linspace(0, 1, W)[None, :]
    L[:] = lchlab(50, 30, 125)
    L[:int(H * 0.35)] = lchlab(72, 22, 255)
    face = ((xx - 0.5) / 0.18) ** 2 + ((yy - 0.55) / 0.3) ** 2 <= 1
    L[face] = (68, 14, 19)
    L[..., 0] = L[..., 0] + (0.5 - yy) * 25 + (0.5 - xx) * 10
    L[..., 1:] += rng.normal(0, 0.8, L[..., 1:].shape)
    return cs.lab_to_srgb8(L)


def gray_scene():
    ramp = np.repeat(np.linspace(30, 230, 240)[None, :, None], 60, 0).repeat(3, 2).astype(np.uint8)
    return np.concatenate([portrait_outdoor(), ramp], 0)


def shifted(img, da, db):
    lab = lab_of(img)
    lab[:, 1] += da
    lab[:, 2] += db
    return cs.lab_to_srgb8(lab).reshape(img.shape)


def hue_ramp_image(C, L0=8.0, L1=95.0, h0=20.0, h1=80.0, H=200, W=200):
    hh = np.linspace(h0, h1, W)[None, :] * np.ones((H, 1))
    L = np.linspace(L0, L1, H)[:, None] * np.ones((1, W))
    return cs.lab_to_srgb8(np.stack([L, C * np.cos(np.radians(hh)), C * np.sin(np.radians(hh))], -1))


# ------------------------------------------------------------------ 1. 肤色偏色 + 目标风格：只修一次
@pytest.mark.parametrize("skin_lab", [(66, 9, 28), (65, 24, 12)])
def test_skin_cast_and_style_curve_do_not_double_correct(skin_lab):
    img = face_scene(skin_lab=skin_lab)[0]
    rep = an.analyze_skin_region(img, CTX, preset_key="natural")
    assert any(f.problem.startswith("肤色整体") for f in rep.findings)
    layers = lp.plan_from_report(rep, img.shape).layers
    px = img[rep.valid_mask]
    tgt_h = rep.skin.target_hue
    h0 = cs.lab_to_lch(lab_of(px).mean(0))[2]
    h1 = cs.lab_to_lch(lab_of(apply_layers(px, layers)).mean(0))[2]
    assert abs(cs.hue_diff(h1, tgt_h)) < 0.35 * abs(cs.hue_diff(h0, tgt_h))  # 修好，而不是修过头到另一边（以前 72°→41°）


# ------------------------------------------------------------------ 2. 同一张卡片上二选一的色相/饱和度不叠加
@pytest.mark.parametrize("img,family,target", [(landscape(), "绿", (50, -35, 30)), (portrait_outdoor(), "蓝", (60, -5, -35))])
def test_unity_alternatives_give_one_huesat(img, family, target):
    rep = an.analyze_palette_report(img, CTX, family=family, family_target_lab=target)
    plan = lp.plan_from_report(rep, img.shape)
    i = next(i for i, f in enumerate(rep.findings) if "统一度" in f.problem)
    glob = [ly for ly in plan.for_finding(i) if ly.kind == "hueSaturation" and ly.mask is None]
    assert len(glob) == 1, [ly.reason for ly in glob]
    if family == "绿":  # 绿色像素实际转的角度不超过一条建议的量（以前两层叠成 45°）
        x = img.reshape(-1, 3) / 255.0
        h, s, _ = _hsl(x)
        g = (h > 70) & (h < 110) & (s > 0.15)
        h2, _, _ = _hsl(hue_sat(x[g], glob[0].params))
        moved = float(np.median((h2 - h[g] + 180) % 360 - 180))
        assert 0 < moved <= nl.PS_HUE_CAP + 1


def test_overlapping_huesat_never_merged_or_stacked_in_one_card():
    a = nl.layer("hueSaturation", nl.huesat(3, (46, 76, 106, 136), hue=27), "global")
    b = nl.layer("hueSaturation", nl.huesat(2, (67, 81, 97, 111), hue=27), "global")
    c = nl.layer("hueSaturation", nl.huesat(5, (195, 225, 255, 285), hue=-10), "global")
    assert lp._merge_pair(a, b) is None  # 重叠：不合并
    assert lp._merge_pair(a, c) is not None  # 不重叠：照常合并

    class F:
        adjustments = (nl.adjustment("hueSaturation", a["params"], "甲 色相 +27", "global", "甲"),
                       nl.adjustment("hueSaturation", b["params"], "乙 色相 +27", "global", "乙"))

    skipped = []
    out = lp._finding_layers(F, skipped)
    assert len(out) == 1 and skipped and skipped[0][0] == "乙 色相 +27"


# ------------------------------------------------------------------ 3. 整体缺色：对比正常时不说“对比偏低”；没用的影调曲线不给
def test_dull_but_contrasty_image_gets_no_contrast_curve():
    img = hue_ramp_image(9)  # 亮暗差（P95−P5）约 79
    rep = an.analyze_palette_report(img, CTX)
    dull = next(f for f in rep.findings if f.problem.startswith("整体缺色"))
    assert not any("对比偏低" in a.instruction for a in dull.adjustments)
    assert not any(ly.kind == "curves" for ly in lp.plan_from_report(rep, img.shape).layers)


def test_dull_flat_image_gets_contrast_only_from_the_tone_style():
    """颜色卡不改明暗：亮暗差小只提示去「影调风格」里选；选了长调的影调风格，曲线只有影调卡那一条。"""
    img = hue_ramp_image(9, L0=42, L1=66)  # 亮暗差约 22
    rep = an.analyze_palette_report(img, CTX)
    dull = next(f for f in rep.findings if f.problem.startswith("整体缺色"))
    assert not any(a.layer and a.layer["kind"] == "curves" for a in dull.adjustments)
    assert any("影调风格" in a.instruction for a in dull.adjustments)
    rep = an.analyze_palette_report(img, CTX, style_key="kodak_ultramax400", tone_style="cinematic")
    curves_ = [ly for ly in lp.plan_from_report(rep, img.shape).layers if ly.kind == "curves"]
    assert len(curves_) == 1 and curves_[0].name.startswith("影调 ")


@pytest.mark.parametrize("style", ["kodak_portra400", "kodak_portra160", "fuji_pro_neg_std"])
def test_no_negligible_or_contradicting_tone_curves(style):
    img = hue_ramp_image(9)
    rep = an.analyze_palette_report(img, CTX, style_key=style)
    plan = lp.plan_from_report(rep, img.shape)
    curves_ = [ly for ly in plan.layers if ly.kind == "curves"]
    assert len(curves_) <= 1
    for ly in curves_:
        assert max(abs(y - x) for x, y in ly.params["channels"]["composite"]) >= 3


# ------------------------------------------------------------------ 4. 选不出皮肤 / 位置时不生成整张图的图层
def test_skin_layers_without_skin_mask_are_not_emitted():
    img = noisy(flat(lab_color(60, -30, 20), 150, 150), 4)  # 没有皮肤：自动抠皮肤失败
    rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    res = bs.analyze_request(rgba, "skin", {}, bs.Store(), bs.Store())
    assert any(f["problem"].startswith("肤色整体") for f in res["findings"])
    assert all(ly["has_mask"] for ly in res["plan"]["layers"])
    notes = [a for f in res["findings"] for a in f["adjustments"] if lp.SKIP_NO_MASK in a["instruction"]]
    assert notes and all(a["layer"] is None for a in notes)  # 建议还在（文字），只是没有图层
    for f in res["findings"]:
        assert f["layers"] == [ly["id"] for ly in res["plan"]["layers"] if ly["finding"] == f["index"]]


def test_plan_never_turns_scoped_layers_into_global_ones():
    img = face_scene(skin_lab=(66, 6, 8))[0]
    for rep in (an.analyze_skin_region(img, CTX, auto_mask=False), an.analyze_skin_region(img, CTX)):
        specs = [(i, s) for i, f in enumerate(rep.findings) for s, _ in lp._finding_layers(f)]
        plan = lp.plan_from_report(rep, img.shape)
        assert len(plan.layers) + len({(i, t) for i, t, _ in plan.skipped}) >= len(specs)
        for ly in plan.layers:
            spec = next(s for i, s in specs if i == ly.finding and s["params"] == ly.params)
            assert spec["scope"] == "global" or ly.mask is not None


def test_compare_mode_layers_are_limited_to_the_problem_region():
    rep = an.compare_regions(skin_region(68, 14, 19), skin_region(64, 18, 14, seed=3), CTX, skin=True)
    plan = lp.plan_from_report(rep, (120, 120, 3))
    assert plan.layers and all(ly.mask is not None for ly in plan.layers)


# ------------------------------------------------------------------ 5. 整图扫描：各处的蒙版互不重叠
def test_scan_spot_masks_are_disjoint():
    rng = np.random.default_rng(1)
    img = noisy(flat(lab_color(55, -30, 35), 600, 800), 2)
    for i in range(100):
        y, x = rng.integers(0, 580), rng.integers(0, 780)
        if i % 2:  # 一小片 4×4 像素的杂色点（大图按区域平均降采样，1 个像素的点会被平均掉）
            for dy in (0, 8):
                for dx in (0, 8):
                    img[y + dy:y + dy + 4, x + dx:x + dx + 4] = lab_color(60, 30, -30)
        else:
            img[y:y + 18, x:x + 18] = lab_color(55, -8, 10)
    rep = an.scan_image(img, CTX)
    plan = lp.plan_from_report(rep, img.shape)
    spots = [ly for ly in plan.layers if ly.id.startswith("opposite")]
    assert len(spots) >= 2
    for a in range(len(spots)):
        for b in range(a + 1, len(spots)):
            assert not ((spots[a].mask > 0) & (spots[b].mask > 0)).any(), (spots[a].id, spots[b].id)
    assert all(ly.mask.max() == 255 for ly in spots)


# ------------------------------------------------------------------ 6/7. 色彩平衡的数值：在这一段的实际颜色上求，不过头、不白调
def _band_tint_change(img, style_key):
    rep = an.analyze_palette_report(img, CTX, style_key=style_key)
    cb = [ly for ly in lp.plan_from_report(rep, img.shape).layers if ly.kind == "colorBalance"]
    sub, _, lab, lch = pl._prep(img, pl.MAX_SAMPLES)
    L, C = lch[..., 0].ravel(), lch[..., 1].ravel()
    lab2 = lab.reshape(-1, 3)
    bands = pl.tone_bands(L, sub)
    style = pl.get_style(style_key)
    out = []
    for tone, band, key in (("shadows", bands["shadow"], "shadow_tint"), ("highlights", bands["highlight"], "highlight_tint")):
        t = None if band is None else pl._band_tint(lab2, C, band, L)
        if t is None or not cb or not any(cb[0].params[tone]):
            out.append((tone, None, None))
            continue
        lim8 = cs.neutral_chroma_limit(L, pl.TINT_NEUTRAL_C)
        neutral = lim8 if pl._tint(lab2, band & (C < lim8), 0.05, band) else cs.neutral_chroma_limit(L, pl.TINT_MAX_C)
        sel = band & (C < neutral)
        lo, hi = style[key]["hue"]
        gh, gc = pl._center(lo, hi), float(np.mean(style[key]["chroma"]))
        need = np.array([gc * np.cos(np.radians(gh)), gc * np.sin(np.radians(gh))]) - t[2][1:]
        got = lab_of(color_balance(sub.reshape(-1, 3)[sel] / 255.0, cb[0].params) * 255).mean(0)[1:] - t[2][1:]
        out.append((tone, float(got @ need / (need @ need)), float(L[sel].mean())))
    return out


@pytest.mark.parametrize("img,style", [(face_scene()[0], "pink_blue"), (face_scene()[0], "teal_orange"),
                                       (morandi(), "black_gold"), (morandi(), "french")])
def test_style_color_balance_reaches_the_target_without_overshoot(img, style):
    res = _band_tint_change(img, style)
    assert any(r is not None for _, r, _ in res)
    for tone, ratio, L in res:
        if ratio is not None:
            assert 0.75 <= ratio <= 1.25, (tone, ratio, L)  # 以前 1.3–2.1 倍


def test_style_shadow_tint_not_judged_without_real_shadows():
    img = morandi()  # 最暗 15% 的明度约 L* 55：这是浅灰，不是阴影（以前按“阴影色调”给了一条曲线去染它）
    rep = an.analyze_palette_report(img, CTX, style_key="pink_blue")
    style = next(f for f in rep.findings if f.problem.startswith("与颜色「"))
    cb = [a for a in style.adjustments if a.layer and a.layer["kind"] == "colorBalance"]
    assert not cb or not any(cb[0].layer["params"]["shadows"])
    assert not any(a.instruction.startswith("阴影") for a in style.adjustments if not a.tool.startswith("常用做法"))
    assert any("几乎没有真正的阴影" in g for g in style.evidence)


def _skin_band(base, where, da, db):
    img, truth = face_scene(skin_lab=base)
    lab = lab_of(img).reshape(img.shape)
    Ls = lab[..., 0][truth]
    thr = np.percentile(Ls, 80 if where == "hl" else 20)
    sel = truth & ((lab[..., 0] >= thr) if where == "hl" else (lab[..., 0] <= thr))
    lab[sel, 1] += da
    lab[sel, 2] += db
    return cs.lab_to_srgb8(lab)


@pytest.mark.parametrize("base,where,da,db", [((45, 12, 16), "sh", -6, 0), ((50, 14, 19), "sh", 0, 8), ((66, 14, 19), "hl", 5, -6)])
def test_skin_band_cast_balance_pulls_band_back_to_midtones(base, where, da, db):
    img = _skin_band(base, where, da, db)
    rep = an.analyze_skin_region(img, CTX)
    plan = lp.plan_from_report(rep, img.shape)
    word = "高光" if where == "hl" else "阴影"
    i = next(i for i, f in enumerate(rep.findings) if f.problem.startswith(f"肤色{word}"))
    cb = [ly for ly in plan.for_finding(i) if ly.kind == "colorBalance"]
    assert cb
    px = img[rep.valid_mask]
    lab = lab_of(px)
    L = lab[:, 0]
    p15, p40, p60, p85 = np.percentile(L, [15, 40, 60, 85])
    band = L >= p85 if where == "hl" else L <= p15
    mid = (L >= p40) & (L <= p60)
    out = lab_of(color_balance(px / 255.0, cb[0].params) * 255)
    rel0 = lab[band].mean(0)[1:] - lab[mid].mean(0)[1:]
    rel1 = out[band].mean(0)[1:] - out[mid].mean(0)[1:]
    assert np.hypot(*rel1) < 0.6 * np.hypot(*rel0)  # 这一段相对中间调的偏色明显变小
    assert float((rel0 - rel1) @ rel0) / float(rel0 @ rel0) <= 1.3  # 不修过头


def test_skin_shadow_cast_out_of_reach_gives_no_layer():
    img = _skin_band((66, 14, 19), "sh", -6, 0)  # 阴影这一段 L* 约 62：色彩平衡“阴影”几乎不起作用
    rep = an.analyze_skin_region(img, CTX)
    i = next(i for i, f in enumerate(rep.findings) if f.problem.startswith("肤色阴影"))
    assert not lp.plan_from_report(rep, img.shape).for_finding(i)
    assert any("不建议用色彩平衡" in a.instruction for a in rep.findings[i].adjustments)


# ------------------------------------------------------------------ 8/9/10. 仿色：白平衡 + 阴影/高光 + 混色器
def _bands(img):
    _, _, lab, _ = pl._prep(img)
    lab = lab.reshape(-1, 3)
    p = np.percentile(lab[:, 0], [15, 85])
    neutral = np.hypot(lab[:, 1], lab[:, 2]) < 12
    return {k: lab[m & neutral, 1:].mean(0) for k, m in (("all", np.ones(len(lab), bool)), ("sh", lab[:, 0] <= p[0]),
                                                          ("hl", lab[:, 0] >= p[1]))}


@pytest.mark.parametrize("da,db", [(3, 8), (-2, -8), (6, 0)])
def test_reference_color_balance_matches_neutral_casts_per_band(da, db):
    mine = gray_scene()
    ref = shifted(mine, da, db)
    rep = an.compare_reference(mine, ref, CTX)
    cb = [ly for ly in lp.plan_from_report(rep, mine.shape).layers if ly.kind == "colorBalance"]
    assert cb
    after = np.rint(apply_layers(mine, cb)).astype(np.uint8).reshape(mine.shape)
    want, start, got = _bands(ref), _bands(mine), _bands(after)
    for k in want:
        d0, d1 = np.hypot(*(start[k] - want[k])), np.hypot(*(got[k] - want[k]))
        assert d1 < max(1.0, 0.35 * d0), (k, start[k], got[k], want[k])  # 每一段都往参考靠（高光以前反而远离）


def test_white_balance_uses_the_full_jacobian():
    """洋红-绿也会改 b*、黄-蓝也会改 a*（在 sRGB 里是真的：G−6 → a* +4.4、b* −3.3）；只要 a* 时 b* 不能跟着跑。"""
    for da, db in ((6, 0), (0, -8), (-4, 4)):
        lv = nl.balance_for_shift(da, db, "midtones", lab=(50, 0, 0))
        gray = cs.lab_to_srgb8(np.array([50.0, 0, 0]))[None].astype(float)
        d = lab_of(color_balance(gray / 255.0, nl.balance(midtones=lv)) * 255)[0] - lab_of(gray)[0]
        assert np.hypot(d[1] - da, d[2] - db) <= 0.25 * np.hypot(da, db), (da, db, lv, d)


def _mean_de(a, b):
    return float(np.mean(cs.delta_e_2000(lab_of(a), lab_of(b))))


@pytest.mark.parametrize("da,db", [(3, 8), (-2, -8), (6, 0)])
def test_reference_balance_and_huesat_do_not_double_correct(da, db):
    mine = gray_scene()
    ref = shifted(mine, da, db)
    rep = an.compare_reference(mine, ref, CTX)
    layers = [ly for ly in lp.plan_from_report(rep, mine.shape).layers if ly.kind in ("colorBalance", "hueSaturation")]
    after = np.rint(apply_layers(mine, layers)).astype(np.uint8).reshape(mine.shape)
    start, end = _mean_de(mine, ref), _mean_de(after, ref)
    assert end < 0.5 * start, (start, end)  # 以前偏冷的参考叠加后反而更远（5.1 → 6.8）
    assert an.compare_reference(after, ref, CTX).reference.similarity > rep.reference.similarity


# ------------------------------------------------------------------ 11. 缺色时不给“杂色降饱和”
def test_ratio_desaturation_not_given_when_image_lacks_colour():
    img = hue_ramp_image(10, L0=45, L1=62, h0=0, h1=300)
    rep = an.analyze_palette_report(img, CTX, style_key="kodak_ultramax400")
    assert any(f.problem.startswith("整体缺色") for f in rep.findings)
    assert an._ratio_layers(rep.palette, rep.palette.ratio)  # 不缺色时这张图会给“杂色降饱和”
    plan = lp.plan_from_report(rep, img.shape)
    assert not [ly for ly in plan.layers if ly.name.startswith("杂色降饱和")]
    assert [ly for ly in plan.layers if ly.kind == "vibrance"]


# ------------------------------------------------------------------ 12. 肤色发灰：真的加色度
@pytest.mark.parametrize("skin_lab", [(66, 6, 8), (60, 8, 10)])
def test_dull_skin_layer_adds_the_missing_chroma(skin_lab):
    img = face_scene(skin_lab=skin_lab)[0]
    rep = an.analyze_skin_region(img, CTX)
    i = next(i for i, f in enumerate(rep.findings) if f.problem.startswith("肤色发灰"))
    ly, = lp.plan_from_report(rep, img.shape).for_finding(i)
    assert ly.kind == "hueSaturation" and ly.mask is not None
    sk = rep.skin
    lo = sk.expected_saturation[0]
    need = 1 / (1 - (lo - sk.saturation_ab) / lo)
    px = img[rep.valid_mask]
    a, b = lab_of(px).mean(0), lab_of(apply_layers(px, [ly])).mean(0)
    ratio = np.hypot(*b[1:]) / np.hypot(*a[1:])
    assert abs((ratio - 1) - (need - 1)) <= 0.25 * (need - 1) + 0.03, (ratio, need)  # 以前可选颜色只加了 3%
    assert abs(b[0] - a[0]) < 1.0 and abs(cs.hue_diff(cs.lab_to_lch(a)[2], cs.lab_to_lch(b)[2])) < 2
