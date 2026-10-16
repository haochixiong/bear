"""整图色彩：ACR 通道模型、色彩丰富度、6:3:1、颜色统一、风格模板。"""

import numpy as np
import pytest

from colordoctor.core import acr_channels as ac
from colordoctor.core import color_spaces as cs
from colordoctor.core import palette as pl
from synth import flat, lch_color, noisy


def blocks(colors_shares, h=200, w=300):
    """按比例横向拼接色块。"""
    img = np.zeros((h, w, 3), np.uint8)
    x = 0
    for color, share in colors_shares:
        n = int(round(share * w))
        img[:, x : x + n] = color
        x += n
    img[:, x:] = colors_shares[-1][0]
    return img


# ------------------------------------------------------------------ ACR 通道
def test_memberships_sum_to_one_and_interpolate():
    w = ac.memberships(np.array([0.0, 15.0, 30.0, 90.0, 350.0]))
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert w[0, 0] == pytest.approx(1.0)
    assert w[1, 0] == pytest.approx(0.5) and w[1, 1] == pytest.approx(0.5)
    assert w[3, 2] == pytest.approx(0.5) and w[3, 3] == pytest.approx(0.5)  # 黄 60 与 绿 120 之间
    assert w[4, 7] > 0 and w[4, 0] > 0  # 洋红-红 跨 0°


def test_channel_stats_orange_image():
    st = {c.name: c for c in ac.channel_stats(flat((230, 120, 20), 20, 20))}
    assert st["橙色"].share > 0.8
    assert st["蓝色"].share == 0


def test_slider_estimates():
    assert ac.hue_slider(30, 45) == 50
    assert ac.hue_slider(30, 0) == -100
    assert ac.sat_slider(40, 20) == -50
    assert ac.lum_slider(5) == 20


def test_tint_to_grading_teal_and_orange():
    h, s = ac.tint_to_grading(30, -6, -6)
    assert 170 <= h <= 230 and s > 0
    h, s = ac.tint_to_grading(80, 4, 8)
    assert 20 <= h <= 50


# ------------------------------------------------------------------ 色彩丰富度
def test_colorfulness_grey_vs_vivid():
    grey = pl.colorfulness(noisy(flat((128, 128, 128), 50, 50), 2))
    vivid = pl.colorfulness(blocks([((230, 30, 30), 0.3), ((30, 200, 40), 0.3), ((30, 60, 230), 0.4)]))
    assert grey.value < 15 and "灰" in grey.label
    assert vivid.value > 82


# ------------------------------------------------------------------ 6:3:1
def test_631_good_composition():
    img = blocks([((60, 90, 170), 0.6), ((235, 235, 235), 0.3), ((220, 40, 40), 0.1)])
    pal, r = pl.ratio_631(pl._prep(img)[2])
    assert [p.family for p in r.roles] == ["蓝", "白/浅灰", "红"]
    assert r.roles[0].share == pytest.approx(0.6, abs=0.02)
    assert r.score >= 90 and r.verdict == "接近 6:3:1"


def test_631_messy_composition_gives_tips():
    cols = [lch_color(55, 45, h) for h in (20, 70, 140, 200, 260, 320)]
    img = blocks([(c, 1 / 6) for c in cols])
    _, r = pl.ratio_631(pl._prep(img)[2])
    assert r.score < 55
    assert any("主色" in t for t in r.tips)
    assert any("其余颜色" in t for t in r.tips)


# ------------------------------------------------------------------ 颜色统一
def test_uniform_green_is_unified():
    img = noisy(flat(lch_color(55, 40, 135), 120, 160), 2)
    u = pl.family_unity(img, "绿")
    assert u.score > 85 and u.hue_spread < 6


def test_mixed_greens_are_flagged_and_located():
    img = flat(lch_color(55, 40, 118), 160, 200)  # 偏黄的绿
    img[:, 100:] = lch_color(55, 40, 160)  # 偏青的绿
    u = pl.family_unity(img, "绿")
    assert u.hue_spread > 15 and u.score < 60
    assert u.patches
    assert any("色相" in a.instruction for a in u.adjustments)


def test_family_unity_against_target():
    img = noisy(flat(lch_color(55, 40, 120), 100, 100), 1)
    u = pl.family_unity(img, "绿", target_lab=tuple(np.round(np.array([50, -30, 20]), 1)))
    assert u.delta_e_to_target is not None and u.delta_e_to_target > 3
    assert any("目标" in a.instruction for a in u.adjustments)


def test_family_absent():
    u = pl.family_unity(flat((40, 80, 200), 50, 50), "绿")
    assert u.share == 0 and "几乎没有" in u.adjustments[0].instruction


# ------------------------------------------------------------------ 风格模板
def test_styles_load():
    keys = {s["key"] for s in pl.load_styles()}
    assert {"teal_orange", "black_gold", "pink_blue", "morandi", "japanese_fresh", "film_warm", "forest"} <= keys


def _teal_orange_image():
    img = flat(lch_color(42, 30, 210), 160, 200)
    img[40:140, 60:140] = lch_color(66, 32, 60)
    return noisy(img, 2)


def test_teal_orange_image_matches_teal_orange_better_than_random_colors():
    good = pl.style_match(_teal_orange_image(), pl.get_style("teal_orange"))
    bad_img = blocks([(lch_color(55, 45, 140), 0.5), (lch_color(45, 45, 300), 0.5)])
    bad = pl.style_match(bad_img, pl.get_style("teal_orange"))
    assert good.on_target > 0.9
    assert bad.on_target < 0.2
    assert good.score > bad.score + 25
    assert any("目标之外" in g for g in bad.gaps)
    names = " ".join(a.instruction for a in bad.adjustments)
    assert "绿色" in names or "紫色" in names


def test_black_gold_checks_black_point_and_cold_colors():
    img = flat(lch_color(8, 2, 60), 160, 200)
    img[60:100, 40:160] = lch_color(70, 55, 80)  # 金色灯光
    good = pl.style_match(noisy(img, 1), pl.get_style("black_gold"))
    img2 = flat(lch_color(40, 35, 260), 160, 200)  # 大面积蓝色、黑位不深
    bad = pl.style_match(img2, pl.get_style("black_gold"))
    assert good.score > bad.score
    # 黑位 / 压暗属于影调：颜色模板不判断，只提示常配的影调风格
    assert not any("黑位" in g for g in bad.gaps)
    assert any("暗调情绪" in n for n in bad.notes)
    assert any("蓝色" in a.instruction and "饱和度" in a.instruction for a in bad.adjustments)


def test_style_shadow_tint_curves_suggestion():
    img = flat(lch_color(70, 30, 60), 160, 200)
    img[100:] = lch_color(20, 2, 60)  # 阴影基本中性
    m = pl.style_match(img, pl.get_style("teal_orange"))
    assert m.shadow_ok is False
    assert any(a.tool == "Camera Raw 颜色分级" and "阴影" in a.instruction for a in m.adjustments)
    assert any(a.tool == "曲线" and "阴影" in a.instruction for a in m.adjustments)


def test_analyze_palette_full():
    r = pl.analyze_palette(_teal_orange_image(), style_key="teal_orange", family="橙")
    assert r.style is not None and r.unity is not None
    assert len(r.tone) == 5
    assert r.ratio.roles


# ------------------------------------------------------------------ 风格库：常见风格 + 富士 / 柯达胶片
CHANNELS = {"红色", "橙色", "黄色", "绿色", "浅绿色", "蓝色", "紫色", "洋红"}
ACTIONS = {"keep", "desat", "desat_dark", "desat_light"}


def test_style_library_has_common_looks_and_film_stocks():
    styles = pl.load_styles()
    keys = {s["key"] for s in styles}
    assert {"french", "morandi", "forest", "japanese_fresh"} <= keys
    fuji = [s for s in styles if s["category"] == "富士胶片"]
    kodak = [s for s in styles if s["category"] == "柯达胶片"]
    assert len(fuji) >= 10 and len(kodak) >= 8
    assert {"fuji_classic_chrome", "fuji_classic_neg", "fuji_nostalgic_neg", "fuji_velvia"} <= keys
    assert {"kodak_portra400", "kodak_ektar100", "kodak_gold200", "kodak_kodachrome64"} <= keys
    assert len(keys) == len(styles)


@pytest.mark.parametrize("style", pl.load_styles(), ids=lambda s: s["key"])
def test_style_entries_are_well_formed(style):
    assert style["category"] in {"色彩风格", "富士胶片", "柯达胶片"}
    sectors = {s["name"] for s in style.get("sectors", [])}
    for sec in style.get("sectors", []):
        assert all(0 <= v <= 360 for v in sec["hue"])
    lo, hi = style["colorfulness"]
    assert 0 <= lo < hi
    if "contrast" in style:
        assert 0 < style["contrast"][0] < style["contrast"][1] <= 100
    for ch, act in style["channel_targets"].items():
        assert ch in CHANNELS
        assert act in ACTIONS or (act.startswith("toward:") and act.split(":", 1)[1] in sectors), (ch, act)
    for key in ("shadow_tint", "highlight_tint"):
        if key in style:
            assert len(style[key]["hue"]) == 2 and style[key]["chroma"][0] <= style[key]["chroma"][1]
    assert style["recipe"] and style["description"]
    if style["category"] != "色彩风格":
        assert style.get("basis"), "胶片模板必须写明依据"
    # 每个模板都能在任意图上跑通
    m = pl.style_match(_teal_orange_image(), style)
    assert 0 <= m.score <= 100 and m.category == style["category"]


def _gradient(lo, hi, h=120, w=160):
    L = np.linspace(lo, hi, w)[None, :].repeat(h, axis=0)
    lab = np.stack([L, np.full_like(L, 8.0), np.full_like(L, 14.0)], axis=-1)
    return cs.lab_to_srgb8(lab)


def test_film_color_template_leaves_blacks_and_contrast_to_the_tone_style():
    """胶片的黑位抬起、低反差属于影调：颜色模板（Portra）不再按黑位 / 对比打分或给曲线，交给「影调风格 · 胶片」。"""
    from colordoctor.core import tone_class as tc

    portra = pl.get_style("kodak_portra400")
    harsh = pl.style_match(_gradient(0, 100), portra)
    soft = pl.style_match(_gradient(12, 80), portra)
    for m in (harsh, soft):
        assert not any("黑位" in g or "对比度" in g for g in m.gaps)
        assert not any(a.tool == "曲线" and a.layer for a in m.adjustments)
        assert any("胶片" in n for n in m.notes)
    assert harsh.contrast > soft.contrast
    # 影调风格「胶片」（中短调）：黑场抬起、亮部收住
    pts, _ = tc.match_curve(tc.quantiles(_gradient(0, 100)), tc.target_quantiles("中短调"))
    assert pts[1][1] > pts[1][0] and pts[-2][1] < pts[-2][0]


def test_style_grading_not_suggested_when_tint_ok_or_neutral_spec():
    import numpy as np

    # 中性阴影 + 黑金（阴影色度 0–6，色相不限）：不应建议往阴影加任何色相
    L = np.linspace(4, 85, 200)[None, :].repeat(120, 0)
    lab = np.stack([L, np.zeros_like(L) + 0.8, np.zeros_like(L) - 0.6], -1)
    img = cs.lab_to_srgb8(lab)
    img[40:80, 120:180] = cs.lab_to_srgb8(np.array([70.0, 20.0, 60.0]))
    m = pl.style_match(img, pl.get_style("black_gold"))
    assert not [a for a in m.adjustments if a.tool == "Camera Raw 颜色分级" and a.instruction.startswith("阴影")]
    # 阴影明显偏色时，只减弱，不换色相
    lab2 = lab.copy()
    lab2[..., 1] = np.where(L < 25, 12.0, 0.0)
    m2 = pl.style_match(cs.lab_to_srgb8(lab2), pl.get_style("black_gold"))
    g = [a for a in m2.adjustments if a.tool == "Camera Raw 颜色分级" and a.instruction.startswith("阴影")]
    assert g
    if g:
        hue = dict(g[0].params)["色相"]
        assert 120 <= hue <= 240, hue  # 往青（红的反方向）减弱红色调


# ------------------------------------------------------------------ 建议的数值与方向（审计回归）
def _lchlab(L, C, h):
    return (L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h)))


def _outdoor_portrait(H=200, W=240):
    """草地 + 淡蓝天空（C* ≈ 22）+ 椭圆肤色。"""
    rng = np.random.default_rng(0)
    lab = np.zeros((H, W, 3))
    yy, xx = np.linspace(0, 1, H)[:, None], np.linspace(0, 1, W)[None, :]
    lab[:] = _lchlab(50, 30, 125)
    lab[: int(H * 0.35)] = _lchlab(72, 22, 255)
    lab[((xx - 0.5) / 0.18) ** 2 + ((yy - 0.55) / 0.3) ** 2 <= 1] = (68, 14, 19)
    lab[..., 0] += (0.5 - yy) * 25 + (0.5 - xx) * 10
    lab[..., 1:] += rng.normal(0, 0.8, lab[..., 1:].shape)
    return cs.lab_to_srgb8(lab)


def _mixer(adjs):
    return [a for a in adjs if a.tool == "Camera Raw 混色器"]


def test_blue_sky_is_not_a_highlight_tint():
    m = pl.style_match(_outdoor_portrait(), pl.get_style("teal_orange"))
    assert not [a for a in m.adjustments if a.tool == "Camera Raw 颜色分级" and a.instruction.startswith("高光")]
    assert m.highlight_tint is None or m.highlight_tint[1] < pl.TINT_MAX_C


def test_grading_saturation_capped():
    L = np.linspace(5, 95, 240)[None, :].repeat(120, 0)
    lab = np.stack([L, np.where(L > 80, -4.0, 0.0), np.where(L > 80, -6.0, 0.0)], -1)  # 高光偏蓝，青橙要暖高光
    m = pl.style_match(cs.lab_to_srgb8(lab), pl.get_style("teal_orange"))
    g = [a for a in m.adjustments if a.tool == "Camera Raw 颜色分级"]
    assert g and all(dict(a.params)["饱和度"] <= pl.GRADING_SAT_MAX for a in g)


def test_low_saturation_style_does_not_desaturate_an_already_matching_image():
    H, W = 200, 240
    lab = np.zeros((H, W, 3))
    lab[: H // 2] = (70, 4, 6)
    lab[H // 2 :] = (60, -3, 4)
    lab[:, : W // 3] = (65, 6, -4)
    lab[..., 0] += np.linspace(-15, 15, H)[:, None]
    m = pl.style_match(cs.lab_to_srgb8(lab), pl.get_style("morandi"))
    assert m.colorfulness_ok
    assert not _mixer(m.adjustments), [a.instruction for a in m.adjustments]


def test_style_mixer_sliders_capped_and_labelled_by_value():
    for key in ("teal_orange", "japanese_fresh", "morandi", "film_warm"):
        m = pl.style_match(_outdoor_portrait(), pl.get_style(key))
        for a in _mixer(m.adjustments):
            vals = [abs(v) for _, v in a.params]
            assert max(vals) <= ac.SLIDER_CAP, a.instruction
            assert a.strength == ac.slider_strength(max(vals)), a.instruction


def test_japanese_fresh_green_moves_toward_aqua_as_its_recipe_says():
    st = pl.get_style("japanese_fresh")
    assert st["channel_targets"]["绿色"] == "toward:浅绿"
    m = pl.style_match(_outdoor_portrait(), st)
    g = [a for a in _mixer(m.adjustments) if a.instruction.startswith("绿色")]
    assert g and dict(g[0].params)["色相"] > 0 and "浅绿" in g[0].instruction


def test_channel_already_inside_target_sector_is_left_alone():
    img = noisy(flat(lch_color(60, 30, 200), 120, 160), 1)  # 青色，本来就在青橙的“青”里
    m = pl.style_match(img, pl.get_style("teal_orange"))
    assert not [a for a in _mixer(m.adjustments) if "色相" in a.instruction]


def _three_greens(target=None):
    H, W = 200, 240
    lab = np.zeros((H, W, 3))
    lab[:] = _lchlab(55, 35, 135)
    lab[40:100, 30:100] = _lchlab(55, 35, 105)
    lab[120:180, 150:220] = _lchlab(55, 35, 165)
    lab[..., 1:] += np.random.default_rng(0).normal(0, 0.7, lab[..., 1:].shape)
    return pl.family_unity(cs.lab_to_srgb8(lab), "绿", target)


def test_unity_mixer_hue_moves_each_channel_toward_target():
    """往共同的色相（这里没有目标，是这种绿的平均色相，约 133）收拢：偏黄的绿（105，黄色通道）只能往右（+），
    主体绿（135，绿色通道）不能往右。以前黄色通道的平均色相被相邻的绿拉偏，算出了反方向的滑块。
    （给了离得远的目标时只给“整体往目标靠”一条，不再逐通道给，见 test_audit14_colour）"""
    u = _three_greens()
    hue = {a.instruction.split("（")[0]: dict(a.params).get("色相", 0) for a in _mixer(u.adjustments)}
    assert hue.get("黄色", 0) > 0, hue
    assert hue.get("绿色", 0) <= 0, hue


def test_unity_patch_notes_use_relative_direction_words():
    u = _three_greens()
    notes = [p.note for p in u.patches]
    assert notes and not any("绿偏绿" in n for n in notes)
    assert any("偏黄" in n for n in notes) and any("偏青" in n for n in notes)


def test_neutral_tint_gap_does_not_show_meaningless_hue():
    L = np.linspace(5, 95, 240)[None, :].repeat(120, 0)
    img = cs.lab_to_srgb8(np.stack([L, np.zeros_like(L), np.zeros_like(L)], -1))
    m = pl.style_match(img, pl.get_style("teal_orange"))
    hl = [g for g in m.gaps if g.startswith("高光色调")]
    assert hl and "接近中性" in hl[0] and "°" not in hl[0].split("，目标")[0]


def test_accent_tip_considers_lightness_and_skips_black_and_white():
    g = np.repeat(np.linspace(10, 245, 240)[None, :], 200, 0)
    mono = np.clip(np.stack([g, g, g], -1), 0, 255).astype(np.uint8)
    _, r = pl.ratio_631(cs.srgb8_to_lab(mono))
    assert not any("点缀色不够" in t for t in r.tips)
    # 点缀色饱和度不高但明度对比很大（暗绿 + 中灰绿 + 亮米白），也算“跳”
    img = blocks([(lch_color(35, 30, 135), 0.6), (lch_color(50, 25, 120), 0.3), (lch_color(92, 12, 80), 0.1)])
    _, r = pl.ratio_631(cs.srgb8_to_lab(img))
    assert not any("点缀色不够" in t for t in r.tips), r.tips
