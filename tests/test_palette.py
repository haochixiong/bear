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
    assert any("黑位" in g for g in bad.gaps)
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


def test_film_look_flags_crushed_blacks_and_high_contrast():
    portra = pl.get_style("kodak_portra400")
    harsh = pl.style_match(_gradient(0, 100), portra)
    soft = pl.style_match(_gradient(12, 80), portra)
    assert any("黑位太深" in g for g in harsh.gaps)
    assert any("对比度" in g and "降低" in g for g in harsh.gaps)
    assert not any("黑位太深" in g or "对比度" in g for g in soft.gaps)
    assert soft.score > harsh.score
    curves = [a.instruction for a in harsh.adjustments if a.tool == "曲线"]
    assert any("黑场抬起" in t for t in curves) and any("降低对比" in t for t in curves)
    assert harsh.contrast > soft.contrast


def test_high_contrast_style_asks_for_more_contrast():
    m = pl.style_match(_gradient(35, 65), pl.get_style("fuji_velvia"))
    assert any("对比度" in g and "提高" in g for g in m.gaps)
