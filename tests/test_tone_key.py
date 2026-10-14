"""影调：暗调（低调）/ 高调画面不被往“白亮透”推；高光、阴影、黑白点的识别要适应画面本身。

规则：明度/影调目标只来自明确选择的风格或肤色风格；没选或选默认的“自然通用”时，肤色建议保持当前明度。
"""

import numpy as np
import pytest
from synth import face_scene, highkey_face_scene, lab_color, lowkey_face_scene

from colordoctor.core import analysis as an
from colordoctor.core import color_spaces as cs
from colordoctor.core import layer_plan as lp
from colordoctor.core import native_layers as nl
from colordoctor.core import palette as pl
from colordoctor.core import portrait_analysis as pa
from colordoctor.core import reference_match as rm
from colordoctor.core import skin_presets as sp
from colordoctor.core.color_management import synthetic_context

CTX = synthetic_context()


def _layers(rep, img):
    return lp.plan_from_report(rep, img.shape).layers


def _composite_lifts(layers) -> list:
    """RGB 复合曲线里把输出抬高的点（提亮）。"""
    out = []
    for ly in layers:
        if ly.kind == "curves" and "composite" in ly.params["channels"]:
            out += [(x, y) for x, y in ly.params["channels"]["composite"] if y > x + 2]
    return out


def _texts(rep):
    return [a.instruction for f in rep.findings for a in f.adjustments] + [e for f in rep.findings for e in f.evidence]


# ------------------------------------------------------------------ 肤色：默认不按深浅判断，暗调保持暗
@pytest.mark.parametrize("preset_key", [None, "natural", "moody"])
def test_dark_portrait_skin_is_not_brightened(preset_key):
    img, _ = lowkey_face_scene()
    rep = an.analyze_skin_region(img, CTX, preset_key=preset_key)
    assert not _composite_lifts(_layers(rep, img))
    texts = _texts(rep)
    assert not [t for t in texts if "明度 L*：偏低" in t or "明度 → 橙色提高" in t], texts
    # 色相、浓淡都正常的暗调皮肤：不出“与目标风格的差距”卡片
    assert not any(f.problem.startswith("与目标风格") for f in rep.findings)


@pytest.mark.parametrize("skin_lab", [(80.0, 10.0, 14.0), (34.0, 11.0, 14.0)])
def test_natural_preset_never_changes_skin_lightness(skin_lab):
    """白皮肤不被压暗、深肤色不被提亮：“自然通用”只看色相和浓淡。"""
    img, _ = face_scene(skin_lab=skin_lab)
    rep = an.analyze_skin_region(img, CTX, preset_key="natural")
    for f in rep.findings:
        for a in f.adjustments:
            assert not (a.layer and "composite" in a.layer["params"].get("channels", {})), a.instruction
            assert "明度：RGB 复合曲线" not in a.instruction and "明亮度" not in a.instruction
    fit = pa.preset_fit(rep.skin, sp.get_preset("natural"))
    assert "保持当前值" in fit["明度 L*"]


def test_preset_lightness_moves_only_to_the_range_edge():
    """明确选了亮一些的肤色风格（港风 L* 55–68）：只拉到区间下沿，不拉到中心。"""
    img, _ = lowkey_face_scene()
    rep = an.analyze_skin_region(img, CTX, preset_key="hongkong")
    card = next(f for f in rep.findings if f.problem.startswith("与目标风格"))
    curve = next(a for a in card.adjustments if a.instruction.startswith("明度：RGB 复合曲线"))
    assert "→ 55.0）" in curve.instruction
    p = sp.get_preset("moody")
    assert p.target_for((40.0, 0, 0))[0] == 42.0 and p.target_for((50.0, 0, 0))[0] == 50.0
    assert p.target_for((70.0, 0, 0))[0] == 58.0  # 选了暗调、皮肤太亮：往暗调区间压


# ------------------------------------------------------------------ 整图色彩：暗调画面不被提亮、不被加饱和
STYLES = pl.load_styles()
COLOURFUL = an.COLOURFUL_STYLE_MIN


def _vibrance_up(layers):
    return [ly for ly in layers if ly.kind == "vibrance" and ly.params["vibrance"] > 0]


@pytest.mark.parametrize("style", [None] + [s["key"] for s in STYLES if not s.get("min_median_L")])
def test_low_key_portrait_is_not_brightened_by_any_style(style):
    img, _ = lowkey_face_scene()
    rep = an.analyze_palette_report(img, CTX, style_key=style)
    assert rep.palette.tone_key == "low"
    layers = _layers(rep, img)
    st = pl.get_style(style) if style else {}
    # 只允许胶片风格明确要求的“黑场抬起”（输入 0 的那个点），中间调和亮部一律不提亮
    lifts = [(x, y) for x, y in _composite_lifts(layers) if not (x == 0 and st.get("black_point_min"))]
    assert not lifts, lifts
    if not st or st["colorfulness"][0] < COLOURFUL:
        assert not _vibrance_up(layers)
        assert not any(f.problem.startswith("整体缺色") for f in rep.findings)
    if st.get("mood") == "dark":
        card = next(f for f in rep.findings if f.problem.startswith("与「"))
        assert any("保持暗部，不要提亮" in e for e in card.evidence)


def test_low_key_contrast_only_deepens_the_shadows():
    img, _ = lowkey_face_scene()
    m = pl.style_match(img, pl.get_style("fuji_provia"))  # 对比目标 68–92，这张暗调图只有约 41
    curve = next(a for a in m.adjustments if a.layer and a.layer["kind"] == "curves")
    pts = [(x, y) for x, y in curve.layer["params"]["channels"]["composite"] if (x, y) not in ((0, 0), (255, 255))]
    assert pts and all(y < x for x, y in pts)  # 只压深暗部
    assert all(y >= nl.gray8(pl.BLACK_FLOOR_L) for _, y in pts)  # 不压成死黑
    assert m.score >= 95  # 暗调的亮暗差小不算差距（以前 82，并要把亮部 L* 50 提到 65）


def test_bright_style_median_lift_is_capped():
    """明确选了「日系清新」：可以提亮，但和其他影调一样一次最多移动 15 L*（以前中位 18 → 60 一步到位）。"""
    img, _ = lowkey_face_scene()
    m = pl.style_match(img, pl.get_style("japanese_fresh"))
    curve = next(a for a in m.adjustments if a.layer and a.layer["kind"] == "curves")
    p50 = float(np.median(cs.srgb8_to_lab(img)[..., 0]))
    (x, y), = [(x, y) for x, y in curve.layer["params"]["channels"]["composite"] if (x, y) not in ((0, 0), (255, 255))]
    assert x == nl.gray8(p50) and y <= nl.gray8(p50 + pl.TONE_MOVE_CAP) + 1
    assert "差得太多" in curve.instruction


@pytest.mark.parametrize("style", [None, "fuji_provia", "kodak_portra400", "french", "fuji_eterna"])
def test_high_key_image_is_not_darkened_or_pushed(style):
    img, _ = highkey_face_scene()
    rep = an.analyze_palette_report(img, CTX, style_key=style)
    assert rep.palette.tone_key == "high"
    for ly in _layers(rep, img):
        if ly.kind == "curves" and "composite" in ly.params["channels"]:
            assert all(y >= x - 2 for x, y in ly.params["channels"]["composite"]), ly.params
    assert not _vibrance_up(_layers(rep, img))


def test_no_style_desaturated_low_key_gets_a_neutral_note_not_a_vibrance_push():
    img, _ = lowkey_face_scene(skin_lab=(40.0, 3.0, 5.0), bg_lab=(12.0, 0.3, -0.5))
    rep = an.analyze_palette_report(img, CTX)
    assert rep.palette.colorfulness.value < 15
    assert not any(f.problem.startswith("整体缺色") for f in rep.findings)
    assert any("颜色少不按缺色处理" in f.problem for f in rep.findings)
    assert not _vibrance_up(_layers(rep, img)) and not _composite_lifts(_layers(rep, img))
    # 唯一不是“正常”的只能是曝光说明卡（这张合成图没有任何高光，会提醒“更像欠曝”，但不提亮）
    assert all(f.severity.value == "正常" for f in rep.findings if not f.problem.startswith("画面整体很暗"))
    # 选了鲜艳风格时仍按缺色处理
    rep2 = an.analyze_palette_report(img, CTX, style_key="fuji_velvia")
    assert any(f.problem.startswith("整体缺色") for f in rep2.findings)


# ------------------------------------------------------------------ 高光 / 阴影 / 黑白点的识别
def test_tone_bands_adapt_to_the_image():
    low = np.linspace(2, 52, 10000)  # 暗调：最亮 15% 是中灰（L* 44–52）
    b = pl.tone_bands(low)
    assert b["highlight"] is None and b["shadow"] is not None
    high = np.linspace(55, 99, 10000)  # 高调：最暗 15% 是浅灰（L* 55–62）
    b = pl.tone_bands(high)
    assert b["shadow"] is None and b["highlight"] is not None
    few = np.r_[np.linspace(5, 40, 9950), np.full(50, 80.0)]  # 亮部只有 0.5%：不够判断
    assert pl.tone_bands(few)["highlight"] is None
    ok = np.r_[np.linspace(5, 40, 9700), np.full(300, 80.0)]  # 3% 的真高光：按它们判断（不是最亮 15% 的中灰）
    hl = pl.tone_bands(ok)["highlight"]
    assert hl is not None and hl.sum() == 300


def test_clipped_whites_do_not_dilute_the_highlight_tint():
    img, _ = highkey_face_scene()  # 背景高光略冷（b* −4），右上角一大块死白
    m = pl.style_match(img, pl.get_style("french"))
    h, c = m.highlight_tint
    assert c > 3 and 200 <= h <= 290, m.highlight_tint  # 以前被死白拉成“接近中性（色度 0.5）”，再按它加暖
    curves = [a.instruction for a in m.adjustments if a.instruction.startswith("高光（平均色）")]
    assert not any("输入 255" in t for t in curves)


def test_dark_tinted_pixels_are_not_neutral_blacks():
    assert cs.neutral_chroma_limit(5.0, 15.0) < 8.0  # L* 5 的深蓝 C* 8 不算“中性的黑”
    assert cs.neutral_chroma_limit(30.0, 15.0) == 15.0
    assert pl.family_of(5.0, 7.0, 260.0) != "黑/深灰" and pl.family_of(20.0, 7.0, 260.0) == "黑/深灰"
    assert cs.clipped(np.array([[255, 250, 240], [0, 0, 1], [128, 128, 128], [10, 0, 0]])).tolist() == [True, True, False, False]


def test_reference_white_balance_ignores_clipped_whites():
    L = np.linspace(30, 90, 200)[None, :].repeat(120, 0)
    def scene(b, clip):
        img = cs.lab_to_srgb8(np.stack([L, np.zeros_like(L), np.full_like(L, b)], -1))
        if clip:
            img[:, :120] = 255  # 60% 死白
        return img
    m = rm.match_reference(scene(-5.0, True), scene(0.0, True))  # 两张影调相同（没有曲线），只差白平衡
    assert m.cast_mine is not None and m.cast_mine[1] == pytest.approx(-5.0, abs=0.7)  # 以前被死白冲淡到约 −2
    cb = [a for a in m.adjustments if a.layer and a.layer["kind"] == "colorBalance"]
    assert cb and cb[0].layer["params"]["midtones"][2] < 0  # 往黄


def _specular_face(spec_lab, frac=0.12, seed=1):
    img, truth = face_scene(skin_lab=(66, 14, 19))
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(truth)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    spec = np.zeros(truth.shape, bool)
    while spec.sum() < frac * truth.sum():
        i = rng.integers(len(ys))
        spec |= ((yy - ys[i]) ** 2 + (xx - xs[i]) ** 2 <= 16) & truth
    img[spec] = lab_color(*spec_lab)
    return img, truth


@pytest.mark.parametrize("spec_lab", [(93, 4, 1), (92, 5, 0), (94, 3, 7)])
def test_specular_reflections_do_not_drive_skin_highlight_cast(spec_lab):
    img, truth = _specular_face(spec_lab)
    rep = an.analyze_skin_region(img, CTX, auto_mask=False, lasso=truth)
    assert abs(rep.skin.highlight_hue_shift) < 3, rep.skin.highlight_shift  # 以前 −15° ~ −22°，报“高光偏洋红/发青”
    assert not any(f.problem.startswith("肤色高光") for f in rep.findings)


def test_flat_skin_has_no_highlight_or_shadow_band():
    img, truth = face_scene(skin_lab=(66, 14, 19))
    lab = cs.srgb8_to_lab(img)
    lab[truth, 0] = 66.0  # 皮肤没有明暗起伏
    lab[truth & (np.arange(img.shape[1])[None, :] > 100), 2] += 6  # 右半边更黄：这是不均匀，不是“高光偏黄”
    flat = cs.lab_to_srgb8(lab)
    rep = an.analyze_skin_region(flat, CTX, auto_mask=False, lasso=truth)
    assert rep.skin.highlight_hue_shift == 0 and rep.skin.shadow_hue_shift == 0
    assert not any(f.problem.startswith(("肤色高光", "肤色阴影")) for f in rep.findings)


def test_clipped_sky_is_not_a_local_cast():
    L = np.linspace(55, 105, 160)[:, None] * np.ones((1, 200))
    lab = np.stack([L, np.full_like(L, 30 * np.cos(np.radians(250))), np.full_like(L, 30 * np.sin(np.radians(250)))], -1)
    img = np.clip(np.rint(cs.lab_to_srgb_float(lab) * 255), 0, 255).astype(np.uint8)  # 上部的天空被裁切
    rep = an.analyze_region(img, CTX)
    assert not any("局部偏色" in f.problem for f in rep.findings)  # 以前：裁切让高光色相偏移，报“高光与中间调不一致”
