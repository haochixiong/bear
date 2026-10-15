"""第八轮审计（影调风格上线后的整体审计）的回归测试。"""

import numpy as np
import pytest
from synth import face_scene, lowkey_face_scene

from colordoctor import cli
from colordoctor.bridge import server as bs
from colordoctor.core import analysis as an
from colordoctor.core import color_spaces as cs
from colordoctor.core import layer_plan as lp
from colordoctor.core import palette as pl
from colordoctor.core import reference_match as rm
from colordoctor.core import skin_presets as sp
from colordoctor.core import tone_class as tc
from colordoctor.core.color_management import synthetic_context
from colordoctor.core.color_models import Severity

CTX = synthetic_context()


def _tone_card(rep):
    return next(f for f in rep.findings if f.problem.startswith("影调："))


def _curves(rep, img):
    return [ly for ly in lp.plan_from_report(rep, img.shape).layers if ly.kind == "curves"]


# ------------------------------------------------------------------ 影调曲线
@pytest.mark.parametrize("q,end", [((2, 71, 120, 160, 226), 0), ((40, 119, 149, 185, 254), 255)])
def test_clipped_ends_move_the_endpoint(q, end):
    """最暗 / 最亮的分位贴着 0 / 255 时，移动黑场 / 白场端点（以前端点不动，调长改不了，重新分析也不会收敛）。"""
    pts, _ = tc.match_curve(q, tc.target_quantiles("中中调"))
    y = dict(map(tuple, pts))[end]
    assert (y > 0) if end == 0 else (y < 255)


def test_curve_points_are_spaced_and_not_too_steep():
    pts, capped = tc.match_curve((4, 9, 14, 30, 240), tc.target_quantiles("高短调"))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    assert xs[0] == 0 and xs[-1] == 255 and capped
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        assert x1 - x0 >= tc.MIN_GAP or x1 == 255 or x0 == 0
        assert y1 > y0 and (y1 - y0) / (x1 - x0) <= tc.MAX_SLOPE + 0.1
    assert ys == sorted(ys)


def test_curve_is_fitted_for_the_luminosity_blend_on_colour_pixels():
    """曲线作用在 RGB 上、明度混合：按实际效果修正后，分位点比只按明度算的更接近目标。"""
    img, _ = face_scene()
    q, t = tc.quantiles(img), tc.target_quantiles("中中调")
    plain, _ = tc.match_curve(q, t)
    fitted, _ = tc.fit_curve(img, None, q, t)
    err = [float(np.abs(np.percentile(tc.apply_curve_luminosity(img.reshape(-1, 3), p), [2, 25, 50, 75, 98]) - t).mean())
           for p in (plain, fitted)]
    assert err[1] <= err[0]


def test_extreme_histogram_gives_no_empty_curve():
    """97% 死黑 + 3% 死白的夜景：曲线拉不动，不生成什么都不改的图层，也没有空文字。"""
    img = np.zeros((100, 100, 3), np.uint8)
    img[:3] = 255
    rep = an.analyze_palette_report(img, CTX, tone_style="film")
    card = _tone_card(rep)
    assert not _curves(rep, img) or all(max(abs(y - x) for x, y in ly.params["channels"]["composite"]) >= 3
                                        for ly in _curves(rep, img))
    assert "（）" not in " ".join(a.instruction for a in card.adjustments)


def test_curve_words_list_every_direction():
    words = tc.curve_words([[0, 0], [44, 58], [91, 106], [164, 150], [183, 198], [255, 255]])
    assert "提亮" in words and "压暗 164→150" in words


def test_tone_card_does_not_inflate_overall_severity():
    img, _ = lowkey_face_scene(skin_lab=(62.0, 13.0, 17.0))
    plain = an.analyze_palette_report(img, CTX)
    target = an.analyze_palette_report(img, CTX, tone_style="japanese")
    assert target.overall_severity == plain.overall_severity and target.overall_risk == plain.overall_risk
    assert _tone_card(target).severity != Severity.NORMAL


def test_header_grid_and_tone_card_use_the_same_classification():
    rng = np.random.default_rng(3)
    for _ in range(20):
        lo, hi = sorted(rng.integers(0, 255, 2))
        img = np.repeat(np.linspace(lo, hi, 1000)[None, :, None], 900, 0).repeat(3, 2).astype(np.uint8)
        rep = an.analyze_palette_report(img, CTX, tone_style="japanese")
        assert _tone_card(rep).problem.startswith(f"影调：{rep.palette.tone_class.name}")


def test_dark_dull_image_is_only_offered_same_key_long_styles():
    img = np.full((80, 80, 3), 60, np.uint8)
    img[:, :40] = (70, 66, 62)
    rep = an.analyze_palette_report(img, CTX)
    tips = " ".join(a.instruction for f in rep.findings for a in f.adjustments)
    assert "白亮通透" not in tips


# ------------------------------------------------------------------ 仿色曲线
def test_reference_curve_keeps_matching_quantiles_fixed():
    L = np.linspace(5, 80, 300)
    zero = np.zeros((60, 300))
    mine = cs.lab_to_srgb8(np.stack([np.tile(L, (60, 1)), zero, zero], -1))
    ref = cs.lab_to_srgb8(np.stack([np.tile(np.where(L > 55, L + 12, L), (60, 1)), zero, zero], -1))
    m = rm.match_reference(mine, ref)
    xs, ys = np.array([(0, 0), *m.curve_points, (255, 255)]).T
    after = np.interp(mine.astype(float), xs, ys).astype(np.uint8)
    p = [np.percentile(cs.srgb8_to_lab(x)[..., 0], [25, 50, 95]) for x in (after, ref)]
    assert np.abs(p[0] - p[1]).max() < 1.5  # 以前 P50 被带亮 11 L*


# ------------------------------------------------------------------ 颜色模板只管颜色
@pytest.mark.parametrize("style", pl.load_styles(), ids=lambda s: s["key"])
def test_colour_template_advice_has_no_tone_moves(style):
    img, _ = lowkey_face_scene()
    m = pl.style_match(img, style)
    for a in m.adjustments:
        if a.tool.startswith("常用做法·") and a.tool.split("·")[1] in pl.TONE_PANELS:
            assert not any(w in a.instruction for w in ("黑场", "对比", "提亮", "S 形")), a.instruction
        if a.layer and a.layer["kind"] == "hueSaturation":  # 颜色模板的色相/饱和度不改明度（以前黑金压暗冷色）
            assert all(r["lightness"] == 0 for r in a.layer["params"]["ranges"])
        assert not any(k == "明亮度" for k, _ in (a.params or ())), a.instruction


def test_skin_preset_lightness_is_capped():
    img, truth = lowkey_face_scene(skin_lab=(45.0, 13.0, 17.0))
    rep = an.analyze_skin_region(img, CTX, preset=sp.get_preset("japanese"), manual_mask=truth)
    curves = [a for f in rep.findings for a in f.adjustments if a.layer and a.layer["name"].endswith("明度 · 曲线")]
    for a in curves:
        (x, y), = [p for p in a.layer["params"]["channels"]["composite"] if p[0] not in (0, 255)]
        assert abs(y - x) <= 40 and "影调风格" in a.instruction


def test_dirty_modes_do_not_add_contrast_curves():
    img = np.full((120, 120, 3), (40, 110, 50), np.uint8)
    img[40:80, 40:80] = (60, 75, 62)  # 一块发灰的绿
    for rep in (an.scan_image(img, CTX), an.analyze_region(img, CTX)):
        assert not any(ly.kind == "curves" for ly in lp.plan_from_report(rep, img.shape).layers)


# ------------------------------------------------------------------ 桥接 / 命令行的参数检查
@pytest.mark.parametrize("q,word", [({"style": "nosuch"}, "颜色模板"), ({"family": "紫"}, "颜色")])
def test_bridge_rejects_unknown_palette_params(q, word):
    img = np.dstack([face_scene()[0], np.full((200, 200), 255, np.uint8)])
    with pytest.raises(ValueError, match=word):
        bs.analyze_request(img, "palette", q, bs.Store(), bs.Store(), layer_kind="pixel")


def test_cli_rejects_palette_options_in_other_modes(tmp_path, capsys):
    from PIL import Image

    p = tmp_path / "a.png"
    Image.fromarray(face_scene()[0]).save(p)
    assert cli.run(["analyze", str(p), "--tone-style", "japanese"]) == 2
    assert "--mode palette" in capsys.readouterr().err
    assert cli.run(["analyze", str(p), "--mode", "palette", "--tone-target", "低短调"]) == 0
    assert capsys.readouterr().out.startswith("[整图色彩] 影调")
