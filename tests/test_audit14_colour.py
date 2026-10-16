"""第 14 轮审查（颜色 / 影调）的回归：仿色曲线、影调曲线端点、跨卡片的色相/饱和度、黑白参考、
Camera Raw 建议重复修正、颜色统一、色相范围跨 0°、斜率限制的端点。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from synth import flat, lch_color, noisy

from colordoctor.core import acr_preset as ap
from colordoctor.core import color_spaces as cs
from colordoctor.core import layer_plan as lp
from colordoctor.core import native_layers as nl
from colordoctor.core import palette as pl
from colordoctor.core import ps_models as pm
from colordoctor.core import reference_match as rm
from colordoctor.core import tone_class as tc
from colordoctor.core.color_models import AnalysisReport


def _grad(L0, L1, a=2.0, b=5.0, h=200, w=200):
    L = np.linspace(L0, L1, h * w).reshape(h, w)
    lab = np.stack([L, np.full_like(L, a), np.full_like(L, b)], -1)
    return cs.lab_to_srgb8(lab.reshape(-1, 3)).reshape(h, w, 3).astype(np.uint8)


def _slopes(pts):
    xs, ys = np.array(pts, float).T
    return np.diff(ys) / np.diff(xs)


# ------------------------------------------------------------------ 1. 仿色曲线：端点跟着移动、斜率有限制
def test_reference_curve_does_not_stretch_highlights():
    """我的 P95 很亮、参考的高光收住（胶片感）：以前端点 (255, 255) 不动，最上面一段斜率约 6、样条还往回走。"""
    m = rm.match_reference(_grad(8, 99.6), _grad(18, 74))
    full = nl.curve_points(m.curve_points)
    assert dict(map(tuple, full))[255] < 255  # 白场跟着压下来
    sl = _slopes(full)
    assert sl.max() <= tc.MAX_SLOPE + 0.15 and sl.min() >= 1 / tc.MAX_SLOPE - 0.15, full
    assert np.diff(tc.spline_lut(full)).min() >= -0.5


def test_reference_curve_fuzz_monotone_and_bounded():
    rng = np.random.default_rng(2)
    n = back = 0
    for _ in range(600):
        tm, tr = np.sort(rng.uniform(0, 100, 5)), np.sort(rng.uniform(0, 100, 5))
        if rng.random() < 0.3:
            tm[0] = rng.uniform(0, 2)
        if rng.random() < 0.3:
            tm[-1] = rng.uniform(98, 100)
        curve = rm.tone_curve(tm, tr)
        if not curve:
            continue
        n += 1
        full = nl.curve_points(curve)
        sl = _slopes(full)
        assert sl.max() <= tc.MAX_SLOPE + 0.15 and sl.min() >= 1 / tc.MAX_SLOPE - 0.15, (tm, tr, full)
        back += int(np.diff(tc.spline_lut(full)).min() < -0.5)
    assert n > 500 and back <= 2  # 以前约 58% 的曲线往回走（PS 曲线点数有上限，极少数仍可能有一点）


def test_reference_curve_moves_black_endpoint_when_p5_is_near_black():
    tm = (1.0, 30.0, 50.0, 70.0, 90.0)
    tr = (12.0, 33.0, 50.0, 70.0, 90.0)
    full = dict(map(tuple, nl.curve_points(rm.tone_curve(tm, tr))))
    assert full[0] > 0  # 抬黑场，而不是在 0–12 之间放一个陡的点


# ------------------------------------------------------------------ 2. 影调曲线：离端点不到 MIN_GAP 的分位移动端点
@pytest.mark.parametrize("p98", list(range(244, 252)))
def test_match_curve_moves_white_endpoint_for_p98_close_to_255(p98):
    pts, _ = tc.match_curve((80, 165, 190, 214, p98), tc.target_quantiles("高短调"))
    assert pts[-1][0] == 255 and pts[-1][1] < 255  # 以前 244–251 的 P98 被悄悄丢掉，白场不动


@pytest.mark.parametrize("p2", list(range(4, 12)))
def test_match_curve_moves_black_endpoint_for_p2_close_to_0(p2):
    pts, _ = tc.match_curve((p2, 40, 128, 200, 240), tc.target_quantiles("中短调"))
    assert pts[0] == [0, pts[0][1]] and pts[0][1] > 0


def test_tone_curve_response_is_smooth_across_p98():
    """t14：高光 P98 从 240 到 255，按曲线调完以后的 P98 应连续（以前 246–251 时调完几乎不动：244 / 247 / 249）。"""
    rng = np.random.default_rng(0)
    got = []
    for p98 in (240, 243, 246, 249, 251, 253):
        y = np.concatenate([rng.uniform(150, 230, 9000), rng.uniform(230, p98, 800), rng.uniform(p98, 255, 200),
                            rng.uniform(60, 150, 1000)])
        im = np.repeat(np.clip(np.rint(y), 0, 255).astype(np.uint8)[:, None], 3, 1).reshape(-1, 1, 3)
        pts, _ = tc.fit_curve(im, None, tc.quantiles(im), tc.target_quantiles("高短调"))
        after = tc.spline_lut(pts)[im[:, 0, 0]]
        got.append(float(np.percentile(after, 98)))
    assert max(got) - min(got) <= 4, got


# ------------------------------------------------------------------ 3. 不同卡片的色相/饱和度：只去掉重叠的范围
def _report(*findings):
    rep = AnalysisReport.__new__(AnalysisReport)
    for f in ("contamination", "uniformity", "palette", "valid_mask", "reference"):
        object.__setattr__(rep, f, None)
    object.__setattr__(rep, "mode", "palette")
    object.__setattr__(rep, "findings", tuple(SimpleNamespace(adjustments=tuple(a)) for a in findings))
    return rep


def _hs(items, text, name):
    return nl.adjustment("hueSaturation", nl.huesat_multi(items), text, "global", name)


def test_later_card_keeps_its_non_overlapping_ranges():
    first = _hs([(180, 8, 20, -5, 0, 0)], "青 色相 -5", "统一")
    second = _hs([(180, 8, 20, 0, -20, 0), (20, 8, 20, 0, -15, 0)], "青 饱和度 -20；红 饱和度 -15", "模板")
    plan = lp.plan_from_report(_report([first], [second]), (20, 20, 3))
    later = plan.for_finding(1)
    assert len(later) == 1, plan.skipped  # 以前一个范围重叠就整层不生成
    ranges = later[0].params["ranges"]
    assert len(ranges) == 1 and ranges[0]["saturation"] == -15
    assert not plan.skipped and plan.partial
    assert "对准色相 180°" in plan.partial[0][2] and "对准色相 180°" in later[0].reason
    fj = lp.annotate_findings([{"adjustments": [{"instruction": first.instruction, "layer": first.layer}]},
                               {"adjustments": [{"instruction": second.instruction, "layer": second.layer}]}], plan)
    a = fj[1]["adjustments"][0]
    assert a["layer"] and "没有放进这一层" in a["instruction"]


def test_fully_overlapping_later_card_is_skipped_with_accurate_note():
    first = _hs([(180, 8, 20, -5, 0, 0)], "青 色相 -5", "统一")
    second = _hs([(185, 8, 20, 0, -27, 0)], "青 饱和度 -27", "模板")
    plan = lp.plan_from_report(_report([first], [second]), (20, 20, 3))
    assert not plan.for_finding(1)
    (i, _, why), = plan.skipped
    assert i == 1 and why == lp.SKIP_OTHER_CARD and "按那一层调整即可" not in why  # 两层改的不是同一件事


# ------------------------------------------------------------------ 4. 黑白参考：影调和色调都按转黑白以后的灰度算
def _colourful_mine():
    h = w = 120
    img = np.zeros((h, w, 3), np.uint8)
    img[:40] = (30, 60, 220)
    img[40:80] = (240, 220, 40)
    img[80:] = np.linspace(20, 235, w)[None, :, None].astype(np.uint8)
    return noisy(img, 3)


def _bw(img):
    x = img.reshape(-1, 3) / 255.0
    return np.clip(np.rint(pm.hue_saturation(x, saturation=-100) * 255), 0, 255).astype(np.uint8)


def test_bw_reference_tone_is_measured_on_the_desaturated_image():
    mine = _colourful_mine()
    ref = np.repeat(np.linspace(10, 245, 120)[None, :, None], 120, 0).repeat(3, 2).astype(np.uint8)
    m = rm.match_reference(mine, ref)
    assert any("黑白作品" in f for f in m.fit)
    bw_q = np.percentile(cs.srgb8_to_lab(_bw(mine))[:, 0], rm.PCTS)
    assert np.abs(np.array(m.tone_mine) - bw_q).max() < 0.6  # 以前按彩色图算：P75 87 vs 转黑白后 59


def test_bw_reference_does_not_correct_a_cast_the_conversion_already_removed():
    """我的图整体偏暖、参考是中性的黑白：转黑白以后已经没有偏色，不能再加一层往蓝的色彩平衡（以前会把灰图推成偏青蓝）。"""
    mine = _colourful_mine()
    lab = cs.srgb8_to_lab(mine.reshape(-1, 3)).astype(float)
    lab[:, 2] += 8
    mine = cs.lab_to_srgb8(lab).reshape(mine.shape).astype(np.uint8)
    ref = np.repeat(np.linspace(10, 245, 120)[None, :, None], 120, 0).repeat(3, 2).astype(np.uint8)
    m = rm.match_reference(mine, ref)
    assert any("黑白作品" in f for f in m.fit)
    assert not [a for a in m.adjustments if a.tool == "Camera Raw" and "色温" in a.instruction]
    assert not [a for a in m.adjustments if a.layer and a.layer["kind"] == "colorBalance"]


# ------------------------------------------------------------------ 5. Camera Raw：同一个偏色不修三次
def _cast_pair():
    H, W = 120, 240
    ref = np.zeros((H, W, 3), np.uint8)
    ref[:, :80] = np.repeat(np.linspace(30, 230, H)[:, None, None], 80, 1).repeat(3, 2).astype(np.uint8)
    for i, (L, C, h) in enumerate(((55, 40, 140), (45, 40, 260), (70, 35, 60), (60, 45, 25))):
        ref[i * 30:(i + 1) * 30, 80:] = lch_color(L, C, h)
    ref = noisy(ref, 2)
    lab = cs.srgb8_to_lab(ref.reshape(-1, 3)).astype(float)
    lab[:, 2] += 8
    return cs.lab_to_srgb8(lab).reshape(ref.shape).astype(np.uint8), ref


def test_uniform_cast_gives_temperature_only_in_camera_raw():
    mine, ref = _cast_pair()
    m = rm.match_reference(mine, ref)
    acr = [a for a in m.adjustments if a.tool.startswith("Camera Raw")]
    assert [a.instruction[:4] for a in acr] == ["色温往蓝"], [a.instruction for a in acr]
    assert ap.params_from_adjustments(m.adjustments) == {}  # 以前：阴影 / 高光各加一次蓝，混色器 6 个通道再修一次


# ------------------------------------------------------------------ 6. 颜色统一
@pytest.mark.parametrize("fam,L,C,h", [("青", 40, 20, 230), ("蓝", 30, 30, 290), ("绿", 35, 25, 140), ("黄", 85, 40, 95)])
def test_unified_family_gets_no_hue_rows_or_layers(fam, L, C, h):
    img = noisy(flat(lch_color(L, C, h), 120, 160), 1)
    img[:, 80:] = noisy(flat(lch_color(L + 8, C, h), 120, 80), 1)
    u = pl.family_unity(img, fam)
    assert u.score >= pl.UNITY_OK_SCORE
    assert not [a for a in u.adjustments if a.tool == "Camera Raw 混色器" or a.layer]  # 以前蓝色给出“色相 +8”


def test_tiny_spread_gives_no_hue_suggestion_even_below_ok_score():
    """色相已经统一（标准差 < 4°）、只是饱和度不齐：不给色相建议。"""
    img = noisy(flat(lch_color(55, 20, 135), 120, 160), 1)
    img[:, 80:] = noisy(flat(lch_color(55, 50, 135), 120, 80), 1)
    u = pl.family_unity(img, "绿")
    assert u.hue_spread < pl.UNITY_HUE_OK
    assert all(dict(a.params).get("色相", 0) == 0 for a in u.adjustments if a.tool == "Camera Raw 混色器" and a.params)


def test_toward_target_replaces_per_channel_rows():
    img = flat(lch_color(55, 35, 112), 120, 180)
    img[:, 60:120] = lch_color(55, 35, 135)
    img[:, 120:] = lch_color(55, 35, 160)
    u = pl.family_unity(noisy(img, 1), "绿", target_lab=(50, -25, 30))
    mixer = [a.instruction for a in u.adjustments if a.tool == "Camera Raw 混色器"]
    assert len(mixer) == 1 and mixer[0].startswith("整体往目标绿靠"), mixer  # 以前还有 3 条逐通道的，同一批像素转两次
    assert not any("收拢" in a.instruction for a in u.adjustments)


# ------------------------------------------------------------------ 8. 色相范围放宽时跨过 0° / 360°
def test_padded_hue_range_wraps_around_zero():
    h = np.array([357.0, 2.0, 33.0, 40.0, 180.0])
    assert pl._in_range_padded(h, 0, 30, 5).tolist() == [True, True, True, False, False]
    assert pl._in_range_padded(h, 330, 358, 5).tolist() == [True, True, False, False, False]
    assert pl._in_range_padded(h, 0, 360, 5).all()


def test_style_sector_starting_at_zero_counts_hues_just_below_360():
    img = noisy(flat(lch_color(50, 40, 357), 60, 60), 1)
    base = pl.get_style("teal_orange")
    style = {**base, "sectors": [{"name": "橙", "hue": [0, 30]}, base["sectors"][1]]}
    m = pl.style_match(img, style)
    assert m.sectors[0].share > 0.9  # 以前放宽成 -5–35 不取模：357° 不算在里面


# ------------------------------------------------------------------ 9. 斜率限制不把端点推出 0–255
def test_limit_slopes_keeps_endpoints_in_range():
    pts, _ = tc.limit_slopes([[0, 0], [240, 252], [255, 255]])
    ys = [y for _, y in pts]
    assert 0 <= min(ys) and max(ys) <= 255  # 以前最后一点被推到 257，取整截到 255 后最后一段斜率只有 0.2
    sl = _slopes(pts)
    assert sl.min() >= 1 / tc.MAX_SLOPE - 1e-9 and sl.max() <= tc.MAX_SLOPE + 1e-9
