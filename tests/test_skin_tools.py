"""精准皮肤选区、经验快速判定、肤色均匀度。"""

import numpy as np
import pytest

from colordoctor.core import color_spaces as cs
from colordoctor.core import portrait_analysis as pa
from colordoctor.core import skin_presets as sp
from colordoctor.core.skin_mask import polygon_mask, skin_mask
from colordoctor.core.skin_uniformity import analyze_uniformity
from synth import face_scene, skin_region


# ------------------------------------------------------------------ 精准选区
@pytest.mark.parametrize("skin_lab", [(68.0, 14.0, 19.0), (36.0, 11.0, 14.0), (82.0, 9.0, 11.0)])
def test_skin_mask_separates_skin_from_hair_lips_eyes_background(skin_lab):
    img, truth = face_scene(skin_lab=skin_lab)
    m, info = skin_mask(img)
    tp = (m & truth).sum() / truth.sum()
    fp = (m & ~truth).sum() / max(1, m.sum())
    assert tp > 0.9, tp
    assert fp < 0.05, fp
    assert info["skin_share"] > 0.3


def test_skin_mask_respects_lasso():
    img, truth = face_scene()
    lasso = polygon_mask([(60, 100), (140, 100), (140, 160), (60, 160)], img.shape[1], img.shape[0])
    m, _ = skin_mask(img, within=lasso)
    assert not (m & ~lasso).any()
    assert m.sum() > 0.8 * (lasso & truth).sum()


def test_polygon_mask_triangle():
    m = polygon_mask([(0, 0), (9, 0), (0, 9)], 10, 10)
    assert m[1, 1] and not m[9, 9]


def test_skin_mask_on_large_region_is_downsampled_but_full_size():
    img, truth = face_scene(h=1600, w=1600)
    m, _ = skin_mask(img)
    assert m.shape == truth.shape
    assert (m & truth).sum() / truth.sum() > 0.85


def test_skin_mask_without_skin():
    m, info = skin_mask(np.full((50, 50, 3), (30, 80, 200), np.uint8))
    assert m.sum() == 0 and info["note"]


# ------------------------------------------------------------------ 经验快速判定（a、b 在 10–20，差值 ≤ 5）
def _lab_px(L, a, b, n=100):
    return np.tile(np.array([L, a, b], float), (n, 1))


def test_quick_rule_default_values():
    assert sp.global_quick_rule() == ((10.0, 20.0), (10.0, 20.0), 5.0)


@pytest.mark.parametrize(
    "lab,passed,tip",
    [
        ((68, 14, 18), True, None),
        ((68, 13, 21), False, "b*"),  # b 超 20
        ((68, 11, 19), False, "相差"),  # 差 8
        ((68, 7, 10), False, "a*"),
        ((68, 22, 18), False, "a*"),
    ],
)
def test_quick_rule(lab, passed, tip):
    r = pa.quick_rule_check(_lab_px(*lab), sp.DEFAULT_QUICK_RULE)
    assert r.passed is passed
    if tip:
        assert any(tip in t for t in r.tips)
    else:
        assert r.verdict == "正常肤色" and r.pixel_pass_share == 1.0


# ------------------------------------------------------------------ 均匀度
def test_uniform_skin_scores_high():
    img = skin_region(h=160, w=160)
    r = analyze_uniformity(img)
    assert r.score > 85
    assert not [p for p in r.patches if p.severity == "明显"]


def test_yellow_patch_on_orange_skin_is_located():
    img = skin_region(68, 14, 19, h=200, w=200)  # 色相 ~54°（橙）
    lab = cs.srgb8_to_lab(img[120:180, 20:90])
    L = lab[..., 0]
    patch = cs.lch_to_lab(np.stack([L, np.full_like(L, 25.0), np.full_like(L, 70.0)], -1))  # 色相 70°（偏黄）
    img[120:180, 20:90] = cs.lab_to_srgb8(patch)
    r = analyze_uniformity(img)
    yellow = [p for p in r.patches if p.kind == "偏黄"]
    assert yellow, r.patches
    x, y, w, h = yellow[0].bbox
    assert x <= 35 and x + w >= 75 and y <= 135 and y + h >= 165
    assert yellow[0].hue_shift > 10
    assert r.score < 85


def test_gray_patch_is_reported_as_missing_color():
    img = skin_region(68, 14, 19, h=200, w=200)
    lab = cs.srgb8_to_lab(img[30:90, 110:180])
    lab[..., 1:] *= 0.5  # 同明度，色度减半
    img[30:90, 110:180] = cs.lab_to_srgb8(lab)
    r = analyze_uniformity(img)
    gray = [p for p in r.patches if p.kind == "缺色发灰"]
    assert gray and gray[0].sat_shift < -0.25


def test_uniformity_uses_mask_and_offset():
    img, truth = face_scene()
    r = analyze_uniformity(img, mask=truth, offset=(1000, 500))
    assert r.score > 80
    for p in r.patches:
        assert p.bbox[0] >= 1000 and p.bbox[1] >= 500


def test_natural_shading_is_not_uneven():
    img = skin_region(60, 13, 18, sigma_L=10, h=200, w=200)  # 明暗起伏很大
    r = analyze_uniformity(img)
    assert not [p for p in r.patches if p.kind in ("缺色发灰", "饱和度偏高") and p.severity == "明显"]


def test_new_presets_include_dark_and_wheat_skin():
    keys = {p.key for p in sp.builtin_presets()}
    assert {"deep", "wheat", "bronze", "korean", "western", "hongkong", "chinese"} <= keys
    deep = sp.get_preset("deep", sp.builtin_presets())
    assert deep.target_lab[0] < 45 and deep.quick_rule is not None
