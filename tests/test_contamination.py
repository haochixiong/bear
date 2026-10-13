"""脏色检测：对立色混入、局部发灰、位置定位，以及误报专项。"""

import numpy as np
import pytest

from colordoctor.core import contamination as ct
from synth import BLUE, YELLOW, flat, lch_color, noisy, patch, skin_region, sprinkle


def run(img, **kw):
    return ct.analyze(img, **kw)


# ------------------------------------------------------------------ 基本
def test_uniform_blue_is_clean():
    r = run(noisy(flat(BLUE), 2))
    assert r.base_hue_name == "蓝"
    assert r.opposite_fraction == 0
    assert r.graying_fraction < 0.01
    assert not r.blobs


@pytest.mark.parametrize("frac", [0.01, 0.05, 0.10])
def test_blue_with_scattered_yellow_detected_with_correct_fraction(frac):
    img = sprinkle(noisy(flat(BLUE), 2), YELLOW, frac, seed=7)
    r = run(img)
    assert r.base_hue_name == "蓝"
    assert r.opposite_hue_name == "黄"
    assert r.opposite_fraction == pytest.approx(frac, rel=0.15)
    assert r.interspersion > 2.0  # 散点混杂
    assert not r.bicolor
    assert r.blobs  # 有位置
    assert all(b.kind == "opposite" for b in r.blobs)


def test_low_saturation_warm_pixels_in_blue_are_detected():
    warm = lch_color(62, 14, 70)  # 低饱和暖色
    img = sprinkle(noisy(flat(BLUE), 2), warm, 0.05, seed=3)
    r = run(img)
    assert r.opposite_fraction == pytest.approx(0.05, rel=0.2)
    assert r.opposite_chroma < 20


def test_location_of_a_contaminated_patch_is_reported():
    base = noisy(flat(BLUE, 200, 200), 2)
    dirty = sprinkle(base[120:180, 20:80], YELLOW, 0.25, seed=1)
    img = base.copy()
    img[120:180, 20:80] = dirty
    r = run(img)
    assert r.blobs
    b = max(r.blobs, key=lambda b: b.area_fraction)
    x, y, w, h = b.bbox
    # bbox 应覆盖 (20..80, 120..180) 且不跑到图像另一端
    assert x <= 30 and x + w >= 70
    assert y <= 130 and y + h >= 170
    assert x + w <= 110 and y >= 90


def test_mask_marks_contaminated_pixels():
    img = sprinkle(noisy(flat(BLUE, 60, 60), 2), YELLOW, 0.05, seed=2)
    r = run(img)
    assert r.mask.shape == (60, 60)
    assert (r.mask == ct.MASK_OPPOSITE).mean() == pytest.approx(0.05, rel=0.2)


# ------------------------------------------------------------------ 误报专项
def test_low_saturation_grey_blue_is_not_dirty():
    img = noisy(flat(lch_color(55, 9, 265)), 3, seed=4)
    r = run(img)
    assert r.opposite_fraction < 0.003
    assert not r.blobs


def test_dark_low_key_region_is_not_dirty():
    img = noisy(flat(lch_color(14, 10, 280)), 2, seed=5)
    r = run(img)
    assert r.opposite_fraction < 0.003
    assert r.graying_fraction < 0.02


def test_fifty_fifty_opposite_colours_is_a_palette_not_contamination():
    img = flat(BLUE)
    img[:, 80:] = YELLOW
    r = run(noisy(img, 2))
    assert r.bicolor
    assert not any(b.kind == "opposite" for b in r.blobs)


def test_teal_orange_grade_is_not_contamination():
    teal = lch_color(45, 30, 205)
    orange = lch_color(70, 35, 60)
    img = flat(teal, 120, 160)
    img[30:110, 50:110] = orange  # 人物（橙）在青色背景上
    r = run(noisy(img, 2), mode="scan")
    opp = [b for b in r.blobs if b.kind == "opposite"]
    assert not opp


def test_small_noise_does_not_create_strong_contamination():
    img = noisy(flat(lch_color(60, 30, 260)), 6, seed=8)
    r = run(img)
    assert r.opposite_fraction < 0.002


def test_hue_gradient_same_hue_different_lightness_is_clean():
    h, w = 80, 200
    L = np.linspace(25, 85, w)
    img = np.stack([np.array(lch_color(l, 0.55 * l, 260)) for l in L])[None].repeat(h, 0).astype(np.uint8)
    r = run(img)
    assert r.opposite_fraction == 0
    assert r.graying_fraction < 0.05


def test_pastel_tint_gradient_is_not_graying():
    # 越亮越淡（混白）是正常的“粉彩”，不是发灰
    h, w = 80, 200
    t = np.linspace(0, 1, w)
    L = 40 + 45 * t
    C = 50 * (1 - t) + 8
    img = np.stack([np.array(lch_color(l, c, 255)) for l, c in zip(L, C)])[None].repeat(h, 0).astype(np.uint8)
    r = run(img)
    assert r.graying_fraction < 0.05


# ------------------------------------------------------------------ 局部发灰
def test_same_lightness_lower_chroma_patch_is_flagged_as_graying():
    clean = lch_color(50, 45, 265)
    grey = lch_color(50, 15, 265)
    img = noisy(flat(clean, 120, 160), 1.5)
    img[40:100, 90:150] = noisy(flat(grey, 60, 60), 1.5, seed=2)
    r = run(img)
    assert r.graying_fraction > 0.15
    g = [b for b in r.blobs if b.kind == "graying"]
    assert g
    x, y, w, h = max(g, key=lambda b: b.area_fraction).bbox
    assert x >= 70 and y >= 25 and x + w <= 160


# ------------------------------------------------------------------ 肤色模式
def test_skin_with_green_and_purple_anomalies_named():
    img = skin_region()
    img = sprinkle(img, lch_color(60, 18, 140), 0.03, seed=1)  # 绿
    img = sprinkle(img, lch_color(55, 20, 320), 0.04, seed=2)  # 紫
    r = ct.analyze(img, mode="skin")
    names = dict(r.anomaly_fractions)
    assert names.get("绿", 0) == pytest.approx(0.03, rel=0.25)
    assert names.get("紫/品红", 0) == pytest.approx(0.04, rel=0.25)


def test_clean_skin_with_natural_shading_is_clean():
    r = ct.analyze(skin_region(), mode="skin")
    assert r.opposite_fraction < 0.002
    assert r.graying_fraction < 0.05


def test_skin_lips_like_red_is_not_an_anomaly():
    img = patch(skin_region(), lch_color(50, 40, 30), (40, 80, 30, 10))
    r = ct.analyze(img, mode="skin")
    assert r.opposite_fraction < 0.002


def test_empty_region_raises():
    with pytest.raises(ValueError):
        ct.analyze(np.zeros((0, 0, 3), np.uint8))


def test_neutral_region_has_no_base_hue():
    r = run(noisy(flat((128, 128, 128)), 1))
    assert r.base_hue is None
    assert r.base_hue_name == "中性灰"
    assert r.opposite_fraction == 0


def test_large_region_is_fast_and_bounded():
    import time

    img = sprinkle(noisy(flat(BLUE, 2000, 3000), 2), YELLOW, 0.03)
    t = time.perf_counter()
    r = run(img)
    assert time.perf_counter() - t < 5.0
    assert r.mask.shape[0] * r.mask.shape[1] <= ct.MAX_SAMPLES
    assert r.mask_scale > 1
    assert r.opposite_fraction == pytest.approx(0.03, rel=0.25)
