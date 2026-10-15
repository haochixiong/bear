"""原图尺寸分析（每个像素都算）与取点精度的回归测试。"""

import numpy as np
import pytest
from synth import BLUE, YELLOW, face_scene, flat, noisy, sprinkle

from colordoctor.bridge import server as bs
from colordoctor.core import color_spaces as cs
from colordoctor.core import palette as pl
from colordoctor.core import reference_match as rm
from colordoctor.core.color_management import synthetic_context
from colordoctor.core.pixel_table import PixelTable
from colordoctor.core.sampling import sample_point


# ------------------------------------------------------------------ 按“不同的颜色 × 像素数”加权 = 逐个像素算
def test_weighted_percentile_equals_numpy_on_all_pixels():
    rng = np.random.default_rng(0)
    v = rng.integers(0, 40, 5000).astype(float)  # 大量重复值
    u, w = np.unique(v, return_counts=True)
    q = [0, 5, 25, 50, 75, 95, 100]
    assert np.allclose(cs.weighted_percentile(u, w, q), np.percentile(v, q))


def test_per_colour_conversion_equals_direct():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (800, 600, 3)).astype(np.uint8)  # 48 万像素：走“每种颜色算一次”的路径
    img[:400] = img[400:]  # 有重复的颜色
    fast = cs.srgb8_to_lab(img)
    direct = cs.xyz_to_lab(cs.srgb8_to_xyz(img.astype(np.float64)))
    assert np.abs(np.asarray(fast, float) - direct).max() < 1e-3


def test_pixel_table_counts_every_pixel():
    img = noisy(flat(BLUE, 700, 700), 3)
    pt = PixelTable(img)
    assert pt.count().sum() == 700 * 700
    assert np.array_equal(pt.colours[pt.index], img)


def test_family_unity_counts_every_pixel():
    img = noisy(flat(BLUE, 600, 800), 2)
    img[:, :8] = (230, 200, 40)  # 窄窄一条黄
    fu = pl.family_unity(img, "黄")
    assert abs(fu.share - 8 / 800) < 1e-3
    lab = cs.srgb8_to_lab(img[:, :8].reshape(-1, 3))
    assert np.allclose(fu.mean_lab, lab.mean(axis=0), atol=0.05)


def test_reference_match_uses_every_pixel():
    """仿色的影调分位数和逐个像素的 np.percentile 一样（以前只看降采样后的像素）。"""
    mine = face_scene()[0]
    ref = np.ascontiguousarray(mine[::-1, ::-1] // 2 + 60)
    r = rm.match_reference(mine, ref)
    lm = cs.srgb8_to_lab(mine.reshape(-1, 3))[:, 0]
    assert np.allclose(r.tone_mine, np.round(np.percentile(lm, rm.PCTS), 1), atol=0.051)


def test_reference_curve_with_endpoint_point_has_no_duplicate_x():
    """参考的暗部是死黑 / 亮部是死白时，曲线点会落在 0 或 255 上：端点只能有一个（以前样条除以 0）。"""
    mine = np.full((60, 60, 3), 128, np.uint8)
    mine[:, :20] = 60
    mine[:, 40:] = 200
    ref = np.full((60, 60, 3), 128, np.uint8)
    ref[:, :20] = 0
    ref[:, 40:] = 255
    with np.errstate(all="raise"):
        r = rm.match_reference(mine, ref)
    xs = [x for x, _ in r.curve_points]
    assert len(xs) == len(set(xs))


def test_single_pixel_speckles_found_at_full_size():
    """3000×2000 的图里零星的单个杂色像素：原图尺寸检查能找到（降采样会把它们平均掉）。"""
    from colordoctor.core import contamination as ct

    img = noisy(flat(BLUE, 2000, 3000), 2)
    img[800:1000, 1200:1400] = sprinkle(img[800:1000, 1200:1400], YELLOW, 0.2, seed=3)
    r = ct.analyze(img, "region")
    assert r.mask_scale == 1 and r.mask.shape == img.shape[:2]
    assert r.opposite_fraction > 0.001


# ------------------------------------------------------------------ 取色 / 取点
def test_patch_uses_exact_foreground_lab():
    px = np.full((1, 1, 3), (200, 150, 120), np.uint8)
    out = bs.patch_values(px, lab=(66.43, 12.37, 18.91))
    assert out["values"]["lab"] == pytest.approx([66.43, 12.37, 18.91], abs=1e-3)


def test_feathered_selection_is_weighted_by_selection():
    px = np.zeros((1, 3, 4), np.uint8)
    px[0, 0] = (200, 100, 100, 255)
    px[0, 1] = (100, 200, 100, 64)  # 羽化边缘：只选中 1/4
    px[0, 2] = (0, 0, 255, 0)  # 没选中
    out = bs.patch_values(px)
    expect = (np.array([200, 100, 100]) * 1.0 + np.array([100, 200, 100]) * (64 / 255)) / (1 + 64 / 255)
    assert np.allclose(out["values"]["rgb"], np.round(expect), atol=1) and out["pixels"] == 2


def test_point_sample_at_edge_reports_real_pixel_count():
    img = np.zeros((20, 20, 3), np.uint8)
    p = sample_point(img, 0, 0, 2, synthetic_context())
    assert p.pixel_count == 9 and "靠边" in p.size_label and "9" in p.size_label
    p = sample_point(img, 10, 10, 2, synthetic_context())
    assert p.pixel_count == 25 and p.size_label == "5×5"


def test_single_odd_pixel_in_a_solid_layer_is_still_analysed():
    """一整块纯色里只有一个杂色像素：以前只抽查 20 万个像素判断“纯色”，会漏掉它、拒绝分析。"""
    from colordoctor.core import pixel_guard as pg

    img = np.zeros((1000, 1000, 3), np.uint8)
    img[:] = BLUE
    img[777, 333] = YELLOW
    pg.prepare(img)
    img[777, 333] = BLUE
    with pytest.raises(pg.BlockedError, match="纯色"):
        pg.prepare(img)
