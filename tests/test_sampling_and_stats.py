import numpy as np
import pytest

from colordoctor.core import clustering, region_statistics as rs, sampling
from colordoctor.core import color_spaces as cs
from colordoctor.core.color_management import synthetic_context
from synth import BLUE, YELLOW, flat, noisy, sprinkle


# ------------------------------------------------------------------ 取样
def test_point_patch_sizes():
    img = np.arange(10 * 10 * 3, dtype=np.uint8).reshape(10, 10, 3)
    assert sampling.point_patch(img, 5, 5, 0).shape == (1, 3)
    assert sampling.point_patch(img, 5, 5, 1).shape == (9, 3)
    assert sampling.point_patch(img, 5, 5, 2).shape == (25, 3)


def test_point_patch_clipped_at_border():
    img = flat((1, 2, 3), 10, 10)
    assert sampling.point_patch(img, 0, 0, 2).shape == (9, 3)
    assert sampling.point_patch(img, 9, 9, 1).shape == (4, 3)


def test_point_outside_image_raises():
    img = flat((1, 2, 3), 10, 10)
    with pytest.raises(sampling.SampleError):
        sampling.point_patch(img, 10, 3, 2)
    with pytest.raises(sampling.SampleError):
        sampling.point_patch(img, -1, 3, 0)


def test_empty_image_raises():
    with pytest.raises(sampling.SampleError):
        sampling.point_patch(np.zeros((0, 0, 3), np.uint8), 0, 0, 0)
    with pytest.raises(sampling.SampleError):
        rs.compute_stats(np.zeros((0, 5, 3), np.uint8))


def test_sample_point_averages_like_photoshop_eyedropper():
    img = flat((100, 100, 100), 5, 5)
    img[2, 2] = (200, 100, 100)
    s1 = sampling.sample_point(img, 2, 2, 0, synthetic_context())
    s5 = sampling.sample_point(img, 2, 2, 2, synthetic_context())
    assert s1.values.rgb == (200, 100, 100)
    assert s5.values.rgb == (104, 100, 100)
    assert s5.size_label == "5×5"
    assert s5.pixel_count == 25


def test_color_values_complete():
    v = sampling.color_values(np.array([255.0, 0.0, 0.0]))
    assert v.hex == "#FF0000"
    assert v.lab == pytest.approx((54.29, 80.81, 69.89), abs=0.02)
    assert v.lab_d65 == pytest.approx((53.24, 80.09, 67.20), abs=0.02)
    assert v.lch[2] == pytest.approx(40.85, abs=0.05)
    assert v.hsb == pytest.approx((0, 100, 100))
    assert v.hsl == pytest.approx((0, 100, 50))
    assert v.luminance == pytest.approx(0.2126729, abs=1e-6)
    assert v.hue_name == "红"


def test_clamp_rect():
    assert sampling.clamp_rect((-5, -5, 20, 20), 10, 10) == (0, 0, 10, 10)
    assert sampling.clamp_rect((5, 5, 100, 2), 10, 10) == (5, 5, 5, 2)
    assert sampling.clamp_rect((20, 20, 5, 5), 10, 10) is None
    assert sampling.clamp_rect((3, 3, 0, 4), 10, 10) is None


def test_grid_downsample_is_deterministic_and_bounded():
    img = np.random.default_rng(0).integers(0, 256, (1000, 1500, 3)).astype(np.uint8)
    a, step_a = sampling.grid_downsample(img, 50_000)
    b, step_b = sampling.grid_downsample(img, 50_000)
    assert np.array_equal(a, b) and step_a == step_b
    assert a.shape[0] * a.shape[1] <= 50_000
    small, step = sampling.grid_downsample(img[:10, :10], 50_000)
    assert step == 1 and small.shape == (10, 10, 3)


# ------------------------------------------------------------------ 圆形色相统计
def test_circular_mean_wraps_around_zero():
    mean, R, spread = rs.weighted_circular_hue(np.array([350.0, 10.0]), np.array([1.0, 1.0]))
    assert mean == pytest.approx(0.0, abs=1e-6) or mean == pytest.approx(360.0, abs=1e-6)
    assert R == pytest.approx(np.cos(np.radians(10)), abs=1e-9)


def test_circular_mean_weighted_by_chroma():
    mean, _, _ = rs.weighted_circular_hue(np.array([0.0, 90.0]), np.array([3.0, 1.0]))
    assert mean == pytest.approx(np.degrees(np.arctan2(1, 3)), abs=1e-6)


def test_opposite_hues_have_low_concentration():
    _, R, spread = rs.weighted_circular_hue(np.array([0.0, 180.0]), np.array([1.0, 1.0]))
    assert R == pytest.approx(0.0, abs=1e-9)
    assert spread > 100


# ------------------------------------------------------------------ 区域统计
def test_stats_uniform_blue():
    st = rs.compute_stats(flat(BLUE))
    assert st.total_pixels == 120 * 160
    assert st.sampled_pixels == st.total_pixels
    assert st.L_std == pytest.approx(0, abs=1e-6)
    assert st.hue_spread == pytest.approx(0, abs=1e-3)
    assert st.hue_concentration == pytest.approx(1.0)
    lab = cs.srgb8_to_lab(np.array(BLUE))
    assert st.lab_mean == pytest.approx(tuple(lab), abs=1e-3)
    assert st.clusters[0].share == pytest.approx(1.0)
    assert st.chromatic_fraction == 1.0


def test_stats_grey_has_undefined_hue():
    st = rs.compute_stats(noisy(flat((128, 128, 128)), 1.0))
    assert st.hue_mean is None
    assert st.chromatic_fraction < 0.05


def test_stats_percentiles_and_highlight_shadow():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:3] = 250
    img[3:6] = 128
    img[6:] = 10
    st = rs.compute_stats(img)
    assert st.highlight_fraction == pytest.approx(0.3)
    assert st.shadow_fraction == pytest.approx(0.4)
    assert st.L_p90 > 95 and st.L_p10 < 5
    assert st.clipped_fraction == pytest.approx(0.0)


def test_stats_large_region_is_sampled_and_reports_counts():
    img = noisy(flat(BLUE, 1200, 1600), 3)
    st = rs.compute_stats(img, max_samples=100_000)
    assert st.total_pixels == 1200 * 1600
    assert st.sampled_pixels <= 100_000
    assert st.sampled_pixels > 50_000


def test_tiny_region_one_pixel():
    st = rs.compute_stats(flat(BLUE, 1, 1))
    assert st.total_pixels == 1 and st.sampled_pixels == 1
    assert st.L_std == 0


def test_clusters_find_two_colours_with_shares():
    img = sprinkle(flat(BLUE), YELLOW, 0.2, seed=1)
    st = rs.compute_stats(img)
    names = [c.hue_name for c in st.clusters]
    assert names[0] == "蓝"
    assert "黄" in names
    yellow = [c for c in st.clusters if c.hue_name == "黄"][0]
    assert yellow.share == pytest.approx(0.2, abs=0.01)


# ------------------------------------------------------------------ k-means
def test_kmeans_deterministic():
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.normal(0, 1, (100, 3)), rng.normal(20, 1, (50, 3))])
    c1, l1 = clustering.kmeans(pts, 2)
    c2, l2 = clustering.kmeans(pts, 2)
    assert np.array_equal(c1, c2) and np.array_equal(l1, l2)
    counts = sorted(np.bincount(l1).tolist())
    assert counts == [50, 100]


def test_dominant_clusters_merge_similar_colours():
    lab = cs.srgb8_to_lab(noisy(flat(BLUE, 50, 50), 2).reshape(-1, 3))
    clusters = clustering.dominant_clusters(lab)
    assert len(clusters) == 1
    assert clusters[0][1] == pytest.approx(1.0)


def test_kmeans_handles_fewer_points_than_k():
    centers, labels = clustering.kmeans(np.zeros((2, 3)), 5)
    assert len(labels) == 2
