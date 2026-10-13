import colorsys
import io

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image, ImageCms

from colordoctor.core import color_spaces as cs

# Sharma, Wu, Dalal (2005) CIEDE2000 test data: (Lab1, Lab2, ΔE00)
SHARMA = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0010), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0011), 7.2195),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0012), 7.2195),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0009, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0010, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0011, -2.4900), 4.7461),
    ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
    ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
    ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2972, 0.0000), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 1.8634, 0.5757), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2592, 0.3350), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


@pytest.mark.parametrize("lab1,lab2,expected", SHARMA)
def test_delta_e_2000_matches_sharma_reference(lab1, lab2, expected):
    assert cs.delta_e_2000(np.array(lab1), np.array(lab2)) == pytest.approx(expected, abs=1e-4)


def test_delta_e_2000_is_vectorised_and_symmetric():
    a = np.array([p[0] for p in SHARMA])
    b = np.array([p[1] for p in SHARMA])
    d1 = cs.delta_e_2000(a, b)
    d2 = cs.delta_e_2000(b, a)
    assert d1.shape == (len(SHARMA),)
    np.testing.assert_allclose(d1, d2, atol=1e-9)


def test_srgb_red_lab_d50_matches_photoshop_style_bradford_value():
    # Bruce Lindbloom sRGB (Bradford-adapted) red in D50 Lab
    lab = cs.srgb8_to_lab(np.array([255, 0, 0]))
    np.testing.assert_allclose(lab, [54.2917, 80.8125, 69.8851], atol=0.02)


def test_neutral_greys_have_zero_ab_in_d50_and_d65():
    for v in (0, 1, 37, 128, 200, 255):
        lab = cs.srgb8_to_lab(np.array([v, v, v]))
        lab65 = cs.srgb8_to_lab(np.array([v, v, v]), white="D65")
        assert abs(lab[1]) < 1e-3 and abs(lab[2]) < 1e-3
        assert abs(lab65[1]) < 1e-3 and abs(lab65[2]) < 1e-3
    assert cs.srgb8_to_lab(np.array([255, 255, 255]))[0] == pytest.approx(100.0, abs=1e-6)


def test_srgb_red_lab_d65_matches_cie_reference():
    lab = cs.srgb8_to_lab(np.array([255, 0, 0]), white="D65")
    np.testing.assert_allclose(lab, [53.2408, 80.0925, 67.2032], atol=0.01)


def test_lab_d50_agrees_with_littlecms_within_8bit_encoding():
    srgb = ImageCms.createProfile("sRGB")
    lab_p = ImageCms.createProfile("LAB", 5000)
    t = ImageCms.buildTransform(srgb, lab_p, "RGB", "LAB")
    rng = np.random.default_rng(1)
    colors = rng.integers(0, 256, (64, 3)).astype(np.uint8)
    img = Image.fromarray(colors.reshape(8, 8, 3), "RGB")
    out = np.asarray(ImageCms.applyTransform(img, t)).reshape(-1, 3).astype(float)
    L = out[:, 0] * 100 / 255
    a = out[:, 1].copy()
    b = out[:, 2].copy()
    a[a > 127] -= 256
    b[b > 127] -= 256
    ours = cs.srgb8_to_lab(colors)
    assert np.max(np.abs(ours[:, 0] - L)) < 1.0
    assert np.max(np.abs(ours[:, 1] - a)) < 1.5
    assert np.max(np.abs(ours[:, 2] - b)) < 1.5


@given(st.tuples(st.integers(0, 255), st.integers(0, 255), st.integers(0, 255)))
@settings(max_examples=200, deadline=None)
def test_lab_roundtrip_returns_same_8bit_rgb(rgb):
    lab = cs.srgb8_to_lab(np.array(rgb))
    back = cs.lab_to_srgb8(lab)
    assert tuple(int(v) for v in back) == rgb


def test_oklab_reference_values():
    np.testing.assert_allclose(cs.srgb8_to_oklab(np.array([255, 255, 255])), [1.0, 0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(
        cs.srgb8_to_oklab(np.array([255, 0, 0])), [0.62796, 0.22486, 0.12585], atol=1e-4
    )
    np.testing.assert_allclose(
        cs.srgb8_to_oklab(np.array([0, 0, 255])), [0.45201, -0.03246, -0.31153], atol=1e-4
    )


def test_oklch_and_lch_hue_in_degrees_0_360():
    lch = cs.lab_to_lch(cs.srgb8_to_lab(np.array([[0, 0, 255], [255, 255, 0]])))
    assert np.all((lch[:, 2] >= 0) & (lch[:, 2] < 360))
    ok = cs.oklab_to_oklch(cs.srgb8_to_oklab(np.array([0, 0, 255])))
    assert 260 < ok[2] < 270


@given(st.tuples(st.integers(0, 255), st.integers(0, 255), st.integers(0, 255)))
@settings(max_examples=200, deadline=None)
def test_hsb_and_hsl_match_colorsys(rgb):
    r, g, b = (c / 255 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    hsb = cs.srgb8_to_hsb(np.array(rgb))
    hsl = cs.srgb8_to_hsl(np.array(rgb))
    if s > 0:
        assert abs(((hsb[0] - h * 360) + 180) % 360 - 180) < 1e-6
    assert hsb[1] == pytest.approx(s * 100, abs=1e-6)
    assert hsb[2] == pytest.approx(v * 100, abs=1e-6)
    assert hsl[1] == pytest.approx(ss * 100, abs=1e-6)
    assert hsl[2] == pytest.approx(ll * 100, abs=1e-6)


@given(st.tuples(st.integers(0, 255), st.integers(0, 255), st.integers(0, 255)))
@settings(max_examples=200, deadline=None)
def test_hsb_and_hsl_roundtrip_exactly(rgb):
    assert tuple(cs.hsb_to_srgb8(cs.srgb8_to_hsb(np.array(rgb)))) == rgb
    assert tuple(cs.hsl_to_srgb8(cs.srgb8_to_hsl(np.array(rgb)))) == rgb


def test_relative_luminance():
    assert cs.relative_luminance(np.array([255, 255, 255])) == pytest.approx(1.0)
    assert cs.relative_luminance(np.array([0, 0, 0])) == pytest.approx(0.0)
    assert cs.relative_luminance(np.array([255, 0, 0])) == pytest.approx(0.2126729, abs=1e-6)


def test_hex():
    assert cs.to_hex(np.array([255, 128, 0])) == "#FF8000"


@pytest.mark.parametrize(
    "rgb,name",
    [
        ((255, 0, 0), "红"),
        ((255, 140, 0), "橙"),
        ((255, 230, 0), "黄"),
        ((0, 200, 0), "绿"),
        ((0, 220, 220), "青"),
        ((0, 0, 255), "蓝"),
        ((200, 0, 200), "紫/品红"),
        ((128, 128, 128), "中性灰"),
    ],
)
def test_hue_names_for_primaries(rgb, name):
    lch = cs.lab_to_lch(cs.srgb8_to_lab(np.array(rgb)))
    assert cs.hue_name(lch[2], lch[1]) == name


def test_circular_hue_difference():
    assert cs.hue_diff(350.0, 10.0) == pytest.approx(20.0)
    assert cs.hue_diff(10.0, 350.0) == pytest.approx(-20.0)
    assert abs(cs.hue_diff(0.0, 180.0)) == pytest.approx(180.0)


def test_arrays_of_images_are_supported():
    img = np.zeros((4, 5, 3), dtype=np.uint8)
    img[..., 2] = 255
    lab = cs.srgb8_to_lab(img)
    assert lab.shape == (4, 5, 3)
