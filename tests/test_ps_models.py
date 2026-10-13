"""Photoshop 调整的近似模型与参数求解。"""

import numpy as np
import pytest

from colordoctor.core import color_spaces as cs
from colordoctor.core import ps_models as pm


def rgbf(*c):
    return np.array([c], dtype=float) / 255.0


def test_color_balance_zero_is_identity():
    x = np.random.default_rng(0).random((50, 3))
    np.testing.assert_allclose(pm.color_balance(x), x, atol=1e-9)


def test_color_balance_midtones_red_raises_red_in_midtones_more_than_shadows():
    mid = rgbf(128, 128, 128)
    dark = rgbf(15, 15, 15)
    m2 = pm.color_balance(mid, midtones=(30, 0, 0), preserve_luminosity=False)
    d2 = pm.color_balance(dark, midtones=(30, 0, 0), preserve_luminosity=False)
    assert m2[0, 0] - mid[0, 0] > 0.1
    assert d2[0, 0] - dark[0, 0] < m2[0, 0] - mid[0, 0]


def test_color_balance_preserve_luminosity_keeps_hsl_lightness():
    x = rgbf(180, 140, 120)
    y = pm.color_balance(x, midtones=(-20, 10, 15), preserve_luminosity=True)
    lx = (x.max() + x.min()) / 2
    ly = (y.max() + y.min()) / 2
    assert ly == pytest.approx(lx, abs=1e-6)


def test_hue_saturation_master_hue_shift():
    y = pm.hue_saturation(rgbf(255, 0, 0), hue=120)
    np.testing.assert_allclose(y, rgbf(0, 255, 0), atol=1e-6)


def test_hue_saturation_desaturate():
    y = pm.hue_saturation(rgbf(200, 100, 50), saturation=-100)
    assert np.ptp(y) < 1e-6


def test_hue_saturation_reds_range_leaves_blue_alone():
    blue = rgbf(30, 60, 200)
    np.testing.assert_allclose(pm.hue_saturation(blue, hue=20, color_range="reds"), blue, atol=1e-9)
    red = rgbf(200, 40, 30)
    assert not np.allclose(pm.hue_saturation(red, hue=20, color_range="reds"), red)


def test_selective_color_reds_cyan_reduces_red_only_in_reds():
    red = rgbf(220, 60, 40)
    grey = rgbf(128, 128, 128)
    blue = rgbf(40, 60, 220)
    assert pm.selective_color(red, "reds", cyan=50)[0, 0] < red[0, 0]
    np.testing.assert_allclose(pm.selective_color(grey, "reds", cyan=50), grey)
    np.testing.assert_allclose(pm.selective_color(blue, "reds", cyan=50), blue)


def test_curves_points_map_mean_exactly():
    pts = pm.curves_points((200, 150, 120), (210, 150, 110))
    assert pts == {"红": (200, 210), "绿": (150, 150), "蓝": (120, 110)}
    x = rgbf(200, 150, 120)
    y = pm.apply_curves(x, pts)
    np.testing.assert_allclose(y * 255, [[210, 150, 110]], atol=1e-6)


def test_apply_curves_is_monotonic_and_keeps_endpoints():
    pts = {"红": (128, 150), "绿": (128, 128), "蓝": (128, 100)}
    ramp = np.stack([np.linspace(0, 1, 256)] * 3, axis=1)
    y = pm.apply_curves(ramp, pts)
    assert np.all(np.diff(y, axis=0) >= -1e-12)
    np.testing.assert_allclose(y[0], 0, atol=1e-9)
    np.testing.assert_allclose(y[-1], 1, atol=1e-9)


def test_solver_recovers_a_color_balance_move():
    rng = np.random.default_rng(1)
    px = np.clip(np.array([200, 150, 125]) / 255 + rng.normal(0, 0.03, (300, 3)), 0, 1)
    moved = pm.color_balance(px, midtones=(12, -6, -10))
    target = cs.srgb8_to_lab(np.rint(moved * 255)).mean(axis=0)
    sol = pm.solve(px, target, "color_balance")
    assert sol.residual_de < 0.8
    assert sol.residual_de < sol.start_de / 4


def test_solver_for_hue_sat_and_selective_color_reduce_error():
    rng = np.random.default_rng(2)
    px = np.clip(np.array([214, 160, 140]) / 255 + rng.normal(0, 0.02, (300, 3)), 0, 1)
    target = np.array([72.0, 12.0, 14.0])
    for tool in ("hue_saturation", "selective_color"):
        sol = pm.solve(px, target, tool)
        assert sol.residual_de < sol.start_de, tool
        assert all(-100 <= v <= 100 for _, v in sol.params)
