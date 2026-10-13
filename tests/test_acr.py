"""ACR 预设导出与滑块实测校准。"""

import xml.dom.minidom

import numpy as np
import pytest

from colordoctor.core import acr_calibration as cal
from colordoctor.core import acr_channels as ac
from colordoctor.core import acr_preset as ap
from colordoctor.core import color_spaces as cs
from colordoctor.core import palette as pl
from colordoctor.core.color_models import Adjustment


@pytest.fixture(autouse=True)
def _no_user_calibration():
    ac.set_calibration({})
    yield
    ac.set_calibration({})


def test_xmp_is_well_formed_and_contains_only_given_settings():
    params = {
        "hsl": {"绿色": {"hue": 60, "sat": -60}, "Blue": {"hue": -30}},
        "grading": {"shadow": {"hue": 200, "sat": 18}, "highlight": {"hue": 40, "sat": 14}, "blending": 60},
        "calibration": {"blue": {"hue": -50, "sat": 20}},
        "curve": {"composite": [[64, 58], [192, 200]]},
        "basic": {"Contrast2012": -15, "Exposure2012": 0.3},
    }
    x = ap.build_xmp("青橙 · 测试", params)
    xml.dom.minidom.parseString(x)  # 合法 XML
    for frag in ('crs:HueAdjustmentGreen="+60"', 'crs:SaturationAdjustmentGreen="-60"', 'crs:HueAdjustmentBlue="-30"',
                 'crs:SplitToningShadowHue="200"', 'crs:SplitToningHighlightSaturation="14"', 'crs:ColorGradeBlending="60"',
                 'crs:BlueHue="-50"', 'crs:BlueSaturation="20"', 'crs:Contrast2012="-15"', 'crs:Exposure2012="+0.30"',
                 "<rdf:li>0, 0</rdf:li>", "<rdf:li>64, 58</rdf:li>", "<rdf:li>255, 255</rdf:li>", "青橙 · 测试"):
        assert frag in x, frag
    assert "HueAdjustmentRed" not in x and "Vibrance" not in x


def test_xmp_clamps_and_rejects_unknown():
    attrs, _ = ap.settings_from_params({"hsl": {"红色": {"hue": 250}}, "grading": {"shadow": {"hue": 400, "sat": 150}}})
    assert attrs["HueAdjustmentRed"] == "+100" and attrs["SplitToningShadowSaturation"] == "100"
    with pytest.raises(ValueError):
        ap.settings_from_params({"hsl": {"橙红": {"hue": 1}}})
    with pytest.raises(ValueError):
        ap.settings_from_params({"basic": {"Foo": 1}})


def test_params_from_style_adjustments():
    img = np.zeros((120, 160, 3), np.uint8)
    img[:] = cs.lab_to_srgb8(cs.lch_to_lab(np.array([55.0, 40.0, 140.0])))  # 大片绿色
    img[:, :60] = cs.lab_to_srgb8(cs.lch_to_lab(np.array([65.0, 30.0, 60.0])))
    m = pl.style_match(img, pl.get_style("teal_orange"))
    p = ap.params_from_adjustments(m.adjustments)
    assert "绿色" in p["hsl"]
    x = ap.build_xmp("t", p)
    assert "AdjustmentGreen" in x
    adj = [Adjustment("Camera Raw 颜色分级", "阴影：色相 200，饱和度 12", None, True, (("色相", 200), ("饱和度", 12)))]
    assert ap.params_from_adjustments(adj) == {"grading": {"shadow": {"hue": 200, "sat": 12}}}


# ------------------------------------------------------------------ 校准
def _fake_acr(img, channel, slider, value, patches, k):
    """模拟 ACR：只改该通道的色块。hue: 每格 k 度；sat: 色度 ×(1+k·v)；lum: 每格 k 个 L*。"""
    out = img.copy()
    for p in patches:
        if p.channel != channel:
            continue
        x, y, w, h = p.rect
        lab = cs.srgb8_to_lab(out[y:y + h, x:x + w])
        if slider == "hue":
            hsb = cs.srgb8_to_hsb(out[y:y + h, x:x + w]).astype(float)
            hsb[..., 0] = (hsb[..., 0] + k * value) % 360
            out[y:y + h, x:x + w] = cs.hsb_to_srgb8(hsb)
            continue
        if slider == "sat":
            lab[..., 1:] *= 1 + k * value
        else:
            lab[..., 0] += k * value
        out[y:y + h, x:x + w] = cs.lab_to_srgb8(lab)
    return out


def test_chart_layout():
    img, patches = cal.chart()
    assert len(patches) == len(ac.CHANNELS) * len(cal.LIGHTNESS) + 3
    m = cal.measure(img, patches)
    for p, lab in zip(patches, m, strict=True):
        assert np.allclose(lab, p.lab, atol=1.0)
    hues = [cs.srgb8_to_hsb(cs.lab_to_srgb8(np.array(p.lab)))[0] for p in patches if p.channel == "绿色"]
    assert all(abs(((h - 120 + 180) % 360) - 180) < 6 for h in hues)


def test_fit_recovers_simulated_slider_response(tmp_path, monkeypatch):
    img, patches = cal.chart()
    samples = []
    for v in (-50, 50):
        samples.append(("橙色", "hue", v, _fake_acr(img, "橙色", "hue", v, patches, 0.2)))
        samples.append(("橙色", "sat", v, _fake_acr(img, "橙色", "sat", v, patches, 0.006)))
        samples.append(("橙色", "lum", v, _fake_acr(img, "橙色", "lum", v, patches, 0.15)))
    fit = cal.fit(img, samples, patches)
    o = fit["橙色"]
    assert o["hue_per_unit"] == pytest.approx(0.2, abs=0.03)
    assert o["sat_per_unit"] == pytest.approx(0.006, abs=0.0015)
    assert o["lum_per_unit"] == pytest.approx(0.15, abs=0.03)
    # 保存后换算改用实测值：同样转 10°，实测 0.2°/格 -> 50 格（经验值 0.3 -> 33 格）
    monkeypatch.setattr("colordoctor.core.skin_presets.user_config_dir", lambda: tmp_path)
    path = cal.store(fit, "17.0")
    assert path.endswith("acr_calibration.json")
    assert ac.is_calibrated("橙色") and not ac.is_calibrated("蓝色")
    assert 47 <= ac.hue_slider(30, 40, channel="橙色") <= 53
    assert ac.hue_slider(30, 40, channel="蓝色") == 33
    ac.set_calibration(None)  # 重新从文件读
    assert 47 <= ac.hue_slider(30, 40, channel="橙色") <= 53
