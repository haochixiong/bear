"""肤色预设与人像肤色分析。"""

import json

import numpy as np
import pytest

from colordoctor.core import color_spaces as cs
from colordoctor.core import portrait_analysis as pa
from colordoctor.core import skin_presets as sp
from synth import lch_color, skin_region, sprinkle


# ------------------------------------------------------------------ 预设
def test_builtin_presets_exist_and_are_consistent():
    presets = sp.builtin_presets()
    keys = [p.key for p in presets]
    for k in ("natural", "japanese", "french", "moody", "cosplay"):
        assert k in keys
    for p in presets:
        L, a, b = p.target_lab
        _, C, h = cs.lab_to_lch(np.array([L, a, b]))
        assert p.hue[0] <= h <= p.hue[1], p.key
        assert p.chroma[0] <= C <= p.chroma[1], p.key
        assert cs.lab_in_srgb_gamut(np.array(p.target_lab))
        assert a > 0 and b > 0


def test_style_presets_are_ordered_sensibly():
    P = {p.key: p for p in sp.builtin_presets()}
    assert P["cosplay"].target_lab[0] > P["japanese"].target_lab[0] > P["natural"].target_lab[0] > P["moody"].target_lab[0]
    assert P["french"].target_lab[2] > P["natural"].target_lab[2] > P["japanese"].target_lab[2]


def test_user_preset_file_overrides_and_adds(tmp_path):
    f = tmp_path / "skin_presets.json"
    f.write_text(
        json.dumps(
            {
                "presets": [
                    {"key": "natural", "name": "我的自然", "L": [60, 70], "a": [10, 16], "b": [14, 22], "hue": [45, 62], "chroma": [17, 27]},
                    {"key": "film", "name": "胶片", "L": [60, 70], "a": [10, 15], "b": [16, 24], "hue": [50, 66], "chroma": [19, 28]},
                ]
            }
        ),
        encoding="utf-8",
    )
    presets = sp.load_presets(f)
    names = {p.key: p.name for p in presets}
    assert names["natural"] == "我的自然"
    assert "film" in names and "japanese" in names


def test_broken_user_preset_file_is_ignored(tmp_path):
    f = tmp_path / "skin_presets.json"
    f.write_text("{not json", encoding="utf-8")
    assert len(sp.load_presets(f)) == len(sp.builtin_presets())


def test_preset_from_sample():
    p = sp.preset_from_sample((70.0, 12.0, 18.0))
    assert p.target_lab == pytest.approx((70.0, 12.0, 18.0))


# ------------------------------------------------------------------ 肤色分析
@pytest.mark.parametrize(
    "lab", [(32.0, 9.0, 12.0), (45.0, 12.0, 16.5), (58.0, 13.0, 18.0), (68.0, 14.0, 19.0), (82.0, 10.5, 14.5)]
)
def test_different_skin_depths_with_natural_hue_are_normal(lab):
    img = skin_region(*lab)  # 色相 ~54°，从深到浅
    r = pa.analyze_skin(img)
    assert r.tendency == "正常"
    assert abs(r.highlight_hue_shift) < 5
    assert abs(r.shadow_hue_shift) < 5
    assert not r.mixed_region


def test_depth_label_uses_ita():
    light = pa.analyze_skin(skin_region(L=80, a=10, b=14))
    dark = pa.analyze_skin(skin_region(L=35, a=10, b=14))
    assert light.ita > dark.ita
    assert light.depth_label != dark.depth_label


@pytest.mark.parametrize(
    "lab,expected",
    [
        ((65, 22, 12), "偏红"),
        ((65, 9, 26), "偏黄"),
        ((65, 1.5, 20), "偏绿"),
        ((65, 16, 2), "偏洋红"),
        ((65, 14, 19), "正常"),
    ],
)
def test_overall_tendency(lab, expected):
    r = pa.analyze_skin(skin_region(*lab))
    assert r.tendency == expected


def test_warm_french_style_skin_is_not_called_yellow_error():
    r = pa.analyze_skin(skin_region(69, 13.5, 22))
    assert r.tendency == "正常"


def _modify_by_lightness(img, select, da=0.0, db=0.0):
    lab = cs.srgb8_to_lab(img)
    L = lab[..., 0]
    lo, hi = np.percentile(L, [15, 85])
    m = (L >= hi) if select == "high" else (L <= lo)
    lab[..., 1][m] += da
    lab[..., 2][m] += db
    return cs.lab_to_srgb8(lab)


def test_yellow_highlights_detected():
    img = _modify_by_lightness(skin_region(), "high", da=-2, db=9)
    r = pa.analyze_skin(img)
    assert r.highlight_hue_shift > 8
    assert r.highlight_shift[2] > 5


def test_cyan_shadows_detected():
    img = _modify_by_lightness(skin_region(), "low", da=-8, db=-2)
    r = pa.analyze_skin(img)
    assert r.shadow_hue_shift > 10
    assert r.shadow_shift[1] < -5


def test_low_saturation_skin_reported_against_expected_range():
    r = pa.analyze_skin(skin_region(68, 6, 8))
    lo, hi = r.expected_saturation
    assert r.saturation_ab < lo


def test_region_with_background_is_marked_mixed():
    img = skin_region()
    img[:, :50] = lch_color(40, 30, 250)  # 框进了蓝色背景
    r = pa.analyze_skin(img)
    assert r.mixed_region


def test_skin_anomalies_reported():
    img = sprinkle(skin_region(), lch_color(60, 18, 200), 0.05, seed=4)  # 青色
    r = pa.analyze_skin(img)
    assert dict(r.anomaly_fractions).get("青", 0) == pytest.approx(0.05, rel=0.25)


def test_compare_to_preset():
    p = sp.get_preset("japanese")
    r = pa.analyze_skin(skin_region(68, 14, 19))
    d = pa.compare_to_preset(r, p)
    assert d.dL > 5  # 需要提亮
    assert d.db < -3  # 需要减黄
    assert d.delta_e00 > 5


def test_tendency_continuous_and_non_negative():
    import math

    from colordoctor.core.portrait_analysis import tendency_of

    def at(h, c=20.0):
        return tendency_of(c * math.cos(math.radians(h)), c * math.sin(math.radians(h)))

    t0, t359 = at(0.5), at(359.5)
    assert t0[0] == t359[0] == "偏洋红"
    assert abs(t0[1] - t359[1]) < 2  # 0° 两侧连续
    assert at(351.5)[1] > at(8.5)[1]  # 越往外超出越多
    assert tendency_of(2.0, 1.0)[1] >= 0 and tendency_of(1.5, 3.0)[1] >= 0
    assert tendency_of(-3.0, -4.0)[0] == "偏青/偏蓝"
    assert tendency_of(-3.0, 6.0)[0] == "偏绿"
    assert at(55)[0] == "正常"


def test_pale_skin_not_flagged_grey_with_default_preset():
    from colordoctor.core import analysis as an
    from colordoctor.core import skin_presets as sp
    from colordoctor.core.color_management import synthetic_context

    img = skin_region(80, 8, 14, h=120, w=120)  # 浅肤色，C*/L* ≈ 0.2（日系 / 韩系常见）
    rep = an.analyze_skin_region(img, synthetic_context(), preset=sp.get_preset("natural", sp.builtin_presets()), auto_mask=False)
    lo, _ = rep.skin.expected_saturation
    assert lo <= 0.16
    assert not any("发灰" in f.problem for f in rep.findings), [f.problem for f in rep.findings]
