"""仿色对比与整图色彩报告。"""

import numpy as np

from colordoctor.core import analysis as an
from colordoctor.core import reference_match as rm
from colordoctor.core.color_management import synthetic_context
from colordoctor.core.color_models import Severity
from synth import flat, lch_color, noisy

CTX = synthetic_context()


def _scene(bg_h=210, bg_L=42, subject=(66, 32, 60), cast=(0.0, 0.0)):
    img = flat(lch_color(bg_L, 30, bg_h), 160, 200)
    img[40:140, 60:140] = lch_color(*subject)
    img[140:] = lch_color(80, 2, 60)  # 浅灰地面（中性区域）
    return noisy(img, 2)


def test_identical_images_are_very_similar():
    a = _scene()
    m = rm.match_reference(a, a)
    assert m.similarity > 95
    assert not m.curve_points
    assert "适合" in m.fit[0]


def test_darker_cooler_reference_gives_curve_and_temperature():
    mine = _scene()
    ref = _scene(bg_L=30)
    ref = (ref.astype(float) * 0.8).astype(np.uint8)  # 参考更暗
    m = rm.match_reference(mine, ref)
    assert m.curve_points and all(b < a for a, b in m.curve_points)
    assert m.similarity < 95


def test_reference_with_missing_main_color_is_flagged():
    mine = _scene(bg_h=140)  # 我的背景是绿色
    ref = _scene(bg_h=300)  # 参考背景是紫色
    m = rm.match_reference(mine, ref)
    assert any("没有接近的颜色" in f for f in m.fit)
    names = " ".join(a.instruction for a in m.adjustments)
    assert "绿色" in names or "紫色" in names


def test_skin_delta_reported():
    a = _scene()
    m = rm.match_reference(a, a, skin_mine=(68, 14, 19), skin_ref=(72, 12, 15))
    assert m.skin_delta[0] > 3 and m.skin_delta[1] == 4.0


def test_compare_reference_report():
    r = an.compare_reference(_scene(bg_h=140), _scene(bg_h=210), CTX)
    assert r.mode == "reference" and r.reference is not None
    assert any("相似度" in f.problem for f in r.findings)
    assert any("合不合适" in f.problem for f in r.findings)


def test_palette_report_flags_missing_color():
    grey = noisy(flat((120, 118, 115), 150, 200), 2)
    r = an.analyze_palette_report(grey, CTX)
    assert any("缺色" in f.problem for f in r.findings)
    assert r.palette is not None


def test_palette_report_with_style_and_unity():
    img = _scene()
    r = an.analyze_palette_report(img, CTX, style_key="teal_orange", family="橙")
    probs = " ".join(f.problem for f in r.findings)
    assert "青橙" in probs and "统一度" in probs and "6:3:1" in probs
    st = [f for f in r.findings if "青橙" in f.problem][0]
    assert st.severity in (Severity.NORMAL, Severity.MILD)


def test_grading_direction_uses_difference():
    """我的阴影偏黄（b* +10），参考阴影 b* +3：应往蓝（色相约 200–300），而不是继续加黄。"""
    import numpy as np

    from colordoctor.core import color_spaces as cs
    from colordoctor.core import reference_match as rm

    def scene(shadow_b):
        img = np.zeros((120, 160, 3), np.uint8)
        L = np.linspace(8, 90, 160)[None, :].repeat(120, 0)
        lab = np.stack([L, np.zeros_like(L), np.where(L < 30, shadow_b, 0.0)], -1)
        img[:] = cs.lab_to_srgb8(lab)
        return img

    m = rm.match_reference(scene(10.0), scene(3.0))
    g = [a for a in m.adjustments if a.tool == "Camera Raw 颜色分级" and a.instruction.startswith("阴影")]
    assert g, [a.instruction for a in m.adjustments]
    hue = dict(g[0].params)["色相"]
    assert 180 <= hue <= 300, hue
