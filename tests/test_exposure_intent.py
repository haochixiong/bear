"""影调意图：刻意暗调 / 欠曝要能区分——自动判断时不提亮但一定说明；用户选「正常曝光」才按欠曝修正。"""

import numpy as np
import pytest
from synth import face_scene, highkey_face_scene, lowkey_face_scene

from colordoctor.bridge import server as bs
from colordoctor.core import analysis as an
from colordoctor.core import layer_plan as lp
from colordoctor.core.color_management import synthetic_context
from colordoctor.core.color_models import Severity

CTX = synthetic_context()


def _moody_with_rim_light():
    """刻意的暗调人像：暗背景、正常的脸，再加一条轮廓光（真正的高光）。"""
    img, _ = lowkey_face_scene()
    img = img.copy()
    img[20:200, 150:156] = (235, 228, 220)  # 轮廓光
    return img


def _underexposed():
    img, _ = face_scene()
    return (img.astype(float) * 0.35).astype(np.uint8)  # 整体欠曝：连最亮的地方也很暗


def _exposure(rep):
    return next((f for f in rep.findings if "画面整体" in f.problem or "整体欠曝" in f.problem or "整体过曝" in f.problem), None)


def _curves(rep, img):
    return [ly for ly in lp.plan_from_report(rep, img.shape).layers if ly.kind == "curves"]


def test_auto_treats_moody_portrait_as_intentional_and_never_brightens():
    img = _moody_with_rim_light()
    rep = an.analyze_palette_report(img, CTX)
    f = _exposure(rep)
    assert f is not None and "刻意的暗调" in f.problem and f.severity == Severity.NORMAL
    assert not _curves(rep, img)  # 不提亮
    assert any("正常曝光" in a.instruction for a in f.adjustments)  # 告诉用户判断不对时怎么改


def test_auto_says_it_looks_underexposed_but_does_not_force_it():
    img = _underexposed()
    rep = an.analyze_palette_report(img, CTX)
    f = _exposure(rep)
    assert f is not None and "欠曝" in f.problem and f.severity == Severity.MILD
    assert not _curves(rep, img)  # 自动判断不替用户决定
    assert any("输入" in a.instruction and "Camera Raw 曝光 +" in a.instruction for a in f.adjustments)


@pytest.mark.parametrize("make", [_underexposed, _moody_with_rim_light])
def test_normal_exposure_brightens_with_a_curve_and_protects_highlights(make):
    img = make()
    rep = an.analyze_palette_report(img, CTX, tone_intent="normal")
    f = _exposure(rep)
    assert f is not None and f.problem.startswith("整体欠曝")
    curves = _curves(rep, img)
    assert len(curves) == 1
    pts = curves[0].params["channels"]["composite"]
    assert pts[0] == [0, 0] and pts[-1] == [255, 255]
    assert all(y > x for x, y in pts[1:-1])  # 提亮
    assert all(y <= 250 for _x, y in pts[1:-1])  # 最亮处不推到死白
    assert [x for x, _ in pts] == sorted(x for x, _ in pts) and [y for _, y in pts] == sorted(y for _, y in pts)
    assert any(f"输入 {pts[1][0]} → 输出 {pts[1][1]}" in a.instruction for a in f.adjustments)  # 卡片数字 = 图层数字


def test_chosen_low_key_has_no_exposure_card_and_no_brightening():
    for img in (_underexposed(), _moody_with_rim_light()):
        rep = an.analyze_palette_report(img, CTX, tone_intent="low")
        assert _exposure(rep) is None and not _curves(rep, img)


def test_high_key_auto_note_and_normal_darkens():
    img, _ = highkey_face_scene()
    rep = an.analyze_palette_report(img, CTX)
    f = _exposure(rep)
    if rep.palette.detected_key == "high":
        assert "刻意的高调" in f.problem and not _curves(rep, img)
    bright = np.clip(face_scene()[0].astype(float) * 1.6 + 60, 0, 255).astype(np.uint8)
    rep = an.analyze_palette_report(bright, CTX, tone_intent="normal")
    f = _exposure(rep)
    assert f is not None and f.problem.startswith("整体过曝")
    pts = _curves(rep, bright)[0].params["channels"]["composite"]
    assert all(y < x for x, y in pts[1:-1])  # 压暗


def test_tone_intent_changes_style_tone_handling():
    img = _underexposed()
    auto = an.analyze_palette_report(img, CTX, style_key="fuji_provia")
    normal = an.analyze_palette_report(img, CTX, style_key="fuji_provia", tone_intent="normal")
    assert auto.palette.tone_key == "low" and normal.palette.tone_key == ""


def test_bridge_tone_param():
    img = _underexposed()
    r = bs.analyze_request(img, "palette", {"tone": "normal"}, bs.Store(), bs.Store(), layer_kind="pixel")
    assert any(f["problem"].startswith("整体欠曝") and f["layers"] for f in r["findings"])
    with pytest.raises(ValueError):
        bs.analyze_request(img, "palette", {"tone": "dark"}, bs.Store(), bs.Store(), layer_kind="pixel")
