"""影调风格 = 预制的目标影调，只管明暗和反差；颜色模板只管颜色，两者互不绑定。

- 不选影调风格：只描述现在是九种影调里的哪一种，不改明暗（不替用户判断“暗调还是欠曝”）。
- 选了影调风格（或直接选了目标影调）：按直方图匹配给一条曲线；同一张图选不同风格，曲线方向不同。
- 颜色模板（青橙、黑金……）不再带影调曲线，可以和任何影调风格搭配。
"""

import numpy as np
import pytest
from synth import face_scene, lowkey_face_scene

from colordoctor.bridge import server as bs
from colordoctor.core import analysis as an
from colordoctor.core import layer_plan as lp
from colordoctor.core import palette as pl
from colordoctor.core import tone_class as tc
from colordoctor.core.color_management import synthetic_context
from colordoctor.core.color_models import Severity

CTX = synthetic_context()


def _under():
    img, _ = face_scene()
    return (img.astype(float) * 0.35).astype(np.uint8)  # 低短调


def _tone_card(rep):
    return next(f for f in rep.findings if f.problem.startswith("影调："))


def _curves(rep, img):
    return [ly for ly in lp.plan_from_report(rep, img.shape).layers if ly.kind == "curves"]


def test_resolve_target():
    assert tc.resolve_target() == ("", (), "")
    assert tc.resolve_target("japanese")[0] == "高短调"
    assert tc.resolve_target("japanese", "中长调")[0] == "中长调"  # 直接选的影调优先
    with pytest.raises(ValueError):
        tc.resolve_target("nope")
    with pytest.raises(ValueError):
        tc.resolve_target("", "超长调")


def test_every_tone_style_explains_itself_for_people_who_do_not_know_the_names():
    for s in tc.TONE_STYLES.values():
        assert s["target"] in tc.TONE_NAMES and all(t in tc.TONE_NAMES for t in s["also"])
        assert len(s["look"]) >= 10 and s["scene"]


def test_path_words():
    assert tc.path_words("中中调", "高中调") == "整体提亮"
    assert tc.path_words("低短调", "高短调") == "整体提亮一大截"
    assert "拉开反差" in tc.path_words("中短调", "中长调")
    assert "压缩反差" in tc.path_words("高长调", "高短调")
    assert tc.path_words("低中调", "低中调") == "已经是这种影调"


def test_suggestions_are_sorted_by_how_much_has_to_change():
    s = tc.suggestions("中中调")
    assert [x["distance"] for x in s] == sorted(x["distance"] for x in s)
    first = {x["name"] for x in s if x["distance"] == 0}
    assert {"森系", "法式"} <= first
    jp = next(x for x in s if x["key"] == "japanese")
    assert jp["change"] == "整体提亮" and jp["distance"] == 1  # 中中调 → 高中调（日系也可以）只差提亮一档


def test_no_tone_style_only_describes():
    img = _under()
    rep = an.analyze_palette_report(img, CTX)
    card = _tone_card(rep)
    assert card.severity == Severity.NORMAL and "低短调" in card.problem
    assert not card.adjustments and not _curves(rep, img)
    assert "影调风格" in card.notes[0] and "稍改就到" in card.notes[0]
    assert rep.findings[0] is card  # 影调卡排第一


def test_same_photo_different_tone_styles_give_opposite_curves():
    img = _under()
    bright = an.analyze_palette_report(img, CTX, tone_style="japanese")
    moody = an.analyze_palette_report(img, CTX, tone_style="moody")
    b_card, m_card = _tone_card(bright), _tone_card(moody)
    assert "高短调" in b_card.problem and "低" in m_card.problem.split("→")[1]
    b_pts = b_card.adjustments[0].layer["params"]["channels"]["composite"]
    m_pts = m_card.adjustments[0].layer["params"]["channels"]["composite"]
    assert all(y >= x for x, y in b_pts)  # 日系：全部提亮
    assert m_pts[1][1] <= m_pts[1][0]  # 暗调情绪：暗部不提亮
    assert any(y > x for x, y in m_pts)  # ……只拉开亮部
    # 曲线是这张卡的图层
    plan = lp.plan_from_report(bright, img.shape)
    i = bright.findings.index(b_card)
    assert [ly.kind for ly in plan.for_finding(i)] == ["curves"]


def test_big_jump_points_to_raw():
    card = _tone_card(an.analyze_palette_report(_under(), CTX, tone_style="japanese"))
    assert any("Camera Raw" in n for n in card.notes)


def test_tone_already_matching_gives_no_curve():
    img = _under()
    rep = an.analyze_palette_report(img, CTX, tone_target="低短调")
    card = _tone_card(rep)
    assert "已经符合" in card.problem and not _curves(rep, img)
    # 影调风格有几种可接受的影调：往离现在最近的走（低短调 + 暗调情绪 → 低中调，不是更远的低长调）
    rep = an.analyze_palette_report(img, CTX, tone_style="moody")
    assert "低中调" in _tone_card(rep).problem.split("→")[1] and rep.palette.tone_target == "低中调"


def test_color_template_never_changes_tone_and_pairs_with_any_tone_style():
    img = _under()
    for style in pl.load_styles():
        m = pl.style_match(img, style)
        assert not any(a.layer and a.layer["kind"] == "curves" and "composite" in a.layer["params"]["channels"]
                       for a in m.adjustments), style["key"]
        assert all(k in tc.TONE_STYLES for k in style.get("tone_pair", ())), style["key"]
    # 港风可以配青橙，也可以配黑金：影调曲线只有影调卡那一条
    for color in ("teal_orange", "black_gold"):
        rep = an.analyze_palette_report(img, CTX, style_key=color, tone_style="hongkong")
        names = [ly.name for ly in _curves(rep, img)]
        assert names and all(n.startswith("影调 ") for n in names), names
        style_card = next(f for f in rep.findings if f.problem.startswith("与颜色"))
        assert not any(a.layer and a.layer["kind"] == "curves" for a in style_card.adjustments)


def test_color_analysis_follows_the_target_key():
    img, _ = lowkey_face_scene(skin_lab=(62.0, 13.0, 17.0))
    assert pl.analyze_palette(img).tone_key == "low"
    assert pl.analyze_palette(img, tone_style="japanese").tone_key == "high"
    assert pl.analyze_palette(img, tone_style="film").tone_key == ""


def test_bridge_tone_params_and_grid():
    img = np.dstack([_under(), np.full(_under().shape[:2], 255, np.uint8)])
    r = bs.analyze_request(img, "palette", {"tone_style": "japanese", "style": "teal_orange"}, bs.Store(), bs.Store(),
                           layer_kind="pixel")
    t = r["tone"]
    assert t["name"] == "低短调" and t["target"] == "高短调" and t["label"] == "日系" and "高中调" in t["also"]
    assert len(t["histogram"]) == 32 and len(t["suggestions"]) == len(tc.TONE_STYLES)
    assert r["findings"][0]["problem"].startswith("影调：") and r["findings"][0]["layers"]
    r2 = bs.analyze_request(img, "palette", {}, bs.Store(), bs.Store(), layer_kind="pixel")
    assert r2["tone"]["target"] == "" and not r2["findings"][0]["layers"]
    for bad in ({"tone_style": "dark"}, {"tone_target": "很暗调"}):
        with pytest.raises(ValueError):
            bs.analyze_request(img, "palette", bad, bs.Store(), bs.Store(), layer_kind="pixel")
    assert bs.analyze_request(img, "dirty", {}, bs.Store(), bs.Store(), layer_kind="pixel")["tone"] is None
