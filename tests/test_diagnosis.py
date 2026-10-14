"""诊断规则、置信度、建议映射与总体流程（含误报专项）。"""

import numpy as np
import pytest

from colordoctor.core import analysis as an
from colordoctor.core import color_spaces as cs
from colordoctor.core import recommendations as rec
from colordoctor.core.color_management import synthetic_context
from colordoctor.core.color_models import ColorContext, Severity
from synth import BLUE, YELLOW, flat, lch_color, noisy, skin_region, sprinkle

CTX = synthetic_context()


def comp(report, key):
    return {c.key: c for c in report.components}[key]


def sev_rank(s):
    return [Severity.NORMAL, Severity.MILD, Severity.NOTABLE, Severity.STRONG].index(s)


# ------------------------------------------------------------------ 通用区域
def test_clean_blue_is_normal():
    r = an.analyze_region(noisy(flat(BLUE), 2), CTX)
    assert r.overall_severity == Severity.NORMAL
    assert not [f for f in r.findings if f.severity != Severity.NORMAL]


def test_severity_grows_with_yellow_fraction_in_blue():
    sev = []
    for f in (0.01, 0.05, 0.10):
        r = an.analyze_region(sprinkle(noisy(flat(BLUE), 2), YELLOW, f, seed=5), CTX)
        c = comp(r, "opposite")
        sev.append(c.severity)
        assert any("黄" in e for e in c.evidence)
    assert sev[0] in (Severity.MILD, Severity.NOTABLE)
    assert sev[1] == Severity.NOTABLE or sev[1] == Severity.STRONG
    assert sev[2] == Severity.STRONG
    assert sev_rank(sev[0]) <= sev_rank(sev[1]) <= sev_rank(sev[2])


def test_finding_has_full_structure_and_numbers():
    r = an.analyze_region(sprinkle(noisy(flat(BLUE), 2), YELLOW, 0.05, seed=5), CTX)
    f = r.findings[0]
    assert "黄" in f.problem and "蓝" in f.problem
    assert f.evidence and any(ch.isdigit() for ch in " ".join(f.evidence))
    assert f.causes and f.adjustments
    assert 0 < f.confidence <= 1
    assert f.notes
    assert "%" in " ".join(f.evidence)


def test_overall_score_is_explained_by_components():
    r = an.analyze_region(sprinkle(noisy(flat(BLUE), 2), YELLOW, 0.05, seed=5), CTX)
    scores = sorted([c.score for c in r.components if c.applicable], reverse=True)
    expected = scores[0] + 0.2 * sum(scores[1:])
    assert r.overall_risk == pytest.approx(min(100, expected), abs=0.5)
    top = max(r.components, key=lambda c: c.score)
    assert r.overall_severity == top.severity or sev_rank(r.overall_severity) > sev_rank(top.severity)


@pytest.mark.parametrize(
    "img",
    [
        noisy(flat(lch_color(55, 9, 265)), 3, seed=4),  # 低饱和灰蓝
        noisy(flat(lch_color(14, 10, 280)), 2, seed=5),  # 暗调
        noisy(flat(lch_color(60, 30, 260)), 6, seed=8),  # 少量噪点
        noisy(flat((128, 128, 128)), 1),  # 灰
    ],
    ids=["low-sat", "low-key", "noise", "grey"],
)
def test_false_positives_stay_normal_or_mild(img):
    r = an.analyze_region(img, CTX)
    assert sev_rank(r.overall_severity) <= 1
    assert sev_rank(comp(r, "opposite").severity) == 0


def test_bicolor_region_explained_as_palette():
    img = flat(BLUE)
    img[:, 80:] = YELLOW
    r = an.analyze_region(noisy(img, 2), CTX)
    c = comp(r, "opposite")
    assert c.severity == Severity.NORMAL
    assert c.note and "配色" in c.note


def test_low_chroma_region_hue_dispersion_not_applicable():
    r = an.analyze_region(noisy(flat((128, 128, 128)), 1), CTX)
    assert not comp(r, "hue_dispersion").applicable


def test_confidence_lower_for_tiny_region_and_assumed_srgb():
    big = an.analyze_region(sprinkle(noisy(flat(BLUE), 2), YELLOW, 0.05), CTX)
    tiny = an.analyze_region(noisy(flat(BLUE, 3, 3), 2), CTX)
    assumed = an.analyze_region(
        sprinkle(noisy(flat(BLUE), 2), YELLOW, 0.05), ColorContext(source="screen", assumed_srgb=True)
    )
    assert tiny.confidence < big.confidence
    assert assumed.confidence < big.confidence


def test_graying_patch_reported_with_location():
    clean = lch_color(50, 45, 265)
    grey = lch_color(50, 15, 265)
    img = noisy(flat(clean, 120, 160), 1.5)
    img[40:100, 90:150] = noisy(flat(grey, 60, 60), 1.5, seed=2)
    r = an.analyze_region(img, CTX, rect=(100, 200, 160, 120))
    c = comp(r, "chroma_compression")
    assert sev_rank(c.severity) >= 1
    blobs = [b for b in r.contamination.blobs if b.kind == "graying"]
    assert blobs and blobs[0].bbox[0] >= 100 + 70  # 坐标已加上选区偏移


# ------------------------------------------------------------------ 肤色
def test_skin_green_contamination_and_suggestions():
    img = sprinkle(skin_region(), lch_color(60, 18, 140), 0.05, seed=1)
    r = an.analyze_skin_region(img, CTX)
    texts = " ".join(f.problem for f in r.findings)
    assert "绿" in texts
    tools = {a.tool for f in r.findings for a in f.adjustments}
    assert tools & {"Camera Raw 混色器", "色相/饱和度", "可选颜色"}


def test_skin_cyan_shadow_finding():
    from test_skin import _modify_by_lightness

    img = _modify_by_lightness(skin_region(), "low", da=-8, db=-2)
    r = an.analyze_skin_region(img, CTX)
    assert any("阴影" in f.problem for f in r.findings)
    assert sev_rank(comp(r, "local_cast").severity) >= 1


def test_clean_skin_of_different_depths_no_problem_findings():
    for lab in [(32.0, 9.0, 12.0), (68.0, 14.0, 19.0), (82.0, 10.5, 14.5)]:
        r = an.analyze_skin_region(skin_region(*lab), CTX)
        assert sev_rank(r.overall_severity) <= 1, lab


def test_skin_preset_target_gives_ps_parameters():
    r = an.analyze_skin_region(skin_region(68, 14, 22), CTX, preset_key="japanese")
    tools = {a.tool for a in r.adjustments}
    assert {"曲线", "色彩平衡", "色相/饱和度", "可选颜色", "Camera Raw"} <= tools
    curves = [a for a in r.adjustments if a.tool == "曲线"][0]
    assert not curves.estimated  # 曲线对平均色是精确的
    cb = [a for a in r.adjustments if a.tool == "色彩平衡"][0]
    assert cb.estimated and cb.params
    acr = [a for a in r.adjustments if a.tool == "Camera Raw"]
    assert any("色温" in a.instruction for a in acr)


def test_skin_mixed_region_warns_reselect_without_auto_mask():
    img = skin_region()
    img[:, :50] = lch_color(40, 30, 250)
    r = an.analyze_skin_region(img, CTX, auto_mask=False)
    assert any("重新" in n for f in r.findings for n in f.notes) or any("重新" in f.problem for f in r.findings)
    assert r.confidence < 0.75


def test_skin_auto_mask_excludes_background():
    img = skin_region()
    img[:, :50] = lch_color(40, 30, 250)
    r = an.analyze_skin_region(img, CTX)
    assert r.valid_mask is not None
    assert not r.valid_mask[:, :50].any()
    assert r.mask_info["skin_share"] == pytest.approx(70 / 120, abs=0.05)
    assert r.skin.tendency == "正常"


def test_skin_report_has_quick_rule_and_uniformity():
    r = an.analyze_skin_region(skin_region(68, 14, 18), CTX)
    assert r.quick_rule.passed
    assert any("经验快速判定" in f.problem for f in r.findings)
    assert r.uniformity.score > 80


def test_face_vs_body_comparison():
    face = skin_region(70, 13, 18)
    body = skin_region(64, 12, 24)  # 身体更暗更黄
    rb = an.analyze_skin_region(body, CTX, part="身体")
    rf = an.analyze_skin_region(face, CTX, part="脸部", other_part=("身体", rb.skin.mean_lab))
    f = [x for x in rf.findings if "脸部与身体" in x.problem][0]
    assert "更亮" in f.problem
    assert f.adjustments


# ------------------------------------------------------------------ 参考对比
def test_reference_vs_problem_only_lightness_difference():
    ref = skin_region(68, 14, 19)
    prob = skin_region(60, 14 * 60 / 68, 19 * 60 / 68)
    r = an.compare_regions(ref, prob, CTX)
    c = r.comparison
    assert c.dL == pytest.approx(-8, abs=0.6)
    assert abs(c.dh) < 2
    assert "明度" in r.findings[0].problem
    assert not any("偏黄" in f.problem or "偏绿" in f.problem for f in r.findings)


def test_reference_vs_problem_only_chroma_difference():
    ref = skin_region(68, 14, 19)
    prob = skin_region(68, 8, 11)
    r = an.compare_regions(ref, prob, CTX)
    c = r.comparison
    assert c.dC < -8
    assert abs(c.dL) < 1
    assert any("色度" in f.problem or "发灰" in f.problem for f in r.findings)


def test_reference_vs_problem_yellow_shift_suggests_moving_back():
    ref = skin_region(68, 14, 19)
    prob = skin_region(68, 12, 27)
    r = an.compare_regions(ref, prob, CTX)
    assert r.comparison.db > 6
    acr = " ".join(a.instruction for a in r.adjustments if a.tool == "Camera Raw")
    assert "蓝" in acr  # 色温往蓝


# ------------------------------------------------------------------ 建议映射
def test_acr_direction_mapping():
    adj = rec.acr_directions(dL=0, da=-5, db=-6, dC=-2, dh=0)
    text = " ".join(a.instruction for a in adj)
    assert "色温" in text and "蓝" in text
    assert "色调" in text and "绿" in text
    assert all(a.estimated for a in adj)


def test_strength_labels():
    assert rec.strength_label(1.0) is None
    assert rec.strength_label(3.0) == "轻微"
    assert rec.strength_label(6.0) == "小幅"
    assert rec.strength_label(12.0) == "中等"


def test_contamination_adjustments_mention_channel():
    adj = rec.contamination_adjustments("黄", "蓝", skin=False)
    assert any("黄色" in a.instruction for a in adj)
    adj = rec.contamination_adjustments("紫/品红", "橙", skin=True)
    assert any("洋红" in a.instruction or "紫色" in a.instruction for a in adj)


# ------------------------------------------------------------------ 整图扫描
def test_scan_reports_dirty_location():
    img = noisy(flat(BLUE, 200, 300), 2)
    img[50:150, 50:150] = sprinkle(img[50:150, 50:150], YELLOW, 0.06, seed=3)
    r = an.scan_image(img, CTX)
    assert r.contamination.blobs
    assert any(f.severity != Severity.NORMAL for f in r.findings)


def test_scan_teal_orange_clean():
    teal = lch_color(45, 30, 205)
    orange = lch_color(70, 35, 60)
    img = flat(teal, 120, 160)
    img[30:110, 50:110] = orange
    r = an.scan_image(noisy(img, 2), CTX)
    assert sev_rank(r.overall_severity) <= 1


def test_manual_mask_overrides_auto():
    img = skin_region()
    m = np.zeros(img.shape[:2], bool)
    m[:, 60:] = True
    r = an.analyze_skin_region(img, CTX, manual_mask=m)
    assert r.mask_info.get("manual") and not r.valid_mask[:, :60].any()


def test_non_skin_compare_uses_matching_range_and_channel():
    from colordoctor.core import recommendations as rec

    blue = np.array([[70, 110, 200]] * 50, np.uint8)
    target = cs.srgb8_to_lab(np.array([60, 130, 210], np.uint8))
    adjs = rec.skin_adjustments(blue, target, skin=False)
    text = " ".join(a.instruction for a in adjs)
    assert "皮肤" not in text and "肤色" not in text
    hs = [a for a in adjs if a.tool == "色相/饱和度"][0]
    assert "蓝色" in hs.instruction or "青色" in hs.instruction


def test_skin_brightening_not_applied_twice():
    from colordoctor.core import recommendations as rec

    skin = np.array([[180, 135, 110]] * 50, np.uint8)
    cur = cs.srgb8_to_lab(skin[0])
    target = np.array([cur[0] + 10, cur[1], cur[2]])
    adjs = rec.skin_adjustments(skin, target)
    curve = [a for a in adjs if a.tool == "曲线" and a.instruction.startswith("颜色")]
    light = [a for a in adjs if a.tool == "曲线" and a.instruction.startswith("明度")]
    assert curve and light
    # 颜色曲线保持明度：各通道输出接近输入
    for name, out in curve[0].params:
        inp = int(name.split()[1].rstrip("→"))
        assert abs(out - inp) <= 4, (name, out)


def test_dispersion_advice_is_meaningful():
    adj = rec.dispersion_adjustments("肤色", True)
    text = " ".join(a.instruction for a in adj)
    assert "“肤色”" not in text and "橙色" in text and "红色" in text and "黄色" in text
    adj = rec.dispersion_adjustments("绿", False)
    assert "黄色" in adj[0].instruction and "浅绿色" in adj[0].instruction


# ------------------------------------------------------------------ 建议文字/数值一致性（审计回归）
_ORDER = [Severity.NORMAL, Severity.MILD, Severity.NOTABLE, Severity.STRONG]


def _sallow_face():
    from synth import face_scene

    return face_scene(skin_lab=(66, 9, 28))[0]


def test_region_compare_names_problem_by_its_own_colour():
    ref = skin_region(68, 14, 19)
    r = an.compare_regions(ref, skin_region(68, 10, 28, seed=3), CTX, skin=True)
    assert "偏黄" in r.findings[0].problem or "偏黄" in " ".join(f.problem for f in r.findings)
    assert not any("偏绿" in f.problem or "偏青" in f.problem for f in r.findings)
    r = an.compare_regions(ref, skin_region(66, 22, 15, seed=3), CTX, skin=True)
    main = next(f for f in r.findings if f.problem.startswith("问题区"))
    assert "偏红" in main.problem and "洋红" not in main.problem


@pytest.mark.parametrize("make", [
    lambda: an.analyze_skin_region(_sallow_face(), CTX, preset_key="natural"),
    lambda: an.analyze_skin_region(skin_region(68, 14, 19), CTX),
    lambda: an.scan_image(sprinkle(noisy(flat(lch_color(60, 35, 260), 160, 200), 2), YELLOW, 0.03, seed=1), CTX),
    lambda: an.analyze_palette_report(noisy(flat(lch_color(55, 30, 135), 120, 160), 1), CTX, style_key="teal_orange"),
])
def test_headline_severity_not_below_any_card(make):
    r = make()
    worst = max(_ORDER.index(f.severity) for f in r.findings)
    assert _ORDER.index(r.overall_severity) >= worst
    from colordoctor.core.color_models import severity_from_score

    assert severity_from_score(r.overall_risk) == r.overall_severity


def test_scan_top_card_matches_component_severity():
    img = sprinkle(noisy(flat(lch_color(60, 35, 260), 160, 200), 2), YELLOW, 0.03, seed=1)
    r = an.scan_image(img, CTX)
    cards = [f for f in r.findings if f.problem.startswith("可疑脏色")]
    assert cards and cards[0].severity == comp(r, "opposite").severity


def test_contamination_card_and_layer_use_same_saturation():
    from colordoctor.core import layer_plan as lp

    img = sprinkle(noisy(flat(BLUE), 2), YELLOW, 0.05, seed=5)
    r = an.analyze_region(img, CTX)
    card = next(a for f in r.findings for a in f.adjustments if a.tool == "色相/饱和度" and "饱和度" in dict(a.params))
    layer = lp.plan_from_report(r, img.shape).get("opposite")
    assert int(dict(card.params)["饱和度"]) == layer.params["ranges"][0]["saturation"]
    assert f"饱和度 {layer.params['ranges'][0]['saturation']}" in card.instruction


def test_uneven_yellow_patch_text_follows_saturation_sign():
    from types import SimpleNamespace

    from colordoctor.core import layer_plan as lp
    from colordoctor.core.skin_uniformity import UniformPatch

    def uni(ds):
        p = UniformPatch("偏黄", (10, 10, 20, 20), 0.1, 9.0, ds, (60.0, 8.0, 20.0), "明显")
        return SimpleNamespace(patches=(p,), median_hue=55.0, score=80.0)

    dull = an._uniformity_findings(uni(-0.3), 0.9, [])[0].adjustments[0].instruction
    vivid = an._uniformity_findings(uni(+0.3), 0.9, [])[0].adjustments[0].instruction
    assert "饱和度略加" in dull and "略降" not in dull
    assert "饱和度略降" in vivid
    _, params, _ = lp._skin_layer("偏黄", 9.0, -0.3, (60.0, 8.0, 20.0))
    assert params["ranges"][0]["saturation"] > 0
    _, params, _ = lp._skin_layer("偏黄", 9.0, 0.3, (60.0, 8.0, 20.0))
    assert params["ranges"][0]["saturation"] < 0


def test_overall_skin_cast_gets_a_layer_matching_the_card():
    from colordoctor.core import layer_plan as lp

    img = _sallow_face()
    r = an.analyze_skin_region(img, CTX)
    card = next(f for f in r.findings if f.problem == "肤色整体偏黄")
    # 肤色整体偏色用可选颜色（修图常用做法，产品决定），卡片数字就是图层数字
    sc = next(a for a in card.adjustments if a.tool == "可选颜色")
    plan = lp.plan_from_report(r, img.shape)
    layer = plan.get("skin_cast")
    assert layer.kind == "selectiveColor" and layer.mask is not None
    assert layer.params == sc.layer["params"] and layer.finding == r.findings.index(card)
    yellows = layer.params["colors"]["yellows"]
    assert yellows["y"] < 0  # 偏黄 -> 黄色里减黄
    assert f"黄色 {yellows['y']:+d}%" in sc.instruction
    assert layer.mask[10, 10] == 0  # 背景不受影响


def test_basically_equal_regions_get_no_adjustments():
    r = an.compare_regions(skin_region(68, 14, 19), skin_region(68, 14.3, 19.2, seed=3), CTX, skin=True)
    assert all(a.tool == "提示" for a in r.adjustments)
    r = an.analyze_skin_region(skin_region(68, 14, 19), CTX, part="脸部", other_part=("身体", (69.5, 13.0, 19.8)))
    f = next(f for f in r.findings if "差异" in f.problem)
    assert "基本一致" in f.problem and all(a.tool == "提示" for a in f.adjustments)
    r = an.analyze_skin_region(skin_region(68.5, 14, 19), CTX, preset_key="natural")
    assert not any(f.problem.startswith("与目标风格") for f in r.findings)
    assert all(a.tool == "提示" for a in r.adjustments)


def test_scan_cards_name_the_patch_colour():
    img = noisy(flat(BLUE, 200, 300), 2)
    img[50:150, 50:150] = sprinkle(img[50:150, 50:150], YELLOW, 0.06, seed=3)
    r = an.scan_image(img, CTX)
    text = " ".join([f.problem for f in r.findings] + [a.instruction for f in r.findings for a in f.adjustments])
    assert "“”" not in text and "相邻色" not in text
    assert "“蓝”中夹杂“黄”" in text
    from colordoctor.core import layer_plan as lp

    names = [ly.name + ly.reason for ly in lp.plan_from_report(r, img.shape).layers]
    assert names and not any("多种颜色" in n for n in names)


def test_skin_cast_card_defers_to_style_card_for_same_target():
    r = an.analyze_skin_region(_sallow_face(), CTX, preset_key="natural")
    cast = next(f for f in r.findings if f.problem.startswith("肤色整体"))
    assert any(f.problem.startswith("与目标风格") for f in r.findings)
    assert not [a for a in cast.adjustments if a.tool == "Camera Raw"]
    r = an.analyze_skin_region(_sallow_face(), CTX)  # 没选风格：卡片自己给方向
    cast = next(f for f in r.findings if f.problem.startswith("肤色整体"))
    assert [a for a in cast.adjustments if a.tool == "Camera Raw"]


def test_explanations_are_accurate():
    adj = rec.acr_directions(dL=6, da=0, db=0, dC=0, dh=0)
    assert not any("只影响肤色" in a.instruction for a in adj)
    adjs = rec.skin_adjustments(skin_region(66, 9, 28), (66.0, 14.0, 19.0), keep_lightness=True)
    hs = next(a for a in adjs if a.tool == "色相/饱和度")
    if int(dict(hs.params)["明度"]):
        assert "只改颜色" not in hs.instruction
    cb = next(a for a in adjs if a.tool == "色彩平衡")
    assert "往蓝" in cb.instruction


def test_flat_graying_patch_is_one_card():
    base = noisy(flat(lch_color(55, 40, 140), 160, 200), 1.5)
    base[40:80, 40:80] = lch_color(55, 15, 140)
    r = an.analyze_region(base, CTX)
    assert any("发灰" in f.problem for f in r.findings)
    assert not any("明度和色度同时被压缩" in f.problem for f in r.findings)


def test_black_and_white_palette_is_not_called_desaturated():
    g = np.repeat(np.linspace(10, 245, 240)[None, :], 200, 0)
    mono = np.clip(np.stack([g, g, g], -1), 0, 255).astype(np.uint8)
    r = an.analyze_palette_report(mono, CTX)
    assert any(f.problem.startswith("黑白作品") for f in r.findings)
    assert not any("缺色" in f.problem and not f.problem.startswith("黑白") for f in r.findings)
    assert not any("自然饱和度" in a.instruction for f in r.findings for a in f.adjustments)
