"""诊断规则、置信度、建议映射与总体流程（含误报专项）。"""

import pytest

from colordoctor.core import analysis as an
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


def test_skin_mixed_region_warns_reselect():
    img = skin_region()
    img[:, :50] = lch_color(40, 30, 250)
    r = an.analyze_skin_region(img, CTX)
    assert any("重新" in n for f in r.findings for n in f.notes) or any("重新" in f.problem for f in r.findings)
    assert r.confidence < 0.75


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
