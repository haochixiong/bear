"""每张问题卡片都带修它的 PS 原生调整图层：图层挂在卡片上、数字与卡片相同、格式符合插件约定。"""

import json
import re

import numpy as np
import pytest
from synth import face_scene, skin_region
from test_bridge import dirty_sky

from colordoctor.bridge import server as bs
from colordoctor.core import acr_preset as ap
from colordoctor.core import analysis as an
from colordoctor.core import color_spaces as cs
from colordoctor.core import layer_plan as lp
from colordoctor.core import native_layers as nl
from colordoctor.core.color_management import synthetic_context
from colordoctor.core.color_models import Severity
from colordoctor.core.perception import DE_LEVELS

CTX = synthetic_context()
TIPS = ("提示", "构图/调色")


def lchlab(L, C, h):
    return (L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h)))


def landscape(H=200, W=240):
    """天空 + 草地 + 中间一块皮肤，带明暗渐变和少量噪声。"""
    rng = np.random.default_rng(0)
    L = np.zeros((H, W, 3))
    yy = np.linspace(0, 1, H)[:, None]
    L[:] = np.array(lchlab(55, 40, 125))
    L[: int(H * 0.4)] = np.array(lchlab(62, 30, 260))
    L[int(H * 0.4):int(H * 0.7), int(W * 0.35):int(W * 0.65)] = (68, 14, 19)
    L[..., 0] += (0.5 - yy) * 30
    L[..., 1:] += rng.normal(0, 1.0, L[..., 1:].shape)
    return cs.lab_to_srgb8(L)


def teal_orange(H=200, W=240):
    L = np.zeros((H, W, 3))
    xx = np.linspace(0, 1, W)[None, :]
    L[..., 0] = 25 + 60 * xx
    L[..., 1] = np.where(xx < 0.5, -18, 22)
    L[..., 2] = np.where(xx < 0.5, -18, 40)
    L[:, int(W * 0.4):int(W * 0.6)] = (68, 14, 19)
    return cs.lab_to_srgb8(L)


def _actionable(f):
    return f.severity != Severity.NORMAL and any(a.tool not in TIPS and not a.tool.startswith("常用做法") for a in f.adjustments)


def _cases():
    sky = dirty_sky()
    face = face_scene(skin_lab=(66, 9, 28))[0]
    land = landscape()
    out = {
        "dirty_scan": (an.scan_image(sky, CTX), sky.shape),
        "skin_sallow_natural": (an.analyze_skin_region(face, CTX, preset_key="natural"), face.shape),
        "skin_sallow": (an.analyze_skin_region(face, CTX), face.shape),
        "reference": (an.compare_reference(land, teal_orange(), CTX), land.shape),
        "compare_regions": (an.compare_regions(skin_region(68, 14, 19), skin_region(64, 18, 14, seed=3), CTX, skin=True),
                            skin_region(64, 18, 14).shape),
    }
    for key in ("teal_orange", "morandi", "fuji_classic_chrome", "kodak_portra400"):
        out[f"style_{key}"] = (an.analyze_palette_report(land, CTX, style_key=key, family="绿"), land.shape)
    return out


CASES = _cases()


# ------------------------------------------------------------------ (a) 每张有可执行建议的卡片都有图层
@pytest.mark.parametrize("name", list(CASES))
def test_every_actionable_finding_has_a_linked_layer(name):
    rep, shape = CASES[name]
    plan = lp.plan_from_report(rep, shape)
    assert plan.layers, name
    for ly in plan.layers:
        assert ly.finding is not None and 0 <= ly.finding < len(rep.findings), (name, ly.id)
    for i, f in enumerate(rep.findings):
        if _actionable(f):
            assert plan.for_finding(i), (name, f.problem, [a.tool for a in f.adjustments])
        if not any(getattr(a, "layer", None) for a in f.adjustments):
            assert not plan.for_finding(i), (name, f.problem)  # 只有提示的卡片不生成图层


def test_scan_gives_each_dirty_spot_its_own_masked_layer():
    rep, shape = CASES["dirty_scan"]
    plan = lp.plan_from_report(rep, shape)
    spots = [i for i, f in enumerate(rep.findings) if f.problem.startswith("可疑脏色")]
    assert spots
    for i in spots:
        ly = plan.for_finding(i)[0]
        assert ly.kind == "hueSaturation" and ly.mask is not None
    assert plan.get("opposite").finding == spots[0]


def test_existing_layers_are_linked_not_duplicated():
    rep, shape = CASES["skin_sallow_natural"]
    plan = lp.plan_from_report(rep, shape)
    cast = next(i for i, f in enumerate(rep.findings) if f.problem.startswith("肤色整体"))
    own = plan.for_finding(cast)
    assert [ly.id for ly in own] == ["skin_cast"] and own[0].kind == "selectiveColor"
    keys = [(ly.finding, ly.kind, json.dumps(ly.params, sort_keys=True)) for ly in plan.layers]
    assert len(keys) == len(set(keys))


# ------------------------------------------------------------------ (b) 图层数字 = 卡片数字
def _numbers_shown(ly):
    """图层参数里的每个非零数值都要出现在对应的卡片文字（reason = 卡片原文）里。"""
    t = ly.reason
    p = ly.params
    if ly.kind == "hueSaturation":
        for r in [p["master"], *p["ranges"]]:
            for key, word in (("hue", "色相"), ("saturation", "饱和度"), ("lightness", "明度")):
                v = r[key]
                if v:
                    assert f"{word} {v:+d}" in t or f"{word} {v}" in t, (ly.id, word, v, t)
    elif ly.kind == "colorBalance":
        assert nl.balance_text(p) in t, (ly.id, t)
    elif ly.kind == "vibrance":
        assert f"自然饱和度 {p['vibrance']:+d}" in t, (ly.id, t)
    elif ly.kind == "curves":
        for pts in p["channels"].values():
            for x, y in pts:
                if (x, y) not in ((0, 0), (255, 255)):
                    assert f"输入 {x} → 输出 {y}" in t, (ly.id, x, y, t)
    elif ly.kind == "selectiveColor":
        assert nl.selective_text(p) in t, (ly.id, t)


@pytest.mark.parametrize("name", list(CASES))
def test_layer_numbers_match_card_text(name):
    rep, shape = CASES[name]
    for ly in lp.plan_from_report(rep, shape).layers:
        _numbers_shown(ly)
        # reason 就是这张卡片上一条（或合并的几条）建议的原文
        texts = [a.instruction for a in rep.findings[ly.finding].adjustments if getattr(a, "layer", None)]
        rest = ly.reason
        while rest:
            t = next((t for t in texts if rest.startswith(t)), None)
            assert t is not None, ly.reason
            rest = rest[len(t):].removeprefix("；")


def test_each_layer_kind_is_produced_somewhere():
    kinds = {ly.kind for rep, shape in CASES.values() for ly in lp.plan_from_report(rep, shape).layers}
    from test_skin import _modify_by_lightness

    img = _modify_by_lightness(skin_region(), "high", da=0, db=10)  # 高光发黄 -> 色彩平衡
    rep = an.analyze_skin_region(img, CTX)
    cb = [ly for ly in lp.plan_from_report(rep, img.shape).layers if ly.kind == "colorBalance"]
    assert cb and cb[0].params["highlights"][2] > 0  # 往蓝
    _numbers_shown(cb[0])
    kinds.add("colorBalance")
    lab = np.dstack([np.full((80, 100), 55.0), np.full((80, 100), 3.0), np.full((80, 100), 4.0)])
    dull = cs.lab_to_srgb8(lab + np.random.default_rng(0).normal(0, 1.5, lab.shape))  # 整体缺色
    rep = an.analyze_palette_report(dull, CTX)
    vib = [ly for ly in lp.plan_from_report(rep, dull.shape).layers if ly.kind == "vibrance"]
    assert vib and 10 <= vib[0].params["vibrance"] <= 25
    kinds.add("vibrance")
    assert kinds == set(nl.LAYER_KINDS)


# ------------------------------------------------------------------ (c) 格式
def test_all_layers_follow_the_plugin_schema():
    for rep, shape in CASES.values():
        for f in rep.findings:
            for a in f.adjustments:
                if a.layer:
                    nl.validate(a.layer)
                    assert a.layer["name"] and a.layer["scope"] in ("skin", "local", "global")
        for ly in lp.plan_from_report(rep, shape).layers:
            nl.validate({"kind": ly.kind, "params": ly.params, "scope": "global"})


def test_selective_color_schema():
    p = nl.skin_selective(da=-4.0, db=-8.0)  # 偏黄偏红：减黄、减洋红、红色里加青
    nl.validate(nl.layer("selectiveColor", p, "skin"))
    assert p["method"] == "relative" and set(p["colors"]) == {"reds", "yellows"}
    for inks in p["colors"].values():
        assert set(inks) == {"c", "m", "y", "k"} and all(-30 <= v <= 30 for v in inks.values())
    assert p["colors"]["yellows"]["y"] < 0 and p["colors"]["reds"]["m"] < 0 and p["colors"]["reds"]["c"] > 0
    assert nl.selective({"greens": {"c": 0, "m": 0, "y": 0, "k": 0}})["colors"] == {}  # 没改的颜色不列出
    with pytest.raises(ValueError):
        nl.validate(nl.layer("selectiveColor", {"method": "relative", "colors": {"skin": {"c": 1, "m": 0, "y": 0, "k": 0}}}))


def test_mixer_rows_on_the_same_hue_are_merged_not_cancelled():
    out = nl.mixer_rows_to_huesat([("黄色", 89.0, -25.0, 0, 0), ("绿色", 93.0, 25.0, 0, 0), ("蓝色", 230.0, 0.0, -30, 0)])
    (params, text), = out
    assert len(params["ranges"]) == 1 and params["ranges"][0]["saturation"] == -30  # 黄/绿往两边拉 = 不动
    assert "蓝色" in text and "黄色" not in text
    rows = [(n, c, 5.0, 0, 0) for n, c in (("红色", 0), ("橙色", 30), ("黄色", 60), ("绿色", 120), ("浅绿色", 180),
                                           ("蓝色", 240), ("紫色", 270), ("洋红", 300))]
    chunks = nl.mixer_rows_to_huesat(rows)
    assert [len(p["ranges"]) for p, _ in chunks] == [6, 2]
    for p, _ in chunks:
        nl.validate(nl.layer("hueSaturation", p))


def test_grading_and_white_balance_translations():
    assert nl.grading_to_balance(0, 30) == [18, -9, -9]  # 红：往红，少一点洋红/黄
    cr, mg, yb = nl.grading_to_balance(210, 30)
    assert cr < 0 and yb > 0  # 青蓝
    _, mg, yb = nl.balance_for_shift(da=-4.0, db=-10.0)  # 往绿、往蓝
    assert mg > 0 and yb > 0


# ------------------------------------------------------------------ (d) 桥接 JSON
@pytest.mark.parametrize("mode,img", [("dirty", dirty_sky()), ("skin", face_scene(skin_lab=(66, 9, 28))[0])])
def test_bridge_findings_carry_index_and_layers(mode, img):
    rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    q = {"preset": "natural"} if mode == "skin" else {}
    res = bs.analyze_request(rgba, mode, q, bs.Store(), bs.Store())
    json.dumps(res, ensure_ascii=False)
    fs, layers = res["findings"], res["plan"]["layers"]
    assert layers
    for i, f in enumerate(fs):
        assert f["index"] == i
        assert f["layers"] == [ly["id"] for ly in layers if ly["finding"] == i]
        for a in f["adjustments"]:
            if a.get("layer"):
                assert set(a["layer"]) == {"kind", "params", "scope", "name"}  # 内部键不导出
    for ly in layers:
        assert isinstance(ly["finding"], int) and 0 <= ly["finding"] < len(fs)
        assert ly["id"] in fs[ly["finding"]]["layers"]


# ------------------------------------------------------------------ (e) ACR 导出不变
def test_acr_params_unchanged():
    rep, _ = CASES["style_teal_orange"]
    assert ap.params_for_report(rep) == {"hsl": {"黄色": {"hue": -40}, "绿色": {"hue": 40}, "浅绿色": {"hue": -40}, "蓝色": {"hue": -40}}}
    rep, _ = CASES["reference"]
    assert ap.params_for_report(rep) == {
        "curve": {"composite": [[98, 67], [113, 95], [138, 163], [186, 203]]},
        "hsl": {"橙色": {"hue": 15, "lum": 21, "sat": 40}, "浅绿色": {"hue": -40, "lum": -40, "sat": 40},
                "红色": {"hue": 10, "lum": 14, "sat": 30}, "绿色": {"sat": -40}, "蓝色": {"hue": -40, "lum": -40, "sat": 40},
                "黄色": {"sat": -40}},
    }


# ------------------------------------------------------------------ (f) 色差带文字
_WORDS = tuple(w for _, w in DE_LEVELS)


def test_delta_e_numbers_come_with_plain_words():
    rep, _ = CASES["skin_sallow_natural"]
    style = next(f for f in rep.findings if f.problem.startswith("与目标风格"))
    assert re.search(r"相差 ΔE00 \d+\.\d（(很明显|明显|差别很大)）", style.problem), style.problem
    texts = []
    for rep, _ in CASES.values():
        for f in rep.findings:
            texts += [f.problem, *f.evidence, *(a.instruction for a in f.adjustments)]
    found = 0
    for t in texts:
        for m in re.finditer(r"ΔE00 (\d+(?:\.\d+)?)", t):
            found += 1
            tail = t[m.end():m.end() + 12]
            assert any(w in tail for w in _WORDS), t
    assert found >= 5
