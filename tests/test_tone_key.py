"""影调：暗调（低调）/ 高调画面不被往“白亮透”推；高光、阴影、黑白点的识别要适应画面本身。

规则：明度/影调目标只来自明确选择的风格或肤色风格；没选或选默认的“自然通用”时，肤色建议保持当前明度。
"""

import numpy as np
import pytest
from synth import face_scene, lowkey_face_scene

from colordoctor.core import analysis as an
from colordoctor.core import layer_plan as lp
from colordoctor.core import portrait_analysis as pa
from colordoctor.core import skin_presets as sp
from colordoctor.core.color_management import synthetic_context

CTX = synthetic_context()
BRIGHT_WORDS = ("提亮", "偏暗", "太暗", "不够透亮", "明度提高")


def _layers(rep, img):
    return lp.plan_from_report(rep, img.shape).layers


def _composite_lifts(layers) -> list:
    """RGB 复合曲线里把输出抬高的点（提亮）。"""
    out = []
    for ly in layers:
        if ly.kind == "curves" and "composite" in ly.params["channels"]:
            out += [(x, y) for x, y in ly.params["channels"]["composite"] if y > x + 2]
    return out


def _texts(rep):
    return [a.instruction for f in rep.findings for a in f.adjustments] + [e for f in rep.findings for e in f.evidence]


# ------------------------------------------------------------------ 肤色：默认不按深浅判断，暗调保持暗
@pytest.mark.parametrize("preset_key", [None, "natural", "moody"])
def test_dark_portrait_skin_is_not_brightened(preset_key):
    img, _ = lowkey_face_scene()
    rep = an.analyze_skin_region(img, CTX, preset_key=preset_key)
    assert not _composite_lifts(_layers(rep, img))
    texts = _texts(rep)
    assert not [t for t in texts if "明度 L*：偏低" in t or "明度 → 橙色提高" in t], texts
    # 色相、浓淡都正常的暗调皮肤：不出“与目标风格的差距”卡片
    assert not any(f.problem.startswith("与目标风格") for f in rep.findings)


@pytest.mark.parametrize("skin_lab", [(80.0, 10.0, 14.0), (34.0, 11.0, 14.0)])
def test_natural_preset_never_changes_skin_lightness(skin_lab):
    """白皮肤不被压暗、深肤色不被提亮：“自然通用”只看色相和浓淡。"""
    img, _ = face_scene(skin_lab=skin_lab)
    rep = an.analyze_skin_region(img, CTX, preset_key="natural")
    for f in rep.findings:
        for a in f.adjustments:
            assert not (a.layer and "composite" in a.layer["params"].get("channels", {})), a.instruction
            assert "明度：RGB 复合曲线" not in a.instruction and "明亮度" not in a.instruction
    fit = pa.preset_fit(rep.skin, sp.get_preset("natural"))
    assert "保持当前值" in fit["明度 L*"]


def test_preset_lightness_moves_only_to_the_range_edge():
    """明确选了亮一些的肤色风格（港风 L* 55–68）：只拉到区间下沿，不拉到中心。"""
    img, _ = lowkey_face_scene()
    rep = an.analyze_skin_region(img, CTX, preset_key="hongkong")
    card = next(f for f in rep.findings if f.problem.startswith("与目标风格"))
    curve = next(a for a in card.adjustments if a.instruction.startswith("明度：RGB 复合曲线"))
    assert "→ 55.0）" in curve.instruction
    p = sp.get_preset("moody")
    assert p.target_for((40.0, 0, 0))[0] == 42.0 and p.target_for((50.0, 0, 0))[0] == 50.0
    assert p.target_for((70.0, 0, 0))[0] == 58.0  # 选了暗调、皮肤太亮：往暗调区间压
