"""九种影调：按直方图描述（基调 × 调长），并能按目标影调做分位点（直方图）匹配。"""

import numpy as np
import pytest
from synth import face_scene, highkey_face_scene, lowkey_face_scene

from colordoctor.core import tone_class as tc


def _from_quantiles(q, n=100000):
    u = np.random.default_rng(0).random(n)
    xs = [0, 0.02, 0.25, 0.5, 0.75, 0.98, 1]
    ys = [max(0, q[0] - 5), *q, min(255, q[-1] + 5)]
    g = np.clip(np.interp(u, xs, ys), 0, 255)
    return np.stack([g, g, g], -1).reshape(-1, 1, 3)


@pytest.mark.parametrize("name", tc.TONE_NAMES)
def test_each_tone_standard_histogram_classifies_as_itself(name):
    assert tc.classify(_from_quantiles(tc.target_quantiles(name))).name == name


def test_examples():
    assert tc.classify(lowkey_face_scene(skin_lab=(62.0, 13.0, 17.0))[0]).key == "低"
    under = (face_scene()[0] * 0.35).astype(np.uint8)
    assert tc.classify(under).name == "低短调"
    assert tc.classify(highkey_face_scene()[0]).key == "高"


def test_match_curve_depends_on_the_target_and_is_monotonic():
    under = (face_scene()[0] * 0.35).astype(np.uint8)
    q = tc.quantiles(under)
    bright, _ = tc.match_curve(q, tc.target_quantiles("高短调"))
    moody, _ = tc.match_curve(q, tc.target_quantiles("低长调"))
    for pts in (bright, moody):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        assert xs == sorted(xs) and ys == sorted(ys) and pts[0] == [0, 0] and pts[-1] == [255, 255]
        assert all(abs(y - x) <= tc.MAX_MOVE + 1 for x, y in pts)
    assert bright[1][1] > bright[1][0]  # 往高短调：暗部也提亮
    assert moody[1][1] <= moody[1][0]  # 往低长调：暗部保持暗（不提亮），只拉开亮部
