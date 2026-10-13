"""AI 抠皮肤：用假的推理会话测试前后处理；有真实模型时（COLORDOCTOR_SEG_MODEL）再跑一次真实推理。"""

import os

import numpy as np
import pytest
from synth import face_scene

from colordoctor.core import segmentation as sg


class FakeSession:
    """把输入图里“像皮肤”的像素判为脸部（上半）或身体（下半）。"""

    def __init__(self):
        self.seen = None

    def get_inputs(self):
        return [type("I", (), {"name": "input_29"})()]

    def get_providers(self):
        return ["FakeExecutionProvider"]

    def run(self, _out, feed):
        x = feed["input_29"][0]
        self.seen = x
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        skin = (r > g) & (g > b) & (r - b > 0.12) & (r > 0.45)
        logits = np.full((256, 256, 6), -4.0, np.float32)
        logits[..., 0] = 4.0
        top = np.zeros((256, 256), bool)
        top[:128] = True
        logits[skin & top, 3] = 6.0
        logits[skin & ~top, 2] = 6.0
        return [logits[None]]


def test_segment_skin_splits_face_and_body_and_refines():
    img, truth = face_scene(200, 200)
    sess = FakeSession()
    seg = sg.segment_skin(img, session=sess)
    assert sess.seen.shape == (256, 256, 3) and 0 <= sess.seen.min() and sess.seen.max() <= 1
    assert seg.face.shape == truth.shape and seg.provider == "FakeExecutionProvider"
    both = seg.part(None)
    inter = (both & truth).sum() / max(1, truth.sum())
    assert inter > 0.8
    assert (both & ~truth).sum() / max(1, both.sum()) < 0.1
    assert seg.info["ai_face"] > 0
    assert seg.part("脸部").sum() == seg.face.sum() and seg.part("身体").sum() == seg.body.sum()


def test_unavailable_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("COLORDOCTOR_SEG_MODEL", str(tmp_path / "none.onnx"))
    monkeypatch.setattr(sg, "model_candidates", lambda: [tmp_path / "none.onnx"])
    ok, why = sg.available()
    assert not ok and ("onnxruntime" in why or "模型" in why)


def test_bridge_falls_back_to_color_rule(monkeypatch):
    from colordoctor.bridge import server as bs

    monkeypatch.setattr(sg, "_session", lambda: (_ for _ in ()).throw(sg.SegmentationUnavailable("找不到模型文件")))
    img, _ = face_scene(160, 160)
    res = bs.analyze_request(img, "skin", {"seg": "ai", "part": "脸部"}, bs.Store(), bs.Store(), layer_kind="pixel")
    assert res["summary"]


@pytest.mark.skipif(not os.environ.get("COLORDOCTOR_SEG_MODEL") or not sg.available()[0], reason="没有真实模型")
def test_real_model_runs():
    img, _ = face_scene(256, 256)
    seg = sg.segment_skin(img)
    assert seg.labels.shape == (256, 256)


def test_segment_skin_keeps_aspect_ratio_for_wide_images():
    """宽幅图按比例缩放后居中补边，人不被压扁；输出蒙版仍与原图同尺寸、位置对应。"""
    img, truth = face_scene(200, 200)
    wide = np.full((200, 800, 3), 40, np.uint8)
    wide[:, 300:500] = img
    t = np.zeros((200, 800), bool)
    t[:, 300:500] = truth
    sess = FakeSession()
    seg = sg.segment_skin(wide, session=sess)
    pad = sess.seen[:, :, 0]
    assert np.allclose(pad[:90], 0.5) and np.allclose(pad[-90:], 0.5)  # 上下补边
    both = seg.part(None)
    assert both.shape == t.shape
    assert (both & t).sum() / max(1, t.sum()) > 0.75
    assert (both & ~t).sum() / max(1, both.sum()) < 0.15
