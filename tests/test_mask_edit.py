import numpy as np

from colordoctor.core import mask_edit as me
from synth import face_scene


def test_paint_add_and_erase():
    m = np.zeros((20, 20), bool)
    me.paint(m, 10, 10, 3, True)
    assert m[10, 10] and m[10, 13] and not m[10, 14]
    me.paint(m, 10, 10, 1, False)
    assert not m[10, 10] and m[10, 13]


def test_paint_at_border_is_clipped():
    m = np.zeros((10, 10), bool)
    me.paint(m, 0, 0, 3, True)
    me.paint(m, 50, 50, 3, True)
    assert m[0, 0] and m.sum() < 20


def test_stroke_is_continuous():
    m = np.zeros((20, 60), bool)
    me.stroke(m, [(2, 10), (57, 10)], 2, True)
    assert m[10, 2:58].all()


def test_magic_wand_selects_connected_similar_region():
    img, truth = face_scene()
    w = me.magic_wand(img, 100, 45, tolerance=6)  # 点在头发上
    assert w[45, 100] and w.sum() > 500
    assert not (w & truth).any()  # 不会选到皮肤
    w2 = me.magic_wand(img, 100, 120, tolerance=12)  # 点在脸颊上
    assert (w2 & truth).sum() > 0.6 * truth.sum()


def test_magic_wand_outside_returns_empty():
    img, _ = face_scene()
    assert not me.magic_wand(img, -1, 5).any()


def test_undo():
    h = me.MaskHistory(np.zeros((5, 5), bool))
    h.push()
    h.mask[2, 2] = True
    assert h.undo() and not h.mask[2, 2]
    assert not h.undo()
