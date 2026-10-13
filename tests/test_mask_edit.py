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


def _bfs(m, y, x):
    from collections import deque

    h, w = m.shape
    out = np.zeros_like(m)
    if not m[y, x]:
        return out
    q = deque([(y, x)])
    out[y, x] = True
    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
            if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                q.append((ny, nx))
    return out


def test_connected_component_matches_flood_fill():
    rng = np.random.default_rng(3)
    for _ in range(200):
        h, w = rng.integers(1, 30, 2)
        m = rng.random((h, w)) < rng.uniform(0.3, 0.8)
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        assert np.array_equal(me.connected_component(m, y, x), _bfs(m, y, x))


def test_magic_wand_large_flat_area_is_fast():
    import time

    img = np.full((1000, 1000, 3), 180, np.uint8)
    t = time.perf_counter()
    m = me.magic_wand(img, 500, 500, 8)
    assert m.all()
    assert time.perf_counter() - t < 1.5  # 以前逐像素膨胀约 2.5 秒
