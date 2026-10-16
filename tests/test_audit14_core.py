"""第 14 轮审查（核心计算）的回归测试：颜色极多时整图扫描 / 均匀度的内存和时间只和像素数有关、
缓存有上限、取样和图的大小无关、并行的分析一项出错其余停下、零星透明像素排除。结果都和原来的算法一样。"""

import threading
import time
import tracemalloc

import numpy as np
import pytest
from synth import BLUE, YELLOW, face_scene, flat, noisy, sprinkle

from colordoctor.core import analysis as an
from colordoctor.core import clustering
from colordoctor.core import color_spaces as cs
from colordoctor.core import contamination as ct
from colordoctor.core import pixel_guard as pg
from colordoctor.core import pixel_table as ptm
from colordoctor.core import recommendations as rec
from colordoctor.core import region_statistics as rs
from colordoctor.core import skin_uniformity as su
from colordoctor.core.color_management import synthetic_context

CTX = synthetic_context()


def _noisy_skin(h, w, amp=60, seed=0):
    """颜色极多（几乎每个像素一种颜色）的肤色噪声图。"""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.int16)
    img[:] = (190, 140, 110)
    img += rng.integers(-amp, amp, img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def _bincount_tally(idx, n):
    """原来的做法：每块一张 n 长的计数表。"""
    cnt = np.bincount(np.asarray(idx).reshape(-1), minlength=n)
    u = np.flatnonzero(cnt)
    return u, cnt[u]


def _same(a, b):
    for f in a.__dataclass_fields__:
        x, y = getattr(a, f), getattr(b, f)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            assert np.array_equal(x, y), f
        elif f != "normal_counts":
            assert x == y, f


# ------------------------------------------------------------------ 按块数颜色：只和这块的大小有关
@pytest.mark.parametrize("n, size", [(50, 1000), (100_000, 300), (100_000, 0), (7, 7)])
def test_tally_equals_bincount(n, size):
    idx = np.random.default_rng(n + size).integers(0, n, size).astype(np.int32)
    u, c = ptm.tally(idx, n)
    u0, c0 = _bincount_tally(idx, n)
    assert np.array_equal(u, u0) and np.array_equal(c, c0) and c.dtype == np.int64


def test_scan_and_uniformity_with_many_colours_match_per_tile_bincount(monkeypatch):
    img = _noisy_skin(300, 420)  # 每块像素比颜色种类少得多：走排序计数
    with ptm.scope():
        assert ptm.table(img).n > 4 * (300 // 12) ** 2
        new_scan, new_uni = ct.analyze(img, mode="scan"), su.analyze_uniformity(img)
        m = np.zeros(img.shape[:2], bool)
        m[20:280, 30:400] = True
        new_scan_m = ct.analyze(img, mode="scan", valid=m)
        new_uni_m = su.analyze_uniformity(img, mask=m)
    monkeypatch.setattr(ct, "tally", _bincount_tally)
    monkeypatch.setattr(su, "tally", _bincount_tally)
    ptm.forget_all()
    with ptm.scope():
        _same(new_scan, ct.analyze(img, mode="scan"))
        _same(new_scan_m, ct.analyze(img, mode="scan", valid=m))
        old_uni, old_uni_m = su.analyze_uniformity(img), su.analyze_uniformity(img, mask=m)
    for a, b in ((new_uni, old_uni), (new_uni_m, old_uni_m)):
        assert a.patches == b.patches and a.score == b.score and np.array_equal(a.tile_map, b.tile_map)


def test_scan_memory_does_not_grow_with_tiles_times_colours():
    img = _noisy_skin(600, 900)  # 约 50 万种颜色、200 多块：原来每块一张 n 长的计数表，要 1 GB 左右
    with ptm.scope():
        pt = ptm.table(img)
        _ = pt.lab
        assert pt.n > 300_000
        tracemalloc.start()
        try:
            ct.analyze(img, mode="scan")
            su.analyze_uniformity(img)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert peak < 200e6, f"peak {peak / 1e6:.0f} MB"


def test_scan_finds_scattered_opposites_on_many_colour_background():
    img = noisy(flat(BLUE, 600, 800), 3)
    img[200:400, 300:500] = sprinkle(img[200:400, 300:500], YELLOW, 0.03, seed=3)
    r = ct.analyze(img, mode="scan")
    assert r.opposite_fraction > 0 and any(b.kind == "opposite" for b in r.blobs)


# ------------------------------------------------------------------ 缓存：有字节上限，不在分析之间占着蒙版 / 像素表
def test_recent_cache_is_bounded_by_bytes(monkeypatch):
    ptm.forget_all()
    imgs = [np.ascontiguousarray(face_scene(seed=i)[0].repeat(4, axis=0).repeat(4, axis=1)) for i in range(3)]
    one = imgs[0].nbytes + imgs[0].shape[0] * imgs[0].shape[1] * 4  # 一张图 + 它的颜色编号
    monkeypatch.setattr(ptm, "RECENT_BYTES", int(1.5 * one))
    for img in imgs:
        an.analyze_region(img, CTX)
        assert ptm._recent_bytes() <= ptm.RECENT_BYTES or len(ptm._recent) == 1
    assert len(ptm._recent) == 1 and ptm._root(next(iter(ptm._recent.values()))["arr"]) is imgs[-1]  # 最近的一张总留着
    monkeypatch.setattr(ptm, "RECENT_BYTES", 1)  # 一张都放不下：仍然留着最近的一项（再分析同一张图不用重新编号）
    an.analyze_region(imgs[0], CTX)
    assert len(ptm._recent) == 1 and ptm._root(next(iter(ptm._recent.values()))["arr"]) is imgs[0]


def test_crops_of_one_image_count_the_image_once():
    ptm.forget_all()
    img = np.ascontiguousarray(face_scene()[0].repeat(5, axis=0).repeat(5, axis=1))
    for x in (0, 100, 200):
        with ptm.scope():
            ptm.table(img[:, x : x + 700])
    roots = {id(ptm._root(e["arr"])) for e in ptm._recent.values()}
    assert len(ptm._recent) == 3 and len(roots) == 1
    tables = sum(e["items"]["table"].nbytes() for e in ptm._recent.values())
    assert ptm._recent_bytes() == img.nbytes + tables


def test_mask_counts_are_dropped_when_the_analysis_ends():
    img = np.ascontiguousarray(face_scene()[0].repeat(4, axis=0).repeat(4, axis=1))
    m = np.zeros(img.shape[:2], bool)
    m[100:600] = True
    with ptm.scope():
        pt = ptm.table(img)
        pt.count(m)
        assert len(pt._mask_counts) == 1
    assert pt._mask_counts == {}


def test_in_place_edit_off_the_old_sample_grid_is_detected():
    h, w = 1200, 1000
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (40, 90, 200)
    img += np.random.default_rng(0).integers(0, 4, img.shape, dtype=np.uint8)
    assert an.analyze_region(img, CTX).contamination.opposite_fraction == 0
    ys = np.arange(1, h, 2)
    ys = ys[ys % 4 != 0]  # 原来只抽查每 4 行、每 3 列：避开这些位置改写
    xs = np.arange(1, w, 7)
    xs = xs[xs % 3 != 0]
    img[np.ix_(ys, xs)] = (220, 200, 40)
    fresh = an.analyze_region(img.copy(), CTX).contamination.opposite_fraction
    assert fresh > 0 and an.analyze_region(img, CTX).contamination.opposite_fraction == fresh


def test_mask_memo_sees_edits_off_the_old_sample_rows():
    img = np.random.default_rng(0).integers(0, 30, (1200, 1000, 3)).astype(np.uint8)
    pt = ptm.PixelTable(img)
    m = np.zeros((1200, 1000), bool)
    m[:600] = True
    pt.count(m)
    m[1] = False  # 个数不变，只是换了几行（原来的指纹只抽查偶数行）
    m[601] = True
    assert np.array_equal(pt.count(m), np.bincount(pt.index[m], minlength=pt.n))


def test_report_does_not_keep_the_pixel_table():
    img = noisy(flat(BLUE, 300, 400), 3)
    img[100:200, 100:200] = sprinkle(img[100:200, 100:200], YELLOW, 0.03, seed=2)
    r = ct.analyze(img, mode="region")
    cols, w = r.normal_counts
    assert isinstance(cols, np.ndarray) and cols.shape[1] == 3 and (w > 0).all()
    fast, scanned = an._bands(img, r.mask, r.normal_counts), an._bands(img, r.mask, None)
    assert fast.keys() == scanned.keys() and all(np.allclose(fast[k], scanned[k]) for k in fast)
    for rep in (an.analyze_region(img, CTX), an.analyze_skin_region(face_scene()[0], CTX), an.scan_image(img, CTX)):
        assert rep.contamination.normal_counts is None


# ------------------------------------------------------------------ 抽样和图的大小（走哪条路）无关
def _with_threshold(monkeypatch, value, fn):
    monkeypatch.setattr(cs, "PER_COLOUR_MIN", value)
    ptm.forget_all()
    with ptm.scope():
        return fn()


@pytest.mark.parametrize("shape", [(300, 400), (700, 800)])  # 小于 / 大于 PER_COLOUR_MIN
def test_clusters_and_adjustments_do_not_depend_on_the_path(monkeypatch, shape):
    rng = np.random.default_rng(3)
    h, w = shape
    base = np.zeros((h, w, 3))
    base[..., 0] = np.linspace(80, 220, w)
    base[..., 1] = 120
    base[..., 2] = np.linspace(40, 160, h)[:, None]
    img = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    skin = np.clip(np.full((h, w, 3), (205, 150, 125)) + rng.normal(0, 5, (h, w, 3)), 0, 255).astype(np.uint8)

    def run():
        cols, wts = an._skin_pixels(skin)
        adj = [(a.tool, a.params) for a in rec.skin_adjustments(cols, (70.0, 14.0, 18.0), weights=wts)]
        return rs.compute_stats(img).clusters, adj

    per_pixel = _with_threshold(monkeypatch, 10**12, run)
    per_colour = _with_threshold(monkeypatch, 1, run)
    assert per_pixel == per_colour


def test_unique_colours_small_and_large_paths_agree(monkeypatch):
    px = np.random.default_rng(0).integers(0, 12, (5000, 3)).astype(np.uint8)
    small = cs.unique_colours(px)
    monkeypatch.setattr(cs, "PER_COLOUR_MIN", 1)
    large = cs.unique_colours(px)
    assert np.array_equal(small[0], large[0]) and np.array_equal(small[1], large[1])
    assert small[1].sum() == 5000 and small[1].dtype == np.int64


def test_weighted_kmeans_equals_repeating_points():
    pts = np.array([[10.0, 0, 0], [12.0, 0, 0], [50.0, 0, 0]])
    w = np.array([3.0, 1.0, 4.0])
    c, _ = clustering.kmeans(pts, 1, weights=w)
    assert np.allclose(c[0], np.repeat(pts, w.astype(int), axis=0).mean(axis=0))
    # 颜色不多时按像素数聚类：一种占 75% 像素的颜色不会被零星的颜色“分票”
    lab = np.array([[60.0, 20, 20]] + [[60.0 + i % 5, -30 + i % 7, 30 - i % 11] for i in range(1, 400)])
    wts = np.ones(len(lab))
    wts[0] = 3 * wts[1:].sum()
    out = clustering.dominant_clusters(lab, weights=wts)
    assert out[0][1] == pytest.approx(0.75, abs=0.01) and np.allclose(out[0][0], lab[0], atol=0.5)


def test_model_sample_is_proportional_to_pixel_counts(monkeypatch):
    seen = {}

    def solve(x, target, model, **kw):
        seen.setdefault("x", x)
        raise StopIteration

    monkeypatch.setattr(rec.pm, "solve", solve)
    # 一种颜色 10000 个像素 + 1000 种只有 1 个像素的零星颜色（真实占比 9%）
    rare = np.stack(np.meshgrid(np.arange(10), np.arange(10), np.arange(10), indexing="ij"), -1).reshape(-1, 3) + (100, 140, 180)
    cols = np.vstack([[[205, 150, 125]], rare]).astype(np.uint8)
    w = np.r_[10000, np.ones(1000)]
    with pytest.raises(StopIteration):
        rec.skin_adjustments(cols, (60.0, 30.0, 10.0), weights=w)
    share = 1 - np.mean(np.all(np.rint(seen["x"] * 255) == cols[0], axis=1))
    assert 0.06 < share < 0.12  # 原来每种颜色至少一个名额：零星颜色占到 27%


# ------------------------------------------------------------------ together：一项出错，其余停下再抛错
def test_together_stops_siblings_when_one_fails():
    state = {}

    def slow(name):
        def run():
            state[name] = "running"
            for _ in range(300):
                ptm.check_cancel()
                time.sleep(0.01)
            state[name] = "finished"
        return run

    def bad():
        time.sleep(0.05)
        raise ValueError("boom")

    t = time.perf_counter()
    with pytest.raises(ValueError):
        ptm.together(bad, slow("a"), slow("b"))
    assert time.perf_counter() - t < 2
    assert state == {"a": "running", "b": "running"}  # 已经停下（不是还在后台跑），也没跑完
    time.sleep(0.1)
    assert state == {"a": "running", "b": "running"}

    state.clear()
    with pytest.raises(ValueError):  # 出错的是后台那一项：本线程做的第一项也停下，抛的是真正的错误
        ptm.together(slow("a"), bad)
    assert state == {"a": "running"}


def test_together_still_honours_outer_cancel():
    ev = threading.Event()

    def stop_later():
        time.sleep(0.05)
        ev.set()
        for _ in range(300):
            ptm.check_cancel()
            time.sleep(0.01)

    with ptm.cancellable(ev), pytest.raises(ptm.Cancelled):
        ptm.together(stop_later, stop_later)
    assert ptm.together(lambda: 1, lambda: 2) == [1, 2]


def test_failed_analysis_leaves_nothing_in_the_cache():
    ptm.forget_all()
    img = np.ascontiguousarray(face_scene()[0].repeat(6, axis=0).repeat(6, axis=1))
    with pytest.raises(Exception):  # noqa: B017 - 空的手动蒙版：统计报错
        an.analyze_skin_region(img, CTX, manual_mask=np.zeros(img.shape[:2], bool))
    time.sleep(0.3)
    assert len(ptm._recent) == 0


# ------------------------------------------------------------------ 透明度、精度、边界
def test_a_few_transparent_pixels_are_excluded():
    rng = np.random.default_rng(0)
    px = np.zeros((1000, 1000, 4), np.uint8)
    px[..., :3] = np.clip(np.array([200, 150, 120]) + rng.integers(-4, 5, (1000, 1000, 3)), 0, 255)
    px[..., 3] = 255
    holes = rng.random((1000, 1000)) < 0.0008  # 0.08 %：擦掉的零星小点，RGB 是别的颜色
    px[holes, 3] = 0
    px[holes, :3] = (40, 160, 200)
    p = pg.prepare(px)
    assert p.valid is not None and np.array_equal(p.valid, ~holes) and p.notes == ()
    assert an.scan_image(p.rgb, CTX, valid=p.valid).contamination.opposite_fraction == 0
    full = pg.prepare(np.dstack([px[..., :3], np.full((1000, 1000), 255, np.uint8)]))
    assert full.valid is None


def test_colour_lists_are_converted_in_float64():
    rng = np.random.default_rng(0)
    cols = rng.integers(0, 256, (cs.PER_COLOUR_MIN + 10, 3)).astype(np.uint8)
    lab = cs.srgb8_to_lab(cols)
    assert lab.dtype == np.float64
    assert np.allclose(lab, cs.xyz_to_lab(cs.srgb8_to_xyz(cols.astype(np.float64))), atol=1e-9)
    img = cols[: 800 * 500].reshape(800, 500, 3)
    assert cs.srgb8_to_lab(img).dtype == np.float32  # 整幅图仍用 float32 省内存
    pt = ptm.PixelTable(img)
    assert pt.lab.dtype == np.float64
    assert np.allclose(pt.lab, cs.xyz_to_lab(cs.srgb8_to_xyz(pt.colours.astype(np.float64))), atol=1e-9)


@pytest.mark.parametrize("mode", ["region", "scan", "skin"])
def test_all_false_mask_on_one_pixel_is_not_treated_as_all_valid(mode):
    r = ct.analyze(np.array([[[200, 40, 40]]], np.uint8), mode=mode, valid=np.zeros((1, 1), bool))
    assert r.mask[0, 0] == ct.MASK_OUTSIDE
