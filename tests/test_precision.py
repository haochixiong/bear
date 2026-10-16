"""分析精度：原图精查前先出快速结果、「快速」精度、同一张图重复分析时复用颜色编号。"""

import numpy as np
import pytest
from PyQt6.QtCore import QSettings
from synth import BLUE, YELLOW, face_scene, flat, noisy, sprinkle

from colordoctor.core import analysis as an
from colordoctor.core import pixel_table as ptm
from colordoctor.core.color_management import synthetic_context
from colordoctor.core.sampling import resolution
from colordoctor.ui import session as sess_mod
from colordoctor.ui.main_window import MainWindow, precision_change

CTX = synthetic_context()


def _speckled(h=600, w=800):
    img = noisy(flat(BLUE, h, w), 2)
    img[200:400, 300:500] = sprinkle(img[200:400, 300:500], YELLOW, 0.02, seed=4)
    return img


def test_resolution_context_marks_and_scales_results():
    img = _speckled()
    with resolution(120_000, "preview"):
        quick = an.analyze_region(img, CTX)
    full = an.analyze_region(img, CTX)
    assert quick.precision == "preview" and full.precision == "full"
    assert quick.contamination.mask_scale > 1 and full.contamination.mask_scale == 1
    assert quick.stats.sampled_pixels < full.stats.sampled_pixels == img.shape[0] * img.shape[1]
    # 位置换算回原图坐标：快速结果的问题位置和原图结果在同一处
    qb = [b for b in quick.contamination.blobs if b.kind == "opposite"]
    fb = [b for b in full.contamination.blobs if b.kind == "opposite"]
    assert fb and all(250 <= b.bbox[0] + b.bbox[2] / 2 <= 550 for b in fb)
    for b in qb:
        assert 250 <= b.bbox[0] + b.bbox[2] / 2 <= 550


def test_small_images_are_not_marked_as_reduced():
    img = _speckled(100, 120)
    with resolution(120_000, "fast"):
        r = an.analyze_region(img, CTX)
    assert r.precision == "full"  # 本来就不超过上限：仍是每个像素


def test_session_stages():
    s = sess_mod.Session()
    big = np.zeros((5000, 5000, 3), np.uint8)
    small = np.zeros((1000, 1000, 3), np.uint8)
    assert [lab for _, lab in s.stages(big)] == ["preview", "full"]
    assert [lab for _, lab in s.stages(small)] == ["full"]
    s.precision = "fast"
    assert [lab for _, lab in s.stages(big)] == ["fast"]


def test_colour_index_is_reused_for_the_same_image(monkeypatch):
    built = []
    orig = ptm.PixelTable.__init__

    def init(self, rgb):
        built.append(rgb.shape)
        orig(self, rgb)

    monkeypatch.setattr(ptm.PixelTable, "__init__", init)
    ptm.forget_all()
    img = np.ascontiguousarray(face_scene()[0].repeat(6, axis=0).repeat(6, axis=1))  # > 40 万像素
    an.analyze_palette_report(img, CTX, style_key="teal_orange")
    n1 = len(built)
    an.analyze_palette_report(img, CTX, style_key="morandi", tone_style="japanese")  # 换模板 / 影调风格重算
    an.analyze_region(img, CTX)
    assert n1 == 1 and len(built) == 1
    img2 = img.copy()
    img2[0, 0] = (1, 2, 3)  # 内容不同（哪怕只差一个像素）不能复用
    an.analyze_region(img2, CTX)
    assert len(built) == 2


def test_modified_array_is_not_reused(monkeypatch):
    ptm.forget_all()
    img = np.ascontiguousarray(face_scene()[0].repeat(6, axis=0).repeat(6, axis=1))
    a = an.analyze_region(img, CTX)
    img[:] = (40, 80, 200)  # 同一个数组被改写
    img[::7, ::7] = (230, 200, 40)
    b = an.analyze_region(img, CTX)
    assert b.stats.lab_mean != a.stats.lab_mean


def test_precision_change_words():
    img = _speckled()
    with resolution(30_000, "preview"):
        quick = an.analyze_region(img, CTX)
    full = an.analyze_region(img, CTX)
    assert precision_change(full, full).startswith("结论和快速结果一致")
    assert isinstance(precision_change(quick, full), str)


@pytest.fixture
def win(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow(settings=settings)
    qtbot.addWidget(w)
    w.show()
    return w


def test_desktop_shows_quick_result_then_replaces_it(win, qtbot, monkeypatch):
    monkeypatch.setattr(sess_mod, "PREVIEW_FROM", 100_000)
    monkeypatch.setattr(sess_mod, "PREVIEW_MAX", 60_000)
    win.set_mode("dirty")
    seen = []
    orig = win._rect_done

    def spy(region, rect, rep, *a, **k):
        seen.append(rep.precision)
        return orig(region, rect, rep, *a, **k)

    monkeypatch.setattr(win, "_rect_done", spy)
    win._run_rect(_speckled(), (0, 0, 800, 600), CTX, "t", 0)
    qtbot.waitUntil(lambda: seen[-1:] == ["full"], timeout=20000)
    assert seen == ["preview", "full"]
    assert len(win.session.history) == 1  # 精查结果替换了快速结果，不是多一条
    assert win.session.history[0].result.precision == "full"
    assert "精查" in win.statusBar().currentMessage()


def test_desktop_fast_precision_setting(win, qtbot):
    win.session.precision = "fast"
    win.set_mode("dirty")
    import colordoctor.core.sampling as smp

    old = smp.FAST_MAX
    try:
        sess_mod.FAST_MAX = 60_000
        win._run_rect(_speckled(), (0, 0, 800, 600), CTX, "t", 0)
        qtbot.waitUntil(lambda: len(win.session.history) == 1, timeout=20000)
    finally:
        sess_mod.FAST_MAX = old
    rep = win.session.history[0].result
    assert rep.precision == "fast" and rep.contamination.mask_scale > 1
    from colordoctor.ui.report_html import report_html

    assert "快速精度" in report_html(rep)


# ------------------------------------------------------------------ 插件服务：精度、先快速后精查、同一张图复用
def _post(port, path, body, headers):
    import urllib.request

    from colordoctor.bridge import server as bs

    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, method="POST",
                                 headers={bs.CLIENT_HEADER: "t", **headers})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, __import__("json").loads(r.read())


def _get(port, path):
    import json
    import urllib.error
    import urllib.request

    from colordoctor.bridge import server as bs

    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers={bs.CLIENT_HEADER: "t"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_bridge_progressive_then_refined(monkeypatch):
    import threading

    from colordoctor.bridge import server as bs

    monkeypatch.setattr(bs, "PREVIEW_FROM", 100_000)
    monkeypatch.setattr("colordoctor.core.sampling.PREVIEW_MAX", 60_000)
    srv = bs.make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        img = _speckled()
        hdr = {"X-Width": "800", "X-Height": "600", "X-Components": "3", "Content-Type": "application/octet-stream"}
        code, quick = _post(port, "/analyze?mode=dirty&progressive=1", img.tobytes(), hdr)
        assert code == 200 and quick["precision"] == "preview" and quick["refine"]
        code, full = _get(port, f"/refined?token={quick['refine']}&wait=60")
        assert code == 200 and full["precision"] == "full" and full["session"] != quick["session"]
        assert "refine" not in full
        assert _get(port, "/refined?token=nosuch")[0] == 404
        # 「快速」时插件已经缩小读取：服务按收到的像素算（不再缩小），progressive 对小图不起作用
        code, small = _post(port, "/analyze?mode=dirty&progressive=1",
                            np.ascontiguousarray(img[:200, :300]).tobytes(),
                            {**hdr, "X-Width": "300", "X-Height": "200"})
        assert small["precision"] == "full" and "refine" not in small
        assert _get(port, "/health")[1]["protocol"] >= 4
    finally:
        srv.shutdown()
        srv.server_close()


def test_bridge_precision_param_and_reuse(monkeypatch):
    from colordoctor.bridge import server as bs

    monkeypatch.setattr("colordoctor.core.sampling.FAST_MAX", 60_000)
    img = _speckled()
    res = bs.analyze_request(img.copy(), "dirty", {"precision": "fast"}, bs.Store(), bs.Store())
    assert res["precision"] == "fast"
    with pytest.raises(ValueError):
        bs.analyze_request(img.copy(), "dirty", {"precision": "nosuch"}, bs.Store(), bs.Store())
    # 插件每次重新传同一张图（新的字节）：用同一份准备好的数组，颜色编号可以复用
    p1 = bs._prepared(img.copy())
    p2 = bs._prepared(img.copy())
    assert p1 is p2
    other = img.copy()
    other[5, 5] = (1, 2, 3)
    assert bs._prepared(other) is not p1


# ------------------------------------------------------------------ 取消：又分析了一次，旧的原图精查马上停下
def test_cancelled_analysis_stops():
    import threading

    from colordoctor.core.pixel_table import Cancelled, cancellable

    ev = threading.Event()
    ev.set()
    with cancellable(ev), pytest.raises(Cancelled):
        an.scan_image(_speckled(), CTX)
    an.scan_image(_speckled(), CTX)  # 不在取消范围内：照常算


def test_bridge_supersede_cancels_old_refine(monkeypatch):
    import time

    from colordoctor.bridge import server as bs
    from colordoctor.core.pixel_table import check_cancel

    monkeypatch.setattr(bs, "PREVIEW_FROM", 10)

    def fake(pixels, mode, q, *a, **k):
        if q.get("precision") == "preview":
            return {"precision": "preview", "session": "s"}
        for _ in range(500):  # 原图精查：很慢，期间不断检查是否被取消
            check_cancel()
            time.sleep(0.01)
        return {"precision": "full"}

    monkeypatch.setattr(bs, "analyze_request", fake)
    refines = bs.Store()
    px = np.zeros((10, 10, 3), np.uint8)
    first = bs.analyze_progressive(px, "dirty", {"progressive": "1"}, bs.Store(), bs.Store(), refines)
    job = refines.get(first["refine"])
    t = time.perf_counter()
    bs.analyze_progressive(px, "dirty", {"progressive": "1", "supersede": first["refine"]}, bs.Store(), bs.Store(), refines)
    assert job.done.wait(3) and time.perf_counter() - t < 3
    assert job.error[0] == 409 and job.result is None


def test_desktop_stops_refine_when_result_is_replaced(win, qtbot):
    import time

    from colordoctor.core.pixel_table import check_cancel

    seen = []

    class Item:
        id = 4242

    def compute(stage):
        if stage[1] == "preview":
            return "quick"
        for _ in range(500):
            check_cancel()
            time.sleep(0.01)
        return "full"

    def on_result(rep, final):
        seen.append(rep)
        return Item()

    win._run_staged([(1, "preview"), (2, "full")], compute, on_result)
    qtbot.waitUntil(lambda: 4242 in win._refining, timeout=5000)
    t = time.perf_counter()
    ev = win._refining[4242]
    win._stop_refine(4242)  # 这条结果被重算 / 替换 / 删除
    qtbot.waitUntil(lambda: ev.is_set(), timeout=1000)
    qtbot.wait(300)
    assert seen == ["quick"] and time.perf_counter() - t < 3  # 精查停下，没有替换，也没有报错弹窗
