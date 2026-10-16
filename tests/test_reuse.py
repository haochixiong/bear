"""同一张图再分析时复用颜色编号、同一个蒙版只数一次、被新的分析取代时停下。结果都和重新算一样。"""

import threading
import time

import numpy as np
import pytest
from PyQt6.QtCore import QSettings
from synth import BLUE, YELLOW, face_scene, flat, noisy, sprinkle

from colordoctor.bridge import server as bs
from colordoctor.core import analysis as an
from colordoctor.core import pixel_table as ptm
from colordoctor.core.color_management import synthetic_context
from colordoctor.ui.main_window import MainWindow

CTX = synthetic_context()


def _big_face():
    return np.ascontiguousarray(face_scene()[0].repeat(6, axis=0).repeat(6, axis=1))  # > 40 万像素：走颜色编号


def test_colour_index_is_reused_for_the_same_image(monkeypatch):
    built = []
    orig = ptm.PixelTable.__init__

    def init(self, rgb):
        built.append(rgb.shape)
        orig(self, rgb)

    monkeypatch.setattr(ptm.PixelTable, "__init__", init)
    ptm.forget_all()
    img = _big_face()
    a = an.analyze_palette_report(img, CTX, style_key="teal_orange")
    an.analyze_palette_report(img, CTX, style_key="morandi", tone_style="japanese")  # 换模板 / 影调风格重算
    an.analyze_region(img, CTX)
    assert len(built) == 1
    ptm.forget_all()
    b = an.analyze_palette_report(img, CTX, style_key="teal_orange")  # 重新编号：结果一样
    assert a.palette.tone == b.palette.tone and a.palette.colorfulness == b.palette.colorfulness
    img2 = img.copy()
    img2[0, 0] = (1, 2, 3)  # 内容不同（哪怕只差一个像素）不能复用
    an.analyze_region(img2, CTX)
    assert len(built) == 3


def test_modified_array_is_not_reused():
    ptm.forget_all()
    img = _big_face()
    a = an.analyze_region(img, CTX)
    img[:] = (40, 80, 200)  # 同一个数组被改写
    img[::7, ::7] = (230, 200, 40)
    b = an.analyze_region(img, CTX)
    assert b.stats.lab_mean != a.stats.lab_mean


def test_mask_counts_are_shared_and_correct():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 40, (900, 800, 3)).astype(np.uint8)
    pt = ptm.PixelTable(img)
    m = rng.random((900, 800)) < 0.3
    assert np.array_equal(pt.count(m), np.bincount(pt.index[m], minlength=pt.n))
    assert np.array_equal(pt.count(m), np.bincount(pt.index[m], minlength=pt.n))  # 第二次用记住的
    m[::3] = ~m[::3]  # 同一个蒙版被改写：要重新数
    assert np.array_equal(pt.count(m), np.bincount(pt.index[m], minlength=pt.n))
    assert np.array_equal(pt.count(np.ones((900, 800), bool)), pt.count())


def test_bridge_recognises_the_same_pixels():
    img = noisy(flat(BLUE, 600, 800), 2)
    p1 = bs._prepared(img.copy())
    p2 = bs._prepared(img.copy())  # 插件每次重新传来同一张图（新的字节）
    assert p1 is p2
    other = img.copy()
    other[5, 5] = (1, 2, 3)
    assert bs._prepared(other) is not p1


def test_bridge_has_no_quick_mode():
    assert bs.PROTOCOL >= 5
    img = noisy(flat(BLUE, 600, 800), 2)
    img[200:400, 300:500] = sprinkle(img[200:400, 300:500], YELLOW, 0.02, seed=4)
    res = bs.analyze_request(img, "dirty", {"progressive": "1", "precision": "fast"}, bs.Store(), bs.Store())
    assert "refine" not in res and "precision" not in res  # 旧插件发来的参数被忽略：一律按原图每个像素算


# ------------------------------------------------------------------ 取消：又分析了一次，旧的马上停下
def test_cancelled_analysis_stops():
    ev = threading.Event()
    ev.set()
    with ptm.cancellable(ev), pytest.raises(ptm.Cancelled):
        an.scan_image(noisy(flat(BLUE, 600, 800), 2), CTX)
    an.scan_image(noisy(flat(BLUE, 600, 800), 2), CTX)  # 不在取消范围内：照常算


@pytest.fixture
def win(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow(settings=settings)
    qtbot.addWidget(w)
    w.show()
    return w


def test_desktop_cancellable_run_stops_quietly(win, qtbot):
    seen = []

    def slow():
        for _ in range(500):
            ptm.check_cancel()
            time.sleep(0.01)
        return "done"

    t = time.perf_counter()
    ev = win._run_cancellable(slow, seen.append)
    qtbot.wait(100)
    ev.set()  # 又改了设置：上一次还没算完的重算停下
    qtbot.wait(400)
    assert seen == [] and time.perf_counter() - t < 3  # 停下了，没有用旧结果，也没有报错弹窗
    ev2 = win._run_cancellable(lambda: "ok", seen.append)
    qtbot.waitUntil(lambda: seen == ["ok"], timeout=5000)
    assert not ev2.is_set()


def test_settings_change_reruns_and_replaces(win, qtbot):
    win.set_mode("palette")
    img = noisy(flat(BLUE, 300, 400), 2)
    win._run_rect(img, (0, 0, 400, 300), CTX, "t", 0)
    qtbot.waitUntil(lambda: len(win.session.history) == 1, timeout=15000)
    item = win.session.history[0]
    for _ in range(3):  # 连续改几次：只算最后一次，结果替换原来那条
        win._rerun(item, "palette")
    qtbot.waitUntil(lambda: win.session.history and win.session.history[0] is not item, timeout=15000)
    qtbot.wait(800)
    assert len(win.session.history) == 1


def test_bench_command_runs(capsys):
    from colordoctor.cli import run

    assert run(["bench", "--mp", "0.6"]) == 0
    out = capsys.readouterr().out
    for name in ("脏色 / 区域", "整图扫描", "人像肤色", "整图色彩"):
        assert name in out
    assert run(["bench", "--mp", "abc"]) == 2
