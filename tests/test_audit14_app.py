"""第 14 轮审计：桌面版、本机服务、命令行的回归测试。"""

import json
import os
import threading

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, QSettings, Qt
from PyQt6.QtGui import QWheelEvent

from colordoctor.bridge import server as bs
from colordoctor.core import skin_presets as sp
from colordoctor.core.color_management import synthetic_context
from colordoctor.ui.main_window import MainWindow, _wheel_points
from colordoctor.ui.session import region_mean_lab
from synth import BLUE, flat, noisy, skin_region

CTX = synthetic_context()


@pytest.fixture
def win(qtbot, tmp_path):
    w = MainWindow(settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    w.show()
    return w


# ------------------------------------------------------------------ 本机服务
@pytest.mark.skipif(os.name == "nt", reason="Windows 用独占端口，不受 TIME_WAIT 影响")
def test_bridge_rebinds_right_after_restart():
    """退出后马上重开：端口上还有 TIME_WAIT 连接，服务也要绑得上（以前要等半分钟，期间插件连不上）。"""
    import urllib.request

    srv = bs.make_server(0)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/health", headers={bs.CLIENT_HEADER: "t"})
    urllib.request.urlopen(req, timeout=5).read()
    srv.shutdown()
    srv.server_close()
    again = bs.make_server(port)
    again.server_close()


def test_second_live_server_on_same_port_still_fails():
    srv = bs.make_server(0)
    try:
        with pytest.raises(OSError):
            bs.make_server(srv.server_address[1])
    finally:
        srv.server_close()


def test_probe_reports_running_service_only():
    from colordoctor.bridge.probe import running_version

    srv = bs.make_server(0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        from colordoctor import __version__

        assert running_version(port) == __version__
    finally:
        srv.shutdown()
        srv.server_close()
    assert running_version(port, timeout=0.3) is None


def test_probe_only_talks_to_loopback():
    from pathlib import Path

    text = (Path(bs.__file__).parent / "probe.py").read_text(encoding="utf-8")
    assert "HTTPConnection(HOST" in text and "0.0.0.0" not in text and "urlopen" not in text


def test_patch_lab_must_be_finite_and_in_range():
    px = np.full((2, 2, 3), 128, np.uint8)
    for bad in ("50,1e400,0", "50,nan,0", "50,0,200"):
        with pytest.raises(ValueError):
            bs.analyze_request(px, "patch", {"lab": bad}, bs.Store(), bs.Store())


def test_panorama_mask_sizes_are_allowed():
    bs._check_size(40000, 2000, "蒙版")  # 8000 万像素的长条全景：插件按原图读，蒙版也要能做
    with pytest.raises(ValueError):
        bs._check_size(20000, 20000, "蒙版")  # 面积上限照旧


def test_transparent_layer_mask_is_stored_compactly_and_unchanged():
    img = np.zeros((300, 400, 4), np.uint8)
    img[..., :3] = noisy(flat(BLUE, 300, 400), 3)
    img[50:250, 100:300, 3] = 255  # 中间不透明
    img[100:120, 150:170, 3] = 0  # 里面有个透明洞
    res = bs.analyze_request(img, "dirty", {}, sessions := bs.Store(), bs.Store())
    sess = sessions.get(res["session"])
    bits, vh, vw = sess.opaque
    assert bits.dtype == np.uint8 and bits.size <= (vh * vw + 7) // 8
    full = sess.opaque_mask()
    assert full.shape == (300, 400)
    assert np.array_equal(full, img[..., 3] >= 8)


def test_reference_from_ps_blocked_message_names_the_reference(qtbot):
    """PS 里的参考文档是纯色时，提示说的是参考图，不是“当前图层”。"""
    import urllib.request

    srv = bs.make_server(0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        px = np.full((40, 40, 3), 90, np.uint8)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/reference", data=px.tobytes(), method="POST",
                                     headers={bs.CLIENT_HEADER: "t", "X-Width": "40", "X-Height": "40",
                                              "X-Components": "3"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        body = json.loads(e.value.read().decode("utf-8"))
        assert e.value.code == 422 and "参考图" in body.get("blocked", body.get("error", "")) \
            and "当前图层" not in json.dumps(body, ensure_ascii=False)
    finally:
        srv.shutdown()
        srv.server_close()


# ------------------------------------------------------------------ 设置 / 预设文件写坏
def test_corrupt_preset_file_does_not_stop_the_app(tmp_path):
    f = tmp_path / "skin_presets.json"
    f.write_text("[]", encoding="utf-8")  # 顶层写成了列表
    assert {p.key for p in sp.load_presets(f)} == {p.key for p in sp.builtin_presets()}
    assert sp.global_quick_rule(f)
    good = {"key": "mine", "name": "我的", "L": [60, 75], "a": [10, 18], "b": [12, 22], "hue": [40, 60], "chroma": [15, 30]}
    bad = dict(good, key="bad", L=[1])  # 范围只写了一个数
    f.write_text(json.dumps({"presets": [good, bad]}), encoding="utf-8")
    keys = {p.key for p in sp.load_presets(f)}
    assert "mine" in keys and "bad" not in keys


def test_list_valued_bool_setting_falls_back_to_default(win):
    win.settings.setValue("seg/ai", ["true", "true"])  # ini 里写成 ai=a, b 读出来是列表
    assert win.setting_bool("seg/ai", True) is True
    win.settings.setValue("raw/use_preview", "false")
    assert win.setting_bool("raw/use_preview", True) is False


# ------------------------------------------------------------------ 重算 / 任务
def _skin_item(win, qtbot):
    win.set_mode("skin")
    win._run_rect(skin_region(), (0, 0, 120, 120), CTX, "test", 0)
    qtbot.waitUntil(lambda: len(win.session.history) == 1, timeout=10000)
    return win.session.history[0]


def test_rerun_result_does_not_take_over_after_user_moved_on(win, qtbot):
    skin = _skin_item(win, qtbot)
    win.set_mode("dirty")
    win._run_rect(noisy(flat(BLUE, 80, 80), 2), (0, 0, 80, 80), CTX, "other", 0)
    qtbot.waitUntil(lambda: len(win.session.history) == 2, timeout=10000)
    other = win._current
    assert other is not skin
    win._rerun(skin, "skin")  # 换了肤色风格，后台重算
    qtbot.waitUntil(lambda: skin.replaced_by is not None, timeout=10000)
    assert win._current is other  # 用户在看别的记录：不抢画面
    assert win.session.get(skin.id) is None and win.session.get(skin.replaced_by.id) is not None


def test_replacement_of_deleted_item_is_dropped(win, qtbot):
    skin = _skin_item(win, qtbot)
    rep = skin.result
    win.session.remove(skin.id)
    assert win._rect_done(skin.region, skin.rect, rep, "x", None, skin, None) is None
    assert win.session.history == []


def test_two_quick_replacements_leave_one_row(win, qtbot):
    skin = _skin_item(win, qtbot)
    rep = skin.result
    a = win._rect_done(skin.region, skin.rect, rep, "x", None, skin, None)
    b = win._rect_done(skin.region, skin.rect, rep, "x", None, skin, None)  # 第二次替换的还是原条目
    assert a is not None and b is not None
    assert [h.id for h in win.session.history] == [b.id]


def test_finished_tasks_release_their_pixels(win, qtbot):
    _skin_item(win, qtbot)
    qtbot.wait(50)
    assert win._tasks == []


def test_stale_rerun_failure_shows_no_dialog(win, qtbot, monkeypatch):
    shown = []
    monkeypatch.setattr("colordoctor.ui.main_window.QMessageBox.warning", lambda *a: shown.append(a[2]))
    ev = win._run_cancellable(lambda: (_ for _ in ()).throw(RuntimeError("boom")), lambda r: None)
    ev.set()
    qtbot.wait(300)
    assert shown == []


def test_ai_failure_keeps_pending_problem(win, monkeypatch):
    monkeypatch.setattr("colordoctor.ui.main_window.QMessageBox.warning", lambda *a: None)
    win._pending_problem = ("x",)
    win._ref_busy = True
    win._failed_quiet("网络错误")
    assert win._pending_problem == ("x",) and win._ref_busy is True


# ------------------------------------------------------------------ 大图在界面线程上的计算
def test_wheel_points_sample_inside_mask_without_full_copy():
    rng = np.random.default_rng(0)
    region = rng.integers(0, 256, (300, 400, 3)).astype(np.uint8)
    mask = np.zeros((300, 400), bool)
    mask[100:200, 50:350] = True
    pts = _wheel_points(region, mask)
    from colordoctor.ui.color_wheel import MAX_POINTS

    assert 0 < len(pts) <= MAX_POINTS
    inside = {tuple(p) for p in region[mask]}
    assert all(tuple(p) in inside for p in pts[:200])


def test_region_mean_lab_equals_per_pixel_mean():
    from colordoctor.core import color_spaces as cs

    region = skin_region(h=700, w=700)  # 49 万像素：走“按不同颜色”算的路径
    direct = cs.srgb8_to_lab(region.reshape(-1, 3)).astype(np.float64).mean(axis=0)
    assert np.allclose(region_mean_lab(region), direct, atol=1e-3)


# ------------------------------------------------------------------ 触控板
def _wheel(widget, angle=(0, 0), pixel=(0, 0)):
    pos = QPointF(widget.width() / 2, widget.height() / 2)
    return QWheelEvent(pos, widget.mapToGlobal(pos), QPoint(*pixel), QPoint(*angle),
                       Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)


def test_trackpad_scroll_pans_and_wheel_zoom_is_proportional(qtbot):
    from colordoctor.ui.selection_widget import ImageSelectWidget

    w = ImageSelectWidget(allow_zoom=True)
    qtbot.addWidget(w)
    w.resize(400, 300)
    w.set_image(flat(BLUE, 200, 300))
    z0 = w.zoom
    w.wheelEvent(_wheel(w, angle=(10, 0)))  # 横向滑动：不缩放
    assert w.zoom == z0
    w.wheelEvent(_wheel(w, angle=(0, 40)))  # 三分之一格：只放大一点
    assert w.zoom == pytest.approx(z0 * 1.15 ** (40 / 120))
    z1, pan = w.zoom, QPointF(w.pan)
    w.wheelEvent(_wheel(w, angle=(0, 30), pixel=(0, 12)))  # 触控板两指滑动：平移
    assert w.zoom == z1 and w.pan != pan


# ------------------------------------------------------------------ 命令行
def test_bench_rejects_bad_sizes(capsys):
    from colordoctor.cli import run

    for bad in ("0", "-5", "nan", "inf", "1e-9", "abc"):
        assert run(["bench", "--mp", bad]) == 2
    assert run(["bench", "/nonexistent/x.jpg"]) == 2
    with pytest.raises(SystemExit):
        run(["serve", "--port", "70000"])


def test_cli_dispatch_ignores_qt_and_xcode_arguments():
    from colordoctor import app

    assert "-platform" not in app.CLI_FLAGS and "-NSDocumentRevisionsDebugMode" not in app.CLI_FLAGS
    assert set(app.CLI_FLAGS) == {"-h", "--help", "--version"}
