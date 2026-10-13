"""界面冒烟与交互测试（offscreen）。真实屏幕、热键、显示器配置文件需在 Windows 上人工验证。"""

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QSettings, Qt

from colordoctor.capture.geometry import ScreenInfo
from colordoctor.capture.screen_capture import ScreenCapture
from colordoctor.core import analysis as an
from colordoctor.core import contamination as ct_mod
from colordoctor.core.color_management import synthetic_context
from colordoctor.core.color_models import AnalysisReport, PointSample
from colordoctor.ui import report_html as rh
from colordoctor.ui.capture_overlay import CaptureSession
from colordoctor.ui.color_wheel import ColorWheel
from colordoctor.ui.main_window import JpgSelectDialog, MainWindow
from colordoctor.ui.result_view import overlay_image
from synth import BLUE, YELLOW, flat, lch_color, noisy, skin_region, sprinkle

CTX = synthetic_context()


@pytest.fixture
def win(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow(settings=settings)
    qtbot.addWidget(w)
    w.show()
    return w


def test_window_starts_with_all_modes(win):
    assert set(win.mode_buttons) == {"point", "dirty", "skin", "compare"}
    assert win.preset_box.count() >= 5
    assert win.radius_box.currentData() == 2  # 默认 5×5


def test_point_selection_shows_values(win):
    img = flat((200, 150, 120), 20, 20)
    win.set_mode("point")
    win._handle_point(img[3:8, 3:8], 5, 5, CTX, "test")
    item = win.session.history[0]
    assert isinstance(item.result, PointSample)
    html = win.point_view.toHtml()
    assert "#C89678" in html and "Lab" in html


def test_dirty_region_analysis_runs_in_background(win, qtbot):
    win.set_mode("dirty")
    img = sprinkle(noisy(flat(BLUE), 2), YELLOW, 0.05)
    win._run_rect(img, (10, 20, 160, 120), CTX, "test", 0)
    qtbot.waitUntil(lambda: len(win.session.history) == 1, timeout=10000)
    rep = win.session.history[0].result
    assert isinstance(rep, AnalysisReport) and rep.mode == "general"
    assert "黄" in win.report_view.toPlainText()
    assert win.wheel.opposite_of is not None
    assert win.result_view.view.image is not None


def test_skin_analysis_and_preset_switch(win, qtbot):
    win.set_mode("skin")
    win.preset_box.setCurrentIndex(win.preset_box.findData("japanese"))
    win._run_rect(skin_region(), (0, 0, 120, 120), CTX, "test", 0)
    qtbot.waitUntil(lambda: len(win.session.history) == 1, timeout=10000)
    assert "日系" in win.report_view.toPlainText()
    assert win.wheel.preset is not None


def test_compare_flow_two_steps_back_to_back(win, qtbot):
    win.set_mode("compare")
    win._run_rect(skin_region(68, 14, 19), (0, 0, 120, 120), CTX, "t", 0)
    win._run_rect(skin_region(68, 12, 27), (200, 0, 120, 120), CTX, "t", 1)  # 参考区可能尚未算完
    qtbot.waitUntil(lambda: any(h.result.mode == "compare" for h in win.session.history if isinstance(h.result, AnalysisReport)), timeout=15000)
    rep = [h.result for h in win.session.history if isinstance(h.result, AnalysisReport) and h.result.mode == "compare"][0]
    assert rep.comparison.db > 6
    assert win.wheel.ref_lab is not None and win.wheel.prob_lab is not None


def test_details_toggle(win, qtbot):
    win.set_mode("dirty")
    win._run_rect(noisy(flat(BLUE), 2), (0, 0, 160, 120), CTX, "t", 0)
    qtbot.waitUntil(lambda: len(win.session.history) == 1, timeout=10000)
    assert "详细数据" not in win.report_view.toPlainText().replace("显示详细数据", "")
    win.details_check.setChecked(True)
    assert "主色聚类" in win.report_view.toPlainText()


def test_open_bad_file_shows_warning(win, tmp_path, monkeypatch):
    shown = {}
    monkeypatch.setattr("colordoctor.ui.main_window.QMessageBox.warning", lambda *a: shown.setdefault("msg", a[2]))
    bad = tmp_path / "x.jpg"
    bad.write_bytes(b"not a jpeg")
    win.open_jpg(str(bad))
    assert "msg" in shown


def test_live_tick_does_not_crash(win):
    win.set_live(True)
    win._live_tick()
    win.set_live(False)


def test_lock_without_live_shows_hint(win):
    win.lock_sample()
    assert "实时取色" in win.statusBar().currentMessage()


def _drag(qtbot, widget, a, b):
    qtbot.mousePress(widget, Qt.MouseButton.LeftButton, pos=QPoint(*a))
    qtbot.mouseMove(widget, QPoint(*b))
    qtbot.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=QPoint(*b))


def test_jpg_select_dialog_rect_and_point(qtbot):
    img = flat((10, 200, 30), 300, 400)
    dlg = JpgSelectDialog(None, img, ["第 1 步", "第 2 步"], 2)
    qtbot.addWidget(dlg)
    dlg.resize(400, 300)
    dlg.show()
    qtbot.waitExposed(dlg)
    _drag(qtbot, dlg.view, (50, 50), (150, 120))
    qtbot.mouseClick(dlg.view, Qt.MouseButton.LeftButton, pos=QPoint(200, 150))
    assert [r[0] for r in dlg.results] == ["rect", "point"]
    kind, (x, y, w, h), step = dlg.results[0]
    assert w > 50 and h > 30 and step == 0


def test_capture_session_with_fake_screens(qtbot):
    info = ScreenInfo("FAKE", 0, 0, 400, 300, 1.0)
    cap = ScreenCapture(info=info, pixels=flat((120, 60, 30), 300, 400))
    s = CaptureSession([cap], ["框选"], radius=2)
    got = []
    s.selected.connect(lambda c, k, coords, step: got.append((k, coords, step)))
    s.show()
    ov = s.overlays[0]
    qtbot.waitExposed(ov)
    _drag(qtbot, ov.view, (10, 10), (110, 60))
    assert got and got[0][0] == "rect"
    x, y, w, h = got[0][1]
    assert 90 <= w <= 110 and 40 <= h <= 60


def test_color_wheel_paints(qtbot):
    w = ColorWheel()
    qtbot.addWidget(w)
    w.resize(300, 300)
    w.set_scatter(skin_region())
    w.set_compare((68, 14, 19), (68, 12, 27))
    w.set_opposite(55)
    from colordoctor.core.skin_presets import get_preset

    w.set_preset(get_preset("japanese"))
    pm = w.grab()
    assert not pm.isNull()


def test_overlay_marks_contaminated_pixels():
    img = sprinkle(noisy(flat(BLUE, 60, 60), 2), YELLOW, 0.05, seed=2)
    rep = an.analyze_region(img, CTX)
    out = overlay_image(img, rep)
    mask = rep.contamination.mask == ct_mod.MASK_OPPOSITE
    assert out.shape == img.shape
    assert np.abs(out[mask].astype(int) - img[mask].astype(int)).mean() > 30
    assert (out[~mask].astype(int) <= img[~mask].astype(int)).all()  # 其余压暗


def test_overlay_alignment_for_downsampled_masks():
    img = noisy(flat(BLUE, 1200, 1600), 2)
    img[600:700, 800:900] = sprinkle(img[600:700, 800:900], YELLOW, 0.5, seed=1)
    rep = an.analyze_region(img, CTX)
    assert rep.contamination.mask_scale > 1
    out = overlay_image(img, rep, dim_normal=False)
    changed = np.abs(out.astype(int) - img.astype(int)).sum(axis=2) > 60
    ys, xs = np.nonzero(changed)
    assert 590 <= ys.min() and ys.max() <= 710 and 790 <= xs.min() and xs.max() <= 910


def test_report_html_contains_structure():
    rep = an.analyze_skin_region(sprinkle(skin_region(), lch_color(60, 18, 140), 0.05), CTX, preset_key="french")
    html = rh.report_html(rep, details=True)
    for word in ("证据", "可能原因", "调整方向", "置信度", "注意事项", "风险分项", "色彩管理"):
        assert word in html
