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
    assert set(win.mode_buttons) == {"point", "dirty", "skin", "compare", "palette", "reference"}
    assert [win.radius_box.itemText(i) for i in range(win.radius_box.count())] == [
        "1×1", "3×3", "5×5", "11×11", "31×31", "51×51", "101×101"]
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


# ------------------------------------------------------------------ 新增功能
def test_palette_mode_flow(win, qtbot):
    win.set_mode("palette")
    win.style_box.setCurrentIndex(win.style_box.findData("teal_orange"))
    win.family_box.setCurrentIndex(win.family_box.findData("蓝"))
    win._run_rect(noisy(flat(BLUE), 2), (0, 0, 160, 120), CTX, "t", 0)
    qtbot.waitUntil(lambda: len(win.session.history) == 1, timeout=15000)
    text = win.report_view.toPlainText()
    assert "6:3:1" in text and "青橙" in text and "统一" in text


def test_reference_mode_two_steps(win, qtbot):
    win.set_mode("reference")
    win._run_rect(noisy(flat(BLUE), 2), (0, 0, 160, 120), CTX, "t", 0)
    win._run_rect(noisy(flat(YELLOW), 2), (0, 0, 160, 120), CTX, "t", 1)
    qtbot.waitUntil(lambda: any(isinstance(h.result, AnalysisReport) and h.result.mode == "reference"
                                for h in win.session.history), timeout=20000)
    assert "相似度" in win.report_view.toPlainText()


def test_skin_lasso_and_part_and_mask_editor(win, qtbot, monkeypatch):
    from synth import face_scene
    from colordoctor.ui import mask_editor

    img, truth = face_scene()
    win.set_mode("skin")
    win.part_box.setCurrentText("身体")
    rect, lasso = win._polygon_rect([(40, 60), (160, 60), (160, 190), (40, 190)], img.shape)
    region = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    win._run_rect(region, rect, CTX, "t", 0, lasso=lasso)
    qtbot.waitUntil(lambda: len(win.session.history) == 1, timeout=15000)
    rep = win.session.history[0].result
    assert rep.part == "身体" and rep.valid_mask is not None
    assert "身体" in win.session.skin_by_part

    # 蒙版编辑：直接修改蒙版后“确定”，应以手动蒙版重新分析
    class FakeDlg:
        def __init__(self, parent, region, mask, auto):
            self.mask = mask.copy()
            self.mask[:, : region.shape[1] // 2] = False

        def exec(self):
            return True

    monkeypatch.setattr(mask_editor, "MaskEditorDialog", FakeDlg)
    win.edit_skin_mask()
    qtbot.waitUntil(lambda: win.session.history[0].result.mask_info.get("manual"), timeout=15000)
    assert len([h for h in win.session.history if h.kind == "rect"]) == 1  # 替换而不是新增
    item = win.session.history[0]
    assert item.lasso is not None and item.lasso.shape == item.region.shape[:2]  # 重新分析后套索仍保留

    # “恢复自动”的蒙版不能超出套索
    seen = {}

    class SpyDlg:
        def __init__(self, parent, region, mask, auto):
            seen["auto"] = auto

        def exec(self):
            return False

    item.lasso = np.zeros(item.region.shape[:2], bool)
    item.lasso[:, : item.region.shape[1] // 2] = True
    monkeypatch.setattr(mask_editor, "MaskEditorDialog", SpyDlg)
    win.edit_skin_mask()
    assert seen["auto"].any() and not seen["auto"][~item.lasso].any()


def test_mask_canvas_brush_and_undo(qtbot):
    from synth import face_scene
    from colordoctor.ui.mask_editor import MaskEditorDialog

    img, truth = face_scene()
    dlg = MaskEditorDialog(None, img, truth.copy(), truth.copy())
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    c = dlg.canvas
    c.tool = "brush_erase"
    c.brush = 20
    before = c.mask.sum()
    center = c.to_view(100, 120).toPoint()
    qtbot.mousePress(c, Qt.MouseButton.LeftButton, pos=center)
    qtbot.mouseRelease(c, Qt.MouseButton.LeftButton, pos=center)
    assert c.mask.sum() < before
    dlg._undo()
    assert c.mask.sum() == before
    c.tool = "wand_add"
    qtbot.mousePress(c, Qt.MouseButton.LeftButton, pos=c.to_view(5, 5).toPoint())
    assert c.mask[5, 5]


def test_mask_editor_large_image_is_fast_and_keeps_precision(qtbot):
    """2400 万像素：在缩小副本上编辑，每一笔只重画一小块；确定时没改动的地方保持原蒙版。"""
    import time

    from colordoctor.ui.mask_editor import MaskEditorDialog

    H, W = 4000, 6000
    img = np.full((H, W, 3), 150, np.uint8)
    start = np.zeros((H, W), bool)
    start[1000:3001, 1001:4999] = True  # 边界不在缩放步长的整数倍上
    start[2000, 5500] = True  # 单个像素：缩小副本里看不到，确定时也不能丢
    dlg = MaskEditorDialog(None, img, start, None)
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    c = dlg.canvas
    assert c.step > 1 and c.mask.size <= 2_100_000
    assert np.array_equal(dlg.mask, start)  # 没动过：原样返回
    c.tool = "brush_erase"
    c.brush = 30
    k = c.step
    a, b = c.to_view(2500 / k, 2000 / k).toPoint(), c.to_view(3500 / k, 2000 / k).toPoint()
    qtbot.mousePress(c, Qt.MouseButton.LeftButton, pos=a)
    t = time.perf_counter()
    steps = 20
    for i in range(1, steps + 1):
        qtbot.mouseMove(c, a + (b - a) * (i / steps))
    per = (time.perf_counter() - t) / steps
    qtbot.mouseRelease(c, Qt.MouseButton.LeftButton, pos=b)
    assert per < 0.25, per
    out = dlg.mask
    assert out.shape == (H, W)
    assert not out[2000, 2600:3400].any()  # 擦掉了
    assert out[2000, 5500]  # 单个像素保留
    far = np.zeros((H, W), bool)
    far[:1500] = True  # 远离笔画的部分逐像素与原蒙版相同
    assert np.array_equal(out[far], start[far])
    dlg._undo()
    assert np.array_equal(dlg.mask, start)


def test_copy_export_and_ai_disabled(win, qtbot, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication

    win.set_mode("dirty")
    win._run_rect(noisy(flat(BLUE), 2), (0, 0, 160, 120), CTX, "t", 0)
    qtbot.waitUntil(lambda: len(win.session.history) == 1, timeout=10000)
    win.copy_for_agent()
    assert "```json" in QApplication.clipboard().text()
    out = tmp_path / "r.json"
    monkeypatch.setattr("colordoctor.ui.main_window.QFileDialog.getSaveFileName", lambda *a, **k: (str(out), ""))
    win.export_json()
    assert out.exists() and '"mode"' in out.read_text(encoding="utf-8")
    shown = {}
    monkeypatch.setattr("colordoctor.ui.main_window.QMessageBox.information", lambda *a: shown.setdefault("t", a[1]))
    win.ask_ai()
    assert "未启用" in shown["t"]


def test_ai_flow_with_fake_command(win, qtbot, tmp_path):
    import sys

    script = tmp_path / "fake.py"
    script.write_text("import sys; d=sys.stdin.read(); print('## 点评\\n收到 %d 字' % len(d))", encoding="utf-8")
    win.settings.setValue("ai/enabled", "true")
    win.settings.setValue("ai/provider", "claude_code")
    win.settings.setValue("ai/command", f'"{sys.executable}" "{script}"')
    img = flat((200, 150, 120), 20, 20)
    win._handle_point(img[3:8, 3:8], 5, 5, CTX, "t")
    win.ask_ai()
    qtbot.waitUntil(lambda: getattr(win, "_ai_dialog", None) is not None, timeout=20000)
    assert "收到" in win._ai_dialog.findChild(type(win.report_view)).toPlainText()


def test_overlay_radius_keys(qtbot):
    info = ScreenInfo("FAKE", 0, 0, 200, 150, 1.0)
    s = CaptureSession([ScreenCapture(info=info, pixels=flat((1, 2, 3), 150, 200))], ["x"], radius=2)
    got = []
    s.radius_changed.connect(got.append)
    s.show()
    v = s.overlays[0].view
    qtbot.keyClick(v, Qt.Key.Key_BracketRight)
    qtbot.keyClick(v, Qt.Key.Key_BracketRight)
    qtbot.keyClick(v, Qt.Key.Key_BracketLeft)
    assert got == [5, 15, 5]
    s.close()


def test_preset_change_in_other_mode_keeps_skin_result(win, qtbot):
    win.set_mode("skin")
    win._run_rect(skin_region(), (0, 0, 120, 120), CTX, "test", 0)
    qtbot.waitUntil(lambda: len(win.session.history) == 1, timeout=10000)
    win.set_mode("dirty")
    win.preset_box.setCurrentIndex(win.preset_box.findData("japanese"))
    qtbot.wait(1500)
    modes = [h.result.mode for h in win.session.history if isinstance(h.result, AnalysisReport)]
    assert modes == ["skin"], modes


def test_reference_image_can_be_reset_and_is_cleared_with_history(win, qtbot):
    win.set_mode("reference")
    win._run_rect(noisy(flat(BLUE), 2), (0, 0, 160, 120), CTX, "t", 0)
    qtbot.waitUntil(lambda: win.ref_image_item is not None, timeout=20000)
    assert win.ref_row.isVisibleTo(win)
    win.reset_reference_image()
    assert win.ref_image_item is None
    win._run_rect(noisy(flat(BLUE), 2), (0, 0, 160, 120), CTX, "t", 0)
    qtbot.waitUntil(lambda: win.ref_image_item is not None, timeout=20000)
    win._history_clear()
    assert win.ref_image_item is None


def test_overlay_closed_by_system_cancels_session(qapp, qtbot):
    from colordoctor.ui.capture_overlay import CaptureSession

    cap = ScreenCapture(ScreenInfo("s", 0, 0, 200, 100, 1.0), np.zeros((100, 200, 3), np.uint8), None)
    sess = CaptureSession([cap], ["选区"])
    got = []
    sess.cancelled.connect(lambda: got.append(1))
    sess.overlays[0].close()  # 模拟 Alt+F4
    assert got == [1] and sess.overlays == []
