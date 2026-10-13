"""操作逻辑回归测试：多步框选、结果页一致、历史删除 / 清空、重算不重复、蒙版编辑器按键（offscreen）。"""

import pytest
from PyQt6.QtCore import QSettings, Qt

from colordoctor.core.color_management import synthetic_context
from colordoctor.core.color_models import AnalysisReport
from colordoctor.ui.main_window import JpgSelectDialog, MainWindow
from synth import BLUE, face_scene, flat, noisy

CTX = synthetic_context()


@pytest.fixture
def win(qtbot, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow(settings=settings)
    qtbot.addWidget(w)
    w.show()
    return w


def _wait_items(qtbot, win, n):
    qtbot.waitUntil(lambda: len(win.session.history) == n, timeout=15000)


def test_single_click_does_not_consume_a_step_in_two_step_modes(qtbot):
    img = noisy(flat(BLUE), 2)
    dlg = JpgSelectDialog(None, img, ["第 1 步", "第 2 步"], 2, labels=["参考区", "问题区"], allow_point=False)
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    v = dlg.view
    p = v.to_view(40, 40).toPoint()
    qtbot.mouseClick(v, Qt.MouseButton.LeftButton, pos=p)
    assert dlg.step == 0 and not dlg.results and "单击无效" in v.notice
    qtbot.mousePress(v, Qt.MouseButton.LeftButton, pos=p)
    qtbot.mouseMove(v, v.to_view(90, 80).toPoint())
    qtbot.mouseRelease(v, Qt.MouseButton.LeftButton, pos=v.to_view(90, 80).toPoint())
    assert dlg.step == 1 and dlg.results[0][0] == "rect"


def test_tiny_selection_is_rejected_with_a_message(win):
    win.set_mode("dirty")
    win._run_rect(noisy(flat(BLUE), 2)[:3, :3], (0, 0, 3, 3), CTX, "t", 0)
    assert not win.session.history
    assert "选区太小" in win.statusBar().currentMessage()


def test_point_then_region_keeps_result_pages_consistent(win, qtbot):
    win.set_mode("dirty")
    win._run_rect(noisy(flat(BLUE), 2), (0, 0, 160, 120), CTX, "t", 0)
    _wait_items(qtbot, win, 1)
    win.set_mode("point")
    win._run_rect(noisy(flat(BLUE), 2)[:40, :60], (0, 0, 60, 40), CTX, "t", 0)
    assert len(win.session.history) == 2
    assert "单点取色" in win.report_view.toPlainText()  # 结论页不再显示上一次的区域报告
    assert "拖框区域平均" in win.point_view.toPlainText()
    win.tabs.setCurrentIndex(0)
    win.details_check.setChecked(True)
    assert win.tabs.currentIndex() == 0  # 勾选详细数据不跳页


def test_remove_current_and_clear_history_reset_current(win, qtbot):
    win.set_mode("dirty")
    win._run_rect(noisy(flat(BLUE), 2), (0, 0, 160, 120), CTX, "t", 0)
    _wait_items(qtbot, win, 1)
    cur = win._current
    win._history_remove(cur.id)
    assert win._current is None and not win.report_view.toPlainText()
    win.export_json()
    assert "请先完成" in win.statusBar().currentMessage()
    img, _ = face_scene()
    win.set_mode("skin")
    win._run_rect(img[40:200, 40:160], (40, 40, 120, 160), CTX, "t", 0)
    _wait_items(qtbot, win, 1)
    assert win.session.skin_by_part
    win._history_clear()
    assert win._current is None and not win.session.skin_by_part


def test_preset_change_rerun_is_coalesced_and_keeps_part_and_reference(win, qtbot):
    img, _ = face_scene()
    win.set_mode("skin")
    win.part_box.setCurrentText("脸部")
    win._run_rect(img[40:200, 40:160], (40, 40, 120, 160), CTX, "t", 0)
    _wait_items(qtbot, win, 1)
    win.part_box.setCurrentText("身体")
    for i in range(1, 4):  # 连续切换 3 次
        win.preset_box.setCurrentIndex(i)
    qtbot.wait(300)
    qtbot.waitUntil(lambda: all(isinstance(h.result, AnalysisReport) for h in win.session.history), timeout=15000)
    qtbot.wait(1500)
    assert len(win.session.history) == 1  # 只替换，不会多出重复记录
    assert win.session.history[0].result.part == "脸部"  # 用原结果的部位重算
    # 参考区不会因改肤色目标而被替换掉
    win.set_mode("compare")
    win.session.set_reference(win.session.history[0])
    win._show_item(win.session.history[0])
    win.preset_box.setCurrentIndex(0)
    qtbot.wait(800)
    assert win.session.reference is win.session.history[0]


def test_live_off_clears_stale_reading(win):
    win._live = ("stale",)
    win.set_live(True)
    win.set_live(False)
    win.lock_sample()
    assert not win.session.history and "实时取色" in win.statusBar().currentMessage()


def test_reference_mode_menu_sets_reference_image(win, qtbot):
    win.set_mode("palette")
    win._run_rect(noisy(flat(BLUE), 2), (0, 0, 160, 120), CTX, "t", 0)
    _wait_items(qtbot, win, 1)
    win.set_mode("reference")
    assert win.history_panel.ref_action_text == "设为参考作品"
    win._history_set_ref(win.session.history[0].id)
    assert win.ref_image_item is win.session.history[0]
    assert "当前参考作品" in win.ref_label.text() and win.ref_reset_btn.isEnabled()


def test_mask_editor_brackets_change_brush_and_small_mask_keeps_dialog_open(qtbot, monkeypatch):
    from colordoctor.ui import mask_editor
    from colordoctor.ui.mask_editor import MaskEditorDialog

    img, truth = face_scene()
    dlg = MaskEditorDialog(None, img, truth.copy(), truth.copy())
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    c = dlg.canvas
    b0 = dlg.brush.value()
    qtbot.keyClick(c, Qt.Key.Key_BracketRight)
    assert dlg.brush.value() > b0 and c.brush == dlg.brush.value()
    warned = {}
    monkeypatch.setattr(mask_editor.QMessageBox, "warning", lambda *a: warned.setdefault("w", a[1]))
    c.history.mask[:] = False
    dlg.accept()
    assert warned and dlg.isVisible()  # 不关闭，修改还在
    assert dlg.result() == 0
