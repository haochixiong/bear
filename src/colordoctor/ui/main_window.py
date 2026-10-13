"""主窗口：悬浮在 Photoshop 旁边的小工具窗口。"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
from PyQt6.QtCore import QSettings, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QCursor, QDesktopServices, QKeySequence
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from colordoctor import __version__
from colordoctor.capture import hotkeys as hk
from colordoctor.capture import screen_capture as sc
from colordoctor.core import portrait_analysis as pa
from colordoctor.core import recommendations as rec
from colordoctor.core import skin_presets as sp
from colordoctor.core.color_management import ImageLoadError, load_jpeg, screen_pixels_to_srgb
from colordoctor.core.color_models import AnalysisReport, PointSample, Severity
from colordoctor.core.sampling import SampleError, color_values
from colordoctor.ui.capture_overlay import CaptureSession
from colordoctor.ui.color_wheel import ColorWheel
from colordoctor.ui.history_panel import HistoryPanel
from colordoctor.ui.report_html import point_html, report_html
from colordoctor.ui.result_view import ResultView
from colordoctor.ui.selection_widget import ImageSelectWidget
from colordoctor.ui.session import MODES, RADIUS_CHOICES, HistoryItem, Session, crop_selection, point_patch_from
from colordoctor.ui.worker import run_in_background

STEPS = {
    "point": ["单击取点（拖动则取框内平均）"],
    "dirty": ["拖动框选要检查的区域（衣服、背景、整张照片都可以）"],
    "skin": ["拖动框选皮肤（脸颊、额头等，避开眉眼、头发和阴影边缘）"],
    "compare": ["第 1 步：框选参考区（你认为干净、正常的皮肤或颜色）", "第 2 步：框选问题区（怀疑发灰、发黄、发绿的地方）"],
}


class JpgSelectDialog(QDialog):
    """在 JPG 上选择点/区域（不是图片浏览器，只用来取样）。"""

    def __init__(self, parent, pixels: np.ndarray, steps: list[str], radius: int):
        super().__init__(parent)
        self.setWindowTitle("在 JPG 上选择（滚轮缩放，右键拖动平移，Esc 取消）")
        self.resize(1100, 760)
        self.steps = steps
        self.step = 0
        self.results: list[tuple[str, tuple, int]] = []
        self.view = ImageSelectWidget(self, magnifier=True, allow_zoom=True)
        self.view.radius = radius
        self.view.set_image(pixels)
        self.view.point_selected.connect(lambda x, y: self._got("point", (x, y)))
        self.view.rect_selected.connect(lambda x, y, w, h: self._got("rect", (x, y, w, h)))
        self.view.cancelled.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        self._text()

    def _text(self):
        self.view.instruction = self.steps[self.step]
        self.view.update()

    def _got(self, kind, coords):
        self.results.append((kind, coords, self.step))
        rect = coords if kind == "rect" else (coords[0], coords[1], 1, 1)
        self.view.add_mark(rect, ["参考区", "问题区"][self.step] if len(self.steps) > 1 else "选区")
        self.step += 1
        if self.step >= len(self.steps):
            self.accept()
        else:
            self._text()


class SettingsDialog(QDialog):
    def __init__(self, parent, hotkeys: dict[str, str]):
        super().__init__(parent)
        self.setWindowTitle("设置")
        form = QFormLayout(self)
        self.lock = QLineEdit(hotkeys["lock_sample"])
        self.cap = QLineEdit(hotkeys["capture"])
        form.addRow("锁定实时取色", self.lock)
        form.addRow("冻结屏幕并框选", self.cap)
        note = QLabel(
            "全局快捷键会优先于 Photoshop 自己的快捷键。默认的 Ctrl+Alt+Shift+Z / X\n"
            "据 Photoshop 默认快捷键表未被占用；如果与你的自定义快捷键冲突请修改。"
        )
        note.setWordWrap(True)
        form.addRow(note)
        btn = QPushButton("打开风格预设文件夹（可放置 skin_presets.json 覆盖/新增风格）")
        btn.clicked.connect(self._open_presets)
        form.addRow(btn)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _open_presets(self):
        d = sp.user_config_dir()
        d.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def values(self) -> dict[str, str]:
        return {"lock_sample": self.lock.text().strip(), "capture": self.cap.text().strip()}


class MainWindow(QMainWindow):
    def __init__(self, session: Session | None = None, settings: QSettings | None = None):
        super().__init__()
        self.session = session or Session()
        self.settings = settings or QSettings("ColorDoctor", "ColorDoctor")
        self.setWindowTitle(f"人像色彩诊断取色器 {__version__}")
        self.resize(480, 940)
        self.setMinimumWidth(440)
        self.setAcceptDrops(True)
        self.jpg = None
        self.capture_session: CaptureSession | None = None
        self._live = None
        self._was_live = False
        self._pending_problem = None
        self._ref_busy = False
        self._current: HistoryItem | None = None
        self._tasks = []
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(70)
        self.live_timer.timeout.connect(self._live_tick)

        self._build()
        self._load_settings()
        self.hotkeys = hk.make_manager(self)
        self.hotkeys.triggered.connect(self._on_hotkey)
        self._register_hotkeys()

    # ================================================================ 界面
    def _build(self):
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # 模式
        row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for key, name in MODES.items():
            b = QToolButton()
            b.setText(name)
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            self.mode_group.addButton(b)
            self.mode_buttons[key] = b
            b.clicked.connect(lambda _c, k=key: self.set_mode(k))
            row.addWidget(b)
        lay.addLayout(row)

        # 取样方式
        row = QHBoxLayout()
        self.live_btn = QToolButton()
        self.live_btn.setText("◎ 实时取色")
        self.live_btn.setCheckable(True)
        self.live_btn.toggled.connect(self.set_live)
        self.capture_btn = QPushButton("▭ 框选屏幕")
        self.capture_btn.clicked.connect(self.start_capture)
        self.jpg_btn = QPushButton("打开 JPG")
        self.jpg_btn.clicked.connect(lambda: self.open_jpg())
        self.reselect_btn = QPushButton("JPG 重选")
        self.reselect_btn.setEnabled(False)
        self.reselect_btn.clicked.connect(self.select_on_jpg)
        for w in (self.live_btn, self.capture_btn, self.jpg_btn, self.reselect_btn):
            row.addWidget(w)
        lay.addLayout(row)

        # 参数
        row = QHBoxLayout()
        self.radius_box = QComboBox()
        for r, label in RADIUS_CHOICES:
            self.radius_box.addItem(label, r)
        self.radius_box.setToolTip("取样大小：与 Photoshop 吸管“取样大小 N×N 平均”一致")
        self.radius_box.currentIndexChanged.connect(self._radius_changed)
        self.preset_box = QComboBox()
        self.preset_box.setToolTip("目标肤色风格（经验区间，可在设置中修改）")
        self.preset_box.currentIndexChanged.connect(self._preset_changed)
        self.source_box = QComboBox()
        self.source_box.addItem("画面来自 PS/ACR（色彩管理）", True)
        self.source_box.addItem("画面来自未做色彩管理的软件", False)
        self.source_box.setToolTip(
            "Photoshop/ACR 会按显示器配置文件转换后再显示，本工具用同一配置文件还原为 sRGB 数值；\n"
            "未做色彩管理的软件显示的就是文件数值，原样使用。"
        )
        row.addWidget(QLabel("采样"))
        row.addWidget(self.radius_box)
        row.addWidget(QLabel("风格"))
        row.addWidget(self.preset_box, 1)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.source_box, 1)
        self.scan_check = QCheckBox("整图扫描")
        self.scan_check.setToolTip("框选整张照片时勾选：在全图中寻找局部的脏色和发灰位置")
        self.scan_check.toggled.connect(lambda v: setattr(self.session, "scan_whole", v))
        self.skin_compare_check = QCheckBox("按肤色对比")
        self.skin_compare_check.setChecked(True)
        self.top_check = QCheckBox("置顶")
        self.top_check.toggled.connect(self.set_on_top)
        self.settings_btn = QToolButton()
        self.settings_btn.setText("⚙")
        self.settings_btn.clicked.connect(self.open_settings)
        for w in (self.scan_check, self.skin_compare_check, self.top_check, self.settings_btn):
            row.addWidget(w)
        lay.addLayout(row)

        # 实时读数
        self.live_label = QLabel("实时取色：关闭")
        self.live_label.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.live_label)

        split = QSplitter(Qt.Orientation.Vertical)
        self.wheel = ColorWheel()
        split.addWidget(self.wheel)
        self.tabs = QTabWidget()
        # 结论
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)
        self.details_check = QCheckBox("显示详细数据")
        self.details_check.toggled.connect(lambda _v: self._show_item(self._current))
        v.addWidget(self.details_check)
        self.report_view = QTextBrowser()
        self.report_view.setOpenExternalLinks(False)
        v.addWidget(self.report_view, 1)
        self.tabs.addTab(w, "结论")
        self.point_view = QTextBrowser()
        self.tabs.addTab(self.point_view, "取色数值")
        self.result_view = ResultView()
        self.tabs.addTab(self.result_view, "问题位置")
        self.history_panel = HistoryPanel()
        self.history_panel.item_activated.connect(self._history_activated)
        self.history_panel.set_reference.connect(self._history_set_ref)
        self.history_panel.set_problem.connect(self._history_compare)
        self.history_panel.use_as_target.connect(self._history_target)
        self.history_panel.remove_item.connect(self._history_remove)
        self.history_panel.clear_all.connect(self._history_clear)
        self.tabs.addTab(self.history_panel, "历史")
        split.addWidget(self.tabs)
        split.setSizes([330, 520])
        lay.addWidget(split, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪：所有计算在本地完成，不上传、不修改任何图片")

        act = QAction(self)
        act.setShortcut(QKeySequence("Ctrl+O"))
        act.triggered.connect(lambda: self.open_jpg())
        self.addAction(act)
        self._fill_presets()

    def _fill_presets(self):
        self.preset_box.blockSignals(True)
        self.preset_box.clear()
        for p in self.session.presets:
            self.preset_box.addItem(p.name, p.key)
        if self.session.custom_preset is not None:
            self.preset_box.addItem(self.session.custom_preset.name, "custom")
        idx = self.preset_box.findData(self.session.preset_key)
        self.preset_box.setCurrentIndex(max(0, idx))
        self.preset_box.blockSignals(False)

    # ================================================================ 设置
    def _hotkey_texts(self) -> dict[str, str]:
        return {k: str(self.settings.value(f"hotkeys/{k}", v)) for k, v in hk.DEFAULT_HOTKEYS.items()}

    def _register_hotkeys(self):
        msgs = []
        for name, text in self._hotkey_texts().items():
            ok, msg = self.hotkeys.register(name, text)
            if not ok:
                msgs.append(f"{text}：{msg}")
        if msgs and hk.is_supported():
            self.statusBar().showMessage("快捷键未生效：" + "；".join(msgs), 8000)

    def _load_settings(self):
        mode = str(self.settings.value("mode", "point"))
        self.set_mode(mode if mode in MODES else "point")
        r = int(self.settings.value("radius", 2))
        idx = self.radius_box.findData(r)
        self.radius_box.setCurrentIndex(idx if idx >= 0 else 2)
        key = str(self.settings.value("preset", sp.DEFAULT_KEY))
        idx = self.preset_box.findData(key)
        if idx >= 0:
            self.preset_box.setCurrentIndex(idx)
        managed = str(self.settings.value("managed", "true")).lower() == "true"
        self.source_box.setCurrentIndex(0 if managed else 1)

    def closeEvent(self, ev):
        self.settings.setValue("mode", self.session.mode)
        self.settings.setValue("radius", self.session.radius)
        self.settings.setValue("preset", self.session.preset_key if self.session.preset_key != "custom" else sp.DEFAULT_KEY)
        self.settings.setValue("managed", "true" if self._managed() else "false")
        self.live_timer.stop()
        self.hotkeys.unregister_all()
        super().closeEvent(ev)

    def open_settings(self):
        dlg = SettingsDialog(self, self._hotkey_texts())
        if dlg.exec():
            for k, v in dlg.values().items():
                try:
                    hk.parse_hotkey(v)
                except hk.HotkeyParseError as e:
                    QMessageBox.warning(self, "快捷键无效", f"{v}：{e}")
                    return
                self.settings.setValue(f"hotkeys/{k}", v)
            self.hotkeys.unregister_all()
            self._register_hotkeys()
        self.session.presets = sp.load_presets()
        self._fill_presets()

    # ================================================================ 状态
    def _managed(self) -> bool:
        return bool(self.source_box.currentData())

    def set_mode(self, key: str):
        self.session.mode = key
        self.mode_buttons[key].setChecked(True)
        self.scan_check.setVisible(key == "dirty")
        self.skin_compare_check.setVisible(key == "compare")
        self.wheel.set_preset(self.session.preset if key in ("skin", "point") else None)
        hints = {
            "point": "单点取色：实时取色时按快捷键锁定读数，或“框选屏幕”后单击取点",
            "dirty": "脏色检查：框选一块颜色（衣服、背景）或整张照片，找出混入的对立色和发灰位置",
            "skin": "人像肤色：框选皮肤，查看偏色、杂色、发灰，并与目标风格比较",
            "compare": "参考对比：先框参考区（干净皮肤），再框问题区",
        }
        self.statusBar().showMessage(hints[key], 6000)

    def _radius_changed(self):
        self.session.radius = int(self.radius_box.currentData())

    def _preset_changed(self):
        self.session.preset_key = str(self.preset_box.currentData())
        self.wheel.set_preset(self.session.preset if self.session.mode in ("skin", "point") else None)
        cur = self._current
        if cur is not None and isinstance(cur.result, AnalysisReport) and cur.result.mode == "skin":
            self._run_rect(cur.region, cur.rect, cur.context, cur.source_label, step=0, replace=cur)
        elif cur is not None and isinstance(cur.result, PointSample):
            self._show_item(cur)

    def set_on_top(self, on: bool):
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on)
        self.show()

    # ================================================================ 实时取色
    def set_live(self, on: bool):
        if on:
            self.live_timer.start()
            self.live_label.setText("实时取色：移动鼠标到 Photoshop 画面上；按 %s 锁定读数" % self._hotkey_texts()["lock_sample"])
        else:
            self.live_timer.stop()
            self.live_label.setText("实时取色：关闭")
        if self.live_btn.isChecked() != on:
            self.live_btn.setChecked(on)

    def _live_tick(self):
        if self.isVisible() and self.frameGeometry().contains(QCursor.pos()):
            return  # 光标在本工具窗口上，不取自己的颜色
        try:
            res = sc.grab_around_cursor(self.session.radius)
        except Exception:
            res = None
        if res is None:
            return
        patch, info, (gx, gy) = res
        srgb, ctx = screen_pixels_to_srgb(patch, sc.icc_for(info.name), self._managed(), device=info.name, scale=info.dpr)
        vals = color_values(srgb.reshape(-1, 3).astype(float).mean(axis=0))
        self._live = (srgb, ctx, info, gx, gy, vals)
        r, g, b = vals.rgb
        self.live_label.setText(
            f"<span style='background:rgb({r},{g},{b})'>&nbsp;&nbsp;&nbsp;&nbsp;</span> "
            f"{vals.hex}  RGB {r},{g},{b}  |  Lab {vals.lab[0]:.0f}, {vals.lab[1]:.0f}, {vals.lab[2]:.0f}"
            f"  |  HSB {vals.hsb[0]:.0f}°, {vals.hsb[1]:.0f}%, {vals.hsb[2]:.0f}%"
        )
        self.wheel.set_point(vals.lab, vals.rgb)

    def lock_sample(self):
        if self._live is None:
            self.statusBar().showMessage("请先打开“实时取色”，把鼠标移到要取的位置", 4000)
            return
        srgb, ctx, info, gx, gy, vals = self._live
        r = self.session.radius
        px, py = sc.logical_to_device(info, gx, gy)
        sample = PointSample(x=px, y=py, radius=r, pixel_count=int(srgb.shape[0] * srgb.shape[1]), values=vals, context=ctx)
        item = self.session.add_point(srgb, sample, f"屏幕 {info.name}")
        self._after_add(item)

    # ================================================================ 屏幕框选
    def start_capture(self):
        if self.capture_session is not None:
            return
        self._was_live = self.live_timer.isActive()
        self.live_timer.stop()
        self.hide()
        QTimer.singleShot(250, self._do_capture)

    def _do_capture(self):
        try:
            caps = sc.freeze_all_screens()
        except Exception as e:
            self.show()
            QMessageBox.warning(self, "截屏失败", str(e))
            return
        steps = STEPS[self.session.mode]
        if self.session.mode == "compare" and self.session.reference is not None:
            steps = STEPS["compare"][1:] if self._ask_reuse_reference() else steps
        self._capture_steps_offset = 2 - len(steps) if self.session.mode == "compare" else 0
        cs_ = CaptureSession(caps, steps, radius=self.session.radius)
        cs_.selected.connect(self._on_capture_selected)
        cs_.finished.connect(self._capture_done)
        cs_.cancelled.connect(self._capture_done)
        self.capture_session = cs_
        cs_.show()

    def _ask_reuse_reference(self) -> bool:
        return True  # 已有参考区时直接框选问题区；右键历史可更换参考区

    def _capture_done(self):
        self.capture_session = None
        self.show()
        self.raise_()
        if self._was_live:
            self.live_timer.start()

    def _on_capture_selected(self, cap, kind, coords, step):
        step += getattr(self, "_capture_steps_offset", 0)
        managed = self._managed()
        label = f"屏幕 {cap.info.name}"
        if kind == "point":
            x, y = coords
            r = self.session.radius
            raw = point_patch_from(cap.pixels, x, y, r)
            srgb, ctx = screen_pixels_to_srgb(raw, cap.icc, managed, device=cap.info.name, scale=cap.info.dpr)
            self._handle_point(srgb, x, y, ctx, label)
            return
        x, y, w, h = coords
        raw = cap.pixels[y : y + h, x : x + w]
        srgb, ctx = screen_pixels_to_srgb(raw, cap.icc, managed, device=cap.info.name, scale=cap.info.dpr)
        self._run_rect(srgb, (x, y, w, h), ctx, label, step)

    # ================================================================ JPG
    def open_jpg(self, path: str | None = None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "打开 JPG（只读取，不会修改原图）", "", "JPG 图片 (*.jpg *.jpeg *.JPG *.JPEG)")
            if not path:
                return
        try:
            self.jpg = load_jpeg(path)
        except ImageLoadError as e:
            QMessageBox.warning(self, "无法打开图片", str(e))
            return
        self.reselect_btn.setEnabled(True)
        warn = "；".join(self.jpg.context.warnings)
        self.statusBar().showMessage(f"已载入 {os.path.basename(path)}  {warn}", 8000)
        self.select_on_jpg()

    def select_on_jpg(self):
        if self.jpg is None:
            return
        steps = STEPS[self.session.mode]
        dlg = JpgSelectDialog(self, self.jpg.pixels, steps, self.session.radius)
        if not dlg.exec() and not dlg.results:
            return
        label = f"文件 {self.jpg.context.device}"
        for kind, coords, step in dlg.results:
            if kind == "point":
                x, y = coords
                patch = point_patch_from(self.jpg.pixels, x, y, self.session.radius)
                self._handle_point(patch, x, y, self.jpg.context, label)
            else:
                region, rect = crop_selection(self.jpg.pixels, coords)
                self._run_rect(region, rect, self.jpg.context, label, step)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for url in ev.mimeData().urls():
            if url.isLocalFile():
                self.open_jpg(url.toLocalFile())
                break

    # ================================================================ 分析
    def _handle_point(self, patch: np.ndarray, x: int, y: int, ctx, label: str):
        r = self.session.radius
        # patch 以 (x, y) 为中心（贴边时被裁切），换算出中心在 patch 中的位置
        cx = min(r, x)
        cy = min(r, y)
        try:
            s = self.session.compute_point(patch, cx, cy, r, ctx)
        except SampleError as e:
            self.statusBar().showMessage(str(e), 5000)
            return
        s = dataclasses.replace(s, x=int(x), y=int(y))
        item = self.session.add_point(patch, s, label)
        self._after_add(item)

    def _run_rect(self, region, rect, ctx, label, step, replace: HistoryItem | None = None):
        mode = self.session.mode
        if region.size == 0:
            return
        if mode == "point":
            # 单点模式下拖动：取框内平均
            vals = color_values(region.reshape(-1, 3).astype(float).mean(axis=0))
            s = PointSample(rect[0] + rect[2] // 2, rect[1] + rect[3] // 2, 0, int(region.shape[0] * region.shape[1]), vals, ctx)
            item = self.session.add_point(region, s, label)
            item.label = f"区域平均 {rect[2]}×{rect[3]} {vals.hex}"
            self._after_add(item)
            return
        if mode == "compare":
            skin = self.skin_compare_check.isChecked()
            if step > 0 and self._ref_busy:
                self._pending_problem = (region, rect, ctx, label)  # 等参考区算完再对比
                return
            if step == 0 or self.session.reference is None:
                self._ref_busy = True
                self.statusBar().showMessage("正在分析参考区……")
                fn = self.session.compute_rect
                self._start(fn, region, ctx, rect, "skin" if skin else "dirty",
                            on_done=lambda rep: self._ref_done(region, rect, rep, label))
            else:
                if self._ref_busy:
                    self._pending_problem = (region, rect, ctx, label)
                    return
                self._start(self.session.compute_compare, self.session.reference, region, ctx, rect, skin,
                            on_done=lambda rep: self._rect_done(region, rect, rep, label, "对比"))
            return
        self.statusBar().showMessage("正在分析……")
        self._start(self.session.compute_rect, region, ctx, rect,
                    on_done=lambda rep: self._rect_done(region, rect, rep, label, None, replace))

    def _start(self, fn, *args, on_done):
        task = run_in_background(fn, *args, on_done=on_done, on_failed=self._failed)
        self._tasks.append(task)
        self._tasks = self._tasks[-20:]

    def _ref_done(self, region, rect, rep, label):
        item = self.session.add_rect(region, rect, rep, label, role="参考区")
        self.session.set_reference(item)
        self._ref_busy = False
        self._after_add(item, quiet=True)
        self.statusBar().showMessage("已设置参考区，请框选问题区", 6000)
        if self._pending_problem is not None:
            region2, rect2, ctx2, label2 = self._pending_problem
            self._pending_problem = None
            self._run_rect(region2, rect2, ctx2, label2, step=1)

    def _rect_done(self, region, rect, rep, label, role=None, replace=None):
        if replace is not None:
            self.session.remove(replace.id)
        item = self.session.add_rect(region, rect, rep, label, role=role)
        self._after_add(item)
        sev = rep.overall_severity.value
        self.statusBar().showMessage(f"分析完成：{sev}（风险 {rep.overall_risk:.0f}，置信度 {rep.confidence:.2f}）", 8000)

    def _failed(self, msg: str):
        self._ref_busy = False
        self.statusBar().showMessage("分析失败", 5000)
        QMessageBox.warning(self, "分析失败", msg)

    def _after_add(self, item: HistoryItem, quiet: bool = False):
        ref = self.session.reference.id if self.session.reference else None
        self.history_panel.set_items(self.session.history, ref)
        if not quiet:
            self._show_item(item)

    # ================================================================ 显示
    def _show_item(self, item: HistoryItem | None):
        self._current = item
        if item is None:
            return
        res = item.result
        if isinstance(res, PointSample):
            preset = self.session.preset
            delta = adjs = None
            if preset is not None:
                delta = pa.compare_lab(res.values.lab, preset.target_lab)
                adjs = rec.skin_adjustments(item.region.reshape(-1, 3), preset.target_lab) if delta.delta_e00 >= 1 else ()
            self.point_view.setHtml(point_html(res, preset, delta, adjs or ()))
            self.wheel.set_scatter(None)
            self.wheel.set_compare(None, None)
            self.wheel.set_opposite(None)
            self.wheel.set_point(res.values.lab, res.values.rgb)
            self.tabs.setCurrentWidget(self.point_view)
            return
        rep: AnalysisReport = res
        self.report_view.setHtml(report_html(rep, self.details_check.isChecked()))
        self.wheel.point = None
        ct = rep.contamination
        emphasis = None
        if ct is not None and ct.base_hue is not None and not ct.bicolor and rep.mode != "scan":
            emphasis = (ct.base_hue, 60.0 if rep.mode in ("skin", "compare") else 90.0, ct.opposite_threshold)
        self.wheel.set_scatter(item.region, emphasis)
        comp = {c.key: c for c in rep.components}.get("opposite")
        show_opp = ct is not None and comp is not None and comp.severity != Severity.NORMAL and ct.opposite_hue is not None
        self.wheel.set_opposite(ct.opposite_hue if show_opp else None)
        self.wheel.set_preset(self.session.preset if rep.mode == "skin" else None)
        if rep.comparison is not None:
            self.wheel.set_compare(rep.comparison.ref_lab, rep.comparison.target_lab)
        else:
            self.wheel.set_compare(None, None)
        self.result_view.show_report(item.region, rep)
        self.tabs.setCurrentIndex(0)

    # ================================================================ 历史
    def _history_activated(self, hid: int):
        self._show_item(self.session.get(hid))

    def _history_set_ref(self, hid: int):
        item = self.session.get(hid)
        if item is not None:
            self.session.set_reference(item)
            self._after_add(item, quiet=True)
            self.statusBar().showMessage("已设为参考区", 4000)

    def _history_compare(self, hid: int):
        item = self.session.get(hid)
        ref = self.session.reference
        if item is None or ref is None:
            self.statusBar().showMessage("请先把某条记录设为参考区", 4000)
            return
        skin = self.skin_compare_check.isChecked()
        self._start(self.session.compute_compare, ref, item.region, item.context, item.rect, skin,
                    on_done=lambda rep: self._rect_done(item.region, item.rect, rep, item.source_label, "对比"))

    def _history_target(self, hid: int):
        item = self.session.get(hid)
        if item is None:
            return
        p = self.session.set_custom_preset_from(item)
        self._fill_presets()
        self.statusBar().showMessage(f"已用该取样生成目标风格：Lab {p.target_lab[0]:.0f}, {p.target_lab[1]:.0f}, {p.target_lab[2]:.0f}", 6000)

    def _history_remove(self, hid: int):
        self.session.remove(hid)
        ref = self.session.reference.id if self.session.reference else None
        self.history_panel.set_items(self.session.history, ref)

    def _history_clear(self):
        self.session.clear()
        self.history_panel.set_items([])
        self.wheel.clear()
        self.report_view.clear()
        self.point_view.clear()
        self.result_view.clear()

    # ================================================================ 快捷键
    def _on_hotkey(self, name: str):
        if name == "lock_sample":
            self.lock_sample()
        elif name == "capture":
            self.start_capture()
