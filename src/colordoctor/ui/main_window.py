"""主窗口：悬浮在 Photoshop 旁边的小工具窗口。"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
from PyQt6.QtCore import QSettings, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QCursor, QDesktopServices, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
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
from colordoctor.core.color_management import RAW_EXTS, ImageLoadError, load_image, screen_pixels_to_srgb
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
    "palette": ["框选整张照片（或要分析的区域）"],
    "reference": ["第 1 步：框选参考作品（你想模仿的照片）", "第 2 步：框选你的照片"],
}
LABELS = {"compare": ["参考区", "问题区"], "reference": ["参考图", "我的照片"]}


class JpgSelectDialog(QDialog):
    """在 JPG 上选择点/区域（不是图片浏览器，只用来取样）。"""

    def __init__(self, parent, pixels: np.ndarray, steps: list[str], radius: int, lasso: bool = False,
                 labels: list[str] | None = None):
        super().__init__(parent)
        self.labels = labels or (["参考区", "问题区"] if len(steps) > 1 else ["选区"])
        self.setWindowTitle("在 JPG 上选择（滚轮缩放，右键拖动平移，Esc 取消）")
        self.resize(1100, 760)
        self.steps = steps
        self.step = 0
        self.results: list[tuple[str, tuple, int]] = []
        self.view = ImageSelectWidget(self, magnifier=True, allow_zoom=True)
        self.view.radius = radius
        self.view.lasso_mode = lasso
        self.view.set_image(pixels)
        self.view.polygon_selected.connect(lambda pts: self._got("polygon", tuple(pts)))
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
        if kind == "polygon":
            xs, ys = [p[0] for p in coords], [p[1] for p in coords]
            rect = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)) + 1, int(max(ys) - min(ys)) + 1)
        else:
            rect = coords if kind == "rect" else (coords[0], coords[1], 1, 1)
        self.view.add_mark(rect, self.labels[self.step] if self.step < len(self.labels) else "选区")
        self.step += 1
        if self.step >= len(self.steps):
            self.accept()
        else:
            self._text()


class SettingsDialog(QDialog):
    def __init__(self, parent, hotkeys: dict[str, str], ai: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        ai = ai or {}
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
        # ---- AI 点评（可选，默认关闭）
        form.addRow(QLabel("<b>AI 点评（可选）</b>"))
        self.ai_enabled = QCheckBox("启用 AI 点评（会把分析数值发送给你选择的 AI）")
        self.ai_enabled.setChecked(bool(ai.get("enabled")))
        form.addRow(self.ai_enabled)
        self.ai_provider = QComboBox()
        from colordoctor.ai_options import PROVIDER_NAMES

        for k, name in PROVIDER_NAMES.items():
            self.ai_provider.addItem(name, k)
        self.ai_provider.setCurrentIndex(max(0, self.ai_provider.findData(ai.get("provider", "claude_code"))))
        form.addRow("方式", self.ai_provider)
        self.ai_image = QCheckBox("同时发送选区缩略图（最长边 768 像素；DeepSeek 不支持）")
        self.ai_image.setChecked(bool(ai.get("send_image")))
        form.addRow(self.ai_image)
        self.ai_command = QLineEdit(ai.get("command", ""))
        self.ai_command.setPlaceholderText("留空用默认：claude -p  /  codex exec --skip-git-repo-check -")
        form.addRow("命令（命令行方式）", self.ai_command)
        self.ai_key = QLineEdit(ai.get("api_key", ""))
        self.ai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.ai_key.setPlaceholderText("留空则读环境变量 DEEPSEEK_API_KEY / ANTHROPIC_API_KEY")
        form.addRow("API Key（API 方式）", self.ai_key)
        self.ai_model = QLineEdit(ai.get("model", ""))
        self.ai_model.setPlaceholderText("留空用默认：deepseek-chat / claude-opus-5-5")
        form.addRow("模型", self.ai_model)
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

    def ai_values(self) -> dict:
        return {
            "enabled": self.ai_enabled.isChecked(),
            "provider": self.ai_provider.currentData(),
            "send_image": self.ai_image.isChecked(),
            "command": self.ai_command.text().strip(),
            "api_key": self.ai_key.text().strip(),
            "model": self.ai_model.text().strip(),
        }


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
        self.ref_image_item: HistoryItem | None = None
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

        # 模式（两行）
        row = QHBoxLayout()
        row2 = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for idx, (key, name) in enumerate(MODES.items()):
            b = QToolButton()
            b.setText(name)
            b.setCheckable(True)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            self.mode_group.addButton(b)
            self.mode_buttons[key] = b
            b.clicked.connect(lambda _c, k=key: self.set_mode(k))
            (row if idx < 3 else row2).addWidget(b)
        lay.addLayout(row)
        lay.addLayout(row2)

        # 取样方式
        row = QHBoxLayout()
        self.live_btn = QToolButton()
        self.live_btn.setText("◎ 实时取色")
        self.live_btn.setCheckable(True)
        self.live_btn.toggled.connect(self.set_live)
        self.capture_btn = QPushButton("▭ 框选屏幕")
        self.capture_btn.clicked.connect(self.start_capture)
        self.jpg_btn = QPushButton("打开 JPG / RAW")
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

        # 人像肤色选项
        self.skin_row = QWidget()
        r = QHBoxLayout(self.skin_row)
        r.setContentsMargins(0, 0, 0, 0)
        self.part_box = QComboBox()
        from colordoctor.ui.session import FAMILIES, PARTS

        for ptxt in PARTS:
            self.part_box.addItem(ptxt)
        self.part_box.setToolTip("分别选取脸部和身体，工具会自动比较两者的肤色差异")
        self.part_box.currentTextChanged.connect(lambda t: setattr(self.session, "part", t))
        self.auto_mask_check = QCheckBox("自动抠皮肤")
        self.auto_mask_check.setChecked(True)
        self.auto_mask_check.setToolTip("只统计选区里的皮肤像素（自动去掉头发、眼睛、嘴唇、背景）")
        self.auto_mask_check.toggled.connect(lambda v: setattr(self.session, "auto_mask", v))
        self.lasso_check = QCheckBox("套索")
        self.lasso_check.setToolTip("框选屏幕/JPG 时用自由套索圈出皮肤，而不是矩形")
        self.edit_mask_btn = QPushButton("编辑皮肤蒙版")
        self.edit_mask_btn.setToolTip("自动抠出的皮肤多了或少了：用画笔/魔棒手动添加或删除")
        self.edit_mask_btn.clicked.connect(self.edit_skin_mask)
        for wdg in (QLabel("部位"), self.part_box, self.auto_mask_check, self.lasso_check, self.edit_mask_btn):
            r.addWidget(wdg)
        lay.addWidget(self.skin_row)
        # 整图色彩选项
        self.palette_row = QWidget()
        r = QHBoxLayout(self.palette_row)
        r.setContentsMargins(0, 0, 0, 0)
        self.style_box = QComboBox()
        self.style_box.addItem("（不比较风格）", None)
        from colordoctor.core import palette as pl

        last_cat = None
        for st in pl.load_styles():
            cat = st.get("category", "")
            if cat != last_cat:
                self.style_box.insertSeparator(self.style_box.count())
                last_cat = cat
            self.style_box.addItem(st["name"], st["key"])
            self.style_box.setItemData(self.style_box.count() - 1, f"{cat}：{st['description']}", Qt.ItemDataRole.ToolTipRole)
        self.style_box.currentIndexChanged.connect(lambda _i: setattr(self.session, "style_key", self.style_box.currentData()))
        self.family_box = QComboBox()
        self.family_box.addItem("（不检查）", None)
        for fam in FAMILIES:
            self.family_box.addItem(f"统一：{fam}", fam)
        self.family_box.setToolTip("检查画面里某一种颜色（例如所有绿色）是不是同一种")
        self.family_box.currentIndexChanged.connect(lambda _i: setattr(self.session, "family", self.family_box.currentData()))
        for wdg in (QLabel("风格"), self.style_box, self.family_box):
            r.addWidget(wdg)
        lay.addWidget(self.palette_row)

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
        top = QHBoxLayout()
        self.details_check = QCheckBox("显示详细数据")
        self.details_check.toggled.connect(lambda _v: self._show_item(self._current))
        top.addWidget(self.details_check, 1)
        self.ai_btn = QPushButton("AI 点评")
        self.ai_btn.setToolTip("把分析数值发给你在设置中选择的 AI（Claude Code / Codex / DeepSeek / Claude API）")
        self.ai_btn.clicked.connect(self.ask_ai)
        self.copy_btn = QPushButton("复制给 Agent")
        self.copy_btn.setToolTip("复制提示词和数值，粘贴到 Claude、Codex、WorkBuddy 等任意 AI 会话")
        self.copy_btn.clicked.connect(self.copy_for_agent)
        self.export_btn = QPushButton("导出 JSON")
        self.export_btn.clicked.connect(self.export_json)
        for wdg in (self.ai_btn, self.copy_btn, self.export_btn):
            top.addWidget(wdg)
        v.addLayout(top)
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

    def _ai_settings(self) -> dict:
        g = self.settings.value
        return {
            "enabled": str(g("ai/enabled", "false")).lower() == "true",
            "provider": str(g("ai/provider", "claude_code")),
            "send_image": str(g("ai/send_image", "false")).lower() == "true",
            "command": str(g("ai/command", "")),
            "api_key": str(g("ai/api_key", "")),
            "model": str(g("ai/model", "")),
        }

    def open_settings(self):
        old_ai = self._ai_settings()
        dlg = SettingsDialog(self, self._hotkey_texts(), old_ai)
        if dlg.exec():
            ai = dlg.ai_values()
            if ai["enabled"] and not old_ai["enabled"]:
                ok = QMessageBox.question(
                    self,
                    "启用 AI 点评",
                    "启用后，点击“AI 点评”时会把分析得到的数值（以及你勾选时的选区缩略图）发送给所选的 AI 服务"
                    "（本机 Claude Code / Codex 会话也会把内容发到它们的云端）。\n\n"
                    "不点击“AI 点评”就不会发送任何内容。确定启用吗？",
                )
                if ok != QMessageBox.StandardButton.Yes:
                    ai["enabled"] = False
            for k, v in ai.items():
                self.settings.setValue(f"ai/{k}", "true" if v is True else ("false" if v is False else v))
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
        self.skin_row.setVisible(key == "skin")
        self.palette_row.setVisible(key == "palette")
        self.wheel.set_preset(self.session.preset if key in ("skin", "point") else None)
        hints = {
            "palette": "整图色彩：框选整张照片，看缺色、6:3:1 比例、某种颜色是否统一、与风格模板的差距",
            "reference": "仿色对比：先框参考作品，再框你的照片（可以在两张图上分别框）",
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
            mm = cur.result.valid_mask if cur.result.mask_info.get("manual") else None
            self._run_rect(cur.region, cur.rect, cur.context, cur.source_label, step=0, replace=cur, manual_mask=mm)
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
        if self.session.mode == "reference" and self.ref_image_item is not None:
            steps = STEPS["reference"][1:]
        self._capture_steps_offset = 2 - len(steps) if self.session.mode in ("compare", "reference") else 0
        labels = LABELS.get(self.session.mode)
        if labels and self._capture_steps_offset:
            labels = labels[self._capture_steps_offset:]
        lasso = self.session.mode == "skin" and self.lasso_check.isChecked()
        cs_ = CaptureSession(caps, steps, radius=self.session.radius, lasso=lasso, labels=labels)
        cs_.radius_changed.connect(self._set_radius)
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
        lasso = None
        if kind == "polygon":
            (x, y, w, h), lasso = self._polygon_rect(coords, cap.pixels.shape)
        else:
            x, y, w, h = coords
        raw = cap.pixels[y : y + h, x : x + w]
        srgb, ctx = screen_pixels_to_srgb(raw, cap.icc, managed, device=cap.info.name, scale=cap.info.dpr)
        self._run_rect(srgb, (x, y, w, h), ctx, label, step, lasso=lasso)

    @staticmethod
    def _polygon_rect(points, shape):
        from colordoctor.core.skin_mask import polygon_mask

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
        x1, y1 = min(shape[1], int(np.ceil(max(xs))) + 1), min(shape[0], int(np.ceil(max(ys))) + 1)
        mask = polygon_mask([(px - x0, py - y0) for px, py in points], x1 - x0, y1 - y0)
        return (x0, y0, x1 - x0, y1 - y0), mask

    def _set_radius(self, r: int):
        idx = self.radius_box.findData(r)
        if idx >= 0:
            self.radius_box.setCurrentIndex(idx)

    # ================================================================ JPG
    def open_jpg(self, path: str | None = None):
        if path is None:
            raw = " ".join(f"*{e} *{e.upper()}" for e in sorted(RAW_EXTS))
            path, _ = QFileDialog.getOpenFileName(self, "打开 JPG / RAW（只读取，不会修改原图）", "",
                                                  f"图片 (*.jpg *.jpeg *.JPG *.JPEG {raw});;JPG (*.jpg *.jpeg *.JPG *.JPEG);;相机 RAW ({raw})")
            if not path:
                return
        try:
            self.jpg = load_image(path, use_raw_preview=self.settings.value("raw/use_preview", False, type=bool))
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
        offset = 0
        if self.session.mode == "reference" and self.ref_image_item is not None:
            steps, offset = STEPS["reference"][1:], 1
        labels = LABELS.get(self.session.mode)
        lasso = self.session.mode == "skin" and self.lasso_check.isChecked()
        dlg = JpgSelectDialog(self, self.jpg.pixels, steps, self.session.radius, lasso=lasso,
                              labels=labels[offset:] if labels else None)
        if not dlg.exec() and not dlg.results:
            return
        label = f"文件 {self.jpg.context.device}"
        for kind, coords, step in dlg.results:
            step += offset
            if kind == "point":
                x, y = coords
                patch = point_patch_from(self.jpg.pixels, x, y, self.session.radius)
                self._handle_point(patch, x, y, self.jpg.context, label)
            elif kind == "polygon":
                rect, lasso_mask = self._polygon_rect(coords, self.jpg.pixels.shape)
                region, rect = crop_selection(self.jpg.pixels, rect)
                self._run_rect(region, rect, self.jpg.context, label, step, lasso=lasso_mask[: region.shape[0], : region.shape[1]])
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

    def _run_rect(self, region, rect, ctx, label, step, replace: HistoryItem | None = None, lasso=None, manual_mask=None):
        mode = self.session.mode
        if region.size == 0:
            return
        if mode == "reference":
            if step > 0 and self._ref_busy:
                self._pending_problem = (region, rect, ctx, label)
                return
            if step == 0 or self.ref_image_item is None:
                self._ref_busy = True
                self.statusBar().showMessage("正在分析参考作品……")
                self._start(self.session.compute_rect, region, ctx, rect, "palette",
                            on_done=lambda rep: self._ref_image_done(region, rect, rep, label))
            else:
                self.statusBar().showMessage("正在对比……")
                self._start(self.session.compute_reference, self.ref_image_item, region, ctx, rect,
                            on_done=lambda rep: self._rect_done(region, rect, rep, label, "仿色"))
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
        self._start(self.session.compute_rect, region, ctx, rect, None, lasso, manual_mask,
                    on_done=lambda rep: self._rect_done(region, rect, rep, label, None, replace))

    def _start(self, fn, *args, on_done):
        task = run_in_background(fn, *args, on_done=on_done, on_failed=self._failed)
        self._tasks.append(task)
        self._tasks = self._tasks[-20:]

    def _ref_image_done(self, region, rect, rep, label):
        item = self.session.add_rect(region, rect, rep, label, role="参考图")
        self.ref_image_item = item
        self._ref_busy = False
        self._after_add(item)
        self.statusBar().showMessage("已记录参考作品，请框选你的照片", 6000)
        if self._pending_problem is not None:
            region2, rect2, ctx2, label2 = self._pending_problem
            self._pending_problem = None
            self._run_rect(region2, rect2, ctx2, label2, step=1)

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
        self.ai_btn.setEnabled(True)
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
        scatter = item.region
        if rep.valid_mask is not None and rep.valid_mask.shape == item.region.shape[:2] and rep.valid_mask.any():
            scatter = item.region[rep.valid_mask][None]  # 只画参与分析的像素（例如皮肤）
        self.wheel.set_scatter(scatter, emphasis)
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

    # ================================================================ 皮肤蒙版
    def edit_skin_mask(self):
        item = self._current
        rep = item.result if item is not None else None
        if not isinstance(rep, AnalysisReport) or rep.mode != "skin":
            self.statusBar().showMessage("请先分析一块皮肤（人像肤色模式），再编辑蒙版", 5000)
            return
        from colordoctor.core.skin_mask import skin_mask
        from colordoctor.ui.mask_editor import MaskEditorDialog

        auto, _ = skin_mask(item.region)
        start = rep.valid_mask if rep.valid_mask is not None else np.ones(item.region.shape[:2], bool)
        dlg = MaskEditorDialog(self, item.region, start, auto)
        if dlg.exec():
            mask = dlg.mask.copy()
            if mask.sum() < 30:
                QMessageBox.warning(self, "蒙版太小", "蒙版里的皮肤像素太少，无法分析")
                return
            self.session.part = rep.part or self.session.part
            self._run_rect(item.region, item.rect, item.context, item.source_label, 0, replace=item, manual_mask=mask)

    # ================================================================ AI / 导出
    def _current_prompt(self) -> str | None:
        item = self._current
        if item is None:
            self.statusBar().showMessage("请先完成一次分析或取色", 4000)
            return None
        from colordoctor.ai.prompts import build_prompt
        from colordoctor.core import report_json as rj

        summary = rj.summary(item.result)
        if self.session.preset is not None and (self.session.mode in ("skin", "point")):
            summary["目标肤色风格"] = {"名称": self.session.preset.name, "Lab中心": self.session.preset.target_lab}
        return build_prompt(summary)

    def copy_for_agent(self):
        prompt = self._current_prompt()
        if prompt is None:
            return
        from colordoctor.ai.prompts import SYSTEM

        QApplication.clipboard().setText(SYSTEM + "\n\n" + prompt)
        self.statusBar().showMessage("已复制提示词和数值：粘贴到 Claude、Codex、WorkBuddy 等任意 AI 会话即可", 6000)

    def export_json(self):
        item = self._current
        if item is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出分析结果", "colordoctor_report.json", "JSON (*.json)")
        if not path:
            return
        from colordoctor.core import report_json as rj

        with open(path, "w", encoding="utf-8") as f:
            f.write(rj.report_to_json(item.result))
        self.statusBar().showMessage(f"已导出：{path}", 5000)

    def ask_ai(self):
        cfg_d = self._ai_settings()
        if not cfg_d["enabled"]:
            QMessageBox.information(self, "AI 点评未启用", "请在 ⚙ 设置中启用 AI 点评并选择方式（Claude Code / Codex / DeepSeek / Claude API）。\n"
                                    "也可以用“复制给 Agent”，粘贴到任意 AI 会话。")
            return
        prompt = self._current_prompt()
        if prompt is None:
            return
        from colordoctor.ai import providers as pv

        cfg = pv.AIConfig(provider=cfg_d["provider"], send_image=cfg_d["send_image"], command=cfg_d["command"],
                          api_key=cfg_d["api_key"], model=cfg_d["model"])
        image_path = None
        if cfg.send_image and pv.SUPPORTS_IMAGE.get(cfg.provider):
            from colordoctor.ai.thumbnail import make_thumbnail

            image_path = make_thumbnail(self._current.region)
        self.ai_btn.setEnabled(False)
        self.statusBar().showMessage(f"正在请求 {pv.PROVIDERS[cfg.provider]}……（可能需要一两分钟）")
        self._start(pv.ask, prompt, cfg, image_path, on_done=self._ai_done)

    def _ai_done(self, result):
        self.ai_btn.setEnabled(True)
        self.statusBar().showMessage("AI 点评完成" if result.ok else "AI 点评失败", 5000)
        dlg = QDialog(self)
        dlg.setWindowTitle("AI 点评" if result.ok else "AI 点评失败")
        dlg.resize(640, 720)
        view = QTextBrowser(dlg)
        view.setMarkdown(result.text)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("以下内容由 AI 生成，仅供参考；数值以本工具的测量为准。"))
        lay.addWidget(view)
        dlg.show()
        self._ai_dialog = dlg

    # ================================================================ 快捷键
    def _on_hotkey(self, name: str):
        if name == "lock_sample":
            self.lock_sample()
        elif name == "capture":
            self.start_capture()
