"""主窗口：悬浮在 Photoshop 旁边的小工具窗口。"""

from __future__ import annotations

import dataclasses
import os
import threading

import numpy as np
from PyQt6.QtCore import QByteArray, QSettings, Qt, QTimer, QUrl
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
from colordoctor.core import color_spaces as cs
from colordoctor.core import portrait_analysis as pa
from colordoctor.core import recommendations as rec
from colordoctor.core import skin_presets as sp
from colordoctor.core.color_management import RAW_EXTS, ImageLoadError, load_image, screen_pixels_to_srgb
from colordoctor.core.color_models import AnalysisReport, PointSample, Severity
from colordoctor.core.pixel_table import Cancelled, cancellable
from colordoctor.core.sampling import SampleError, color_values
from colordoctor.ui.capture_overlay import CaptureSession
from colordoctor.ui.color_wheel import ColorWheel
from colordoctor.ui.history_panel import HistoryPanel
from colordoctor.ui.keys import CMD, MAC
from colordoctor.ui.report_html import point_html, report_html
from colordoctor.ui.result_view import ResultView
from colordoctor.ui.selection_widget import ImageSelectWidget
from colordoctor.ui.session import (MODES, RADIUS_CHOICES, HistoryItem, Session, crop_selection, point_patch_from,
                                    region_mean_lab)
from colordoctor.ui.theme import TEXT_DIM
from colordoctor.ui.worker import run_in_background

MIN_SIDE = 8  # 区域分析的最小边长（像素）；更小的选区统计没有意义，有的计算也会出错


def hk_norm(text: str) -> str:
    return "+".join(sorted(t.strip().lower() for t in text.split("+")))


def _looks_like_skin(lab) -> bool:
    """宽松的肤色范围：明度适中、偏暖（a*、b* 为正），色相在红到黄之间。用来避免把蓝天当肤色比较。"""
    L, a, b = (float(v) for v in lab[:3])
    if not (20 <= L <= 95) or a < -2 or b < 0:
        return False
    h = float(np.degrees(np.arctan2(b, a))) % 360
    return 10 <= h <= 85 and np.hypot(a, b) >= 4


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
                 labels: list[str] | None = None, allow_point: bool = True, multi_point: bool = False):
        super().__init__(parent)
        self.labels = labels or (["参考区", "问题区"] if len(steps) > 1 else ["选区"])
        self.setWindowTitle("在图片上选择（滚轮缩放，右键拖动平移；连续取点时回车或 Esc 结束，其余 Esc 取消）")
        self.resize(1100, 760)
        self.steps = steps
        self.step = 0
        self.results: list[tuple[str, tuple, int]] = []
        self.view = ImageSelectWidget(self, magnifier=True, allow_zoom=True)
        self.view.radius = radius
        self.view.lasso_mode = lasso
        self.view.allow_point = allow_point
        self.view.set_image(pixels)
        self.view.polygon_selected.connect(lambda pts: self._got("polygon", tuple(pts)))
        self.view.point_selected.connect(lambda x, y: self._got("point", (x, y)))
        self.view.rect_selected.connect(lambda x, y, w, h: self._got("rect", (x, y, w, h)))
        self.multi_point = multi_point  # 单点模式：连续取多个点，回车或 Esc 结束
        self.view.cancelled.connect(self._esc)
        self.view.finished.connect(self._enter)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        self._text()

    def _text(self):
        if self.multi_point:
            n = len(self.results)
            self.view.instruction = (f"已取 {n} 个，继续单击取点；回车或 Esc 结束（两个点以上会自动对比）" if n
                                     else "单击取点，可以连续点多个；拖动则取框内平均")
        else:
            self.view.instruction = self.steps[self.step]
        self.view.update()

    def _esc(self):
        if self.multi_point and self.results:
            self.accept()  # 已经取的点保留
        else:
            self.reject()

    def _enter(self):
        if self.multi_point and self.results:  # 两步框选时回车不能把对话框关掉
            self.accept()

    def _got(self, kind, coords):
        if self.multi_point:
            self.results.append((kind, coords, 0))
            rect = coords if kind == "rect" else (coords[0], coords[1], 1, 1)
            self.view.add_mark(rect, f"点 {len(self.results)}")
            self._text()
            return
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
    def __init__(self, parent, hotkeys: dict[str, str], ai: dict | None = None, other: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        ai = ai or {}
        other = other or {}
        form = QFormLayout(self)
        self.lock = QLineEdit(hotkeys["lock_sample"])
        self.cap = QLineEdit(hotkeys["capture"])
        form.addRow("锁定实时取色", self.lock)
        form.addRow("冻结屏幕并框选", self.cap)
        note = QLabel(
            "全局快捷键会优先于 Photoshop 自己的快捷键。默认的 Ctrl+Alt+Shift+Z / X\n"
            "据 Photoshop 默认快捷键表未被占用；如果与你的自定义快捷键冲突请修改。"
            + ("\n全局快捷键目前只支持 Windows；Mac 上请用窗口里的按钮。" if MAC else "")
        )
        note.setWordWrap(True)
        form.addRow(note)
        btn = QPushButton("打开风格预设文件夹（可放置 skin_presets.json 覆盖/新增风格）")
        btn.clicked.connect(self._open_presets)
        form.addRow(btn)
        # ---- RAW 与 PS 插件
        self.raw_preview = QCheckBox("打开 RAW 时使用内嵌的 JPG 预览（相机直出的颜色；否则用 LibRaw 解码）")
        self.raw_preview.setChecked(bool(other.get("raw_preview")))
        form.addRow(self.raw_preview)
        self.bridge = QCheckBox("允许 Photoshop 插件连接（本机服务，只监听 127.0.0.1；重启后生效）")
        self.bridge.setChecked(bool(other.get("bridge", True)))
        form.addRow(self.bridge)
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

    def other_values(self) -> dict:
        return {"raw_preview": self.raw_preview.isChecked(), "bridge": self.bridge.isChecked()}

    def ai_values(self) -> dict:
        return {
            "enabled": self.ai_enabled.isChecked(),
            "provider": self.ai_provider.currentData(),
            "send_image": self.ai_image.isChecked(),
            "command": self.ai_command.text().strip(),
            "api_key": self.ai_key.text().strip(),
            "model": self.ai_model.text().strip(),
        }


RERUN_DELAY_MS = 250  # 改设置后等这么久再重算：连续改几个选项只算一次



def _wheel_points(region: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """色轮只画几千个点：先挑出均匀分布的像素位置再取值，不把整张大图（或蒙版内全部像素）复制一遍。"""
    from colordoctor.ui.color_wheel import MAX_POINTS

    px = region.reshape(-1, region.shape[-1])[:, :3]
    if mask is not None and mask.shape == region.shape[:2] and mask.any():
        idx = np.flatnonzero(mask)  # 只画参与分析的像素（例如皮肤）
    else:
        idx = np.arange(px.shape[0])
    if idx.size > MAX_POINTS:
        step = int(np.ceil(idx.size / MAX_POINTS))
        idx = idx[step // 2 :: step]
    return px[idx]

class MainWindow(QMainWindow):
    def __init__(self, session: Session | None = None, settings: QSettings | None = None):
        super().__init__()
        self.session = session or Session()
        self.settings = settings or QSettings("ColorDoctor", "ColorDoctor")
        self.setWindowTitle(f"人像色彩诊断取色器 {__version__}")
        self.resize(480, 940)
        scr = self.screen().availableGeometry() if self.screen() is not None else None
        if scr is not None and scr.height() < 980:  # 笔记本 / 高缩放：第一次打开不要超出屏幕
            self.resize(480, max(600, scr.height() - 60))
        self.setMinimumWidth(440)  # _build 之后按实际布局再调大
        self.setAcceptDrops(True)
        self.jpg = None
        self.capture_session: CaptureSession | None = None
        self._live = None
        self._was_live = False
        self._pending_problem = None
        self._batch_points: list | None = None  # 连续取点时本轮取到的点
        self._ref_busy = False
        self._current: HistoryItem | None = None
        self.ref_image_item: HistoryItem | None = None
        self._tasks = []
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(70)
        self.live_timer.timeout.connect(self._live_tick)

        self._build()
        self._load_settings()
        self._palette_option_changed()  # 恢复的选择同步到分析参数，并显示影调风格说明
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
        self.jpg_btn = QPushButton("打开图片")
        self.jpg_btn.setToolTip(f"JPG / PNG / TIFF / WebP / 相机 RAW（{CMD}O，也可以把文件拖进窗口）")
        self.jpg_btn.clicked.connect(lambda: self.open_jpg())
        self.reselect_btn = QPushButton("在图片上重选")
        self.reselect_btn.setToolTip("在已打开的 JPG / RAW 上重新框选或取点")
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
        self.preset_box.setToolTip("肤色风格：皮肤往哪种肤色靠（经验区间，可在设置中修改）")
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
        self.preset_label = QLabel("肤色风格")
        row.addWidget(self.preset_label)
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
        self.auto_mask_check.toggled.connect(self._toggle_auto_mask)
        from colordoctor.core import segmentation as sg

        seg_ok, seg_why = sg.available()
        self.ai_seg_check = QCheckBox("AI 抠皮肤")
        self.ai_seg_check.setEnabled(seg_ok)
        self.ai_seg_check.setToolTip(f"用 AI 人像分割区分脸部 / 身体皮肤（本机运行：{seg_why}）" if seg_ok else seg_why)
        self.ai_seg_check.setChecked(seg_ok and self.setting_bool("seg/ai", True))
        self.session.ai_seg = self.ai_seg_check.isChecked()
        self.ai_seg_check.toggled.connect(self._toggle_ai_seg)
        self.lasso_check = QCheckBox("套索")
        self.lasso_check.setToolTip("框选屏幕/JPG 时用自由套索圈出皮肤，而不是矩形")
        self.edit_mask_btn = QPushButton("编辑皮肤蒙版")
        self.edit_mask_btn.setToolTip("自动抠出的皮肤多了或少了：用画笔/魔棒手动添加或删除")
        self.edit_mask_btn.clicked.connect(self.edit_skin_mask)
        for wdg in (QLabel("部位"), self.part_box, self.auto_mask_check, self.ai_seg_check, self.lasso_check, self.edit_mask_btn):
            r.addWidget(wdg)
        lay.addWidget(self.skin_row)
        # 整图色彩选项
        self.palette_row = QWidget()
        r = QHBoxLayout(self.palette_row)
        r.setContentsMargins(0, 0, 0, 0)
        from colordoctor.core import palette as pl
        from colordoctor.core import tone_class as tcl

        # 影调风格（只管明暗和反差）和颜色（只管颜色）互相独立：港风可以配青橙，也可以配黑金
        self.tone_style_box = QComboBox()
        self.tone_style_box.addItem("（不改影调，只看）", "")
        for k, ts in tcl.TONE_STYLES.items():
            self.tone_style_box.addItem(f"{ts['name']} · {ts['target']}", f"s:{k}")
            self.tone_style_box.setItemData(self.tone_style_box.count() - 1, f"{ts['look']}\n常用于：{ts['scene']}",
                                            Qt.ItemDataRole.ToolTipRole)
        self.tone_style_box.insertSeparator(self.tone_style_box.count())
        for n in tcl.TONE_NAMES:
            self.tone_style_box.addItem(f"直接选：{n}", f"t:{n}")
            self.tone_style_box.setItemData(self.tone_style_box.count() - 1, tcl.TONE_DESC[n], Qt.ItemDataRole.ToolTipRole)
        self.tone_style_box.setToolTip("影调风格 = 预制的目标影调（明暗和反差往哪里走），和颜色互不绑定。\n"
                                       "不选时只说明这张图现在是九种影调里的哪一种，不改明暗")
        self.tone_style_box.currentIndexChanged.connect(self._palette_option_changed)
        self.style_box = QComboBox()
        self.style_box.addItem("（不改颜色）", None)
        last_cat = None
        for st in pl.load_styles():
            cat = st.get("category", "")
            if cat != last_cat:
                self.style_box.insertSeparator(self.style_box.count())
                last_cat = cat
            self.style_box.addItem(st["name"], st["key"])
            pair = "、".join(tcl.TONE_STYLES[k]["name"] for k in st.get("tone_pair", ()) if k in tcl.TONE_STYLES)
            self.style_box.setItemData(self.style_box.count() - 1,
                                       f"{cat}：{st['description']}" + (f"\n常配影调风格：{pair}（只是常见搭配，可以随意换）" if pair else ""),
                                       Qt.ItemDataRole.ToolTipRole)
        self.style_box.currentIndexChanged.connect(self._palette_option_changed)
        self.family_box = QComboBox()
        self.family_box.addItem("（不检查）", None)
        for fam in FAMILIES:
            self.family_box.addItem(f"统一：{fam}", fam)
        self.family_box.setToolTip("检查画面里某一种颜色（例如所有绿色）是不是同一种")
        self.family_box.currentIndexChanged.connect(self._palette_option_changed)
        for wdg in (QLabel("影调风格"), self.tone_style_box, QLabel("颜色"), self.style_box, QLabel("统一色"), self.family_box):
            r.addWidget(wdg)
        lay.addWidget(self.palette_row)
        # 影调风格的说明（不熟悉“日系”“港风”指什么的人看这里）
        self.tone_hint = QLabel()
        self.tone_hint.setWordWrap(True)
        self.tone_hint.setStyleSheet(f"color: {TEXT_DIM};")
        lay.addWidget(self.tone_hint)

        # 仿色：更换参考作品
        self.ref_row = QWidget()
        rr = QHBoxLayout(self.ref_row)
        rr.setContentsMargins(0, 0, 0, 0)
        self.ref_reset_btn = QPushButton("更换参考作品")
        self.ref_reset_btn.setToolTip("下一次框选 / 打开图片时重新选择参考")
        self.ref_reset_btn.clicked.connect(self.reset_reference_image)
        self.ref_label = QLabel()
        self.ref_label.setWordWrap(True)
        rr.addWidget(self.ref_label, 1)
        rr.addWidget(self.ref_reset_btn)
        lay.addWidget(self.ref_row)

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
        # 只有区域结论受“详细数据”影响；当前是取色 / 多点对比时不重画，避免把多点对比表冲掉
        self.details_check.toggled.connect(
            lambda _v: isinstance(getattr(self._current, "result", None), AnalysisReport)
            and self._show_item(self._current, switch_tab=False))
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
        self.history_panel.compare_points.connect(lambda ids: self.show_multipoint([h for h in (self.session.get(i) for i in ids) if h]))
        self.history_panel.use_as_target.connect(self._history_target)
        self.history_panel.remove_item.connect(self._history_remove)
        self.history_panel.clear_all.connect(self._history_clear)
        self.tabs.addTab(self.history_panel, "历史")
        split.addWidget(self.tabs)
        split.setSizes([330, 520])
        lay.addWidget(split, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪：所有计算在本地完成，不上传、不修改任何图片")

        # 最小宽度不能小于布局需要的宽度，否则按钮文字被截断
        self.setMinimumWidth(max(440, central.minimumSizeHint().width() + 12))
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
        try:
            r = int(self.settings.value("radius", 2))
        except (TypeError, ValueError):  # 设置文件损坏时不能让程序起不来
            r = 2
        idx = self.radius_box.findData(r)
        self.radius_box.setCurrentIndex(idx if idx >= 0 else 2)
        key = str(self.settings.value("preset", sp.DEFAULT_KEY))
        idx = self.preset_box.findData(key)
        if idx >= 0:
            self.preset_box.setCurrentIndex(idx)
        managed = str(self.settings.value("managed", "true")).lower() == "true"
        self.source_box.setCurrentIndex(0 if managed else 1)
        # 其余界面选择：改动时立即保存，下次启动恢复
        for w, key in ((self.top_check, "on_top"), (self.auto_mask_check, "auto_mask"), (self.lasso_check, "lasso"),
                       (self.scan_check, "scan_whole"), (self.skin_compare_check, "skin_compare"),
                       (self.details_check, "details"), (self.part_box, "part"), (self.style_box, "style"),
                       (self.family_box, "family"), (self.tone_style_box, "tone_style")):
            self._persist(w, f"ui/{key}")
        geo = self.settings.value("ui/geometry")
        if isinstance(geo, (QByteArray, bytes, bytearray)) and geo:  # 设置文件损坏时不能让程序起不来
            self.restoreGeometry(QByteArray(geo))

    def _persist(self, w, key: str):
        """恢复一个控件上次的值，并在用户改动时立即保存（程序意外退出也不丢）。"""
        v = self.settings.value(key)
        if isinstance(w, QCheckBox):
            if v is not None:
                w.setChecked(str(v).lower() == "true")
            w.toggled.connect(lambda on: self.settings.setValue(key, "true" if on else "false"))
            return
        # 下拉框：有数据的（影调风格、颜色、统一色）按数据保存，“（不改颜色）”这类空选项存 ""；只有文字的（部位）按文字保存
        by_text = all(w.itemData(i) is None for i in range(w.count()))
        if v is not None:
            v = str(v)
            idx = w.findText(v) if by_text else (0 if v == "" and w.itemData(0) is None else w.findData(v))
            if idx >= 0:
                w.setCurrentIndex(idx)

        def save(_i):
            d = w.currentData()
            self.settings.setValue(key, w.currentText() if by_text else ("" if d is None else str(d)))

        w.currentIndexChanged.connect(save)

    def closeEvent(self, ev):
        self.settings.setValue("mode", self.session.mode)
        self.settings.setValue("radius", self.session.radius)
        self.settings.setValue("preset", self.session.preset_key if self.session.preset_key != "custom" else sp.DEFAULT_KEY)
        self.settings.setValue("managed", "true" if self._managed() else "false")
        self.settings.setValue("ui/geometry", self.saveGeometry())
        self.live_timer.stop()
        self.hotkeys.close()
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

    def reset_reference_image(self):
        if self.session.mode == "compare":
            self.session.reference = None
            self._after_add(None, quiet=True)
            self.statusBar().showMessage("下一次框选时请先框参考区", 6000)
        else:
            self.ref_image_item = None
            self.statusBar().showMessage("下一次框选或打开图片时请先选参考作品", 6000)
        self._update_ref_label()

    def _update_ref_label(self):
        mode = self.session.mode
        if mode == "compare":
            ref = self.session.reference
            self.ref_reset_btn.setText("更换参考区")
            self.ref_label.setText(f"当前参考区：{ref.label}（{ref.time_text}）" if ref is not None
                                   else "参考对比：先框参考区（干净皮肤），再框问题区；参考区会一直沿用")
        else:
            ref = self.ref_image_item
            self.ref_reset_btn.setText("更换参考作品")
            self.ref_label.setText(f"当前参考作品：{ref.label}（{ref.time_text}）" if ref is not None
                                   else "仿色：先选参考作品，再选你的照片（可以是两张不同的图）；参考作品会一直沿用")
        self.ref_reset_btn.setEnabled(ref is not None)

    def _toggle_auto_mask(self, v: bool):
        self.session.auto_mask = v
        # AI 抠皮肤只在“自动抠皮肤”打开时起作用
        from colordoctor.core import segmentation as sg

        self.ai_seg_check.setEnabled(v and sg.available()[0])

    def _palette_option_changed(self, _i=None):
        self.session.style_key = self.style_box.currentData()
        self.session.family = self.family_box.currentData()
        tone = self.tone_style_box.currentData() or ""
        self.session.tone_style = tone[2:] if tone.startswith("s:") else ""
        self.session.tone_target = tone[2:] if tone.startswith("t:") else ""
        self._update_tone_hint()
        cur = self._current
        # 当前显示的是整图色彩结果：按新的风格 / 统一色重新计算（和改肤色目标的行为一致）
        if cur is not None and isinstance(cur.result, AnalysisReport) and cur.result.mode == "palette" \
                and cur is not self.ref_image_item:
            self._rerun(cur, "palette")

    def _update_tone_hint(self):
        """影调风格的说明：看起来是什么样、常用在哪；当前结果是整图色彩时，再说这张图从现在的影调要怎么改。"""
        from colordoctor.core import tone_class as tcl

        st, tt = self.session.tone_style, self.session.tone_target
        if st:
            s = tcl.TONE_STYLES[st]
            also = f"（也可以是{'、'.join(s['also'])}，会往离现在最近的那种走）" if s["also"] else ""
            text = f"「{s['name']}」= 目标 {s['target']}{also}：{s['look']}。常用于{s['scene']}。"
        elif tt:
            text = f"目标 {tt}：{tcl.TONE_DESC[tt]}。"
        else:
            text = "不改影调：只说明这张图现在是哪种影调。想往某种影调走，在「影调风格」里选（鼠标停在选项上有说明）。"
        cur = self._current
        rep = cur.result if cur is not None else None
        if (isinstance(rep, AnalysisReport) and rep.mode == "palette" and rep.palette is not None
                and rep.palette.tone_class is not None and cur is not self.ref_image_item):
            p = rep.palette
            now = p.tone_class.name
            if self.session.report_choice(rep) != self.session.palette_choice():
                # 显示的是按别的选择算的结果（例如从历史里打开的旧结果）：不拿现在的选择去描述它
                tone = p.tone_label or "不改影调"
                color = p.style.name if p.style else "不改颜色"
                text += f"下面这条结果是按「{tone} · {color}」算的，和现在的选择不同；改一下选择就会按新选择重算。"
            else:
                goal = tcl.pick_goal(now, *tcl.resolve_target(st, tt)[:2])
                text += f"这张图现在是{now}" + (f"，要{tcl.path_words(now, goal)}。" if goal and goal != now else "。")
        self.tone_hint.setText(text)

    def _toggle_ai_seg(self, v: bool):
        self.session.ai_seg = v
        self.settings.setValue("seg/ai", "true" if v else "false")

    def open_settings(self):
        old_ai = self._ai_settings()
        other = {"raw_preview": self.setting_bool("raw/use_preview", False),
                 "bridge": self.setting_bool("bridge/enabled", True)}
        dlg = SettingsDialog(self, self._hotkey_texts(), old_ai, other)
        if dlg.exec():
            o = dlg.other_values()
            self.settings.setValue("raw/use_preview", "true" if o["raw_preview"] else "false")
            self.settings.setValue("bridge/enabled", "true" if o["bridge"] else "false")
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
            new_keys = dlg.values()
            bad = []
            if len({hk_norm(v) for v in new_keys.values()}) < len(new_keys):
                bad.append("两个快捷键不能相同")
            for v in new_keys.values():
                try:
                    hk.parse_hotkey(v)
                except hk.HotkeyParseError as e:
                    bad.append(f"{v}：{e}")
            if bad:
                QMessageBox.warning(self, "快捷键无效", "快捷键没有保存：\n" + "\n".join(bad))
            else:
                for k, v in new_keys.items():
                    self.settings.setValue(f"hotkeys/{k}", v)
                self.hotkeys.unregister_all()
                self._register_hotkeys()
        self.session.presets = sp.load_presets()
        self._fill_presets()

    def setting_bool(self, key: str, default: bool) -> bool:
        """读开关设置；设置文件被改坏（例如 ini 里写成列表）时用默认值，而不是启动失败。"""
        v = self.settings.value(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().lower() in ("true", "1", "yes", "false", "0", "no"):
            return v.strip().lower() in ("true", "1", "yes")
        if isinstance(v, int):
            return bool(v)
        return default

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
        self.tone_hint.setVisible(key == "palette")
        self.ref_row.setVisible(key in ("reference", "compare"))
        self.history_panel.ref_action_text = "设为参考作品" if key == "reference" else "设为参考区"
        self._update_ref_label()
        # 肤色目标只在肤色 / 单点模式下起作用
        self.preset_label.setVisible(key in ("skin", "point"))
        self.preset_box.setVisible(key in ("skin", "point"))
        self.wheel.set_preset(self.session.preset if key in ("skin", "point") else None)
        hints = {
            "palette": "整图色彩：框选整张照片，看影调（九种之一，可选影调风格往目标影调走）、缺色、6:3:1 比例、某种颜色是否统一、与颜色模板的差距",
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
        # 只重算普通的肤色结果；参考区 / 参考作品不动（它们是对比的基准）
        if cur is not None and isinstance(cur.result, AnalysisReport) and cur.result.mode == "skin" \
                and cur is not self.session.reference and cur is not self.ref_image_item:
            self._rerun(cur, "skin")
        elif cur is not None and isinstance(cur.result, PointSample):
            self._show_item(cur, switch_tab=False)

    def _rerun(self, item: HistoryItem, mode: str):
        """按新设置重算一条历史结果并替换它。连续改设置时只采用最后一次的结果，不会出现重复记录。"""
        self._rerun_gen = getattr(self, "_rerun_gen", 0) + 1
        gen = self._rerun_gen
        rep = item.result
        mm = rep.valid_mask if mode == "skin" and rep.mask_info.get("manual") else None

        def compute():
            if gen != self._rerun_gen:  # 排队时又改了设置：这次不用算了（大图每次要一两秒）
                return None
            return self.session.compute_rect(item.region, item.context, item.rect, mode, item.lasso, mm,
                                             part=rep.part if mode == "skin" else None)

        def done(new_rep):
            if gen != self._rerun_gen:
                return  # 已有更新的请求
            if self.session.get(item.id) is None:  # 原条目已被删除
                self.statusBar().clearMessage()
                return
            self._rect_done(item.region, item.rect, new_rep, item.source_label, None, item, item.lasso)

        def start():
            if gen != self._rerun_gen:
                return
            self._rerun_ev = self._run_cancellable(compute, done)

        # 又改了设置：上一次按旧设置还没算完的重算马上停掉（不和这次抢 CPU）
        if getattr(self, "_rerun_ev", None) is not None:
            self._rerun_ev.set()

        self.statusBar().showMessage("正在按新设置重新计算……")
        QTimer.singleShot(RERUN_DELAY_MS, start)  # 连续改几个选项时只算最后一次

    def set_on_top(self, on: bool):
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on)
        if visible:  # 改窗口标志会把窗口藏起来；启动时恢复设置还没显示，不能提前弹出
            self.show()

    # ================================================================ 实时取色
    def set_live(self, on: bool):
        if on and not self._screen_capture_ok():
            on = False
        if on:
            self.live_timer.start()
            lock = (f"按 {self._hotkey_texts()['lock_sample']} 锁定读数" if hk.is_supported()
                    else "这个系统不支持全局快捷键，不能在 Photoshop 里锁定读数；要记下颜色请用框选 / 单点取色")
            self.live_label.setText(f"实时取色：移动鼠标到 Photoshop 画面上；{lock}")
        else:
            self.live_timer.stop()
            self.live_label.setText("实时取色：关闭")
            self._live = None  # 关掉后不能再锁定之前的旧读数
            self.wheel.point = None
            self.wheel.update()
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
        srgb, ctx = screen_pixels_to_srgb(patch, sc.icc_for(info.name, info=info), self._managed(), device=info.name, scale=info.dpr)
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

    def _screen_capture_ok(self) -> bool:
        """macOS：没有“屏幕录制”权限时截到的只有桌面背景（系统不报错），先提示去打开权限。"""
        from colordoctor.capture import mac_display

        if mac_display.screen_capture_allowed():
            return True
        mac_display.request_screen_capture()
        QMessageBox.information(
            self, "需要屏幕录制权限",
            "macOS 需要允许本工具“录制屏幕”，才能读取 Photoshop 画面上的颜色。\n\n"
            "请打开 系统设置 → 隐私与安全性 → 屏幕录制，勾选 ColorDoctor，然后退出并重新打开本工具。\n"
            "（每次更新本工具后可能需要重新勾选。）")
        return False

    # ================================================================ 屏幕框选
    def start_capture(self):
        if self.capture_session is not None:
            return
        if not self._screen_capture_ok():
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
            if getattr(self, "_was_live", False):  # 截屏失败也要恢复之前开着的实时取色
                self.live_timer.start()
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
        multi = self.session.mode in ("compare", "reference")  # 多步框选：单击不能吃掉一步
        point_mode = self.session.mode == "point"
        self._batch_points = [] if point_mode else None
        cs_ = CaptureSession(caps, steps, radius=self.session.radius, lasso=lasso, labels=labels, allow_point=not multi,
                             multi_point=point_mode)
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
        self._finish_batch()
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
            other = "*.png *.PNG *.tif *.tiff *.TIF *.TIFF *.webp *.WEBP"
            path, _ = QFileDialog.getOpenFileName(self, "打开图片（只读取，不会修改原图）", "",
                                                  f"图片 (*.jpg *.jpeg *.JPG *.JPEG {other} {raw});;JPG (*.jpg *.jpeg *.JPG *.JPEG);;"
                                                  f"PNG / TIFF / WebP ({other});;相机 RAW ({raw})")
            if not path:
                return
        try:
            self.jpg = load_image(path, use_raw_preview=self.setting_bool("raw/use_preview", False))
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
        mode = self.session.mode
        steps = STEPS[mode]
        offset = 0
        # 已有参考时只选第二步（屏幕框选也是这样）；更换参考用“更换参考区 / 参考作品”按钮
        if mode == "reference" and self.ref_image_item is not None:
            steps, offset = STEPS["reference"][1:], 1
        elif mode == "compare" and self.session.reference is not None:
            steps, offset = STEPS["compare"][1:], 1
        elif mode == "reference":
            # 参考作品和自己的照片通常是两个文件：这张图上只选参考作品，然后去打开自己的照片
            steps = STEPS["reference"][:1]
        labels = LABELS.get(mode)
        lasso = mode == "skin" and self.lasso_check.isChecked()
        dlg = JpgSelectDialog(self, self.jpg.pixels, steps, self.session.radius, lasso=lasso,
                              labels=labels[offset:] if labels else None, allow_point=mode not in ("compare", "reference"),
                              multi_point=mode == "point")
        if not dlg.exec() and not dlg.results:
            return
        label = f"文件 {self.jpg.context.device}"
        self._batch_points = [] if mode == "point" else None
        for kind, coords, step in dlg.results:
            step += offset
            try:
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
            except SampleError as e:
                self.statusBar().showMessage(f"{e}，请在图像上重新框选", 6000)
                return
        if mode == "reference" and offset == 0:
            self.statusBar().showMessage("已记录参考作品：现在打开你的照片（或框选屏幕）选你的照片", 10000)
        self._finish_batch()

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
        if getattr(self, "_batch_points", None) is not None:
            self._batch_points.append(item)
        self._after_add(item)

    def _finish_batch(self):
        """连续取了两个点以上：自动对比，显示在“取色数值”页。"""
        items, self._batch_points = getattr(self, "_batch_points", None) or [], None
        if len(items) >= 2:
            for i, h in enumerate(items):  # 只在真的取了多个点时编号，历史里能对上表格
                h.label = f"点 {i + 1}  {h.label}"
            self._after_add(None, quiet=True)
            self.show_multipoint(items)

    def show_multipoint(self, items: list[HistoryItem]):
        from colordoctor.core.multipoint import compare_points
        from colordoctor.ui.multipoint_view import multipoint_html

        pts = sorted((h for h in items if isinstance(h.result, PointSample)), key=lambda h: h.id)  # 最早取的是基准
        if len(pts) < 2:
            self.statusBar().showMessage("至少选两个取色点才能对比", 4000)
            return
        # 按位置编号，再附上取样时间，避免两批都叫“点 1”
        names = [f"点 {i + 1}（{h.time_text}）" for i, h in enumerate(pts)]
        res = compare_points([h.result.values.lab for h in pts], names)
        self.point_view.setHtml(multipoint_html(res))
        self.tabs.setCurrentWidget(self.point_view)
        self.wheel.set_scatter(None)
        self.wheel.set_point(pts[-1].result.values.lab, pts[-1].result.values.rgb)
        self.statusBar().showMessage(res.summary, 10000)

    def _run_rect(self, region, rect, ctx, label, step, replace: HistoryItem | None = None, lasso=None, manual_mask=None,
                  mode: str | None = None, part: str | None = None):
        mode = mode or self.session.mode
        if region.size == 0:
            return
        if mode != "point" and min(region.shape[:2]) < MIN_SIDE:
            self.statusBar().showMessage(f"选区太小（{region.shape[1]}×{region.shape[0]} 像素），请框大一些（至少 {MIN_SIDE}×{MIN_SIDE}）", 6000)
            return
        if mode == "reference":
            if step > 0 and self._ref_busy:
                self._pending_problem = (region, rect, ctx, label)
                return
            if step == 0 or self.ref_image_item is None:
                self._ref_busy = True
                self.statusBar().showMessage("正在分析参考作品……")
                self._start(self.session.compute_rect, region, ctx, rect, "palette", plain=True,
                            on_done=lambda rep: self._ref_image_done(region, rect, rep, label))
            else:
                self.statusBar().showMessage("正在对比……")
                self._start(self.session.compute_reference, self.ref_image_item, region, ctx, rect,
                            on_done=lambda rep: self._rect_done(region, rect, rep, label, "仿色"))
            return
        if mode == "point":
            # 单点模式下拖动：取框内平均
            vals = color_values(region.reshape(-1, 3).mean(axis=0, dtype=np.float64))  # 不先转成整幅 float（大图很慢）
            s = PointSample(rect[0] + rect[2] // 2, rect[1] + rect[3] // 2, 0, int(region.shape[0] * region.shape[1]), vals, ctx)
            item = self.session.add_point(region, s, label)
            item.label = f"区域平均 {rect[2]}×{rect[3]} {vals.hex}"
            if getattr(self, "_batch_points", None) is not None:  # 连续取点时也算一个点
                self._batch_points.append(item)
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
        if lasso is None and replace is not None:
            lasso = replace.lasso
        self._start(self.session.compute_rect, region, ctx, rect, mode, lasso, manual_mask, part=part,
                    on_done=lambda rep: self._rect_done(region, rect, rep, label, None, replace, lasso))

    def _run_cancellable(self, compute, on_done) -> threading.Event:
        """在后台计算 compute()，算完调用 on_done(结果)。返回取消用的 Event：设置后，计算在下一个检查点停下
        （不再白占 CPU），也不再调用 on_done。"""
        ev = threading.Event()

        def guarded():
            try:
                with cancellable(ev):
                    return compute()
            except Cancelled:
                return None

        def done(rep):
            if rep is not None and not ev.is_set():
                on_done(rep)

        def failed(msg):
            if not ev.is_set():  # 已被取代的计算出错就不打扰用户了
                self._failed(msg)

        self._start(guarded, on_done=done, on_fail=failed)
        return ev

    def _start(self, fn, *args, on_done, on_fail=None, **kwargs):
        task = None

        def forget():  # 结果交出后不再留着任务：断开回调（回调里引用着选区像素），引用环也就没了
            if task in self._tasks:
                self._tasks.remove(task)

            def cut(t=task):  # 不在信号正在送达时断开：留到下一轮事件循环
                for sig in (t.signals.done, t.signals.failed):
                    try:
                        sig.disconnect()
                    except (TypeError, RuntimeError):
                        pass

            QTimer.singleShot(0, cut)

        def done(rep):
            # 窗口已关闭（例如分析还没结束就退出）时丢弃结果
            from PyQt6 import sip

            if sip.isdeleted(self) or sip.isdeleted(self.history_panel):
                return
            forget()
            on_done(rep)

        def failed(msg):
            from PyQt6 import sip

            if not sip.isdeleted(self):
                forget()
                (on_fail or self._failed)(msg)

        task = run_in_background(fn, *args, on_done=done, on_failed=failed, **kwargs)
        self._tasks.append(task)  # 保持任务（和它的信号对象）活到结果送回界面

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

    def _rect_done(self, region, rect, rep, label, role=None, replace=None, lasso=None):
        if replace is not None:
            # 要替换的条目在计算期间已被另一个重算结果替换（例如连改两次蒙版）：替换那个新条目；
            # 已被删除（或因内存上限被清掉）：结果丢弃，不让删掉的记录又冒出来
            while replace.replaced_by is not None:
                replace = replace.replaced_by
            if self.session.get(replace.id) is None:
                self.statusBar().clearMessage()
                return None
        was_ref = replace is not None and replace is self.session.reference
        was_current = replace is None or replace is self._current
        pos = None
        if replace is not None:
            pos = next((i for i, h in enumerate(self.session.history) if h.id == replace.id), None)
            self.session.remove(replace.id)
        item = self.session.add_rect(region, rect, rep, label, role=role)
        if replace is not None:
            replace.replaced_by = item
        if pos is not None:  # 按新设置重算：留在原来的位置，不跳到历史最上面
            self.session.history.remove(item)
            self.session.history.insert(pos, item)
        item.lasso = lasso
        if was_ref:
            self.session.set_reference(item)  # 替换的是参考区：新结果继续作为参考区
        # 重算的结果只在用户还看着原条目时显示；已经切到别的记录就只替换历史，不抢走画面
        self._after_add(item, quiet=not was_current)
        # 计算过程中改了影调风格 / 颜色 / 统一色：结果是按旧选择算的，马上按现在的选择再算一次
        choice = self.session.report_choice(rep)
        if choice is not None and choice != self.session.palette_choice() and item is self._current:
            self._rerun(item, "palette")
        sev = rep.overall_severity.value
        self.statusBar().showMessage(f"分析完成：{sev}（风险 {rep.overall_risk:.0f}，置信度 {rep.confidence:.2f}）", 8000)
        return item

    def _failed_quiet(self, msg: str):
        """只报错、不清掉正在进行的框选流程（参考区 / 问题区）的失败处理。"""
        self.statusBar().showMessage("分析失败", 5000)
        QMessageBox.warning(self, "分析失败", msg)

    def _failed(self, msg: str):
        self._ref_busy = False
        self._pending_problem = None
        # AI 点评按钮只由 AI 请求自己恢复：别的分析失败时不能把它提前打开（否则会同时发出第二个请求）
        self.statusBar().showMessage("分析失败", 5000)
        QMessageBox.warning(self, "分析失败", msg)

    def _after_add(self, item: HistoryItem | None, quiet: bool = False):
        keep = {i.id for i in (item, self.ref_image_item, self._current) if i is not None}
        if self.session.trim(keep):
            self.statusBar().showMessage("历史记录太多（占用内存），已自动删掉最旧的几条", 6000)
        ref = self.session.reference.id if self.session.reference else None
        self.history_panel.set_items(self.session.history, ref, self.ref_image_item.id if self.ref_image_item else None)
        if not quiet and item is not None:
            self._show_item(item)
        cur = self._current
        self.history_panel.highlight(cur.id if cur is not None else None)
        self._update_ref_label()

    # ================================================================ 显示
    def _clear_views(self):
        self._current = None
        self.wheel.clear()
        self.report_view.clear()
        self.point_view.clear()
        self.result_view.clear()

    def _show_item(self, item: HistoryItem | None, switch_tab: bool = True):
        """显示一条结果。三个结果页始终对应同一条：显示单点时清掉区域结论，反之亦然。"""
        if item is None:
            self._clear_views()
            return
        self._current = item
        self.history_panel.highlight(item.id)
        self._update_tone_hint()
        res = item.result
        if isinstance(res, PointSample):
            preset = self.session.preset
            delta = adjs = None
            # 只在肤色 / 单点模式下和目标肤色比较；颜色明显不是肤色时不给“调成肤色”的建议
            if preset is not None and item.mode in ("skin", "point") and _looks_like_skin(res.values.lab):
                target = preset.target_for(res.values.lab)  # 明度：区间内不动，“自然通用”保持当前值
                delta = pa.compare_lab(res.values.lab, target)
                adjs = (rec.skin_adjustments(item.region.reshape(-1, 3), target, keep_lightness=preset.keeps_lightness)
                        if delta.delta_e00 >= 2 else ())
            self.point_view.setHtml(point_html(res, preset if delta is not None else None, delta, adjs or ()))
            self.report_view.setHtml("<p>当前是单点取色，读数见「取色数值」页。</p>")
            self.result_view.clear()
            self.wheel.set_scatter(None)
            self.wheel.set_compare(None, None)
            self.wheel.set_opposite(None)
            self.wheel.set_point(res.values.lab, res.values.rgb)
            if switch_tab:
                self.tabs.setCurrentWidget(self.point_view)
            return
        rep: AnalysisReport = res
        self.point_view.setHtml("<p>当前是区域分析结果，见「结论」页。</p>")
        self.report_view.setHtml(report_html(rep, self.details_check.isChecked()))
        self.wheel.point = None
        ct = rep.contamination
        emphasis = None
        if ct is not None and ct.base_hue is not None and not ct.bicolor and rep.mode != "scan":
            emphasis = (ct.base_hue, 60.0 if rep.mode in ("skin", "compare") else 90.0, ct.opposite_threshold)
        self.wheel.set_scatter(_wheel_points(item.region, rep.valid_mask), emphasis)
        comp = {c.key: c for c in rep.components}.get("opposite")
        show_opp = ct is not None and comp is not None and comp.severity != Severity.NORMAL and ct.opposite_hue is not None
        self.wheel.set_opposite(ct.opposite_hue if show_opp else None)
        self.wheel.set_preset(self.session.preset if rep.mode == "skin" else None)
        if rep.comparison is not None:
            self.wheel.set_compare(rep.comparison.ref_lab, rep.comparison.target_lab)
        else:
            self.wheel.set_compare(None, None)
        self.result_view.show_report(item.region, rep)
        if switch_tab:
            self.tabs.setCurrentIndex(0)

    # ================================================================ 历史
    def _history_activated(self, hid: int):
        # 在历史里逐条查看时不跳走标签页
        self._show_item(self.session.get(hid), switch_tab=False)

    def _history_set_ref(self, hid: int):
        item = self.session.get(hid)
        if item is None:
            return
        if self.session.mode == "reference":
            self.ref_image_item = item  # 仿色模式：设为参考作品
            self._after_add(item, quiet=True)
            self.statusBar().showMessage("已设为仿色的参考作品", 4000)
        else:
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
                    on_done=lambda rep: self._rect_done(item.region, item.rect, rep, item.source_label, "对比"),
                    on_fail=self._failed_quiet)

    def _history_target(self, hid: int):
        item = self.session.get(hid)
        if item is None:
            return
        lab = region_mean_lab(item.region)
        if not _looks_like_skin(lab):
            ok = QMessageBox.question(self, "这不像肤色",
                                      f"这条记录的平均颜色 Lab {lab[0]:.0f}, {lab[1]:.0f}, {lab[2]:.0f} 不在常见肤色范围内。\n"
                                      "仍然用它作为目标肤色吗？")
            if ok != QMessageBox.StandardButton.Yes:
                return
        p = self.session.set_custom_preset_from(item)
        self._fill_presets()
        self.statusBar().showMessage(f"已用该取样生成目标肤色：Lab {p.target_lab[0]:.0f}, {p.target_lab[1]:.0f}, {p.target_lab[2]:.0f}", 6000)
        self._preset_changed()  # 当前的肤色结果按新目标重算

    def _history_remove(self, hid: int):
        if self.ref_image_item is not None and self.ref_image_item.id == hid:
            self.ref_image_item = None
        self.session.remove(hid)
        if self._current is not None and self._current.id == hid:
            self._clear_views()  # 删掉的正在显示：不能再导出 / 点评它
        self._after_add(None, quiet=True)

    def _history_clear(self):
        self.ref_image_item = None
        self._pending_problem = None
        self._ref_busy = False
        self.session.clear()
        self._clear_views()
        self._after_add(None, quiet=True)

    # ================================================================ 皮肤蒙版
    def edit_skin_mask(self):
        item = self._current
        rep = item.result if item is not None else None
        if not isinstance(rep, AnalysisReport) or rep.mode != "skin":
            self.statusBar().showMessage("请先分析一块皮肤（人像肤色模式），再编辑蒙版", 5000)
            return
        from colordoctor.core.skin_mask import skin_mask
        from colordoctor.ui.mask_editor import MaskEditorDialog
        from colordoctor.ui.session import ai_skin_mask

        # “恢复自动”与分析时用同一种方式抠皮肤，并限制在套索之内
        auto = None
        if self.session.ai_seg:
            auto, _ = ai_skin_mask(item.region, item.context, rep.part or self.session.part)
        if auto is None:
            auto, _ = skin_mask(item.region, within=item.lasso)
        elif item.lasso is not None:
            auto = auto & item.lasso
        start = rep.valid_mask if rep.valid_mask is not None else np.ones(item.region.shape[:2], bool)
        dlg = MaskEditorDialog(self, item.region, start, auto)  # 蒙版太小时对话框自己提示、不关闭
        if dlg.exec():
            mask = dlg.mask.copy()
            # 按原结果的部位重算，不改界面上当前选的部位
            self._run_rect(item.region, item.rect, item.context, item.source_label, 0, replace=item, manual_mask=mask,
                           mode="skin", part=rep.part)

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
            self.statusBar().showMessage("请先完成一次分析或取色", 4000)
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出分析结果", "colordoctor_report.json", "JSON (*.json)")
        if not path:
            return
        from colordoctor.core import report_json as rj

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(rj.report_to_json(item.result))
        except OSError as e:
            QMessageBox.warning(self, "导出失败", f"无法写入文件：{path}\n（{e.strerror or e}）\n请换一个文件夹，或检查是否有写入权限。")
            return
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
        def ask_and_cleanup(prompt, cfg, image_path):
            try:
                return pv.ask(prompt, cfg, image_path)
            finally:
                if image_path:
                    import os

                    try:
                        os.remove(image_path)
                    except OSError:
                        pass

        def ai_failed(msg):
            # 不走 _failed：那会清掉正在进行的对比 / 参考区状态，AI 出错和它们无关
            self.ai_btn.setEnabled(True)
            self.statusBar().showMessage("AI 点评失败", 5000)
            QMessageBox.warning(self, "AI 点评失败", msg)

        self._start(ask_and_cleanup, prompt, cfg, image_path, on_done=self._ai_done, on_fail=ai_failed)

    def _ai_done(self, result):
        self.ai_btn.setEnabled(True)
        self.statusBar().showMessage("AI 点评完成" if result.ok else "AI 点评失败", 5000)
        dlg = QDialog(self)
        dlg.setWindowTitle("AI 点评" if result.ok else "AI 点评失败")
        dlg.resize(640, 720)
        view = QTextBrowser(dlg)
        from PyQt6.QtGui import QTextDocument

        view.document().setMarkdown(result.text, QTextDocument.MarkdownFeature.MarkdownNoHTML)
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
