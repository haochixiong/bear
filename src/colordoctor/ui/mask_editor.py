"""皮肤蒙版编辑：自动抠出的皮肤多了或少了时，用画笔/魔棒手动添加或删除。"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
)

from colordoctor.core import mask_edit as me
from colordoctor.ui.selection_widget import ImageSelectWidget

TOOLS = {
    "brush_add": "画笔添加",
    "brush_erase": "画笔擦除",
    "wand_add": "魔棒添加",
    "wand_erase": "魔棒删除",
}


def mask_preview(region: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """皮肤保持原样；非皮肤压暗并带红色，方便看清蒙版范围。"""
    out = region[..., :3].astype(np.float32).copy()
    off = ~mask
    out[off] = out[off] * 0.35 + np.array([120.0, 20.0, 20.0]) * 0.65
    return np.clip(out, 0, 255).astype(np.uint8)


class MaskCanvas(ImageSelectWidget):
    def __init__(self, region: np.ndarray, mask: np.ndarray, parent=None):
        super().__init__(parent, magnifier=False, allow_zoom=True)
        self.region = region
        self.history = me.MaskHistory(mask.astype(bool))
        self.tool = "brush_add"
        self.brush = 12.0
        self.tolerance = 8.0
        self._stroke: list[tuple[float, float]] = []
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.refresh()

    @property
    def mask(self) -> np.ndarray:
        return self.history.mask

    def refresh(self):
        z, pan = self.zoom, self.pan
        self.set_image(mask_preview(self.region, self.mask))
        self.zoom, self.pan = z, pan
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(ev)
        x, y = self.to_image(ev.position())
        self.history.push()
        if self.tool.startswith("wand"):
            region = me.magic_wand(self.region, int(x), int(y), self.tolerance)
            if self.tool == "wand_add":
                self.history.mask |= region
            else:
                self.history.mask &= ~region
            self.refresh()
        else:
            self._stroke = [(x, y)]
            me.paint(self.history.mask, x, y, self.brush, self.tool == "brush_add")
            self.refresh()

    def mouseMoveEvent(self, ev):
        self._cur = ev.position()
        if self._stroke and ev.buttons() & Qt.MouseButton.LeftButton:
            x, y = self.to_image(ev.position())
            me.stroke(self.history.mask, [self._stroke[-1], (x, y)], self.brush, self.tool == "brush_add")
            self._stroke.append((x, y))
            self.refresh()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._stroke = []
            return
        super().mouseReleaseEvent(ev)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._cur is None or self.tool.startswith("wand"):
            return
        s, _, _ = self._transform()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawEllipse(QPointF(self._cur), self.brush * s, self.brush * s)
        p.end()


class MaskEditorDialog(QDialog):
    def __init__(self, parent, region: np.ndarray, mask: np.ndarray, auto_mask: np.ndarray | None = None):
        super().__init__(parent)
        self.setWindowTitle("编辑皮肤蒙版（红色 = 不计入皮肤）")
        self.resize(1000, 760)
        self.auto_mask = auto_mask
        self.canvas = MaskCanvas(region, mask, self)
        tools = QHBoxLayout()
        group = QButtonGroup(self)
        for key, name in TOOLS.items():
            b = QToolButton()
            b.setText(name)
            b.setCheckable(True)
            b.setChecked(key == "brush_add")
            b.clicked.connect(lambda _c, k=key: setattr(self.canvas, "tool", k))
            group.addButton(b)
            tools.addWidget(b)
        tools.addWidget(QLabel("画笔"))
        self.brush = QSlider(Qt.Orientation.Horizontal)
        self.brush.setRange(2, 120)
        self.brush.setValue(12)
        self.brush.setMinimumWidth(110)
        self.brush.valueChanged.connect(lambda v: setattr(self.canvas, "brush", float(v)))
        tools.addWidget(self.brush)
        tools.addWidget(QLabel("魔棒容差"))
        self.tol = QSlider(Qt.Orientation.Horizontal)
        self.tol.setRange(2, 30)
        self.tol.setValue(8)
        self.tol.setMinimumWidth(90)
        self.tol.valueChanged.connect(lambda v: setattr(self.canvas, "tolerance", float(v)))
        tools.addWidget(self.tol)
        undo = QPushButton("撤销")
        undo.clicked.connect(self._undo)
        tools.addWidget(undo)
        reset = QPushButton("恢复自动")
        reset.setEnabled(auto_mask is not None)
        reset.clicked.connect(self._reset)
        tools.addWidget(reset)
        self.info = QLabel()
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("用这个蒙版重新分析")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(tools)
        lay.addWidget(self.canvas, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(self.info, 1)
        bottom.addWidget(bb)
        lay.addLayout(bottom)
        self.info.setText("左键涂抹/点击；滚轮缩放，右键拖动平移；Ctrl+Z 撤销")

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Z and ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._undo()
        else:
            super().keyPressEvent(ev)

    def _undo(self):
        if self.canvas.history.undo():
            self.canvas.refresh()

    def _reset(self):
        self.canvas.history.push()
        self.canvas.history.mask = self.auto_mask.copy()
        self.canvas.refresh()

    @property
    def mask(self) -> np.ndarray:
        return self.canvas.mask
