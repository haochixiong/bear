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

from colordoctor.capture.screen_capture import array_to_qimage
from colordoctor.core import mask_edit as me
from colordoctor.ui.selection_widget import ImageSelectWidget

EDIT_PIXELS = 2_000_000  # 编辑用的工作分辨率上限；大图在缩小的副本上编辑，确定时再放大回原尺寸

TOOLS = {
    "brush_add": "画笔添加",
    "brush_erase": "画笔擦除",
    "wand_add": "魔棒添加",
    "wand_erase": "魔棒删除",
}


def work_step(h: int, w: int) -> int:
    return max(1, int(np.ceil(np.sqrt(h * w / EDIT_PIXELS))))


def upscale(small: np.ndarray, step: int, shape: tuple[int, int]) -> np.ndarray:
    if step == 1:
        return small
    return np.repeat(np.repeat(small, step, axis=0), step, axis=1)[: shape[0], : shape[1]]


def mask_preview(region: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """皮肤保持原样；非皮肤压暗并带红色，方便看清蒙版范围。"""
    out = region[..., :3].astype(np.float32).copy()
    off = ~mask
    out[off] = out[off] * 0.35 + np.array([120.0, 20.0, 20.0]) * 0.65
    return np.clip(out, 0, 255).astype(np.uint8)


class MaskCanvas(ImageSelectWidget):
    """在不超过 EDIT_PIXELS 的工作副本上编辑；每次落笔只重画笔刷经过的那一小块。"""

    def __init__(self, region: np.ndarray, mask: np.ndarray, parent=None):
        super().__init__(parent, magnifier=False, allow_zoom=True)
        self.step = work_step(*region.shape[:2])
        k = self.step
        self.region = np.ascontiguousarray(region[::k, ::k, :3])
        self._lit = self.region.astype(np.uint8)
        self._dark = mask_preview(self.region, np.zeros(self.region.shape[:2], bool))
        self.full_shape = tuple(mask.shape[:2])
        # base：原尺寸的起始蒙版（打开时的蒙版或“恢复自动”）。确定时没改动的地方原样沿用 base，保留原精度
        self.base = mask.astype(bool)
        self.base_small = self.base[::k, ::k].copy()
        self.history = me.MaskHistory(self.base_small)
        self._bases: list[tuple[np.ndarray, np.ndarray]] = []
        self.tool = "brush_add"
        self.brush = 12.0
        self.tolerance = 8.0
        self._stroke: list[tuple[float, float]] = []
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.set_image(self._compose())

    @property
    def mask(self) -> np.ndarray:
        """工作分辨率下的蒙版。"""
        return self.history.mask

    def push(self):
        self.history.push()
        self._bases.append((self.base, self.base_small))
        self._bases = self._bases[-self.history.limit :]

    def undo(self) -> bool:
        if not self.history.undo():
            return False
        if self._bases:
            self.base, self.base_small = self._bases.pop()
        self.refresh()
        return True

    def reset_to(self, full: np.ndarray):
        self.push()
        k = self.step
        self.base = full.astype(bool)
        self.base_small = self.base[::k, ::k].copy()
        self.history.mask = self.base_small.copy()
        self.refresh()

    def full_mask(self) -> np.ndarray:
        """原尺寸蒙版：改动过的地方取编辑结果（放大），没改动的地方保持 base 原样。"""
        if self.step == 1:
            return self.mask.copy()
        changed = self.mask != self.base_small
        out = self.base.copy()
        if changed.any():
            ch = upscale(changed, self.step, self.full_shape)
            out[ch] = upscale(self.mask, self.step, self.full_shape)[ch]
        return out

    def _compose(self, y0: int = 0, y1: int | None = None, x0: int = 0, x1: int | None = None) -> np.ndarray:
        m = self.mask[y0:y1, x0:x1, None]
        return np.where(m, self._lit[y0:y1, x0:x1], self._dark[y0:y1, x0:x1])

    def refresh(self, box: tuple[float, float, float, float] | None = None):
        """box = (x0, y0, x1, y1) 时只重画这一块，否则整幅重画（不改变缩放和平移）。"""
        if box is None or self.image is None:
            self.arr = np.ascontiguousarray(self._compose())
            self.image = array_to_qimage(self.arr)
        else:
            h, w = self.mask.shape
            x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
            x1, y1 = min(w, int(np.ceil(box[2])) + 1), min(h, int(np.ceil(box[3])) + 1)
            if x0 >= x1 or y0 >= y1:
                return
            patch = np.ascontiguousarray(self._compose(y0, y1, x0, x1))
            self.arr[y0:y1, x0:x1] = patch
            p = QPainter(self.image)
            p.drawImage(x0, y0, array_to_qimage(patch))
            p.end()
        self.update()

    def _brush_box(self, ax, ay, bx, by):
        r = self.brush + 2
        return (min(ax, bx) - r, min(ay, by) - r, max(ax, bx) + r, max(ay, by) + r)

    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(ev)
        x, y = self.to_image(ev.position())
        self.push()
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
            self.refresh(self._brush_box(x, y, x, y))

    def mouseMoveEvent(self, ev):
        self._cur = ev.position()
        if self._stroke and ev.buttons() & Qt.MouseButton.LeftButton:
            x, y = self.to_image(ev.position())
            px, py = self._stroke[-1]
            me.stroke(self.history.mask, [(px, py), (x, y)], self.brush, self.tool == "brush_add")
            self._stroke.append((x, y))
            self.refresh(self._brush_box(px, py, x, y))
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
        self.canvas.undo()

    def _reset(self):
        self.canvas.reset_to(self.auto_mask)

    @property
    def mask(self) -> np.ndarray:
        """原尺寸（与传入的 region 一样大）的蒙版。"""
        return self.canvas.full_mask()
