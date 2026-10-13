"""冻结屏幕后的全屏选择层（每块显示器一个窗口）。"""

from __future__ import annotations

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout

from colordoctor.capture.screen_capture import ScreenCapture
from colordoctor.ui.selection_widget import ImageSelectWidget


class _ScreenOverlay(QWidget):
    def __init__(self, capture: ScreenCapture, index: int):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.capture = capture
        self.index = index
        self.view = ImageSelectWidget(self, magnifier=True, allow_zoom=False)
        self.view.set_image(capture.pixels)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        info = capture.info
        self.setGeometry(info.x, info.y, info.width, info.height)


class CaptureSession(QObject):
    """管理所有显示器上的选择层。

    steps：每一步的提示文字。参考对比模式为两步（参考区、问题区），其他模式一步。
    结果信号：(屏幕截图, 类型 'point'/'rect', 坐标元组, 步骤序号)
    """

    selected = pyqtSignal(object, str, tuple, int)
    finished = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self, captures: list[ScreenCapture], steps: list[str], radius: int = 2, allow_point: bool = True):
        super().__init__()
        self.captures = captures
        self.steps = steps
        self.step = 0
        self.allow_point = allow_point
        self.overlays: list[_ScreenOverlay] = []
        for i, cap in enumerate(captures):
            ov = _ScreenOverlay(cap, i)
            ov.view.radius = radius
            ov.view.point_selected.connect(lambda x, y, i=i: self._on_point(i, x, y))
            ov.view.rect_selected.connect(lambda x, y, w, h, i=i: self._on_rect(i, x, y, w, h))
            ov.view.cancelled.connect(self.cancel)
            self.overlays.append(ov)
        self._update_text()

    def show(self):
        for ov in self.overlays:
            ov.show()
            ov.raise_()
            ov.activateWindow()
            ov.view.setFocus()

    def _update_text(self):
        text = self.steps[self.step] if self.step < len(self.steps) else ""
        for ov in self.overlays:
            ov.view.instruction = text + "   （Esc 取消）"
            ov.view.update()

    def _on_point(self, i, x, y):
        if not self.allow_point:
            return
        self.selected.emit(self.captures[i], "point", (x, y), self.step)
        self._advance(i, (x, y, 1, 1))

    def _on_rect(self, i, x, y, w, h):
        self.selected.emit(self.captures[i], "rect", (x, y, w, h), self.step)
        self._advance(i, (x, y, w, h))

    def _advance(self, i, rect):
        label = ["参考区", "问题区"][self.step] if len(self.steps) > 1 and self.step < 2 else "选区"
        self.overlays[i].view.add_mark(rect, label)
        self.step += 1
        if self.step >= len(self.steps):
            self.close()
            self.finished.emit()
        else:
            self._update_text()

    def cancel(self):
        self.close()
        self.cancelled.emit()

    def close(self):
        for ov in self.overlays:
            ov.close()
            ov.deleteLater()
        self.overlays = []
