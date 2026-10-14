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
        self.on_close = None  # 由 CaptureSession 设置：被系统 / Alt+F4 关掉时取消整个会话

    def closeEvent(self, ev):
        cb, self.on_close = self.on_close, None
        super().closeEvent(ev)
        if cb is not None:
            cb()


class CaptureSession(QObject):
    """管理所有显示器上的选择层。

    steps：每一步的提示文字。参考对比模式为两步（参考区、问题区），其他模式一步。
    结果信号：(屏幕截图, 类型 'point'/'rect', 坐标元组, 步骤序号)
    """

    selected = pyqtSignal(object, str, tuple, int)
    radius_changed = pyqtSignal(int)
    finished = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(
        self, captures: list[ScreenCapture], steps: list[str], radius: int = 2, allow_point: bool = True,
        lasso: bool = False, labels: list[str] | None = None, multi_point: bool = False,
    ):
        super().__init__()
        self.captures = captures
        self.steps = steps
        self.step = 0
        self.allow_point = allow_point
        self.multi_point = multi_point  # 单点模式：可以连续点多个，回车或 Esc 结束
        self.points = 0
        self.labels = labels or (["参考区", "问题区"] if len(steps) > 1 else ["选区"])
        self.overlays: list[_ScreenOverlay] = []
        for i, cap in enumerate(captures):
            ov = _ScreenOverlay(cap, i)
            ov.view.radius = radius
            ov.view.lasso_mode = lasso
            ov.view.allow_point = allow_point
            ov.view.radius_changed.connect(self._sync_radius)
            ov.view.polygon_selected.connect(lambda pts, i=i: self._on_polygon(i, pts))
            ov.view.point_selected.connect(lambda x, y, i=i: self._on_point(i, x, y))
            ov.view.rect_selected.connect(lambda x, y, w, h, i=i: self._on_rect(i, x, y, w, h))
            ov.view.cancelled.connect(self._esc)
            ov.view.finished.connect(self._enter)
            ov.on_close = self._overlay_closed
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
        if self.multi_point:
            text = (f"已取 {self.points} 个点，继续单击取点；回车或 Esc 结束（两个点以上会自动对比）" if self.points
                    else "单击取点，可以连续点多个；拖动则取框内平均")
        for ov in self.overlays:
            ov.view.instruction = text + ("" if self.multi_point else "   （Esc 取消）")
            ov.view.update()

    def _on_point(self, i, x, y):
        if not self.allow_point:
            return
        if self.multi_point:
            self.points += 1
            self.selected.emit(self.captures[i], "point", (x, y), self.step)
            self.overlays[i].view.add_mark((x, y, 1, 1), f"点 {self.points}")
            self._update_text()
            return
        self.selected.emit(self.captures[i], "point", (x, y), self.step)
        self._advance(i, (x, y, 1, 1))

    def _esc(self):
        # 连续取点时 Esc 也是“结束”（已经取的点保留），否则是取消
        if self.multi_point and self.points:
            self._enter()
        else:
            self.cancel()

    def _enter(self):
        if self.multi_point and self.points:
            self.close()
            self.finished.emit()

    def _sync_radius(self, r):
        for ov in self.overlays:
            ov.view.radius = r
            ov.view.update()
        self.radius_changed.emit(r)

    def _on_polygon(self, i, pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.selected.emit(self.captures[i], "polygon", tuple(pts), self.step)
        self._advance(i, (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)) + 1, int(max(ys) - min(ys)) + 1))

    def _on_rect(self, i, x, y, w, h):
        self.selected.emit(self.captures[i], "rect", (x, y, w, h), self.step)
        if self.multi_point:  # 连续取点时拖框 = 一个“框内平均”的点，继续取
            self.points += 1
            self.overlays[i].view.add_mark((x, y, w, h), f"点 {self.points}")
            self._update_text()
            return
        self._advance(i, (x, y, w, h))

    def _advance(self, i, rect):
        label = self.labels[self.step] if self.step < len(self.labels) else "选区"
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

    def _overlay_closed(self):
        if self.overlays:  # 不是由 close() 主动关闭的
            self.cancel()

    def close(self):
        overlays, self.overlays = self.overlays, []
        for ov in overlays:
            ov.on_close = None
            ov.close()
            ov.deleteLater()
        self.overlays = []
