"""在一张图像（冻结的屏幕截图或 JPG）上单击取点、拖动框选。

坐标全部换算为图像像素坐标后通过信号发出。带放大镜与十字准星，
可显示带标签的已选矩形（参考区 / 问题区）。
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from colordoctor.capture.geometry import view_to_image
from colordoctor.capture.screen_capture import array_to_qimage

DRAG_THRESHOLD = 4  # 逻辑像素


class ImageSelectWidget(QWidget):
    point_selected = pyqtSignal(int, int)
    rect_selected = pyqtSignal(int, int, int, int)
    cancelled = pyqtSignal()

    def __init__(self, parent=None, magnifier: bool = True, allow_zoom: bool = False):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.image: QImage | None = None
        self.arr: np.ndarray | None = None
        self.magnifier = magnifier
        self.allow_zoom = allow_zoom
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.instruction = ""
        self.marks: list[tuple[tuple[int, int, int, int], str]] = []
        self._press: QPointF | None = None
        self._cur: QPointF | None = None
        self._panning: QPointF | None = None
        self.radius = 2

    # ---------------------------------------------------------------- 数据
    def set_image(self, arr: np.ndarray):
        self.arr = np.ascontiguousarray(arr[..., :3])
        self.image = array_to_qimage(self.arr)
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.update()

    def add_mark(self, rect, label):
        self.marks.append((tuple(int(v) for v in rect), label))
        self.update()

    def clear_marks(self):
        self.marks.clear()
        self.update()

    # ---------------------------------------------------------------- 坐标
    def _transform(self):
        if self.image is None:
            return 1.0, 0.0, 0.0
        s, ox, oy = view_to_image(0, 0, self.width(), self.height(), self.image.width(), self.image.height())
        s *= self.zoom
        cx, cy = self.width() / 2, self.height() / 2
        ox = cx + (ox - cx) * self.zoom + self.pan.x()
        oy = cy + (oy - cy) * self.zoom + self.pan.y()
        return s, ox, oy

    def to_image(self, pos: QPointF) -> tuple[float, float]:
        s, ox, oy = self._transform()
        return (pos.x() - ox) / s, (pos.y() - oy) / s

    def to_view(self, x: float, y: float) -> QPointF:
        s, ox, oy = self._transform()
        return QPointF(ox + x * s, oy + y * s)

    # ---------------------------------------------------------------- 事件
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._press = ev.position()
            self._cur = ev.position()
        elif ev.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton) and self.allow_zoom:
            self._panning = ev.position()
        self.update()

    def mouseMoveEvent(self, ev):
        if self._panning is not None:
            self.pan += ev.position() - self._panning
            self._panning = ev.position()
        self._cur = ev.position()
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._panning is not None and ev.button() != Qt.MouseButton.LeftButton:
            self._panning = None
            return
        if ev.button() != Qt.MouseButton.LeftButton or self._press is None or self.image is None:
            return
        a, b = self._press, ev.position()
        self._press = None
        x0, y0 = self.to_image(a)
        x1, y1 = self.to_image(b)
        W, H = self.image.width(), self.image.height()
        if abs(b.x() - a.x()) < DRAG_THRESHOLD and abs(b.y() - a.y()) < DRAG_THRESHOLD:
            xi, yi = int(x1), int(y1)
            if 0 <= xi < W and 0 <= yi < H:
                self.point_selected.emit(xi, yi)
        else:
            lx, rx = sorted((x0, x1))
            ty, by = sorted((y0, y1))
            lx, ty = max(0, int(lx)), max(0, int(ty))
            rx, by = min(W, int(np.ceil(rx))), min(H, int(np.ceil(by)))
            if rx > lx and by > ty:
                self.rect_selected.emit(lx, ty, rx - lx, by - ty)
        self.update()

    def wheelEvent(self, ev):
        if not self.allow_zoom:
            return
        f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        before = self.to_image(ev.position())
        self.zoom = float(np.clip(self.zoom * f, 0.2, 40))
        after = self.to_view(*before)
        self.pan += ev.position() - after
        self.update()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
        elif ev.key() == Qt.Key.Key_0 and self.allow_zoom:
            self.zoom, self.pan = 1.0, QPointF(0, 0)
            self.update()
        else:
            super().keyPressEvent(ev)

    # ---------------------------------------------------------------- 绘制
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        if self.image is None:
            p.end()
            return
        s, ox, oy = self._transform()
        target = QRectF(ox, oy, self.image.width() * s, self.image.height() * s)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, s < 1)
        p.drawImage(target, self.image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        for rect, label in self.marks:
            x, y, w, h = rect
            tl = self.to_view(x, y)
            r = QRectF(tl.x(), tl.y(), w * s, h * s)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawRect(r)
            p.setPen(QPen(QColor(0, 0, 0), 1, Qt.PenStyle.DashLine))
            p.drawRect(r)
            self._label(p, r.topLeft() + QPointF(4, -6), label)

        if self._press is not None and self._cur is not None:
            r = QRectF(self._press, self._cur).normalized()
            if r.width() >= DRAG_THRESHOLD or r.height() >= DRAG_THRESHOLD:
                p.setPen(QPen(QColor(255, 255, 255), 1.5))
                p.drawRect(r)
                p.setPen(QPen(QColor(0, 0, 0), 1, Qt.PenStyle.DashLine))
                p.drawRect(r)
                x0, y0 = self.to_image(r.topLeft())
                x1, y1 = self.to_image(r.bottomRight())
                self._label(p, r.bottomRight() + QPointF(6, 14), f"{int(x1 - x0)}×{int(y1 - y0)}")

        if self._cur is not None and self.magnifier:
            self._draw_magnifier(p, self._cur)
        if self.instruction:
            f = QFont(self.font())
            f.setPointSize(12)
            p.setFont(f)
            r = QRectF(0, 8, self.width(), 30)
            p.fillRect(QRectF(self.width() / 2 - 260, 6, 520, 32), QColor(0, 0, 0, 170))
            p.setPen(QColor(240, 240, 240))
            p.drawText(r, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, self.instruction)
        p.end()

    def _label(self, p, pos, text):
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 8
        p.fillRect(QRectF(pos.x() - 4, pos.y() - fm.ascent() - 2, w, fm.height() + 4), QColor(0, 0, 0, 170))
        p.setPen(QColor(255, 255, 255))
        p.drawText(pos, text)

    def _draw_magnifier(self, p, pos: QPointF):
        if self.arr is None:
            return
        x, y = self.to_image(pos)
        xi, yi = int(x), int(y)
        H, W = self.arr.shape[:2]
        if not (0 <= xi < W and 0 <= yi < H):
            return
        n, cell = 11, 11
        half = n // 2
        size = n * cell
        mx = pos.x() + 24
        my = pos.y() + 24
        if mx + size > self.width():
            mx = pos.x() - 24 - size
        if my + size + 40 > self.height():
            my = pos.y() - 24 - size - 40
        for j in range(n):
            for i in range(n):
                px, py = xi - half + i, yi - half + j
                if 0 <= px < W and 0 <= py < H:
                    r, g, b = (int(v) for v in self.arr[py, px])
                    p.fillRect(QRectF(mx + i * cell, my + j * cell, cell, cell), QColor(r, g, b))
        # 取样框（半径）
        rr = self.radius
        p.setPen(QPen(QColor(255, 255, 255), 1.5))
        p.drawRect(QRectF(mx + (half - rr) * cell, my + (half - rr) * cell, (2 * rr + 1) * cell, (2 * rr + 1) * cell))
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.drawRect(QRectF(mx, my, size, size))
        r, g, b = (int(v) for v in self.arr[yi, xi])
        self._label(p, QPointF(mx + 4, my + size + 16), f"x {xi}, y {yi}  RGB {r},{g},{b}")
