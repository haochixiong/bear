"""色轮：CIELab a*b* 平面（角度 = 色相 h°，半径 = 色度 C*）。

显示：选区像素散点、取样点、肤色风格目标扇区、参考区→问题区箭头、对立色扇区。
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QToolTip, QWidget

from colordoctor.core import color_spaces as cs
from colordoctor.core.color_models import SkinPreset
from colordoctor.ui import theme

MAX_POINTS = 4000


class ColorWheel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.c_max = 80.0
        self.scatter_ab: np.ndarray | None = None
        self.scatter_rgb: np.ndarray | None = None
        self.point: tuple[np.ndarray, tuple[int, int, int]] | None = None
        self.preset: SkinPreset | None = None
        self.ref_lab = None
        self.prob_lab = None
        self.opposite_of: float | None = None  # 检测到的杂色色相（扇区中心）
        self.emphasis = None  # (主色相, 最小角度差, 最低色度)：满足条件的散点加粗显示
        self._ring = self._ring_colors()

    # ---------------------------------------------------------------- 数据
    @staticmethod
    def _ring_colors():
        hs = np.arange(0, 360, 3.0)
        lab = np.stack([np.full_like(hs, 62.0), 38 * np.cos(np.radians(hs)), 38 * np.sin(np.radians(hs))], axis=1)
        return list(zip(hs, cs.lab_to_srgb8(lab)))

    def clear(self):
        self.scatter_ab = self.scatter_rgb = None
        self.point = None
        self.ref_lab = self.prob_lab = None
        self.opposite_of = None
        self.update()

    def set_scatter(self, rgb_pixels: np.ndarray | None, emphasis=None):
        self.emphasis = emphasis
        if rgb_pixels is None or rgb_pixels.size == 0:
            self.scatter_ab = self.scatter_rgb = None
        else:
            px = rgb_pixels.reshape(-1, 3)
            if px.shape[0] > MAX_POINTS:
                step = int(math.ceil(px.shape[0] / MAX_POINTS))
                px = px[step // 2 :: step]
            lab = cs.srgb8_to_lab(px)
            self.scatter_ab = lab[:, 1:3]
            self.scatter_rgb = px
            c = float(np.percentile(np.hypot(lab[:, 1], lab[:, 2]), 99))
            self.c_max = 40.0 if c < 30 else (80.0 if c < 70 else 120.0)
        self.update()

    def set_point(self, lab, rgb):
        self.point = (np.asarray(lab, float), tuple(int(v) for v in rgb))
        if self.scatter_ab is None:
            c = math.hypot(lab[1], lab[2])
            self.c_max = 40.0 if c < 30 else (80.0 if c < 70 else 120.0)
        self.update()

    def set_preset(self, preset: SkinPreset | None):
        self.preset = preset
        self.update()

    def set_compare(self, ref_lab, prob_lab):
        self.ref_lab = None if ref_lab is None else np.asarray(ref_lab, float)
        self.prob_lab = None if prob_lab is None else np.asarray(prob_lab, float)
        self.update()

    def set_opposite(self, contaminant_hue: float | None):
        self.opposite_of = contaminant_hue
        self.update()

    # ---------------------------------------------------------------- 坐标
    def _geom(self):
        side = min(self.width(), self.height()) - 36
        cx, cy = self.width() / 2, self.height() / 2
        return cx, cy, max(10.0, side / 2)

    def to_xy(self, a, b):
        cx, cy, R = self._geom()
        k = R / self.c_max
        return QPointF(cx + a * k, cy - b * k)

    def from_xy(self, x, y):
        cx, cy, R = self._geom()
        k = R / self.c_max
        return (x - cx) / k, (cy - y) / k

    # ---------------------------------------------------------------- 绘制
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.WHEEL_BG))
        cx, cy, R = self._geom()
        # 色相环（细，只作方向参考）
        ring_w = 6
        for h, rgb in self._ring:
            p.setPen(QPen(QColor(*[int(v) for v in rgb]), ring_w, cap=Qt.PenCapStyle.FlatCap))
            rect = QRectF(cx - R - ring_w, cy - R - ring_w, 2 * (R + ring_w), 2 * (R + ring_w))
            p.drawArc(rect, int(h * 16), int(3.2 * 16))
        # 网格
        grid = QPen(QColor(theme.WHEEL_GRID), 1)
        p.setPen(grid)
        step = 10 if self.c_max <= 40 else 20
        c = step
        while c <= self.c_max + 1e-6:
            r = R * c / self.c_max
            p.drawEllipse(QPointF(cx, cy), r, r)
            c += step
        p.drawLine(QPointF(cx - R, cy), QPointF(cx + R, cy))
        p.drawLine(QPointF(cx, cy - R), QPointF(cx, cy + R))
        p.setPen(QColor(theme.TEXT_DIM))
        f = QFont(self.font())
        f.setPointSize(8)
        p.setFont(f)
        p.drawText(QPointF(cx + R - 34, cy - 4), "+a* 红")
        p.drawText(QPointF(cx - R + 2, cy - 4), "−a* 绿")
        p.drawText(QPointF(cx + 4, cy - R + 12), "+b* 黄")
        p.drawText(QPointF(cx + 4, cy + R - 4), "−b* 蓝")
        p.drawText(QPointF(4, self.height() - 6), f"C* 刻度 {step}/格")

        if self.preset is not None:
            self._draw_sector(p, self.preset.hue, self.preset.chroma, QColor(255, 255, 255, 40), QColor(255, 255, 255, 150))
            mid = (self.preset.hue[0] + self.preset.hue[1]) / 2
            pos = self.to_xy(
                self.preset.chroma[1] * math.cos(math.radians(mid)), self.preset.chroma[1] * math.sin(math.radians(mid))
            )
            p.setPen(QColor(230, 230, 230))
            p.drawText(pos + QPointF(6, -4), f"目标：{self.preset.name}")
        if self.opposite_of is not None:
            oh = self.opposite_of
            pen = QColor(230, 120, 120, 200)
            self._draw_sector(p, (oh - 25, oh + 25), (5, self.c_max), QColor(230, 120, 120, 28), pen, dashed=True)
            pos = self.to_xy(self.c_max * 0.9 * math.cos(math.radians(oh)), self.c_max * 0.9 * math.sin(math.radians(oh)))
            p.setPen(QColor(235, 170, 170))
            p.drawText(pos + QPointF(-14, 4), "杂色")

        if self.scatter_ab is not None:
            p.setPen(Qt.PenStyle.NoPen)
            flagged = np.zeros(len(self.scatter_ab), dtype=bool)
            if self.emphasis is not None:
                base, ang, cthr = self.emphasis
                C = np.hypot(self.scatter_ab[:, 0], self.scatter_ab[:, 1])
                h = np.degrees(np.arctan2(self.scatter_ab[:, 1], self.scatter_ab[:, 0])) % 360
                flagged = (np.abs(cs.hue_diff(base, h)) >= ang) & (C >= cthr)
            for (a, b), rgb in zip(self.scatter_ab[~flagged], self.scatter_rgb[~flagged]):
                p.setBrush(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
                p.drawEllipse(self.to_xy(a, b), 1.6, 1.6)
            p.setPen(QPen(QColor(255, 255, 255), 1))
            for (a, b), rgb in zip(self.scatter_ab[flagged], self.scatter_rgb[flagged]):
                p.setBrush(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
                p.drawEllipse(self.to_xy(a, b), 3.2, 3.2)
            p.setPen(Qt.PenStyle.NoPen)

        if self.point is not None:
            lab, rgb = self.point
            pt = self.to_xy(lab[1], lab[2])
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.setBrush(QColor(*rgb))
            p.drawEllipse(pt, 7, 7)
            p.setPen(QPen(QColor(255, 255, 255, 90), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(cx, cy), pt)

        if self.ref_lab is not None:
            pr = self.to_xy(self.ref_lab[1], self.ref_lab[2])
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(pr, 8, 8)
            p.drawText(pr + QPointF(10, 14), "参考")
            if self.prob_lab is not None:
                pp = self.to_xy(self.prob_lab[1], self.prob_lab[2])
                tri = QPolygonF([pp + QPointF(0, -9), pp + QPointF(8, 6), pp + QPointF(-8, 6)])
                p.drawPolygon(tri)
                p.drawText(pp + QPointF(10, 14), "问题")
                self._arrow(p, pr, pp)
        p.end()

    def _draw_sector(self, p, hue_rng, c_rng, fill, edge, dashed=False):
        h0, h1 = hue_rng
        c0, c1 = c_rng
        path = QPainterPath()
        steps = 24
        pts = []
        for i in range(steps + 1):
            h = math.radians(h0 + (h1 - h0) * i / steps)
            pts.append(self.to_xy(c1 * math.cos(h), c1 * math.sin(h)))
        for i in range(steps, -1, -1):
            h = math.radians(h0 + (h1 - h0) * i / steps)
            pts.append(self.to_xy(c0 * math.cos(h), c0 * math.sin(h)))
        path.addPolygon(QPolygonF(pts))
        path.closeSubpath()
        pen = QPen(edge, 1.2, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        p.setPen(pen)
        p.setBrush(QBrush(fill))
        p.drawPath(path)

    @staticmethod
    def _arrow(p, a: QPointF, b: QPointF):
        p.setPen(QPen(QColor(255, 255, 255, 200), 1.5))
        p.drawLine(a, b)
        ang = math.atan2(b.y() - a.y(), b.x() - a.x())
        for d in (2.6, -2.6):
            p.drawLine(b, b - QPointF(10 * math.cos(ang + d / 6), 10 * math.sin(ang + d / 6)))

    def mouseMoveEvent(self, ev):
        a, b = self.from_xy(ev.position().x(), ev.position().y())
        C = math.hypot(a, b)
        h = math.degrees(math.atan2(b, a)) % 360
        QToolTip.showText(ev.globalPosition().toPoint(), f"a* {a:.1f}  b* {b:.1f}\nC* {C:.1f}  h {h:.0f}°", self)
