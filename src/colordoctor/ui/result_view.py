"""“问题位置”视图：显示被分析的区域，并标出杂色像素、发灰位置和编号框。"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from colordoctor.core import color_spaces as cs
from colordoctor.core import contamination as ct_mod
from colordoctor.core.color_models import AnalysisReport
from colordoctor.ui import theme
from colordoctor.ui.selection_widget import ImageSelectWidget


PREVIEW_PIXELS = 4_000_000  # “问题位置”按这么多像素显示（屏幕上也看不出更多）：大图不用全分辨率去算标记


def preview_stride(shape) -> int:
    """显示用的抽样步长：45 MP 的图按 4 MP 左右显示。"""
    h, w = shape[:2]
    return max(1, int(np.ceil(np.sqrt(h * w / PREVIEW_PIXELS))))


def overlay_image(region: np.ndarray, report: AnalysisReport, dim_normal: bool = True, stride: int = 1) -> np.ndarray:
    """生成叠加标记后的 RGB 图像。stride > 1 时按 region[::stride, ::stride] 的尺寸生成（显示用，不必全分辨率）。"""
    base = region[::stride, ::stride, :3]
    vm = report.valid_mask
    vm = vm[::stride, ::stride] if vm is not None and vm.shape == region.shape[:2] else None
    out = base.astype(np.float32)
    ct = report.contamination
    if ct is None or ct.mask is None:
        if vm is not None:
            out[~vm] *= 0.25
            return np.clip(out, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(base)
    m = ct.mask
    step = int(round(ct.mask_scale))
    # 网格降采样的第 i 个样本代表 [i*step, (i+1)*step) 这一格：按显示像素在原图里的位置查它属于哪一格
    ys = np.arange(base.shape[0]) * stride // step
    xs = np.arange(base.shape[1]) * stride // step
    full = np.full(base.shape[:2], ct_mod.MASK_IGNORED, np.uint8)
    iy, ix = ys < m.shape[0], xs < m.shape[1]
    full[np.ix_(iy, ix)] = m[np.ix_(ys[iy], xs[ix])]
    opp = full == ct_mod.MASK_OPPOSITE
    gray = full == ct_mod.MASK_GRAYING
    outside = full == ct_mod.MASK_OUTSIDE
    if vm is not None:
        outside = ~vm  # 全分辨率的皮肤蒙版更准确
    if outside.any():
        out[outside] *= 0.25  # 不计入分析的像素（非皮肤/套索外）
    if dim_normal and (opp.any() or gray.any()):
        dim = ~(opp | gray)
        a = theme.OVERLAY_DIM[3] / 255
        out[dim] *= (1 - a)
    if opp.any():
        # 用与杂色色相相反、明亮饱和的颜色标出，保证看得见
        hue = ct.opposite_hue if ct.opposite_hue is not None else 0.0
        mark_h = (hue + 180) % 360
        mark = cs.lab_to_srgb8(cs.lch_to_lab(np.array([70.0, 70.0, mark_h]))).astype(np.float32)
        out[opp] = 0.25 * out[opp] + 0.75 * mark
    if gray.any():
        r, g, b, a = theme.OVERLAY_GRAYING
        out[gray] = out[gray] * (1 - a / 255) + np.array([r, g, b]) * (a / 255)
    return np.clip(out, 0, 255).astype(np.uint8)


class ResultView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = ImageSelectWidget(self, magnifier=False, allow_zoom=True)
        self.view.setCursor(Qt.CursorShape.ArrowCursor)
        self.view.select_enabled = False  # 只用来查看问题位置，左键框选没有用途
        self.toggle = QCheckBox("显示标记")
        self.toggle.setChecked(True)
        self.toggle.toggled.connect(self._refresh)
        self.legend = QLabel("")
        self.legend.setWordWrap(True)
        top = QHBoxLayout()
        top.addWidget(self.toggle)
        top.addWidget(self.legend, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)
        self.region = None
        self.report = None
        self.view.instruction = ""
        self.legend.setText("滚轮缩放，右键拖动平移，数字键 0 复位")

    def show_report(self, region: np.ndarray, report: AnalysisReport):
        self.region, self.report = region, report
        self._refresh()

    def clear(self):
        self.region = self.report = None
        self.view.image = None
        self.view.clear_marks()
        self.view.update()

    def _refresh(self):
        if self.region is None or self.report is None:
            return
        f = preview_stride(self.region.shape)  # 大图按显示分辨率生成，不在界面线程上算几千万像素
        img = overlay_image(self.region, self.report, stride=f) if self.toggle.isChecked() else self.region[::f, ::f]
        self.view.set_image(img)
        self.view.clear_marks()

        def mark(r, label):  # 框的坐标跟着显示尺寸缩小
            self.view.add_mark(tuple(v / f for v in r), label)

        ct = self.report.contamination
        if ct is None:
            if self.report.palette is not None and self.report.palette.unity is not None and self.toggle.isChecked():
                ox, oy = (self.report.region or (0, 0, 0, 0))[:2]
                for i, p in enumerate(self.report.palette.unity.patches, 1):
                    x, y, w, h = p.bbox
                    mark((x - ox, y - oy, w, h), f"{i} {p.note}")
                self.legend.setText("框出的是颜色不统一的位置。滚轮缩放，右键拖动平移，0 复位")
            else:
                self.legend.setText("此结果没有位置信息")
            return
        ox, oy = (self.report.region or (0, 0, 0, 0))[:2]
        i = 0
        if self.toggle.isChecked():
            for b in ct.blobs:
                i += 1
                x, y, w, h = b.bbox
                kind = f"“{b.contaminant_hue_name}”杂色" if b.kind == "opposite" else "发灰"
                mark((x - ox, y - oy, w, h), f"{i} {kind}")
            uni = self.report.uniformity
            for p in (uni.patches if uni is not None else ()):
                i += 1
                x, y, w, h = p.bbox
                mark((x - ox, y - oy, w, h), f"{i} 肤色{p.kind}")
            pal = self.report.palette
            if pal is not None and pal.unity is not None:
                for p in pal.unity.patches:
                    i += 1
                    x, y, w, h = p.bbox
                    mark((x - ox, y - oy, w, h), f"{i} {p.note}")
        n_o = sum(1 for b in ct.blobs if b.kind == "opposite")
        n_g = sum(1 for b in ct.blobs if b.kind == "graying")
        n_u = len(self.report.uniformity.patches) if self.report.uniformity is not None else 0
        extra = f"，肤色不均匀 {n_u} 处（编号框）；很暗的区域是未计入的非皮肤像素" if self.report.uniformity is not None else ""
        self.legend.setText(
            f"杂色像素用对比色高亮，发灰处用白色覆盖，其余区域压暗。对立色 {n_o} 处，发灰 {n_g} 处{extra}。"
            "滚轮缩放，右键拖动平移，0 复位"
        )
