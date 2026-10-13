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


def overlay_image(region: np.ndarray, report: AnalysisReport, dim_normal: bool = True) -> np.ndarray:
    """生成叠加标记后的 RGB 图像（与 region 同尺寸）。"""
    out = region[..., :3].astype(np.float32).copy()
    ct = report.contamination
    if ct is None or ct.mask is None:
        if report.valid_mask is not None and report.valid_mask.shape == region.shape[:2]:
            out[~report.valid_mask] *= 0.25
            return np.clip(out, 0, 255).astype(np.uint8)
        return region
    m = ct.mask
    step = int(round(ct.mask_scale))
    big = np.repeat(np.repeat(m, step, axis=0), step, axis=1)
    # 网格降采样的第 i 个样本位于 i*step + step//2，正好代表 [i*step, (i+1)*step) 这一格
    full = np.full(region.shape[:2], ct_mod.MASK_IGNORED, np.uint8)
    h = min(full.shape[0], big.shape[0])
    w = min(full.shape[1], big.shape[1])
    full[:h, :w] = big[:h, :w]
    opp = full == ct_mod.MASK_OPPOSITE
    gray = full == ct_mod.MASK_GRAYING
    outside = full == ct_mod.MASK_OUTSIDE
    if report.valid_mask is not None and report.valid_mask.shape == region.shape[:2]:
        outside = ~report.valid_mask  # 全分辨率的皮肤蒙版更准确
    if outside.any():
        out[outside] = out[outside] * 0.25  # 不计入分析的像素（非皮肤/套索外）
    if dim_normal and (opp.any() or gray.any()):
        dim = ~(opp | gray)
        a = theme.OVERLAY_DIM[3] / 255
        out[dim] = out[dim] * (1 - a)
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
        img = overlay_image(self.region, self.report) if self.toggle.isChecked() else self.region
        self.view.set_image(img)
        self.view.clear_marks()
        ct = self.report.contamination
        if ct is None:
            if self.report.palette is not None and self.report.palette.unity is not None and self.toggle.isChecked():
                ox, oy = (self.report.region or (0, 0, 0, 0))[:2]
                for i, p in enumerate(self.report.palette.unity.patches, 1):
                    x, y, w, h = p.bbox
                    self.view.add_mark((x - ox, y - oy, w, h), f"{i} {p.note}")
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
                self.view.add_mark((x - ox, y - oy, w, h), f"{i} {kind}")
            uni = self.report.uniformity
            for p in (uni.patches if uni is not None else ()):
                i += 1
                x, y, w, h = p.bbox
                self.view.add_mark((x - ox, y - oy, w, h), f"{i} 肤色{p.kind}")
            pal = self.report.palette
            if pal is not None and pal.unity is not None:
                for p in pal.unity.patches:
                    i += 1
                    x, y, w, h = p.bbox
                    self.view.add_mark((x - ox, y - oy, w, h), f"{i} {p.note}")
        n_o = sum(1 for b in ct.blobs if b.kind == "opposite")
        n_g = sum(1 for b in ct.blobs if b.kind == "graying")
        n_u = len(self.report.uniformity.patches) if self.report.uniformity is not None else 0
        extra = f"，肤色不均匀 {n_u} 处（编号框）；很暗的区域是未计入的非皮肤像素" if self.report.uniformity is not None else ""
        self.legend.setText(
            f"杂色像素用对比色高亮，发灰处用白色覆盖，其余区域压暗。对立色 {n_o} 处，发灰 {n_g} 处{extra}。"
            "滚轮缩放，右键拖动平移，0 复位"
        )
