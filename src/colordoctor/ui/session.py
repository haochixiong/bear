"""界面状态与分析调度（不依赖 Qt 控件，可单元测试）。"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from colordoctor.core import analysis as an
from colordoctor.core import color_spaces as cs
from colordoctor.core import skin_presets as sp
from colordoctor.core.color_models import AnalysisReport, ColorContext, PointSample, SkinPreset
from colordoctor.core.sampling import SampleError, clamp_rect, crop

MODES = {
    "point": "单点取色",
    "dirty": "脏色检查",
    "skin": "人像肤色",
    "compare": "参考对比",
    "palette": "整图色彩",
    "reference": "仿色对比",
}
PARTS = ("脸部", "身体", "手臂/腿", "其他")
FAMILIES = ("红", "橙", "黄", "绿", "青", "蓝", "紫/品红")
# 与 Photoshop 吸管“取样大小”一致：取样点、3×3、5×5、11×11、31×31、51×51、101×101 平均（默认 5×5）
RADIUS_CHOICES = [(0, "1×1"), (1, "3×3"), (2, "5×5"), (5, "11×11"), (15, "31×31"), (25, "51×51"), (50, "101×101")]


@dataclass
class HistoryItem:
    id: int
    label: str
    kind: str  # "point" | "rect"
    mode: str
    region: np.ndarray  # 选区像素（sRGB）
    rect: tuple[int, int, int, int]
    context: ColorContext
    result: PointSample | AnalysisReport
    swatch: tuple[int, int, int]
    source_label: str = ""
    created: float = field(default_factory=time.time)

    @property
    def time_text(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.created))


class Session:
    def __init__(self, presets: list[SkinPreset] | None = None):
        self.mode = "point"
        self.radius = 2
        self.preset_key = sp.DEFAULT_KEY
        self.scan_whole = False
        self.presets = presets if presets is not None else sp.load_presets()
        self.custom_preset: SkinPreset | None = None
        self.history: list[HistoryItem] = []
        self.reference: HistoryItem | None = None
        self.part = PARTS[0]
        self.auto_mask = True
        self.style_key: str | None = None
        self.family: str | None = None
        self.family_target_lab = None
        self.skin_by_part: dict[str, tuple[float, float, float]] = {}
        self._ids = itertools.count(1)

    # ---------------------------------------------------------------- 预设
    @property
    def preset(self) -> SkinPreset | None:
        if self.preset_key == "custom":
            return self.custom_preset
        for p in self.presets:
            if p.key == self.preset_key:
                return p
        return None

    def set_custom_preset_from(self, item: HistoryItem) -> SkinPreset:
        lab = cs.srgb8_to_lab(item.region.reshape(-1, 3)).mean(axis=0)
        self.custom_preset = sp.preset_from_sample(lab)
        self.preset_key = "custom"
        return self.custom_preset

    # ---------------------------------------------------------------- 计算（可在后台线程调用）
    @staticmethod
    def compute_point(source: np.ndarray, x: int, y: int, radius: int, ctx: ColorContext) -> PointSample:
        return an.analyze_point(source, x, y, radius, ctx)

    def other_part(self, part: str):
        """“脸部 vs 身体”：返回另一个已分析部位的 (名称, 平均 Lab)。"""
        pairs = {"脸部": ("身体", "手臂/腿"), "身体": ("脸部",), "手臂/腿": ("脸部", "身体")}
        for other in pairs.get(part, ()) + tuple(k for k in self.skin_by_part if k != part):
            if other in self.skin_by_part and other != part:
                return (other, self.skin_by_part[other])
        return None

    def compute_rect(
        self, region: np.ndarray, ctx: ColorContext, rect, mode: str | None = None, lasso=None, manual_mask=None
    ) -> AnalysisReport:
        mode = mode or self.mode
        if mode == "skin":
            return an.analyze_skin_region(
                region, ctx, preset=self.preset, rect=rect, auto_mask=self.auto_mask, lasso=lasso,
                part=self.part, other_part=self.other_part(self.part), manual_mask=manual_mask,
            )
        if mode == "palette":
            return an.analyze_palette_report(region, ctx, self.style_key, self.family, self.family_target_lab, rect=rect)
        if mode == "dirty" and self.scan_whole:
            return an.scan_image(region, ctx, rect=rect)
        return an.analyze_region(region, ctx, rect=rect)

    @staticmethod
    def compute_reference(ref: HistoryItem, mine_region: np.ndarray, ctx: ColorContext, rect, skin_mine=None):
        skin_ref = None
        if isinstance(ref.result, AnalysisReport) and ref.result.skin is not None:
            skin_ref = ref.result.skin.mean_lab
        return an.compare_reference(mine_region, ref.region, ctx, skin_mine=skin_mine if skin_ref else None,
                                    skin_ref=skin_ref if skin_mine else None, rect=rect)

    @staticmethod
    def compute_compare(ref: HistoryItem, prob_region: np.ndarray, ctx: ColorContext, rect, skin: bool = True):
        return an.compare_regions(ref.region, prob_region, ctx, skin=skin, ref_rect=ref.rect, prob_rect=rect)

    # ---------------------------------------------------------------- 历史
    def add_point(self, patch: np.ndarray, sample: PointSample, label: str) -> HistoryItem:
        """patch：取样点周围已裁好的像素（sRGB），sample 的 x/y 为源坐标。"""
        r = sample.radius
        item = HistoryItem(
            id=next(self._ids),
            label=f"点 {sample.size_label} {sample.values.hex}",
            kind="point",
            mode=self.mode,
            region=np.ascontiguousarray(patch).copy(),
            rect=(sample.x - r, sample.y - r, patch.shape[1], patch.shape[0]),
            context=sample.context,
            result=sample,
            swatch=sample.values.rgb,
            source_label=label,
        )
        self.history.insert(0, item)
        return item

    def add_rect(self, region: np.ndarray, rect, report: AnalysisReport, label: str, role: str | None = None) -> HistoryItem:
        mean = np.rint(region.reshape(-1, 3).astype(float).mean(axis=0)).astype(int)
        title = {"general": "区域", "skin": "肤色", "compare": "对比", "scan": "整图", "palette": "整图色彩",
                 "reference": "仿色"}.get(report.mode, "区域")
        if report.mode == "skin" and report.part:
            title = f"肤色·{report.part}"
            self.skin_by_part[report.part] = report.skin.mean_lab
        if role:
            title = role
        item = HistoryItem(
            id=next(self._ids),
            label=f"{title} {rect[2]}×{rect[3]} · {report.overall_severity.value}",
            kind="rect",
            mode=self.mode,
            region=region,
            rect=tuple(int(v) for v in rect),
            context=report.context,
            result=report,
            swatch=tuple(int(v) for v in mean),
            source_label=label,
        )
        self.history.insert(0, item)
        return item

    def remove(self, item_id: int) -> None:
        self.history = [h for h in self.history if h.id != item_id]
        if self.reference is not None and self.reference.id == item_id:
            self.reference = None

    def clear(self) -> None:
        self.history.clear()
        self.reference = None

    def get(self, item_id: int) -> HistoryItem | None:
        for h in self.history:
            if h.id == item_id:
                return h
        return None

    def set_reference(self, item: HistoryItem) -> None:
        self.reference = item


def point_patch_from(source: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    x0, y0 = max(0, x - radius), max(0, y - radius)
    return source[y0 : y + radius + 1, x0 : x + radius + 1].copy()


def crop_selection(source: np.ndarray, rect) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r = clamp_rect(rect, source.shape[1], source.shape[0])
    if r is None:
        raise SampleError("选区不在图像范围内")
    return crop(source, r).copy(), r
