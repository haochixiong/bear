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
from colordoctor.core.sampling import FAST_MAX, FULL_RES_MAX, PREVIEW_MAX, SampleError, clamp_rect, crop, resolution

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
    lasso: np.ndarray | None = None  # 套索蒙版（与 region 同尺寸），编辑皮肤蒙版时“恢复自动”要用

    @property
    def time_text(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.created))


def ai_skin_mask(region: np.ndarray, ctx: ColorContext, part: str | None):
    """AI 抠皮肤；不可用或没找到皮肤时返回 (None, 带提示的 ctx)，交给肤色规则处理。"""
    import dataclasses

    from colordoctor.core import segmentation as sg

    try:
        mask = sg.segment_skin(region[..., :3]).part(part)
    except sg.SegmentationUnavailable as e:
        return None, dataclasses.replace(ctx, warnings=(*ctx.warnings, f"{e}；已改用肤色规则抠皮肤"))
    if not mask.any():
        return None, dataclasses.replace(ctx, warnings=(*ctx.warnings, f"AI 没有在选区里找到{part or ''}皮肤，已改用肤色规则"))
    return mask, ctx


PREVIEW_FROM = 20_000_000  # 原图精度下，比这大的图先出快速结果、再按原图精查

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
        self.ai_seg = False  # 用 AI 人像分割抠皮肤（需要 onnxruntime + 模型）
        self.style_key: str | None = None
        self.tone_style = ""  # 影调风格（tone_class.TONE_STYLES 的键），空 = 不改影调
        self.tone_target = ""  # 直接选的目标影调（九种之一），优先于影调风格
        self.precision = "full"  # 分析精度："full" 原图每个像素（大图先出快速结果再精查）；"fast" 大图按约 2400 万像素
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

    def palette_choice(self) -> tuple:
        """整图色彩的当前选择（颜色模板、统一色、影调）：和结果里记下的比较，判断结果是不是按现在的选择算的。"""
        tone = f"t:{self.tone_target}" if self.tone_target else (f"s:{self.tone_style}" if self.tone_style else "")
        return (self.style_key or None, self.family or None, tone)

    @staticmethod
    def report_choice(rep: AnalysisReport) -> tuple | None:
        p = rep.palette if isinstance(rep, AnalysisReport) else None
        if p is None:
            return None
        return (p.style.key if p.style else None, p.unity.family if p.unity else None, p.tone_choice)

    def stages(self, *images: np.ndarray, preview: bool = True) -> list[tuple[int, str]]:
        """这次分析分几步：[(最多看多少像素, 精度名称)]。

        「快速」精度：大图按约 2400 万像素一步算完。原图精度：图大于 PREVIEW_FROM 时先按约 1200 万像素出结果，
        再按原图精查（第二步的结果替换第一步）；不大的图直接按原图算。
        """
        n = max((im.shape[0] * im.shape[1] for im in images if im is not None), default=0)
        if self.precision == "fast":
            return [(FAST_MAX, "fast")]
        if preview and n > PREVIEW_FROM:
            return [(PREVIEW_MAX, "preview"), (FULL_RES_MAX, "full")]
        return [(FULL_RES_MAX, "full")]

    def compute_rect(
        self, region: np.ndarray, ctx: ColorContext, rect, mode: str | None = None, lasso=None, manual_mask=None,
        part: str | None = None, plain: bool = False, stage: tuple[int, str] | None = None,
    ) -> AnalysisReport:
        """part：肤色模式下的部位；None 时用当前选择（重算已有结果时传原结果的部位）。

        plain：整图色彩不套用影调风格、颜色模板和统一色（分析仿色的参考作品：它不是要改的图）。
        stage：(最多看多少像素, 精度名称)，见 stages()；None = 按当前精度一步算完。
        """
        stage = stage or self.stages(region, preview=False)[-1]
        with resolution(*stage):
            return self._compute_rect(region, ctx, rect, mode, lasso, manual_mask, part, plain)

    def _compute_rect(self, region, ctx, rect, mode, lasso, manual_mask, part, plain) -> AnalysisReport:
        mode = mode or self.mode
        if mode == "skin":
            part = part or self.part
            if self.ai_seg and self.auto_mask and manual_mask is None:
                manual_mask, ctx = ai_skin_mask(region, ctx, part)
            return an.analyze_skin_region(
                region, ctx, preset=self.preset, rect=rect, auto_mask=self.auto_mask, lasso=lasso,
                part=part, other_part=self.other_part(part), manual_mask=manual_mask,
            )
        if mode == "palette" and plain:
            return an.analyze_palette_report(region, ctx, rect=rect)
        if mode == "palette":
            return an.analyze_palette_report(region, ctx, self.style_key, self.family, self.family_target_lab, rect=rect,
                                             tone_style=self.tone_style, tone_target=self.tone_target)
        if mode == "dirty" and self.scan_whole:
            return an.scan_image(region, ctx, rect=rect)
        return an.analyze_region(region, ctx, rect=rect)

    def compute_reference(self, ref: HistoryItem, mine_region: np.ndarray, ctx: ColorContext, rect, skin_mine=None,
                          stage: tuple[int, str] | None = None):
        skin_ref = None
        if isinstance(ref.result, AnalysisReport) and ref.result.skin is not None:
            skin_ref = ref.result.skin.mean_lab
        with resolution(*(stage or self.stages(mine_region, ref.region, preview=False)[-1])):
            return an.compare_reference(mine_region, ref.region, ctx, skin_mine=skin_mine if skin_ref else None,
                                        skin_ref=skin_ref if skin_mine else None, rect=rect)

    def compute_compare(self, ref: HistoryItem, prob_region: np.ndarray, ctx: ColorContext, rect, skin: bool = True,
                        stage: tuple[int, str] | None = None):
        with resolution(*(stage or self.stages(prob_region, ref.region, preview=False)[-1])):
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
        step = max(1, int(np.sqrt(region.shape[0] * region.shape[1] / 250_000)))  # 色块用抽样平均：大图不在界面线程上转整幅 float
        mean = np.rint(region[::step, ::step, :3].reshape(-1, 3).mean(axis=0)).astype(int)
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

    HISTORY_BYTES = 1_500_000_000  # 历史里保存的选区像素 + 蒙版超过这么多时，丢掉最旧的（45 MP 的整图每条约 170 MB）
    HISTORY_MAX = 60

    def trim(self, keep: set[int] = frozenset()) -> list[int]:
        """历史太大时从最旧的开始丢（参考区、参考作品和 keep 里的不丢）。返回被丢掉的 id。"""
        protect = set(keep) | ({self.reference.id} if self.reference is not None else set())
        total, dropped = 0, []
        for i, h in enumerate(self.history):  # 新的在前
            rep = h.result
            size = h.region.nbytes + (rep.valid_mask.nbytes if isinstance(rep, AnalysisReport) and rep.valid_mask is not None else 0)
            total += size
            if h.id not in protect and (total > self.HISTORY_BYTES or i >= self.HISTORY_MAX):
                dropped.append(h.id)
                total -= size
        for i in dropped:
            self.remove(i)
        return dropped

    def remove(self, item_id: int) -> None:
        self.history = [h for h in self.history if h.id != item_id]
        if self.reference is not None and self.reference.id == item_id:
            self.reference = None
        # 各部位的肤色改用剩下的、最近一次的结果；没有了就不再比较
        self.skin_by_part.clear()
        for h in self.history:
            r = h.result
            if isinstance(r, AnalysisReport) and r.mode == "skin" and r.part and r.skin is not None:
                self.skin_by_part[r.part] = r.skin.mean_lab

    def clear(self) -> None:
        self.history.clear()
        self.reference = None
        self.skin_by_part.clear()  # 清空后“脸部 vs 身体”不能再拿已删的结果比较

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
