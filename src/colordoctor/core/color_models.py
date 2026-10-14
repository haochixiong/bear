"""分析用的数据结构（全部是不可变的 dataclass，不依赖 Qt）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    NORMAL = "正常"
    MILD = "轻微"
    NOTABLE = "明显"
    STRONG = "较强"


def severity_from_score(score: float) -> Severity:
    if score < 20:
        return Severity.NORMAL
    if score < 45:
        return Severity.MILD
    if score < 70:
        return Severity.NOTABLE
    return Severity.STRONG


def confidence_label(conf: float) -> str:
    if conf >= 0.75:
        return "高"
    if conf >= 0.5:
        return "中"
    return "低"


@dataclass(frozen=True)
class ColorContext:
    """像素的来源与色彩管理信息。"""

    source: str  # "screen" | "file" | "synthetic"
    analysis_space: str = "sRGB IEC61966-2.1"
    source_profile: str | None = None  # 原始 ICC 描述
    converted: bool = False  # 是否做过 ICC -> sRGB 转换
    assumed_srgb: bool = False  # 没有 ICC，按 sRGB 处理
    managed_source: bool = True  # 屏幕取样时：画面是否来自色彩管理软件
    profile_near_srgb: bool | None = None
    device: str | None = None  # 显示器名 / 文件名
    scale: float | None = None  # 屏幕缩放比例
    clipped_fraction: float = 0.0  # 转换后被裁切的像素比例（估算）
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceImage:
    pixels: object  # np.ndarray (H, W, 3) uint8，sRGB 分析缓冲区
    context: ColorContext
    original_size: tuple[int, int] = (0, 0)  # (宽, 高)，EXIF 旋转之后


@dataclass(frozen=True)
class ColorValues:
    """单个颜色在各色彩空间的数值（L*a*b* 默认 D50，与 Photoshop 一致）。"""

    rgb: tuple[int, int, int]
    hex: str
    hsb: tuple[float, float, float]
    hsl: tuple[float, float, float]
    lab: tuple[float, float, float]
    lab_d65: tuple[float, float, float]
    lch: tuple[float, float, float]  # L*, C*, h°
    oklab: tuple[float, float, float]
    oklch: tuple[float, float, float]
    luminance: float
    hue_name: str


@dataclass(frozen=True)
class PointSample:
    x: int
    y: int
    radius: int  # 0 = 1×1, 1 = 3×3, 2 = 5×5 ...
    pixel_count: int
    values: ColorValues
    context: ColorContext

    @property
    def size_label(self) -> str:
        n = 2 * self.radius + 1
        return f"{n}×{n}"


@dataclass(frozen=True)
class Cluster:
    lab: tuple[float, float, float]
    rgb: tuple[int, int, int]
    share: float
    chroma: float
    hue: float
    hue_name: str


@dataclass(frozen=True)
class RegionStats:
    total_pixels: int
    sampled_pixels: int
    rgb_mean: tuple[float, float, float]
    rgb_median: tuple[float, float, float]
    lab_mean: tuple[float, float, float]
    lab_median: tuple[float, float, float]
    oklab_mean: tuple[float, float, float]
    oklch_mean: tuple[float, float, float]
    lch_mean: tuple[float, float, float]  # 由平均 Lab 计算
    L_std: float
    C_std: float
    L_p10: float
    L_p50: float
    L_p90: float
    C_p10: float
    C_p50: float
    C_p90: float
    sat_p50: float  # HSB S 中位数
    hue_mean: float | None  # 按色度加权的圆形平均（无彩色区域为 None）
    hue_concentration: float  # R，0–1，越接近 1 越集中
    hue_spread: float | None  # 圆形标准差（度）
    chromatic_fraction: float
    highlight_fraction: float
    shadow_fraction: float
    clipped_fraction: float
    clusters: tuple[Cluster, ...] = ()


@dataclass(frozen=True)
class Blob:
    """一处被标记的问题位置（源图像素坐标）。"""

    kind: str  # "opposite" | "graying"
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area_fraction: float  # 占分析区域的比例
    base_hue_name: str
    contaminant_hue_name: str | None
    score: float
    evidence: tuple[str, ...]
    contaminant_hue: float | None = None  # 杂色的 Lab 色相角（发灰处为 None）


@dataclass(frozen=True)
class ContaminationResult:
    """脏色（对立色混入 / 局部发灰）检测结果。"""

    base_hue: float | None
    base_hue_name: str
    base_share: float
    opposite_fraction: float  # 对立色像素 / 有彩像素
    opposite_hue: float | None
    opposite_hue_name: str | None
    opposite_chroma: float
    base_chroma: float
    interspersion: float  # 对立像素与主色像素的混杂程度（0–4）
    bicolor: bool  # 两种颜色都占大比例 -> 视为配色而非污染
    graying_fraction: float  # 局部发灰像素 / 主色族像素
    graying_depth: float  # 发灰处 s_ab 相对同色干净处的下降比例
    chromatic_fraction: float
    noise_like: bool
    hue_spread: float | None = None  # 去掉对立色后，有彩像素的圆形标准差（度）
    opposite_threshold: float = 0.0  # 判为对立色所需的最低色度
    anomaly_fractions: tuple[tuple[str, float], ...] = ()  # 按色相名统计的异常像素占比
    mask: object = None  # np.ndarray (h, w) uint8：0 正常 1 无彩/忽略 2 对立色 3 发灰
    mask_scale: float = 1.0  # mask 像素到源像素的倍数
    blobs: tuple[Blob, ...] = ()


@dataclass(frozen=True)
class RiskComponent:
    key: str
    name: str
    score: float  # 0–100
    severity: Severity
    evidence: tuple[str, ...]
    applicable: bool = True
    note: str | None = None


@dataclass(frozen=True)
class Adjustment:
    tool: str  # 例如 "曲线"、"色彩平衡"、"色相/饱和度"、"可选颜色"、"Camera Raw"
    instruction: str
    strength: str | None = None  # 轻微 / 小幅 / 中等（估计）
    estimated: bool = True
    params: tuple[tuple[str, float], ...] = ()
    # 能直接生成的 PS 原生调整图层（数值与 instruction 相同），格式见 native_layers；不参与比较/哈希
    layer: dict | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Finding:
    """统一表达结构：问题 / 证据 / 可能原因 / 调整方向 / 置信度 / 注意事项。"""

    problem: str
    severity: Severity
    evidence: tuple[str, ...]
    causes: tuple[str, ...]
    adjustments: tuple[Adjustment, ...]
    confidence: float
    notes: tuple[str, ...] = ()

    @property
    def confidence_text(self) -> str:
        return f"{confidence_label(self.confidence)}（{self.confidence:.2f}）"


@dataclass(frozen=True)
class SkinPreset:
    key: str
    name: str
    description: str
    L: tuple[float, float]
    a: tuple[float, float]
    b: tuple[float, float]
    hue: tuple[float, float]
    chroma: tuple[float, float]
    notes: tuple[str, ...] = ()
    quick_rule: tuple | None = None  # ((a 下限, a 上限), (b 下限, b 上限), |a-b| 上限)；None 用全局规则

    @property
    def target_lab(self) -> tuple[float, float, float]:
        return (
            (self.L[0] + self.L[1]) / 2,
            (self.a[0] + self.a[1]) / 2,
            (self.b[0] + self.b[1]) / 2,
        )


@dataclass(frozen=True)
class QuickRuleResult:
    """经验快速判定：a*、b* 是否都在区间内且差值不超过上限。"""

    a: float
    b: float
    a_range: tuple[float, float]
    b_range: tuple[float, float]
    max_diff: float
    a_ok: bool
    b_ok: bool
    diff_ok: bool
    pixel_pass_share: float  # 皮肤像素中满足规则的比例
    verdict: str
    tips: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.a_ok and self.b_ok and self.diff_ok


@dataclass(frozen=True)
class SkinReport:
    mean_lab: tuple[float, float, float]
    mean_lch: tuple[float, float, float]
    ita: float
    depth_label: str
    tendency: str  # 偏红 / 偏黄 / 偏洋红 / 偏绿 / 正常
    tendency_amount: float
    highlight_shift: tuple[float, float, float]  # 高光相对中间调 (ΔL, Δa, Δb)
    shadow_shift: tuple[float, float, float]
    highlight_hue_shift: float
    shadow_hue_shift: float
    saturation_ab: float  # C*/L*
    expected_saturation: tuple[float, float]
    mixed_region: bool
    anomaly_fractions: dict = field(default_factory=dict)  # {"紫/品红": 0.03, "青": 0.01 ...}
    hue_range: tuple[float, float] = (40.0, 68.0)  # 判断整体偏色用的色相区间（选了风格时用风格的区间）
    hue_ref: str = "自然肤色"  # 上面区间的名字
    target_hue: float | None = None  # 整体偏色要拉回的目标色相（风格中心或自然肤色中心）


@dataclass(frozen=True)
class Comparison:
    ref_lab: tuple[float, float, float]
    target_lab: tuple[float, float, float]
    delta_e00: float
    dL: float
    dC: float
    dH: float  # ΔH*（度量值）
    dh: float  # 色相角差（度）
    da: float
    db: float
    along_ref_hue: float  # 沿参考色相方向的变化（色度增减）
    across_ref_hue: float  # 垂直方向（色相偏移量，正值为逆时针）
    opposite_component_change: float  # 问题区对立色比例 - 参考区对立色比例


@dataclass(frozen=True)
class AnalysisReport:
    mode: str
    context: ColorContext
    stats: RegionStats | None = None
    contamination: ContaminationResult | None = None
    skin: SkinReport | None = None
    comparison: Comparison | None = None
    components: tuple[RiskComponent, ...] = ()
    overall_risk: float = 0.0
    overall_severity: Severity = Severity.NORMAL
    confidence: float = 0.0
    findings: tuple[Finding, ...] = ()
    adjustments: tuple[Adjustment, ...] = ()
    region: tuple[int, int, int, int] | None = None
    part: str | None = None  # 肤色部位：脸部 / 身体 / 手臂腿 / 其他
    quick_rule: object = None  # QuickRuleResult
    uniformity: object = None  # skin_uniformity.UniformityResult
    valid_mask: object = field(default=None, repr=False)  # 参与分析的像素（皮肤蒙版/套索），与选区同尺寸
    mask_info: dict = field(default_factory=dict)
    palette: object = None  # palette.PaletteReport（整图色彩）
    reference: object = None  # reference_match.MatchReport（仿色）
