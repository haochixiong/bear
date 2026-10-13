"""把分析结果翻译成 Photoshop 调整图层方案：每条问题一个调整图层 + 灰度蒙版。

只生成“方案”（参数 + 蒙版数组），真正在 PS 里建图层由 UXP 插件完成。
参数单位与 PS 调整图层面板一致（色相/饱和度 ±180/±100、色彩平衡 ±100、自然饱和度 ±100、
曲线 0–255 点）。全部是估计值，图层建好后可以直接在属性面板里改。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from colordoctor.core import acr_channels as ac
from colordoctor.core import contamination as ct_mod
from colordoctor.core.color_models import AnalysisReport

# PS 色相/饱和度的六个颜色范围：(localRange 编号, 名称, HSB 中心色相, 默认渐变 beginRamp..endRamp)
PS_RANGES = (
    (1, "红色", 0, (315, 345, 15, 45)),
    (2, "黄色", 60, (15, 45, 75, 105)),
    (3, "绿色", 120, (75, 105, 135, 165)),
    (4, "青色", 180, (135, 165, 195, 225)),
    (5, "蓝色", 240, (195, 225, 255, 285)),
    (6, "洋红", 300, (255, 285, 315, 345)),
)

LAYER_KINDS = ("hueSaturation", "colorBalance", "vibrance", "curves")


@dataclass
class LayerSpec:
    id: str
    name: str  # 图层名（中文，PS 里显示）
    kind: str  # LAYER_KINDS 之一
    params: dict
    reason: str  # 对应的问题（给面板卡片用）
    mask: np.ndarray | None = field(default=None, repr=False)  # uint8 (H, W)，255 = 作用；None = 全图
    estimated: bool = True

    def to_json(self) -> dict:
        cov = None if self.mask is None else round(float((self.mask > 127).mean()), 4)
        return {"id": self.id, "name": self.name, "kind": self.kind, "params": self.params,
                "reason": self.reason, "has_mask": self.mask is not None, "mask_coverage": cov,
                "estimated": self.estimated}


@dataclass
class LayerPlan:
    group_name: str
    width: int
    height: int
    layers: list[LayerSpec]

    def to_json(self) -> dict:
        return {"group_name": self.group_name, "width": self.width, "height": self.height,
                "layers": [ly.to_json() for ly in self.layers]}

    def full_mask(self, layer_id: str) -> np.ndarray | None:
        """该图层蒙版放大到分析图像尺寸（height × width）。大图的蒙版是在缩小的分辨率上算的。"""
        m = self.get(layer_id).mask
        if m is None or m.shape == (self.height, self.width):
            return m
        return resize_mask(m, self.width, self.height)

    def get(self, layer_id: str) -> LayerSpec:
        for ly in self.layers:
            if ly.id == layer_id:
                return ly
        raise KeyError(layer_id)


# ---------------------------------------------------------------- 蒙版工具
def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return img
    k = 2 * r + 1
    out = img.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
        hi = np.take(c, np.arange(k, k + out.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, out.shape[axis]), axis=axis)
        out = (hi - lo) / k
    return out


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.astype(bool)
    return _box_blur(mask.astype(np.float32), r) > 1e-6


def feather(mask: np.ndarray, radius: int) -> np.ndarray:
    """布尔蒙版 -> 边缘羽化的 uint8 蒙版（两次盒式模糊≈高斯）。"""
    m = mask.astype(np.float32) * 255.0
    r = max(0, int(radius) // 2)
    m = _box_blur(_box_blur(m, r), r)
    return np.clip(np.rint(m), 0, 255).astype(np.uint8)


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """双线性缩放 uint8 蒙版到文档尺寸（插件按文档像素写入图层蒙版）。用 Pillow，几十 MP 也只需零点几秒。"""
    from PIL import Image

    h0, w0 = mask.shape
    if (w0, h0) == (width, height):
        return mask
    img = Image.fromarray(np.ascontiguousarray(mask, dtype=np.uint8), mode="L")
    return np.asarray(img.resize((int(width), int(height)), Image.BILINEAR))


MASK_WORK_PIXELS = 2_000_000  # 蒙版在不超过约 200 万像素的分辨率上计算和羽化，返回时再放大


def _pool_max(mask: np.ndarray, k: int) -> np.ndarray:
    """k×k 块内取“任一为真”：缩小时不丢掉零星的杂色点。"""
    if k <= 1:
        return mask.astype(bool)
    h, w = mask.shape
    H, W = -(-h // k), -(-w // k)
    pad = np.zeros((H * k, W * k), bool)
    pad[:h, :w] = mask
    return pad.reshape(H, k, W, k).any(axis=(1, 3))


def _soft_mask(mask: np.ndarray, rad: int, k: int, grow: bool = False) -> np.ndarray:
    """全分辨率布尔蒙版 -> 工作分辨率上的羽化 uint8 蒙版（大图快很多）。"""
    small = _pool_max(mask, k)
    r = max(1, rad // k)
    if grow:
        small = dilate(small, max(1, r // 2))
    return feather(small, r)


def _feather_radius(shape) -> int:
    return max(2, int(round(min(shape[:2]) * 0.012)))


def _contamination_flags(report: AnalysisReport, shape) -> np.ndarray | None:
    ct = report.contamination
    if ct is None or ct.mask is None:
        return None
    step = int(round(ct.mask_scale)) or 1
    big = np.repeat(np.repeat(ct.mask, step, axis=0), step, axis=1)
    full = np.full(shape[:2], ct_mod.MASK_IGNORED, np.uint8)
    h, w = min(shape[0], big.shape[0]), min(shape[1], big.shape[1])
    full[:h, :w] = big[:h, :w]
    return full


def _bbox_mask(shape, boxes, offset, within=None) -> np.ndarray:
    m = np.zeros(shape[:2], bool)
    ox, oy = offset
    for x, y, w, h in boxes:
        x0, y0 = max(0, x - ox), max(0, y - oy)
        m[y0:y0 + h, x0:x0 + w] = True
    if within is not None and within.shape == m.shape:
        m &= within
    return m


def ps_range_for_lab_hue(lab_hue: float) -> tuple[int, str, tuple[int, int, int, int]]:
    """Lab 色相角 -> PS 色相/饱和度里最接近的颜色范围。"""
    hsb = ac.lab_hue_to_hsb_hue(lab_hue)
    best = min(PS_RANGES, key=lambda r: abs(((hsb - r[2] + 180) % 360) - 180))
    return best[0], best[1], best[3]


def hsb_shift(lab_from: float, lab_to: float, L: float = 62.0, C: float = 30.0) -> float:
    """Lab 色相从 lab_from 变到 lab_to，对应 PS（HSB）色相要转多少度。

    两种色相刻度差别很大（绿色区域约 2 倍、肤色区域约 0.5 倍），不能直接拿 Lab 度数当 PS 滑块值。
    """
    a = ac.lab_hue_to_hsb_hue(lab_from % 360, L=L, C=max(C, 8.0))
    b = ac.lab_hue_to_hsb_hue(lab_to % 360, L=L, C=max(C, 8.0))
    return ((b - a + 180) % 360) - 180


def custom_range(hsb_center: float, half_core: int = 10, half_ramp: int = 30) -> tuple[int, tuple[int, int, int, int]]:
    """以某个 HSB 色相为中心的自定义颜色范围（PS 色相/饱和度里可以拖动范围滑块，效果相同）。"""
    idx = min(PS_RANGES, key=lambda r: abs(((hsb_center - r[2] + 180) % 360) - 180))[0]
    c = int(round(hsb_center))
    return idx, ((c - half_ramp) % 360, (c - half_core) % 360, (c + half_core) % 360, (c + half_ramp) % 360)


def _huesat(range_idx: int, ramps, hue=0, sat=0, light=0, master=False) -> dict:
    if master:
        return {"master": {"hue": int(hue), "saturation": int(sat), "lightness": int(light)}, "ranges": []}
    b0, b1, e0, e1 = ramps
    return {"master": {"hue": 0, "saturation": 0, "lightness": 0},
            "ranges": [{"localRange": range_idx, "beginRamp": b0, "beginSustain": b1, "endSustain": e0,
                        "endRamp": e1, "hue": int(hue), "saturation": int(sat), "lightness": int(light)}]}


def _clip(v, lo, hi) -> int:
    return int(np.clip(round(float(v)), lo, hi))


# ---------------------------------------------------------------- 方案生成
def plan_from_report(report: AnalysisReport, shape, offset=(0, 0), group_name: str | None = None) -> LayerPlan:
    """shape：分析图像 (H, W[, C])；offset：分析区域在文档中的左上角（插件一般整图分析，为 (0, 0)）。"""
    h, w = shape[:2]
    rad = _feather_radius(shape)
    k = max(1, int(np.ceil(np.sqrt(h * w / MASK_WORK_PIXELS))))  # 大图在缩小 k 倍的分辨率上做蒙版
    layers: list[LayerSpec] = []
    flags = _contamination_flags(report, shape)
    ct = report.contamination
    skin = report.valid_mask if report.valid_mask is not None and report.valid_mask.shape == (h, w) else None

    # 1) 对立色混入：在杂色所在颜色范围降饱和，只作用于杂色位置
    if ct is not None and flags is not None and ct.opposite_hue is not None and not ct.bicolor:
        opp = flags == ct_mod.MASK_OPPOSITE
        if opp.mean() > 0.0005:
            idx, rname, ramps = ps_range_for_lab_hue(ct.opposite_hue)
            sat = -_clip(20 + ct.opposite_chroma, 20, 60)
            layers.append(LayerSpec(
                "opposite", f"{ct.base_hue_name}里的{ct.opposite_hue_name} · 色相/饱和度", "hueSaturation",
                _huesat(idx, ramps, sat=sat),
                f"{ct.base_hue_name}中混入{ct.opposite_hue_name}（占分析像素 {ct.opposite_fraction:.1%}）：{rname} 饱和度 {sat}",
                _soft_mask(opp, rad, k, grow=True)))
    # 2) 局部发灰：自然饱和度，只作用于发灰位置
    if ct is not None and flags is not None and ct.graying_fraction > 0:
        gray = flags == ct_mod.MASK_GRAYING
        if gray.mean() > 0.0005:
            vib = _clip(12 + 60 * ct.graying_depth, 10, 40)
            layers.append(LayerSpec(
                "graying", f"{ct.base_hue_name}发灰 · 自然饱和度", "vibrance",
                {"vibrance": vib, "saturation": 0},
                f"{ct.base_hue_name}局部发灰（{ct.graying_fraction:.0%}）：自然饱和度 +{vib}",
                _soft_mask(gray, rad, k, grow=True)))
    # 3) 肤色不均匀：同一类问题合成一层（偏黄 / 偏红橙 是方向相反的两层）
    uni = report.uniformity
    if uni is not None and uni.patches:
        by_kind: dict[str, list] = {}
        for p in uni.patches:
            by_kind.setdefault(p.kind, []).append(p)
        for kind, ps in by_kind.items():
            area = sum(p.area_fraction for p in ps)
            dh = float(np.average([p.hue_shift for p in ps], weights=[p.area_fraction for p in ps]))
            ds = float(np.average([p.sat_shift for p in ps], weights=[p.area_fraction for p in ps]))
            mask = _soft_mask(_bbox_mask(shape, [p.bbox for p in ps], offset, skin), rad, k)
            lab = np.average([p.lab for p in ps], axis=0, weights=[p.area_fraction for p in ps])
            spec = _skin_layer(kind, dh, ds, tuple(float(v) for v in lab))
            if spec is None:
                continue
            lkind, params, text = spec
            layers.append(LayerSpec(f"skin_{len(layers)}", f"肤色{kind} · {_KIND_NAME[lkind]}", lkind, params,
                                    f"肤色{kind}（占皮肤 {area:.0%}，色相 {dh:+.0f}°）：{text}", mask))
    # 4) 某种颜色不统一：把偏离的那一簇推回主色
    pal = report.palette
    if pal is not None and pal.unity is not None and pal.unity.patches:
        u = pal.unity
        for i, p in enumerate(u.patches):
            L0, a0, b0 = u.mean_lab
            main_h = float(np.degrees(np.arctan2(b0, a0))) % 360
            patch_h = main_h + p.hue_shift
            C0 = float(np.hypot(a0, b0))
            hue = _clip(hsb_shift(patch_h, main_h, L0, C0), -30, 30)  # 把偏离的一簇推回主色
            if hue == 0:
                continue
            idx, ramps = custom_range(ac.lab_hue_to_hsb_hue(patch_h % 360, L=L0, C=max(C0, 8.0)))
            rname = PS_RANGES[idx - 1][1]
            layers.append(LayerSpec(f"unity_{i}", f"{u.family}统一 {i + 1} · 色相/饱和度", "hueSaturation",
                                    _huesat(idx, ramps, hue=hue), f"{p.note}：{rname}（范围已对准这一簇）色相 {hue:+d}",
                                    _soft_mask(_bbox_mask(shape, [p.bbox], offset), rad, k)))
    # 5) 仿色：把影调拉向参考的 RGB 复合曲线（全图，无蒙版）
    ref = getattr(report, "reference", None)
    if ref is not None and getattr(ref, "curve_points", None):
        # 端点放前面：参考点（例如 255 → 215）覆盖默认端点，而不是被它覆盖
        pts = [(0, 0), (255, 255), *[(int(a), int(b)) for a, b in ref.curve_points]]
        pts = sorted({p[0]: p for p in pts}.values())
        layers.append(curves_layer("ref_curve", "仿色影调 · 曲线", {"composite": pts},
                                   "把影调拉向参考：" + "；".join(f"{a}→{b}" for a, b in ref.curve_points)))
    name = group_name or f"ColorDoctor · {_MODE_NAME.get(report.mode, report.mode)}"
    return LayerPlan(name, w, h, layers)


_KIND_NAME = {"hueSaturation": "色相/饱和度", "colorBalance": "色彩平衡", "vibrance": "自然饱和度", "curves": "曲线"}
_MODE_NAME = {"region": "脏色", "scan": "脏色", "skin": "肤色均匀", "palette": "整图色彩", "compare": "对比", "point": "取色", "reference": "仿色"}


def _skin_layer(kind: str, dh: float, ds: float, lab=(68.0, 14.0, 19.0)):
    """肤色问题块 -> (图层类型, 参数, 说明)。dh：相对整体的 Lab 色相偏移（度，正 = 偏黄）；lab：问题块平均色。"""
    L, a, b = lab
    hp = float(np.degrees(np.arctan2(b, a))) % 360
    C = float(np.hypot(a, b))
    if kind in ("偏黄", "偏洋红"):
        shift = hsb_shift(hp, hp - dh, L, C)  # 把问题块拉回整体肤色
        hue = int(np.sign(shift) or (-1 if kind == "偏黄" else 1)) * _clip(abs(shift), 2, 15)
        idx, ramps = custom_range(ac.lab_hue_to_hsb_hue(hp, L=L, C=max(C, 8.0)))
        rname = PS_RANGES[idx - 1][1]
        if kind == "偏黄":
            sat = -_clip(abs(ds) * 60, 0, 20) if ds > 0 else 0
            return "hueSaturation", _huesat(idx, ramps, hue=hue, sat=sat), f"{rname}（范围已对准问题色）色相 {hue:+d}，饱和度 {sat}"
        return "hueSaturation", _huesat(idx, ramps, hue=hue), f"{rname}（范围已对准问题色）色相 {hue:+d}（往黄）"
    if kind == "偏红橙":
        cr = -_clip(abs(dh) / 2, 2, 15)
        yb = -_clip(abs(dh) / 3, 1, 10)  # 负值 = 往黄
        return ("colorBalance", {"shadows": [0, 0, 0], "midtones": [cr, 0, yb], "highlights": [0, 0, 0], "preserveLuminosity": True},
                f"中间调 青-红 {cr}，黄-蓝 {yb}")
    if kind == "偏青绿":
        cr = _clip(abs(dh) / 2, 3, 15)
        return ("colorBalance", {"shadows": [0, 0, 0], "midtones": [cr, -_clip(cr / 2, 1, 8), 0], "highlights": [0, 0, 0],
                                 "preserveLuminosity": True}, f"中间调 青-红 +{cr}，洋红-绿 {-_clip(cr / 2, 1, 8)}")
    if kind == "缺色发灰":
        vib = _clip(abs(ds) * 60, 8, 30)
        return "vibrance", {"vibrance": vib, "saturation": 0}, f"自然饱和度 +{vib}"
    if kind == "饱和度偏高":
        sat = -_clip(abs(ds) * 60, 5, 25)
        return "hueSaturation", _huesat(0, None, sat=sat, master=True), f"全图 饱和度 {sat}"
    return None


def curves_layer(lid: str, name: str, points: dict[str, list[tuple[int, int]]], reason: str) -> LayerSpec:
    """曲线图层：points 的键为 composite / red / green / blue。"""
    return LayerSpec(lid, name, "curves", {"channels": {k: [list(map(int, p)) for p in v] for k, v in points.items()}}, reason)
