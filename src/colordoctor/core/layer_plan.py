"""把分析结果翻译成 Photoshop 调整图层方案：每张问题卡片上带图层的建议 -> 一个调整图层 + 灰度蒙版。

只生成“方案”（参数 + 蒙版数组），真正在 PS 里建图层由 UXP 插件完成。
图层参数来自建议本身（Adjustment.layer，见 native_layers），所以卡片上的数字就是图层里的数字；
每个图层记下它修的是第几条问题（finding = report.findings 的序号），插件把图层放进对应的卡片。
参数单位与 PS 调整图层面板一致。全部是估计值，图层建好后可以直接在属性面板里改。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from colordoctor.core import contamination as ct_mod
from colordoctor.core import native_layers as nl
from colordoctor.core.color_models import AnalysisReport

# 兼容旧名字（测试和其他模块在用）
PS_RANGES = nl.PS_RANGES
LAYER_KINDS = nl.LAYER_KINDS
ps_range_for_lab_hue = nl.ps_range_for_lab_hue
hsb_shift = nl.hsb_shift
custom_range = nl.custom_range
patch_sat = nl.patch_sat
_huesat = nl.huesat
_clip = nl.clip
_skin_layer = nl.skin_patch_layer
_KIND_NAME = nl.KIND_NAME


@dataclass
class LayerSpec:
    id: str
    name: str  # 图层名（中文，PS 里显示）
    kind: str  # LAYER_KINDS 之一
    params: dict
    reason: str  # 卡片上的建议原文（数字与参数相同）
    mask: np.ndarray | None = field(default=None, repr=False)  # uint8 (H, W)，255 = 作用；None = 全图
    estimated: bool = True
    finding: int | None = None  # 修的是 report.findings 里的第几条；None = 不属于任何一条

    def to_json(self) -> dict:
        cov = None if self.mask is None else round(float((self.mask > 127).mean()), 4)
        return {"id": self.id, "name": self.name, "kind": self.kind, "params": self.params,
                "reason": self.reason, "has_mask": self.mask is not None, "mask_coverage": cov,
                "estimated": self.estimated, "finding": self.finding}


@dataclass
class LayerPlan:
    group_name: str
    width: int
    height: int
    layers: list[LayerSpec]
    # 没有生成图层的建议：(问题序号, 建议原文, 原因)。例如作用范围是皮肤 / 问题位置，但没能单独选出来
    skipped: list[tuple[int, str, str]] = field(default_factory=list)

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

    def for_finding(self, index: int) -> list[LayerSpec]:
        return [ly for ly in self.layers if ly.finding == index]


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


def _soft_small(small: np.ndarray, rad: int, k: int, grow: bool = False) -> np.ndarray:
    r = max(1, rad // k)
    if grow:
        small = dilate(small, max(1, r // 2))
    return feather(small, r)


def _soft_mask(mask: np.ndarray, rad: int, k: int, grow: bool = False) -> np.ndarray:
    """全分辨率布尔蒙版 -> 工作分辨率上的羽化 uint8 蒙版（大图快很多）。"""
    return _soft_small(_pool_max(mask, k), rad, k, grow)


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


class _Masks:
    """按图层的作用范围取蒙版（工作分辨率，缓存）：skin = 皮肤蒙版，local = 该问题的位置，global = 无蒙版。"""

    _FLAG = {"opposite": ct_mod.MASK_OPPOSITE, "graying": ct_mod.MASK_GRAYING}

    def __init__(self, report: AnalysisReport, shape, offset, spots: dict[str, list[int]] | None = None):
        self.report, self.shape, self.offset = report, shape, offset
        self.spots = spots or {}  # 整图扫描：每类问题位置用到了哪几处（蒙版要互不重叠）
        h, w = shape[:2]
        self.rad = _feather_radius(shape)
        self.k = max(1, int(np.ceil(np.sqrt(h * w / MASK_WORK_PIXELS))))  # 大图在缩小 k 倍的分辨率上做蒙版
        vm = getattr(report, "valid_mask", None)
        self.skin = vm if vm is not None and vm.shape == (h, w) else None
        self._flags = None
        self._cache: dict = {}

    def _small(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def _small_flags(self, value: int) -> np.ndarray | None:
        def make():
            if self._flags is None:
                self._flags = _contamination_flags(self.report, self.shape)
            return None if self._flags is None else _pool_max(self._flags == value, self.k)
        return self._small(("flag", value), make)

    def _small_skin(self):
        return None if self.skin is None else self._small("skin", lambda: _pool_max(self.skin, self.k))

    def _small_boxes(self, boxes, pad: int = 0) -> np.ndarray:
        """问题位置（源图坐标）的矩形（四周再放宽 pad 像素）-> 工作分辨率上的布尔蒙版。"""
        h, w = self.shape[:2]
        k = self.k
        m = np.zeros((-(-h // k), -(-w // k)), bool)
        ox, oy = self.offset
        for x, y, bw, bh in boxes:
            x, y, bw, bh = x - pad, y - pad, bw + 2 * pad, bh + 2 * pad
            x0, y0 = max(0, x - ox), max(0, y - oy)
            x1, y1 = min(w, x - ox + bw), min(h, y - oy + bh)
            if x1 > x0 and y1 > y0:
                m[y0 // k:-(-y1 // k), x0 // k:-(-x1 // k)] = True
        return m

    def get(self, spec: dict) -> np.ndarray | None:
        scope = spec.get("scope")
        if scope == "skin":
            s = self._small_skin()
            return None if s is None else self._small(("soft", "skin"), lambda: _soft_small(s, self.rad, self.k))
        if scope != "local" or not spec.get("_at"):
            return None
        at = spec["_at"]
        return self._small(("soft", at), lambda: self._local(at))

    def _local(self, at: str) -> np.ndarray | None:
        stem, _, arg = at.partition("@")
        rep = self.report
        if stem in self._FLAG:
            flags = self._small_flags(self._FLAG[stem])
            if flags is None:
                return None
            if not arg:
                return _soft_small(flags, self.rad, self.k, grow=True) if flags.any() else None
            # 整图扫描：只作用于这一处问题位置里的杂色 / 发灰像素。几处位置的矩形可能套在一起（一大片里有一小块），
            # 每个像素只归“包含它的最小那一处”，各处的蒙版（含羽化边）互不重叠，同一批像素不会被调两次
            blobs = rep.contamination.blobs if rep.contamination is not None else ()
            j = int(arg)
            if j >= len(blobs):
                return None
            area = lambda i: blobs[i].bbox[2] * blobs[i].bbox[3]  # noqa: E731
            order = sorted({i for i in self.spots.get(stem, [j]) if i < len(blobs)} | {j}, key=lambda i: (area(i), i))
            smaller = [blobs[i].bbox for i in order[:order.index(j)]]
            m = flags & self._small_boxes([blobs[j].bbox])
            if smaller:
                m &= ~self._small_boxes(smaller)
            if not m.any():
                return None
            soft = _soft_small(m, self.rad, self.k, grow=True)
            pad = 2 * self.rad
            own = self._small_boxes([blobs[j].bbox], pad)
            if smaller:
                own &= ~self._small_boxes(smaller, pad)
            soft[~own] = 0
            return soft if soft.any() else None
        if stem == "skin" and arg:  # 肤色不均匀：这一类小块
            uni = getattr(rep, "uniformity", None)
            # 只作用于真正偏离的小块（不是外接框：框里还有正常的皮肤）
            boxes = [b for p in (uni.patches if uni is not None else ()) if p.kind == arg for b in (p.cells or (p.bbox,))]
            if not boxes:
                return None
            m = self._small_boxes(boxes)
            s = self._small_skin()
            if s is not None:
                m &= s
            return _soft_small(m, self.rad, self.k)
        if stem == "region":  # 整个分析区域（例如对比模式的问题区）
            return self._small(("soft", "region"), lambda: np.full(self._small_boxes([]).shape, 255, np.uint8))
        if stem == "unity" and arg:  # 某种颜色不统一：偏离的这一块
            pal = getattr(rep, "palette", None)
            patches = pal.unity.patches if pal is not None and pal.unity is not None else ()
            j = int(arg)
            if j >= len(patches):
                return None
            return _soft_small(self._small_boxes([patches[j].bbox]), self.rad, self.k)
        return None


# ---------------------------------------------------------------- 同一张卡片里的图层合并
def _merge_pair(a: dict, b: dict) -> dict | None:
    """同类型、同作用范围的两个图层能自然合成一层时返回合并后的参数，否则 None。"""
    if a["kind"] != b["kind"] or a["scope"] != b["scope"] or a.get("_at") != b.get("_at"):
        return None
    pa, pb = a["params"], b["params"]
    if a["kind"] == "curves":
        if set(pa["channels"]) != set(pb["channels"]):
            return None  # 例如“颜色曲线”和“明度曲线”：卡片要求分开
        out = {}
        for ch, pts in pa["channels"].items():
            xa = {x for x, _ in pts if x not in (0, 255)}
            xb = {x for x, _ in pb["channels"][ch] if x not in (0, 255)}
            if xa & xb:
                return None
            ends = {x: y for x, y in pts + pb["channels"][ch] if x in (0, 255) and (x, y) not in ((0, 0), (255, 255))}
            mid = [(x, y) for x, y in pts + pb["channels"][ch] if x not in (0, 255)]
            merged = nl.curve_points(mid + list(ends.items()))
            ys = [y for _, y in merged]
            if any(y1 < y0 for y0, y1 in zip(ys, ys[1:])):
                return None  # 合并后不再单调：分开放
            out[ch] = merged
        return {"channels": out}
    if a["kind"] == "colorBalance":
        tones = [t for t, _ in nl.CB_TONES]
        if any(any(pa[t]) and any(pb[t]) for t in tones):
            return None
        return nl.balance(*[pa[t] if any(pa[t]) else pb[t] for t in tones])
    if a["kind"] == "hueSaturation":
        if any(pa["master"].values()) or any(pb["master"].values()) or len(pa["ranges"]) + len(pb["ranges"]) > 6:
            return None
        if {r["localRange"] for r in pa["ranges"]} & {r["localRange"] for r in pb["ranges"]}:
            return None
        if nl.huesat_overlap(pa, pb):
            return None  # 范围有重叠：合成一层后重叠处的颜色会被转两次
        return {"master": dict(pa["master"]), "ranges": pa["ranges"] + pb["ranges"]}
    return None


SKIP_SAME_PIXELS = "同一张卡片上已经有一层色相/饱和度改这些颜色（二选一），没有再生成一层"
SKIP_NO_MASK = "没能单独选出要改的区域（皮肤或问题位置），直接生成会改到整张图，所以没有生成图层；可以先手动建选区或蒙版，再按这里的数值调"


def _finding_layers(f, skipped: list | None = None) -> list[tuple[dict, list[str]]]:
    """一张卡片上带图层的建议 -> [(图层字典, 对应的建议原文)]：完全相同的只留一个，能自然合并的合成一层。

    同一张卡片上、同一作用范围的两层色相/饱和度改到同一批颜色时（例如两种二选一的做法），只生成第一层，
    后一条记入 skipped：[(建议原文, 原因)]。
    """
    out: list[tuple[dict, list[str]]] = []
    for a in f.adjustments:
        spec = getattr(a, "layer", None)
        if not spec:
            continue
        same = next((o for o in out if o[0]["kind"] == spec["kind"] and o[0]["params"] == spec["params"]
                     and o[0]["scope"] == spec["scope"] and o[0].get("_at") == spec.get("_at")), None)
        if same is not None:
            if a.instruction not in same[1]:
                same[1].append(a.instruction)
            continue
        for i, (o, texts) in enumerate(out):
            merged = _merge_pair(o, spec)
            if merged is not None:
                out[i] = ({**o, "params": merged, "name": o["name"]}, [*texts, a.instruction])
                break
        else:
            clash = spec["kind"] == "hueSaturation" and any(
                o["kind"] == "hueSaturation" and o["scope"] == spec["scope"] and o.get("_at") == spec.get("_at")
                and nl.huesat_overlap(o["params"], spec["params"]) for o, _ in out)
            if clash:
                if skipped is not None:
                    skipped.append((a.instruction, SKIP_SAME_PIXELS))
                continue
            out.append((dict(spec), [a.instruction]))
    return out


_ID_STEM = {"hueSaturation": "huesat", "colorBalance": "balance", "vibrance": "vibrance", "curves": "curves",
            "selectiveColor": "selective"}


def _unique(stem: str, used: set[str]) -> str:
    lid, n = stem, 2
    while lid in used:
        lid, n = f"{stem}_{n}", n + 1
    used.add(lid)
    return lid


def ref_curve_points(curve_points) -> list[tuple[int, int]]:
    """仿色曲线点 + 端点：参考点（例如 255 → 215）覆盖默认端点，而不是被它覆盖。"""
    return [tuple(p) for p in nl.curve_points(curve_points)]


# ---------------------------------------------------------------- 方案生成
def plan_from_report(report: AnalysisReport, shape, offset=(0, 0), group_name: str | None = None) -> LayerPlan:
    """shape：分析图像 (H, W[, C])；offset：分析区域在文档中的左上角（插件一般整图分析，为 (0, 0)）。

    每条问题（report.findings[i]）上带 layer 的建议各生成一层（相同的只生成一次，同类可合并的合成一层），
    图层的 finding = i；蒙版按作用范围：skin = 皮肤蒙版，local = 这条问题的位置，global = 全图。
    """
    h, w = shape[:2]
    findings = list(getattr(report, "findings", ()) or ())
    per_finding, skipped = [], []
    for i, f in enumerate(findings):
        sk: list = []
        per_finding.append(_finding_layers(f, sk))
        skipped += [(i, t, why) for t, why in sk]
    spots: dict[str, list[int]] = {}
    for specs in per_finding:
        for spec, _ in specs:
            stem, _, arg = (spec.get("_at") or "").partition("@")
            if arg.isdigit():
                spots.setdefault(stem, []).append(int(arg))
    masks = _Masks(report, shape, offset, spots)
    layers: list[LayerSpec] = []
    used: set[str] = set()
    for i, specs in enumerate(per_finding):
        for spec, texts in specs:
            mask = masks.get(spec)
            if mask is not None and not mask.any():
                mask = None
            if spec["scope"] != "global" and mask is None:
                skipped += [(i, t, SKIP_NO_MASK) for t in texts]  # 作用范围选不出来：不能变成整张图的图层
                continue
            at = spec.get("_at")
            stem = at.partition("@")[0] if at else _ID_STEM[spec["kind"]]
            layers.append(LayerSpec(_unique(stem, used), spec["name"], spec["kind"], spec["params"], "；".join(texts),
                                    mask, finding=i))
    # 没有对应卡片的仿色曲线（例如只有 MatchReport 的报告）：仍给出，不挂在任何卡片上
    ref = getattr(report, "reference", None)
    if ref is not None and getattr(ref, "curve_points", None) and "ref_curve" not in used:
        pts = ref_curve_points(ref.curve_points)
        layers.append(curves_layer(_unique("ref_curve", used), "仿色影调 · 曲线", {"composite": pts},
                                   "把影调拉向参考：" + "；".join(f"{a}→{b}" for a, b in ref.curve_points)))
    name = group_name or f"ColorDoctor · {_MODE_NAME.get(report.mode, report.mode)}"
    return LayerPlan(name, w, h, layers, skipped)


def annotate_findings(findings_json: list[dict], plan: LayerPlan) -> list[dict]:
    """问题卡片的 JSON（顺序与 report.findings 相同）加上 index 和修它的图层 id；
    没有生成图层的建议去掉 layer（插件就把它当文字建议显示），并在文字后注明原因。"""
    skip = {(i, t): why for i, t, why in plan.skipped}
    for i, fj in enumerate(findings_json):
        fj["index"] = i
        fj["layers"] = [ly.id for ly in plan.layers if ly.finding == i]
        for a in fj.get("adjustments") or ():
            why = skip.get((i, a.get("instruction")))
            if why and a.get("layer"):
                a["layer"] = None
                a["instruction"] = f"{a['instruction']}（{why}）"
    return findings_json


_MODE_NAME = {"region": "脏色", "scan": "脏色", "skin": "肤色均匀", "palette": "整图色彩", "compare": "对比", "point": "取色",
              "reference": "仿色", "general": "区域检查"}


def curves_layer(lid: str, name: str, points: dict[str, list[tuple[int, int]]], reason: str) -> LayerSpec:
    """曲线图层：points 的键为 composite / red / green / blue。"""
    return LayerSpec(lid, name, "curves", {"channels": {k: [list(map(int, p)) for p in v] for k, v in points.items()}}, reason)
