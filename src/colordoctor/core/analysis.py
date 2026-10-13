"""分析入口：单点、通用区域、人像肤色、参考对比、整图扫描。

输入是 sRGB uint8 NumPy 数组（已经过色彩管理），输出是不可变的 AnalysisReport。
本模块不依赖任何界面库，可在后台线程中调用。
"""

from __future__ import annotations

import numpy as np

from colordoctor.core import color_spaces as cs
from colordoctor.core import contamination as ct_mod
from colordoctor.core import diagnosis_rules as dr
from colordoctor.core import portrait_analysis as pa
from colordoctor.core import recommendations as rec
from colordoctor.core import region_statistics as rs
from colordoctor.core import skin_presets as sp
from colordoctor.core.color_models import (
    AnalysisReport,
    ColorContext,
    Finding,
    PointSample,
    RiskComponent,
    Severity,
    severity_from_score,
)
from colordoctor.core.sampling import grid_downsample, sample_point

MAX_SAMPLES = rs.MAX_SAMPLES

CAUSES_OPPOSITE = (
    "环境反光或色溢（绿植、蓝天、彩色墙面、衣服反射到皮肤）",
    "HSL / 可选颜色 / 颜色分级的作用范围过大，波及了这个区域",
    "白平衡或分离色调让高光、阴影染上了对立色",
    "JPEG 压缩、降噪或锐化产生的彩色噪点与光晕",
)
CAUSES_GRAY = (
    "降低饱和度、去朦胧为负、清晰度/纹理过度或降噪过强",
    "对立色以“像素内混合”的方式叠加（例如蓝色上叠了一层黄），表现为变灰",
    "曲线压低了对比，局部被“压闷”",
)


def analyze_point(pixels: np.ndarray, x: int, y: int, radius: int, context: ColorContext) -> PointSample:
    return sample_point(pixels, x, y, radius, context)


def _offset(rect):
    return (int(rect[0]), int(rect[1])) if rect else (0, 0)


def _blob_text(blobs, limit=3) -> list[str]:
    out = []
    for i, b in enumerate(blobs[:limit], 1):
        x, y, w, h = b.bbox
        label = f"“{b.contaminant_hue_name}”杂色" if b.contaminant_hue_name else ("发灰" if b.kind == "graying" else "异常")
        out.append(f"位置 {i}：x {x}, y {y}，{w}×{h} 像素（{label}）")
    if len(blobs) > limit:
        out.append(f"……共 {len(blobs)} 处，见“问题位置”视图")
    return out


def _bands(region: np.ndarray, mask: np.ndarray | None) -> dict:
    sub, _ = grid_downsample(region, MAX_SAMPLES)
    lab = cs.srgb8_to_lab(sub).reshape(-1, 3)
    if mask is not None and mask.size == lab.shape[0]:
        lab = lab[mask.reshape(-1) == ct_mod.MASK_NORMAL]
    if lab.shape[0] < 20:
        return {}
    L = lab[:, 0]
    p15, p40, p60, p85 = np.percentile(L, [15, 40, 60, 85])
    out = {}
    for key, sel in (("shadow", L <= p15), ("mid", (L >= p40) & (L <= p60)), ("highlight", L >= p85)):
        if sel.sum() >= 5:
            out[key] = lab[sel].mean(axis=0)
    return out


def _finalize(mode, context, stats, components, findings, extra_conf_reasons, conf, **kw) -> AnalysisReport:
    risk = dr.overall(components)
    sev = severity_from_score(risk)
    if not [f for f in findings if f.severity != Severity.NORMAL]:
        findings = [
            Finding(
                problem="未发现明显的脏色、发灰或偏色问题",
                severity=Severity.NORMAL,
                evidence=tuple(e for c in components if c.applicable for e in c.evidence[:1]),
                causes=(),
                adjustments=(),
                confidence=conf,
                notes=tuple(extra_conf_reasons) or ("结论基于当前选区；换一个区域可能得到不同结果",),
            )
        ] + list(findings)
    order = [Severity.NORMAL, Severity.MILD, Severity.NOTABLE, Severity.STRONG]
    # 真实问题优先；“与目标风格的差距”不是错误，排在问题之后
    findings = sorted(findings, key=lambda f: (f.problem.startswith("与目标风格"), -order.index(f.severity)))
    return AnalysisReport(
        mode=mode,
        context=context,
        stats=stats,
        components=tuple(components),
        overall_risk=risk,
        overall_severity=sev,
        confidence=round(conf, 3),
        findings=tuple(findings),
        **kw,
    )


def _component_findings(components: list[RiskComponent], ct, conf, base_notes, skin: bool, skin_report=None):
    findings = []
    comp = {c.key: c for c in components}
    c = comp.get("opposite")
    if c and c.applicable and c.severity != Severity.NORMAL:
        if skin and ct.anomaly_fractions:
            names = "、".join(n for n, v in ct.anomaly_fractions if v >= 0.002) or ct.opposite_hue_name
            problem = f"肤色中混入了“{names}”杂色（颜色脏）"
            main = ct.anomaly_fractions[0][0]
        else:
            problem = f"“{ct.base_hue_name}”区域中混入了“{ct.opposite_hue_name}”（对立色污染，颜色发脏）"
            main = ct.opposite_hue_name
        notes = ["如果这些颜色本来就是图案、妆容或道具的一部分，则不需要处理"] + list(base_notes)
        if c.note:
            notes.insert(0, c.note)
        findings.append(
            Finding(
                problem=problem,
                severity=c.severity,
                evidence=c.evidence + tuple(_blob_text([b for b in ct.blobs if b.kind == "opposite"])),
                causes=CAUSES_OPPOSITE,
                adjustments=tuple(rec.contamination_adjustments(main, ct.base_hue_name if not skin else "橙", skin)),
                confidence=conf,
                notes=tuple(notes),
            )
        )
    c = comp.get("chroma_compression")
    if c and c.severity != Severity.NORMAL:
        problem = "肤色发灰、不够透亮（色度偏低）" if skin else f"“{ct.base_hue_name}”区域局部发灰（比同色干净处更灰）"
        findings.append(
            Finding(
                problem=problem,
                severity=c.severity,
                evidence=c.evidence + tuple(_blob_text([b for b in ct.blobs if b.kind == "graying"])),
                causes=CAUSES_GRAY,
                adjustments=tuple(rec.graying_adjustments("橙" if skin else ct.base_hue_name)),
                confidence=conf,
                notes=("低饱和本身不是错误；只有在你希望它更干净通透时才需要处理",) + tuple(base_notes),
            )
        )
    c = comp.get("hue_dispersion")
    if c and c.applicable and c.severity != Severity.NORMAL:
        findings.append(
            Finding(
                problem="色相杂乱（同一区域里颜色倾向不统一）",
                severity=c.severity,
                evidence=c.evidence,
                causes=("多光源（色温不同的灯/窗光）", "局部调色叠加不均匀", "选区包含了不同物体"),
                adjustments=tuple(rec.contamination_adjustments(ct.base_hue_name, ct.base_hue_name, skin)[:1]),
                confidence=conf * 0.9,
                notes=("确认选区内只有同一种材质/物体",) + tuple(base_notes),
            )
        )
    c = comp.get("local_cast")
    if c and c.applicable and c.severity != Severity.NORMAL:
        if skin and skin_report is not None:
            findings += _skin_cast_findings(skin_report, c, conf, base_notes)
        else:
            findings.append(
                Finding(
                    problem="高光或阴影的颜色倾向与中间调不一致（局部偏色）",
                    severity=c.severity,
                    evidence=c.evidence,
                    causes=("颜色分级/分离色调给高光或阴影加了颜色", "混合光源"),
                    adjustments=(),
                    confidence=conf,
                    notes=("胶片风格常刻意让高光、阴影偏色；如果是风格选择可以忽略",) + tuple(base_notes),
                )
            )
    c = comp.get("lightness_compression")
    if c and c.severity != Severity.NORMAL:
        findings.append(
            Finding(
                problem="明度和色度同时被压缩，显得发闷",
                severity=c.severity,
                evidence=c.evidence,
                causes=("曲线压低了对比且饱和度偏低",),
                adjustments=tuple(rec.graying_adjustments("橙" if skin else ct.base_hue_name)[1:2]),
                confidence=conf * 0.8,
                notes=("暗调本身不是问题",),
            )
        )
    c = comp.get("outlier")
    if c and c.applicable and c.severity != Severity.NORMAL:
        findings.append(
            Finding(
                problem="存在少量但突出的异常颜色",
                severity=c.severity,
                evidence=c.evidence,
                causes=("杂物、反光点、色噪或选区边缘混入",),
                adjustments=(),
                confidence=conf * 0.8,
                notes=("用“问题位置”视图确认它们在哪里",),
            )
        )
    return findings


def _skin_cast_findings(skin, comp, conf, base_notes):
    out = []
    for where, hue_shift, shift in (
        ("高光", skin.highlight_hue_shift, skin.highlight_shift),
        ("阴影", skin.shadow_hue_shift, skin.shadow_shift),
    ):
        if abs(hue_shift) < 6:
            continue
        da, db = shift[1], shift[2]
        if hue_shift > 0 and da < -3:
            name, label = "青", "发青/发绿"
        elif hue_shift > 0:
            name, label = "黄", "发黄"
        elif db < -3:
            name, label = "洋红", "偏洋红/偏紫"
        else:
            name, label = "红", "偏红"
        sev = severity_from_score(dr._interp(abs(hue_shift), dr.CAST_CURVE))
        if sev == Severity.NORMAL:
            continue
        out.append(
            Finding(
                problem=f"肤色{where}{label}",
                severity=sev,
                evidence=(
                    f"{where}（L* 最{'高' if where == '高光' else '低'} 15%）相对中间调：色相 {hue_shift:+.1f}°，"
                    f"Δa* {da:+.1f}，Δb* {db:+.1f}",
                ),
                causes=(
                    "颜色分级/分离色调作用到了皮肤",
                    "环境光：阴影里常有天空/绿植的反射，高光里常有暖光源",
                ),
                adjustments=tuple(rec.cast_adjustments(where, name)),
                confidence=conf,
                notes=("如果是刻意的风格（例如高光暖、阴影冷），可以忽略",) + tuple(base_notes),
            )
        )
    return out


# ------------------------------------------------------------------ 通用区域
def analyze_region(region: np.ndarray, context: ColorContext, rect=None) -> AnalysisReport:
    stats = rs.compute_stats(region)
    ct = ct_mod.analyze(region, mode="region", offset=_offset(rect))
    conf, reasons = dr.confidence(stats, context, ct)
    chroma = dr.chroma_component_region(ct)
    components = [
        dr.opposite_component(ct),
        dr.hue_dispersion_component(ct),
        chroma,
        dr.local_cast_bands(_bands(region, ct.mask)),
        dr.lightness_component(stats, chroma.score),
        dr.outlier_component(stats, ct),
    ]
    findings = _component_findings(components, ct, conf, reasons, skin=False)
    return _finalize("general", context, stats, components, findings, reasons, conf, contamination=ct, region=rect)


# ------------------------------------------------------------------ 人像肤色
def analyze_skin_region(
    region: np.ndarray, context: ColorContext, preset_key: str | None = None, presets=None, rect=None, preset=None
) -> AnalysisReport:
    stats = rs.compute_stats(region)
    if preset is None and preset_key:
        preset = sp.get_preset(preset_key, presets)
    skin = pa.analyze_skin(region, preset)
    ct = ct_mod.analyze(region, mode="skin", offset=_offset(rect))
    conf, reasons = dr.confidence(stats, context, ct, skin)
    chroma = dr.chroma_component_skin(ct, skin)
    components = [
        dr.opposite_component(ct, skin=True),
        dr.hue_dispersion_component(ct),
        chroma,
        dr.local_cast_skin(skin),
        dr.lightness_component(stats, chroma.score),
        dr.outlier_component(stats, ct),
        dr.skin_cast_component(skin),
    ]
    findings = _component_findings(components, ct, conf, reasons, skin=True, skin_report=skin)
    sc = {c.key: c for c in components}["skin_cast"]
    if sc.severity != Severity.NORMAL:
        target = (sp.get_preset("natural", presets) if preset is None else preset).target_lab
        findings.append(
            Finding(
                problem=f"肤色整体{skin.tendency}",
                severity=sc.severity,
                evidence=sc.evidence,
                causes=("白平衡（色温/色调）偏移", "HSL 橙色/红色的色相被移动"),
                adjustments=tuple(a for a in rec.skin_adjustments(_skin_pixels(region), target) if a.tool == "Camera Raw"),
                confidence=conf,
                notes=(sc.note,) + tuple(reasons),
            )
        )
    if skin.mixed_region:
        findings.append(
            Finding(
                problem="选区不够纯净，建议重新框选皮肤",
                severity=Severity.MILD,
                evidence=("选区中符合肤色色相的像素不足 75%",),
                causes=("框进了头发、眼睛、眉毛、背景或衣服",),
                adjustments=(),
                confidence=conf,
                notes=("请重新选择脸颊、额头等较平整、无妆容遮盖的皮肤区域",),
            )
        )
    adjustments = ()
    if preset is not None:
        cmp = pa.compare_to_preset(skin, preset)
        fit = pa.preset_fit(skin, preset)
        adjustments = tuple(rec.skin_adjustments(_skin_pixels(region), preset.target_lab))
        if cmp.delta_e00 >= 2:
            findings.append(
                Finding(
                    problem=f"与目标风格「{preset.name}」相差 ΔE00 {cmp.delta_e00:.1f}",
                    severity=Severity.MILD if cmp.delta_e00 < 6 else Severity.NOTABLE,
                    evidence=tuple(f"{k}：{v}" for k, v in fit.items())
                    + (f"需要的变化：ΔL* {cmp.dL:+.1f}，Δa* {cmp.da:+.1f}，Δb* {cmp.db:+.1f}，ΔC* {cmp.dC:+.1f}",),
                    causes=("风格目标与当前调色不同（不是错误）",),
                    adjustments=adjustments,
                    confidence=conf,
                    notes=("风格区间是经验值，可在设置中修改或用参考作品取样生成目标",) + tuple(preset.notes),
                )
            )
    return _finalize(
        "skin", context, stats, components, findings, reasons, conf, contamination=ct, skin=skin, adjustments=adjustments, region=rect
    )


def _skin_pixels(region: np.ndarray) -> np.ndarray:
    sub, _ = grid_downsample(region, 20_000)
    px = sub.reshape(-1, 3)
    lch = cs.lab_to_lch(cs.srgb8_to_lab(px))
    sel = (lch[:, 1] >= 3) & (lch[:, 2] <= 110) & (lch[:, 0] >= 10) & (lch[:, 0] <= 98)
    return px[sel] if sel.sum() >= 10 else px


# ------------------------------------------------------------------ 参考对比
def compare_regions(
    ref: np.ndarray, prob: np.ndarray, context: ColorContext, skin: bool = True, ref_rect=None, prob_rect=None
) -> AnalysisReport:
    stats = rs.compute_stats(prob)
    mode = "skin" if skin else "region"
    ct_ref = ct_mod.analyze(ref, mode=mode)
    ct_prob = ct_mod.analyze(prob, mode=mode, offset=_offset(prob_rect))
    if skin:
        ref_lab = cs.srgb8_to_lab(_skin_pixels(ref)).mean(axis=0)
        prob_lab = cs.srgb8_to_lab(_skin_pixels(prob)).mean(axis=0)
    else:
        ref_lab = np.array(rs.compute_stats(ref, with_clusters=False).lab_mean)
        prob_lab = np.array(stats.lab_mean)
    c = pa.compare_lab(ref_lab, prob_lab, ct_prob.opposite_fraction - ct_ref.opposite_fraction)
    conf, reasons = dr.confidence(stats, context, ct_prob)
    chroma = dr.chroma_component_compare(c)
    components = [
        dr.opposite_component(ct_prob, skin=skin),
        dr.hue_dispersion_component(ct_prob),
        chroma,
        dr.local_cast_compare(c),
        dr.lightness_component(stats, chroma.score),
        dr.outlier_component(stats, ct_prob),
    ]
    parts, sev_score = [], dr._interp(c.delta_e00, ([0, 2, 4, 8, 12], [0, 15, 35, 60, 80]))
    if abs(c.dL) >= 3:
        parts.append(f"明度{'更暗' if c.dL < 0 else '更亮'} {abs(c.dL):.1f}（ΔL*）")
    if c.dC <= -3:
        parts.append(f"色度降低 {abs(c.dC):.1f}（发灰、不透亮）")
    elif c.dC >= 3:
        parts.append(f"色度更高 {c.dC:.1f}（更艳）")
    if abs(c.across_ref_hue) >= 2.5 and abs(c.dh) >= 3:
        ref_h = cs.lab_to_lch(ref_lab)[2]
        if skin or 0 <= ref_h <= 110:
            word = ("偏黄" if c.dh > 0 and c.da > -3 else "偏绿/偏青") if c.dh > 0 else ("偏红" if c.db > -3 else "偏洋红/偏紫")
        else:
            word = f"色相偏向{cs.hue_name(cs.lab_to_lch(prob_lab)[2])}"
        parts.append(f"{word}（Δh {c.dh:+.1f}°，Δa* {c.da:+.1f}，Δb* {c.db:+.1f}）")
    if not parts:
        parts.append(f"两个区域非常接近（ΔE00 {c.delta_e00:.1f}）")
    adjustments = tuple(rec.skin_adjustments(_skin_pixels(prob) if skin else prob.reshape(-1, 3), ref_lab))
    main = Finding(
        problem="问题区相对参考区：" + "；".join(parts),
        severity=severity_from_score(sev_score),
        evidence=(
            f"色差 ΔE00 {c.delta_e00:.2f}",
            f"ΔL* {c.dL:+.2f}，ΔC* {c.dC:+.2f}，Δh {c.dh:+.1f}°，Δa* {c.da:+.2f}，Δb* {c.db:+.2f}",
            f"沿参考色相方向的色度变化 {c.along_ref_hue:+.2f}，垂直方向（色相偏移）{c.across_ref_hue:+.2f}",
            f"对立色像素比例变化 {c.opposite_component_change:+.2%}",
            f"参考区 Lab ({ref_lab[0]:.1f}, {ref_lab[1]:.1f}, {ref_lab[2]:.1f})，问题区 Lab ({prob_lab[0]:.1f}, {prob_lab[1]:.1f}, {prob_lab[2]:.1f})",
        ),
        causes=("局部光线/反光不同", "局部调整只作用到了一部分区域", "阴影/高光区本身的颜色变化（属正常时可忽略）"),
        adjustments=adjustments,
        confidence=conf,
        notes=("有参考区时优先使用相对比较的结论；调整建议的目标是把问题区拉回参考区",) + tuple(reasons),
    )
    findings = [main] + _component_findings(
        [x for x in components if x.key in ("opposite", "hue_dispersion")], ct_prob, conf, reasons, skin=skin
    )
    return _finalize(
        "compare", context, stats, components, findings, reasons, conf, contamination=ct_prob, comparison=c, adjustments=adjustments, region=prob_rect
    )


# ------------------------------------------------------------------ 整图扫描
def scan_image(image: np.ndarray, context: ColorContext, rect=None) -> AnalysisReport:
    stats = rs.compute_stats(image)
    ct = ct_mod.analyze(image, mode="scan", offset=_offset(rect))
    conf, reasons = dr.confidence(stats, context, ct)
    conf *= 0.85  # 整图没有语义，结论更保守
    reasons = ["整图扫描不知道哪里是同一物体，只报告“散点状混杂”和“同色局部发灰”，请逐处确认"] + reasons
    f = ct.opposite_fraction
    opp_score = dr._interp(f * 3, dr.OPP_CURVE) if f else 0.0
    components = [
        dr._rc(
            "opposite",
            opp_score,
            [f"散点状对立色像素约占全图 {f:.2%}"] + _blob_text([b for b in ct.blobs if b.kind == "opposite"]),
        ),
        dr.chroma_component_region(ct),
    ]
    findings = []
    for b in ct.blobs[:8]:
        sev = Severity.MILD if b.score < 0.08 else Severity.NOTABLE
        if b.kind == "opposite":
            problem = f"可疑脏色：局部主色中夹杂“{b.contaminant_hue_name}”散点"
            adj = rec.contamination_adjustments(b.contaminant_hue_name or "", "", skin=False)
            causes = CAUSES_OPPOSITE
        else:
            problem = "可疑发灰：比同色区域更灰"
            adj = rec.graying_adjustments("")
            causes = CAUSES_GRAY
        x, y, w, h = b.bbox
        findings.append(
            Finding(
                problem=problem + f"（x {x}, y {y}，{w}×{h}）",
                severity=sev,
                evidence=b.evidence,
                causes=causes,
                adjustments=tuple(adj),
                confidence=conf,
                notes=tuple(reasons[:1]) + ("对可疑位置再用“区域检查”框选，可得到更可靠的结论",),
            )
        )
    return _finalize("scan", context, stats, components, findings, reasons, conf, contamination=ct, region=rect)
