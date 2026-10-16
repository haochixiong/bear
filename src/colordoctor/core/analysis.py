"""分析入口：单点、通用区域、人像肤色、参考对比、整图扫描。

输入是 sRGB uint8 NumPy 数组（已经过色彩管理），输出是不可变的 AnalysisReport。
本模块不依赖任何界面库，可在后台线程中调用。
"""

from __future__ import annotations

import dataclasses

import numpy as np

from colordoctor.core import color_spaces as cs
from colordoctor.core import contamination as ct_mod
from colordoctor.core import diagnosis_rules as dr
from colordoctor.core import layer_plan as lp
from colordoctor.core import native_layers as nl
from colordoctor.core import portrait_analysis as pa
from colordoctor.core import recommendations as rec
from colordoctor.core import region_statistics as rs
from colordoctor.core import skin_mask as skin_mask_mod
from colordoctor.core import skin_presets as sp
from colordoctor.core import skin_uniformity as su
from colordoctor.core.color_models import (
    Adjustment,
    AnalysisReport,
    ColorContext,
    Finding,
    PointSample,
    RiskComponent,
    Severity,
    severity_from_score,
)
from colordoctor.core.perception import de_words
from colordoctor.core.pixel_table import check_cancel, colour_counts, scope, together
from colordoctor.core.sampling import grid_downsample, resolution_limit, sample_point

MAX_SAMPLES = rs.MAX_SAMPLES
SEVERITY_FLOOR = {Severity.MILD: 20.0, Severity.NOTABLE: 45.0, Severity.STRONG: 70.0}  # 与 severity_from_score 的分档一致

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


def _shared_pixels(fn):
    """整个分析过程中同一张图只给颜色编号一次（脏色、统计、分段、肤色、影调共用）。"""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        check_cancel()
        with scope():
            rep = fn(*args, **kwargs)
        check_cancel()
        limit, label = resolution_limit()
        big = any(isinstance(a, np.ndarray) and a.ndim == 3 and a.shape[0] * a.shape[1] > limit for a in args)
        if label != "full" and big and isinstance(rep, AnalysisReport):
            rep = dataclasses.replace(rep, precision=label)  # 这次确实是按缩小的图算的
        return rep

    return wrapper


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


BAND_MIN_DL = 6.0  # 阴影 / 高光段的平均明度与中间调至少差这么多，才算“另一段”（明暗起伏很小的区域不分段比较）


def _bands(region: np.ndarray, mask: np.ndarray | None) -> dict:
    sub, _ = grid_downsample(region, MAX_SAMPLES)
    keep_px = None
    if mask is not None and mask.size == sub.shape[0] * sub.shape[1]:
        keep_px = mask.reshape(sub.shape[:2]) == ct_mod.MASK_NORMAL
    # 原图尺寸的每个像素：按“不同的颜色 × 像素数”加权（和脏色检查共用同一份颜色编号）
    cols, w = colour_counts(sub, keep_px)
    keep = ~cs.clipped(cols)  # 被裁切的纯白 / 死黑没有颜色信息
    cols, w = cols[keep], w[keep].astype(np.float64)
    if w.sum() < 20:
        return {}
    lab = np.asarray(cs.srgb8_to_lab(cols), dtype=np.float64)
    L = lab[:, 0]
    p15, p40, p60, p85 = cs.weighted_percentile(L, w, [15, 40, 60, 85])
    out = {}
    for key, sel in (("shadow", L <= p15), ("mid", (L >= p40) & (L <= p60)), ("highlight", L >= p85)):
        if w[sel].sum() >= 5:
            out[key] = (lab[sel] * w[sel][:, None]).sum(axis=0) / w[sel].sum()
    mid = out.get("mid")
    for key in ("shadow", "highlight"):
        if mid is not None and key in out and abs(out[key][0] - mid[0]) < BAND_MIN_DL:
            del out[key]
    return out


def _finalize(mode, context, stats, components, findings, extra_conf_reasons, conf, **kw) -> AnalysisReport:
    risk = dr.overall(components)
    sev = severity_from_score(risk)
    order = [Severity.NORMAL, Severity.MILD, Severity.NOTABLE, Severity.STRONG]
    # “影调”卡说的是离用户选的目标影调多远，不是脏色 / 偏色问题：不抬高总体程度，也不挡住“未发现明显问题”
    # 6:3:1 是构图经验、不是错误（单一色系的画面本来就“不符合”）：也不抬高总体程度
    problems = [f for f in findings if not f.problem.startswith(("影调：", "色彩比例"))]
    shown = max((order.index(f.severity) for f in problems), default=0)
    if shown > order.index(sev):  # 总体程度不低于任何一张卡片，风险分随之抬到该档的下限
        sev = order[shown]
        risk = max(risk, SEVERITY_FLOOR[sev])
    if not [f for f in problems if f.severity != Severity.NORMAL]:
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
    # 整图的“影调”卡固定在最前（先定明暗，再看颜色）；其余真实问题优先，“与目标风格的差距”不是错误，排在问题之后
    findings = sorted(findings, key=lambda f: (not f.problem.startswith("影调："), f.problem.startswith("与目标风格"),
                                               -order.index(f.severity)))
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
                adjustments=tuple(rec.contamination_adjustments(main, ct.base_hue_name if not skin else "橙", skin, ct.opposite_chroma,
                                                                hue=ct.opposite_hue)),
                confidence=conf,
                notes=tuple(notes),
            )
        )
    c = comp.get("chroma_compression")
    lc = comp.get("lightness_compression")
    graying_card = bool(c and c.severity != Severity.NORMAL)
    if graying_card:
        problem = "肤色发灰、不够透亮（色度偏低）" if skin else f"“{ct.base_hue_name}”区域局部发灰（比同色干净处更灰）"
        deficit = None
        if skin and skin_report is not None:
            lo = skin_report.expected_saturation[0]
            deficit = max(0.0, (lo - skin_report.saturation_ab) / lo) if lo > 0 else 0.0
        # 明度也被压缩时并进这一条（建议相同，不再单独出一张卡）
        flat = lc.evidence if lc and lc.severity != Severity.NORMAL else ()
        findings.append(
            Finding(
                problem=problem,
                severity=c.severity,
                evidence=c.evidence + tuple(flat) + tuple(_blob_text([b for b in ct.blobs if b.kind == "graying"])),
                causes=CAUSES_GRAY,
                adjustments=tuple(rec.graying_adjustments("橙" if skin else ct.base_hue_name, ct.graying_depth, skin_deficit=deficit,
                                                         skin_lab=skin_report.mean_lab if skin_report is not None else None)),
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
                adjustments=tuple(rec.dispersion_adjustments(ct.base_hue_name, skin, ct.base_hue, ct.hue_spread, C=ct.base_chroma)),
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
    c = lc
    if c and c.severity != Severity.NORMAL and not graying_card:
        findings.append(
            Finding(
                problem="明度和色度同时被压缩，显得发闷",
                severity=c.severity,
                evidence=c.evidence,
                causes=("曲线压低了对比且饱和度偏低",),
                adjustments=(rec.contrast_adjustment("skin" if skin else "global"),),
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
                adjustments=tuple(rec.cast_adjustments(where, name, shift, mid_lab=skin.mean_lab)),
                confidence=conf,
                notes=("如果是刻意的风格（例如高光暖、阴影冷），可以忽略",) + tuple(base_notes),
            )
        )
    return out


# ------------------------------------------------------------------ 通用区域
@_shared_pixels
def analyze_region(region: np.ndarray, context: ColorContext, rect=None) -> AnalysisReport:
    stats, ct = together(lambda: rs.compute_stats(region), lambda: ct_mod.analyze(region, mode="region", offset=_offset(rect)))
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
@_shared_pixels
def analyze_skin_region(
    region: np.ndarray,
    context: ColorContext,
    preset_key: str | None = None,
    presets=None,
    rect=None,
    preset=None,
    auto_mask: bool = True,
    lasso: np.ndarray | None = None,
    part: str | None = None,
    other_part: tuple[str, tuple[float, float, float]] | None = None,
    quick_rule: tuple | None = None,
    manual_mask: np.ndarray | None = None,
) -> AnalysisReport:
    """人像肤色分析。

    manual_mask：用户手动修正后的皮肤蒙版（优先于自动抠皮肤）。

    auto_mask：在选区内自动只保留皮肤像素（去掉头发、眼睛、嘴唇、背景）。
    lasso：可选的套索蒙版（与 region 同尺寸）。
    part / other_part：部位名称，以及另一个部位的 (名称, 平均 Lab)，用于“脸部 vs 身体”对比。
    """
    valid = lasso
    mask_info: dict = {}
    if manual_mask is not None:
        valid = manual_mask & lasso if lasso is not None else manual_mask
        mask_info = {"skin_share": float(valid.mean()), "note": "使用手动修正后的皮肤蒙版", "manual": True}
        auto_mask = False
    if auto_mask:
        m, mask_info = skin_mask_mod.skin_mask(region, within=lasso)
        if m.sum() >= 30:
            valid = m
        else:
            mask_info["note"] = (mask_info.get("note") or "") + " 自动抠皮肤失败，改用整个选区"
    if preset is None and preset_key:
        preset = sp.get_preset(preset_key, presets)
    stats, ct, uni = together(
        lambda: rs.compute_stats(region, valid=valid),
        lambda: ct_mod.analyze(region, mode="skin", offset=_offset(rect), valid=valid),
        lambda: su.analyze_uniformity(region, mask=valid, offset=_offset(rect)),
    )
    skin = pa.analyze_skin(region, preset, valid=valid, contamination=ct)
    rule = quick_rule or (preset.quick_rule if preset is not None and preset.quick_rule else sp.global_quick_rule())
    skin_px, skin_w = _skin_pixels(region, valid)
    qr = pa.quick_rule_check(cs.srgb8_to_lab(skin_px), rule, weights=skin_w)
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
        dr.uniformity_component(uni),
    ]
    findings = _component_findings(components, ct, conf, reasons, skin=True, skin_report=skin)
    findings.append(_quick_rule_finding(qr, part, conf))
    findings += _uniformity_findings(uni, conf, reasons)
    preset_lab = (sp.get_preset("natural", presets) if preset is None else preset).target_lab
    tgt_h = float(cs.lab_to_lch(np.asarray(preset_lab, dtype=float))[2])
    skin = dataclasses.replace(skin, target_hue=round(tgt_h, 3))
    cmp = pa.compare_to_preset(skin, preset) if preset is not None else None
    style_card = cmp is not None and cmp.delta_e00 >= rec.NEAR_DE
    sc = {c.key: c for c in components}["skin_cast"]
    if sc.severity != Severity.NORMAL:
        # 这条只管“偏色”（色相）：保留当前的明度和色度，只把色相拉向目标。
        # 否则会按“自然肤色”的明度要求深肤色提亮、浅肤色压暗，违背“不按深浅判断肤色”的原则。
        cur_lch = cs.lab_to_lch(np.asarray(skin.mean_lab, dtype=float))
        target = cs.lch_to_lab(np.array([cur_lch[0], cur_lch[1], tgt_h]))
        adjs = []
        sc_params = nl.skin_selective(float(target[1] - skin.mean_lab[1]), float(target[2] - skin.mean_lab[2]))
        if style_card:
            # 风格那一条的“拉向目标”曲线已经把色相拉到同一个目标：这里再给一层会修两次（叠加后反而偏到另一边）
            adjs.append(Adjustment(
                "提示", f"这条偏色的修正已包含在“与目标风格「{preset.name}」”那一条的“肤色拉向目标 · 曲线”图层里（同一个目标）："
                "只生成那一条的图层，不要再另加可选颜色或色相调整，否则会修两次", None, False))
        elif sc_params["colors"]:
            adjs.append(nl.adjustment(
                "selectiveColor", sc_params,
                f"方法“相对”：{nl.selective_text(sc_params)}（把肤色色相拉回{skin.hue_ref}，明暗和浓淡基本不变；"
                "用皮肤蒙版只作用于皮肤，生成的调整图层就是这一层）",
                "skin", f"肤色整体{skin.tendency} · 可选颜色", "skin_cast",
                strength=rec.strength_label(float(cs.hue_diff(cur_lch[2], tgt_h)) / 2.0)))
        if not style_card:
            adjs += [a for a in rec.skin_adjustments(skin_px, target, keep_lightness=True, weights=skin_w) if a.tool == "Camera Raw"]
        findings.append(
            Finding(
                problem=f"肤色整体{skin.tendency}",
                severity=sc.severity,
                evidence=sc.evidence,
                causes=("白平衡（色温/色调）偏移", "HSL 橙色/红色的色相被移动"),
                adjustments=tuple(adjs),
                confidence=conf,
                notes=(sc.note,) + tuple(reasons),
            )
        )
    if skin.mixed_region and valid is None:
        findings.append(
            Finding(
                problem="选区不够纯净，建议重新框选皮肤",
                severity=Severity.MILD,
                evidence=("选区中符合肤色色相的像素不足 75%",),
                causes=("框进了头发、眼睛、眉毛、背景或衣服",),
                adjustments=(),
                confidence=conf,
                notes=("请重新选择脸颊、额头等较平整的皮肤区域，或开启“自动抠皮肤”/使用套索",),
            )
        )
    if other_part is not None:
        findings.append(_part_compare_finding(part or "当前部位", skin.mean_lab, other_part, (skin_px, skin_w), conf))
    adjustments = ()
    if preset is not None:
        fit = pa.preset_fit(skin, preset)
        # 卡片说“基本一致”（ΔE00 < 2）时不给调整
        # 目标的明度：风格区间内不动、超出才拉回区间边缘；“自然通用”保持当前明度（不因为画面暗就提亮皮肤）
        # 皮肤的明度一次最多拉 SKIN_MAX_DL（整张图的明暗由整图色彩的「影调风格」管，两边不要叠成一大步）
        tgt = preset.target_for(skin.mean_lab)
        tgt = (skin.mean_lab[0] + float(np.clip(tgt[0] - skin.mean_lab[0], -SKIN_MAX_DL, SKIN_MAX_DL)), tgt[1], tgt[2])
        adjustments = (tuple(rec.skin_adjustments(skin_px, tgt, keep_lightness=preset.keeps_lightness, weights=skin_w))
                       if style_card
                       else (rec.near_hint(cmp.delta_e00),))
        if style_card:
            findings.append(
                Finding(
                    problem=f"与目标风格「{preset.name}」相差 ΔE00 {cmp.delta_e00:.1f}（{de_words(cmp.delta_e00)}）",
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
        "skin",
        context,
        stats,
        components,
        findings,
        reasons,
        conf,
        contamination=ct,
        skin=skin,
        adjustments=adjustments,
        region=rect,
        part=part,
        quick_rule=qr,
        uniformity=uni,
        valid_mask=valid,
        mask_info=mask_info,
    )


def _quick_rule_finding(qr, part, conf) -> Finding:
    (alo, ahi), (blo, bhi) = qr.a_range, qr.b_range
    ok = "✓" if qr.passed else "✗"
    return Finding(
        problem=f"{ok} 经验快速判定（{part or '肤色'}）：{qr.verdict}",
        severity=Severity.NORMAL if qr.passed else Severity.MILD,
        evidence=(
            f"平均 a* {qr.a:.1f}（{alo:g}–{ahi:g}）{'✓' if qr.a_ok else '✗'}，"
            f"b* {qr.b:.1f}（{blo:g}–{bhi:g}）{'✓' if qr.b_ok else '✗'}，"
            f"|a*−b*| {abs(qr.a - qr.b):.1f}（≤{qr.max_diff:g}）{'✓' if qr.diff_ok else '✗'}",
            f"皮肤像素中满足该规则的比例：{qr.pixel_pass_share:.0%}",
        ),
        causes=(),
        adjustments=tuple(Adjustment("提示", t, None, True) for t in qr.tips),
        confidence=conf,
        notes=("规则可在 skin_presets.json 的 quick_rule 中修改；部分风格（如法式、小麦、深肤色）有自己的区间",),
    )


def _uniformity_findings(uni, conf, reasons) -> list[Finding]:
    out = []
    if uni is None or not uni.patches:
        return out
    words = {
        "偏黄": ("比其他皮肤更黄", "局部画笔：色温往蓝一点，或在该处 HSL 黄色/橙色色相往红{sat}"),
        "偏红橙": ("比其他皮肤更红/更橙", "局部画笔：色调往绿一点，或 HSL 红色/橙色色相往黄{sat}"),
        "偏洋红": ("偏洋红/偏紫", "局部画笔：色调往绿；或 色相/饱和度 选洋红降饱和"),
        "偏青绿": ("偏青、偏绿", "局部画笔：色调往洋红、色温往黄；或 可选颜色 青色 减青"),
        "缺色发灰": ("色度明显低于同明度的皮肤（缺色、发灰）", "局部画笔：饱和度/自然饱和度加一点，或 HSL 橙色饱和度提高（配合蒙版）"),
        "饱和度偏高": ("比其他皮肤更艳", "局部画笔：饱和度降一点，或 HSL 橙色/红色饱和度降低（配合蒙版）"),
    }
    for kind in dict.fromkeys(p.kind for p in uni.patches):
        ps = [p for p in uni.patches if p.kind == kind]
        area = sum(p.area_fraction for p in ps)
        sev = Severity.NOTABLE if any(p.severity == "明显" for p in ps) else Severity.MILD
        desc, fix = words[kind]
        # 饱和度方向跟着小块实际的饱和度走（与生成的图层同一个数）：更艳才降，更灰就加
        ds = float(np.average([p.sat_shift for p in ps], weights=[p.area_fraction for p in ps]))
        sat = lp.patch_sat(ds)
        fix = fix.format(sat="" if not sat else f"、饱和度略{'降' if sat < 0 else '加'}（约 {sat:+d}）")
        adjs = [Adjustment("Camera Raw / PS 局部", fix, "轻微")]
        lab = tuple(float(v) for v in np.average([p.lab for p in ps], axis=0, weights=[p.area_fraction for p in ps]))
        dh = float(np.average([p.hue_shift for p in ps], weights=[p.area_fraction for p in ps]))
        spec = nl.skin_patch_layer(kind, dh, ds, lab)
        if spec is not None:
            lkind, params, text = spec
            adjs.append(nl.adjustment(lkind, params, f"{text}（只作用于这些位置）", "local",
                                      f"肤色{kind} · {nl.KIND_NAME[lkind]}", f"skin@{kind}", strength="轻微"))
        out.append(
            Finding(
                problem=f"肤色不均匀：{len(ps)} 处{kind}（占皮肤 {area:.0%}）",
                severity=sev,
                evidence=tuple(
                    f"位置 x {p.bbox[0]}, y {p.bbox[1]}，{p.bbox[2]}×{p.bbox[3]}：{desc}，"
                    f"色相 {p.hue_shift:+.1f}°，饱和度 {p.sat_shift:+.0%}，Lab ({p.lab[0]:.0f}, {p.lab[1]:.1f}, {p.lab[2]:.1f})"
                    for p in ps[:4]
                )
                + (f"整体色相中位 h {uni.median_hue:.1f}°，均匀度 {uni.score:.0f}/100",),
                causes=("光线不均（窗光/反光/不同光源）", "局部调整（画笔、蒙版）边缘没过渡好", "皮肤本身的红血丝、晒痕或妆容"),
                adjustments=tuple(adjs),
                confidence=conf,
                notes=("“问题位置”页会标出这些区域",) + tuple(reasons[:1]),
            )
        )
    return out


def _part_compare_finding(part, lab, other, skin_px, conf) -> Finding:
    other_name, other_lab = other
    c = pa.compare_lab(other_lab, lab)
    parts = []
    if abs(c.dL) >= 3:
        parts.append(f"{'更暗' if c.dL < 0 else '更亮'} {abs(c.dL):.1f}")
    if abs(c.dh) >= 4 and np.hypot(c.da, c.db) >= 2:
        parts.append("更黄" if c.dh > 0 else "更红")
    if abs(c.dC) >= 3:
        parts.append(f"{'更灰' if c.dC < 0 else '更艳'} {abs(c.dC):.1f}")
    sev = severity_from_score(dr._interp(c.delta_e00, ([0, 3, 5, 8, 12], [0, 20, 45, 65, 80])))
    return Finding(
        problem=f"{part}与{other_name}的肤色差异：ΔE00 {c.delta_e00:.1f}（{de_words(c.delta_e00)}"
        + (f"；{part}{'、'.join(parts)}）" if parts else "，基本一致）"),
        severity=sev,
        evidence=(
            f"{other_name} Lab ({other_lab[0]:.1f}, {other_lab[1]:.1f}, {other_lab[2]:.1f})，{part} Lab ({lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f})",
            f"ΔL* {c.dL:+.1f}，Δa* {c.da:+.1f}，Δb* {c.db:+.1f}，ΔC* {c.dC:+.1f}，Δh {c.dh:+.1f}°",
        ),
        causes=("脸部和身体受光不同", "只对脸部做了局部调整", "化妆（粉底色号）与身体肤色不同"),
        adjustments=tuple(rec.skin_adjustments(skin_px[0], other_lab, weights=skin_px[1])) if parts else (rec.near_hint(c.delta_e00),),
        confidence=conf,
        notes=(f"调整方向是把{part}拉向{other_name}；如果想反过来，请在另一部位上分析",),
    )


def _skin_pixels(region: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """皮肤像素（原图尺寸的每一个）-> (不同的颜色, 每种的像素数)。"""
    if valid is not None and np.count_nonzero(valid) >= 10:
        return colour_counts(region, valid)
    cols, w = colour_counts(region)
    lch = cs.lab_to_lch(cs.srgb8_to_lab(cols))
    sel = (lch[:, 1] >= 3) & (lch[:, 2] <= 110) & (lch[:, 0] >= 10) & (lch[:, 0] <= 98)
    return (cols[sel], w[sel]) if w[sel].sum() >= 10 else (cols, w)


def _mean_lab(cw) -> np.ndarray:
    cols, w = cw
    w = np.asarray(w, dtype=np.float64)
    return (np.asarray(cs.srgb8_to_lab(cols), dtype=np.float64) * w[:, None]).sum(axis=0) / w.sum()


# ------------------------------------------------------------------ 参考对比
@_shared_pixels
def compare_regions(
    ref: np.ndarray, prob: np.ndarray, context: ColorContext, skin: bool = True, ref_rect=None, prob_rect=None
) -> AnalysisReport:
    stats = rs.compute_stats(prob)
    mode = "skin" if skin else "region"
    ct_ref = ct_mod.analyze(ref, mode=mode)
    ct_prob = ct_mod.analyze(prob, mode=mode, offset=_offset(prob_rect))
    if skin:
        ref_lab = _mean_lab(_skin_pixels(ref))
        prob_lab = _mean_lab(_skin_pixels(prob))
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
            # 按问题区本身的颜色命名：a* 仍为正就是“更黄 / 更红”，只有 a* < 0 才算偏绿、b* < 0 才算偏青/偏紫
            pa_, pb_ = float(prob_lab[1]), float(prob_lab[2])
            if c.dh > 0:
                word = "偏黄" if pa_ >= 0 else ("偏绿" if pb_ >= 0 else "偏青")
            else:
                word = "偏红" if pb_ >= 0 else "偏洋红/偏紫"
        else:
            word = f"色相偏向{cs.hue_name(cs.lab_to_lch(prob_lab)[2])}"
        parts.append(f"{word}（Δh {c.dh:+.1f}°，Δa* {c.da:+.1f}，Δb* {c.db:+.1f}）")
    if not parts:
        parts.append(f"两个区域非常接近（ΔE00 {c.delta_e00:.1f}，{de_words(c.delta_e00)}）")
        adjustments = (rec.near_hint(c.delta_e00, skin),)
    else:
        # 图层只作用于问题区（分析的就是这块区域）
        cols, w = _skin_pixels(prob) if skin else colour_counts(prob)
        adjustments = tuple(rec.skin_adjustments(cols, ref_lab, skin=skin, scope="local", at="region", weights=w))
    main = Finding(
        problem="问题区相对参考区：" + "；".join(parts),
        severity=severity_from_score(sev_score),
        evidence=(
            f"色差 ΔE00 {c.delta_e00:.2f}（{de_words(c.delta_e00)}）",
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
@_shared_pixels
def scan_image(image: np.ndarray, context: ColorContext, rect=None, valid: np.ndarray | None = None) -> AnalysisReport:
    """valid：可选的有效像素蒙版（例如半透明图层的不透明部分），蒙版外不计入。"""
    stats, ct = together(lambda: rs.compute_stats(image, valid=valid),
                         lambda: ct_mod.analyze(image, mode="scan", offset=_offset(rect), valid=valid))
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
    # 每类里最主要的一处用该分项的程度（与总体程度一致），其余按该处自身的浓度，但不超过分项
    comp_sev = {"opposite": components[0].severity, "graying": components[1].severity}
    order = [Severity.NORMAL, Severity.MILD, Severity.NOTABLE, Severity.STRONG]
    seen: set[str] = set()
    for j, b in enumerate(ct.blobs[:8]):
        own = Severity.MILD if b.score < 0.08 else Severity.NOTABLE
        top = comp_sev[b.kind]
        sev = top if b.kind not in seen else min(own, top, key=order.index)
        if top != Severity.NORMAL:  # 分项整体正常时，零星的可疑位置也按正常列出，不把总体程度抬上去
            sev = max(sev, Severity.MILD, key=order.index)
        seen.add(b.kind)
        if b.kind == "opposite":
            problem = f"可疑脏色：“{b.base_hue_name}”中夹杂“{b.contaminant_hue_name}”散点"
            adj = rec.contamination_adjustments(b.contaminant_hue_name or "", b.base_hue_name, False, ct.opposite_chroma,
                                                hue=b.contaminant_hue, at=f"opposite@{j}",
                                                name=f"{b.base_hue_name}里的{b.contaminant_hue_name}散点")
            causes = CAUSES_OPPOSITE
        else:
            problem = f"可疑发灰：这块“{b.base_hue_name}”比同色区域更灰"
            adj = rec.graying_adjustments(b.base_hue_name, ct.graying_depth, at=f"graying@{j}")
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


# ------------------------------------------------------------------ 整图色彩
@_shared_pixels
def analyze_palette_report(
    image: np.ndarray, context: ColorContext, style_key: str | None = None, family: str | None = None,
    family_target_lab=None, rect=None, tone_style: str = "", tone_target: str = "",
) -> AnalysisReport:
    """整图：影调（九种之一，选了影调风格时给出往目标影调走的曲线）、缺色/丰富度、6:3:1、颜色统一、与颜色模板的差距。

    影调和颜色互相独立：tone_style（影调风格，见 tone_class.TONE_STYLES）/ tone_target（直接选的目标影调）只管明暗和反差，
    style_key（颜色模板）只管颜色。两个都可以不选。
    """
    from colordoctor.core import palette as pl
    from colordoctor.core import tone_class as tcl

    stats = rs.compute_stats(image, with_clusters=False)
    pr = pl.analyze_palette(image, style_key, family, family_target_lab, offset=_offset(rect),
                            tone_style=tone_style, tone_target=tone_target)
    conf, reasons = dr.confidence(stats, context)
    # 和 analyze_palette 用同一批像素：标题 / 九宫格里的影调和影调卡、曲线一定一致
    sub, _ = grid_downsample(image, pl.MAX_SAMPLES)
    tone_card, _tc, _target = tcl.tone_finding(sub[..., :3], None, tone_style, tone_target, conf)
    findings: list[Finding] = [tone_card]
    cf = pr.colorfulness
    style = pr.style
    lo, hi = style.colorfulness_range if style else (15.0, 200.0)
    mono = pl.is_monochrome(cf)
    # 暗调 / 高调画面颜色少是常见的特点：没选颜色模板、或选的不是鲜艳的模板时不按“缺色”加饱和
    keyed_soft = bool(pr.tone_key) and not mono and cf.value < lo and lo < COLOURFUL_STYLE_MIN
    if mono:  # 黑白作品：没有颜色可缺，不按“缺色/发灰”评估
        findings.append(Finding(
            problem="黑白作品，不评估缺色",
            severity=Severity.NORMAL,
            evidence=(f"有彩像素（C* ≥ {pl.CHROMA_MIN:g}）只占 {cf.chromatic_share:.1%}，P90 色度 {cf.p90_chroma:.1f}",),
            causes=(),
            adjustments=(),
            confidence=conf,
            notes=("如果这是想要彩色的照片，检查是否误开了黑白/去色",),
        ))
        lo = min(lo, cf.value)
    elif keyed_soft:
        findings.append(Finding(
            problem=(f"目标影调是{pr.tone_target}（{pl.KEY_NAME[pr.tone_key]}），颜色少不按缺色处理" if pr.tone_target
                     else f"{pl.KEY_NAME[pr.tone_key]}画面，颜色少不按缺色处理"),
            severity=Severity.NORMAL,
            evidence=(f"色彩丰富度 {cf.value:.0f}（{cf.label}），中位明度 {pr.tone[2]:.0f}",
                      f"有彩像素平均色度 C* {cf.mean_chroma:.1f}，P90 色度 {cf.p90_chroma:.1f}"),
            causes=(),
            adjustments=(),
            confidence=conf,
            notes=(f"{'暗' if pr.tone_key == 'low' else '亮'}调画面的颜色天然更少；想要更鲜艳时请在「颜色」里选鲜艳的模板再判断",),
        ))
        lo = min(lo, cf.value)
    elif cf.value < lo:
        sev = Severity.NOTABLE if cf.value < lo * 0.6 else Severity.MILD
        vib = nl.clip(10 + 40 * (lo - cf.value) / lo, 10, 25)  # 缺得越多加得越多（+10 ~ +25）
        # 颜色卡不改明暗：亮暗差偏小时只提示，反差由影调卡（影调风格）的曲线管，避免两条曲线叠加或互相抵消
        contrast = float(pr.tone[4] - pr.tone[0])
        tone_has_curve = any(x.layer for x in tone_card.adjustments)
        if contrast >= LOW_CONTRAST:
            tone_adj = ()
        elif tone_has_curve:
            tone_adj = (Adjustment("提示", f"亮暗差偏小（P95−P5 明度 {contrast:.0f}）：反差按上面“影调”那张卡的曲线调，不要再另加 S 曲线",
                                   None, False),)
        elif pr.tone_target:  # 已经符合目标影调：短调本来就反差小，不再提反差
            tone_adj = ()
        else:
            # 不猜意图：只列出和现在同一基调的长调风格（暗的图不推荐往亮走）
            cls = pr.tone_class
            same = [v["name"] for v in tcl.TONE_STYLES.values() if v["target"] == f"{cls.key}长调"]
            tone_adj = (Adjustment("提示", f"亮暗差偏小（P95−P5 明度 {contrast:.0f}，现在是{cls.name}）。如果想加反差，可以在「影调风格」里选"
                                         f"{'、'.join(same) if same else f'{cls.key}长调'}（保持现在的{cls.key}调），会生成对应的曲线；"
                                         "刻意的柔和效果就保持不动", None, False),)
        findings.append(Finding(
            problem=f"整体缺色/发灰：色彩丰富度 {cf.value:.0f}（{cf.label}）" + (f"，低于「{style.name}」的 {lo:g}" if style else ""),
            severity=sev,
            evidence=(f"有彩像素平均色度 C* {cf.mean_chroma:.1f}，P90 色度 {cf.p90_chroma:.1f}", f"有彩像素占 {cf.chromatic_share:.0%}"),
            causes=("整体降饱和过多、去朦胧为负", "曝光/对比偏低导致发闷", "白平衡把颜色中和掉了"),
            adjustments=(
                nl.adjustment("vibrance", {"vibrance": vib, "saturation": 0},
                              f"自然饱和度 +{vib}，饱和度 0（比饱和度更保护肤色；Camera Raw 里同名滑块也可以）", "global",
                              "整体缺色 · 自然饱和度"),
                *tone_adj,
                Adjustment("Camera Raw 混色器", "只提高主色所在通道的饱和度，其他颜色保持", "小幅"),
            ),
            confidence=conf,
            notes=("低饱和的颜色（莫兰迪、日系清新）本身不是错误；请在「颜色」里选择对应模板再判断",),
        ))
    elif style and cf.value > hi:
        findings.append(Finding(
            problem=f"颜色过艳：色彩丰富度 {cf.value:.0f}，高于「{style.name}」的 {hi:g}",
            severity=Severity.MILD, evidence=(f"P90 色度 {cf.p90_chroma:.1f}",), causes=("饱和度/自然饱和度过高",),
            adjustments=(nl.adjustment("vibrance", {"vibrance": -nl.clip(10 + 40 * (cf.value - hi) / cf.value, 10, 20), "saturation": 0},
                                       f"自然饱和度 {-nl.clip(10 + 40 * (cf.value - hi) / cf.value, 10, 20):+d}，饱和度 0"
                                       "（Camera Raw 里同名滑块也可以）", "global", "颜色过艳 · 自然饱和度"),), confidence=conf,
        ))
    r = pr.ratio
    findings.append(Finding(
        problem=f"色彩比例（6:3:1）：{r.verdict}，得分 {r.score:.0f}",
        severity=Severity.NORMAL if r.score >= 75 else (Severity.MILD if r.score >= 55 else Severity.NOTABLE),
        evidence=tuple(f"{role}：{p.family} {p.share:.0%}（Lab {p.lab[0]:.0f}, {p.lab[1]:.0f}, {p.lab[2]:.0f}）"
                       for role, p in zip(("主色", "辅色", "点缀色"), r.roles)) + (f"其余颜色 {r.other_share:.0%}",),
        causes=(),
        # 已经判定“整体缺色”时不再给“杂色降饱和”的图层（与加饱和度那一层方向相反）
        adjustments=tuple(Adjustment("构图/调色", t, None, True) for t in r.tips)
        + (() if any(f.problem.startswith("整体缺色") for f in findings) else _ratio_layers(pr, r)),
        confidence=conf,
        notes=("6:3:1 是设计和电影常用的经验比例，不是硬性规则；中性色（黑白灰）也算一种颜色",),
    ))
    if pr.unity is not None:
        u = pr.unity
        findings.append(Finding(
            problem=f"“{u.family}”统一度 {u.score:.0f}/100" + ("（不统一）" if u.score < 70 else "（基本统一）"),
            severity=Severity.NORMAL if u.score >= 80 else (Severity.MILD if u.score >= 60 else Severity.NOTABLE),
            evidence=(f"{u.family}占画面 {u.share:.0%}，色相标准差 {u.hue_spread:.1f}°，饱和度变异 {u.sat_cv:.0%}，明度标准差 {u.lightness_std:.1f}",)
            + ((f"与目标相差 ΔE00 {u.delta_e_to_target:.1f}（{de_words(u.delta_e_to_target)}）",) if u.delta_e_to_target is not None else ())
            + tuple(f"位置 x {p.bbox[0]}, y {p.bbox[1]}，{p.bbox[2]}×{p.bbox[3]}：{p.note}" for p in u.patches[:4]),
            causes=("不同光源/阴影里的同一种颜色偏向不同", "混色器只调了部分通道"),
            adjustments=u.adjustments,
            confidence=conf,
        ))
    if style is not None:
        findings.append(Finding(
            problem=f"与颜色「{style.name}」的差距：匹配度 {style.score:.0f}/100",
            severity=Severity.NORMAL if style.score >= 80 else (Severity.MILD if style.score >= 60 else Severity.NOTABLE),
            evidence=style.gaps,
            causes=(),
            adjustments=style.adjustments,
            confidence=conf,
            notes=("滑块数值为估算（Adobe 未公开算法），先按方向调整，再重新分析看差距",) + style.notes,
        ))
    comps = [
        dr._rc("chroma_compression", dr._interp(max(0.0, lo - cf.value), ([0, 5, 15, 30], [0, 25, 50, 75])),
               [f"色彩丰富度 {cf.value:.0f}（{cf.label}）"]),
    ]
    return _finalize("palette", context, stats, comps, findings, reasons, conf, palette=pr, region=rect)


SKIN_MAX_DL = 12.0  # 肤色风格一次最多把皮肤明度拉这么多 L*（约 30 级）
LOW_CONTRAST = 50.0  # 整图亮暗差（P95 − P5 明度）低于此值才算“对比偏低”
COLOURFUL_STYLE_MIN = 28.0  # 颜色模板的色彩丰富度下限不低于此值才算“鲜艳”（暗调 / 高调画面只有选了这种模板才按缺色处理）

_NO_DESAT = ("橙", "黑/深灰", "白/浅灰", "灰")  # 不自动降饱和：肤色所在的橙、无彩色


def _ratio_layers(pr, r) -> tuple[Adjustment, ...]:
    """6:3:1 里“其余颜色偏杂”：把这些颜色降饱和（一层色相/饱和度，范围各自对准；主色、辅色、点缀色和肤色不动）。"""
    if r.score >= 75 or r.other_share <= 0.1:
        return ()
    keep = [nl.ps_hsb_of_lab(p.lab) for p in r.roles if p.chroma >= 8]
    items, words = [], []
    for p in pr.palette[3:6]:
        if p.family in _NO_DESAT or p.chroma < 8:
            continue
        c = nl.ps_hsb_of_lab(p.lab)
        if any(abs(((c - k + 180) % 360) - 180) < 30 for k in keep):
            continue  # 离主色/辅色/点缀色太近，会一起被降掉
        items.append((c, 8, 25, 0, -25, 0))
        words.append(f"{p.family}（范围对准色相 {c:.0f}°）饱和度 -25")
    if not items:
        return ()
    return (nl.adjustment("hueSaturation", nl.huesat_multi(items), "其余的杂色降饱和：" + "；".join(words) + "（主色、辅色、点缀色不动）",
                          "global", "杂色降饱和 · 色相/饱和度", strength="小幅"),)


@_shared_pixels
def compare_reference(mine: np.ndarray, ref: np.ndarray, context: ColorContext, skin_mine=None, skin_ref=None, rect=None) -> AnalysisReport:
    """仿色：参考是什么色彩，我的图差多少，适不适合照着调。"""
    from colordoctor.core import reference_match as rm

    m, stats = together(lambda: rm.match_reference(mine, ref, skin_mine, skin_ref),
                        lambda: rs.compute_stats(mine, with_clusters=False))
    conf, reasons = dr.confidence(stats, context)
    sev = Severity.NORMAL if m.similarity >= 80 else (Severity.MILD if m.similarity >= 60 else Severity.NOTABLE)
    findings = [
        Finding(
            problem=f"与参考的相似度 {m.similarity:.0f}/100",
            severity=sev,
            evidence=m.gaps + tuple(f"参考主色 {f}（{s:.0%}）↔ 我的 {mf}，ΔE00 {de:.0f}（{de_words(de)}）" for f, s, mf, de in m.palette_match),
            causes=(),
            adjustments=m.adjustments,
            confidence=conf,
            notes=("曲线点对影调分位数是精确的；颜色分级和混色器数值为估算",),
        ),
        Finding(
            problem="合不合适照着调：" + ("比较合适" if len(m.fit) == 1 and "适合" in m.fit[0] else "有需要注意的地方"),
            severity=Severity.NORMAL if (len(m.fit) == 1 and "适合" in m.fit[0]) else Severity.MILD,
            evidence=m.fit,
            causes=(),
            adjustments=(),
            confidence=conf * 0.8,
            notes=("这是基于数值的规则判断；需要审美层面的意见可以用“AI 点评”",),
        ),
    ]
    comps = [dr._rc("chroma_compression", 0.0, [f"色彩丰富度 参考 {m.colorfulness_ref:.0f} / 我的 {m.colorfulness_mine:.0f}"])]
    return _finalize("reference", context, stats, comps, findings, reasons, conf, reference=m, region=rect)
