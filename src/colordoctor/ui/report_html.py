"""把分析结果渲染为 HTML（QTextBrowser 显示）。纯函数，可单元测试。

默认先显示结论和关键证据；勾选“详细数据”后才展开全部统计。
"""

from __future__ import annotations

from html import escape

from colordoctor.core.color_models import AnalysisReport, ColorContext, Finding, PointSample, SkinPreset
from colordoctor.ui.theme import SEVERITY_COLORS, TEXT_DIM


def _badge(sev: str) -> str:
    c = SEVERITY_COLORS.get(sev, "#999")
    return f'<span style="background:{c};color:#111;padding:1px 6px;border-radius:3px">{escape(sev)}</span>'


def _swatch(rgb, size=14) -> str:
    r, g, b = rgb
    return (
        f'<span style="background:rgb({r},{g},{b});border:1px solid #777">'
        f'{"&nbsp;" * max(2, size // 4)}</span>'
    )


def context_html(ctx: ColorContext) -> str:
    rows = [
        ("数据来源", {"screen": "屏幕取样", "file": "JPG 文件", "synthetic": "合成数据"}.get(ctx.source, ctx.source)),
        ("分析空间", ctx.analysis_space + "；Lab 为 D50（与 Photoshop 一致）"),
        ("原始配置文件", ctx.source_profile or "无"),
        ("色彩空间转换", "已转换到 sRGB" if ctx.converted else ("按 sRGB 假定" if ctx.assumed_srgb else "未转换（原样使用）")),
    ]
    if ctx.source == "screen":
        rows.append(("画面来源", "色彩管理软件（Photoshop/ACR）" if ctx.managed_source else "未做色彩管理的软件"))
        if ctx.device:
            rows.append(("显示器", ctx.device + (f"，缩放 {ctx.scale:.0%}" if ctx.scale else "")))
    elif ctx.device:
        rows.append(("文件", ctx.device))
    html = "".join(f"<tr><td style='color:{TEXT_DIM}'>{escape(k)}</td><td>{escape(str(v))}</td></tr>" for k, v in rows)
    warn = "".join(f"<li>{escape(w)}</li>" for w in ctx.warnings)
    return f"<table cellpadding=2>{html}</table>" + (f"<ul style='color:#d0b070'>{warn}</ul>" if warn else "")


def point_html(s: PointSample, preset: SkinPreset | None = None, preset_delta=None, adjustments=()) -> str:
    v = s.values
    rows = [
        ("坐标", f"x {s.x}, y {s.y}"),
        ("采样", f"{s.size_label} 平均（{s.pixel_count} 像素）"),
        ("HEX", v.hex),
        ("RGB", "%d, %d, %d" % v.rgb),
        ("HSB（PS 拾色器）", "H %.0f°  S %.0f%%  B %.0f%%" % v.hsb),
        ("HSL", "H %.0f°  S %.0f%%  L %.0f%%" % v.hsl),
        ("CIE Lab（D50，同 PS）", "L* %.1f   a* %.1f   b* %.1f" % v.lab),
        ("明度 L* / 色度 C* / 色相角 h°", "%.1f / %.1f / %.1f°" % v.lch),
        ("CIE Lab（D65）", "L* %.1f   a* %.1f   b* %.1f" % v.lab_d65),
        ("OKLab", "L %.3f   a %.3f   b %.3f" % v.oklab),
        ("OKLCH", "L %.3f   C %.3f   h %.1f°" % v.oklch),
        ("相对亮度 Y", f"{v.luminance:.4f}"),
        ("色相名", v.hue_name),
    ]
    body = "".join(f"<tr><td style='color:{TEXT_DIM}'>{escape(k)}</td><td><b>{escape(val)}</b></td></tr>" for k, val in rows)
    head = f"<h3>{_swatch(v.rgb, 40)} {escape(v.hex)}</h3>"
    extra = ""
    if preset is not None and preset_delta is not None:
        d = preset_delta
        extra = (
            f"<p>相对目标风格「{escape(preset.name)}」：ΔE00 {d.delta_e00:.1f}；"
            f"需要 ΔL* {d.dL:+.1f}，Δa* {d.da:+.1f}，Δb* {d.db:+.1f}</p>"
        ) + adjustments_html(adjustments)
    return head + f"<table cellpadding=3>{body}</table>" + extra + "<hr>" + context_html(s.context)


def adjustments_html(adjs) -> str:
    if not adjs:
        return ""
    items = ""
    for a in adjs:
        tag = "" if a.tool == "提示" else (" <i style='color:%s'>（估计）</i>" % TEXT_DIM if a.estimated else " <i>（精确）</i>")
        strength = f" <span style='color:{TEXT_DIM}'>[{escape(a.strength)}]</span>" if a.strength else ""
        items += f"<li><b>{escape(a.tool)}</b>：{escape(a.instruction)}{strength}{tag}</li>"
    return f"<p><b>调整方向：</b></p><ul>{items}</ul>"


def _finding_html(f: Finding) -> str:
    ev = "".join(f"<li>{escape(e)}</li>" for e in f.evidence)
    causes = "".join(f"<li>{escape(c)}</li>" for c in f.causes)
    adj = adjustments_html(f.adjustments)
    notes = "".join(f"<li>{escape(n)}</li>" for n in f.notes if n)
    parts = [
        f"<h3>{_badge(f.severity.value)} {escape(f.problem)}</h3>",
        f"<p><b>证据：</b></p><ul>{ev}</ul>" if ev else "",
        f"<p><b>可能原因：</b></p><ul>{causes}</ul>" if causes else "",
        adj,
        f"<p><b>置信度：</b>{escape(f.confidence_text)}</p>",
        f"<p><b>注意事项：</b></p><ul>{notes}</ul>" if notes else "",
    ]
    return "<div>" + "".join(parts) + "</div><hr>"


def report_html(r: AnalysisReport, details: bool = False) -> str:
    mode_name = {"general": "脏色检查（区域）", "skin": "人像肤色", "compare": "参考对比", "scan": "脏色检查（整图扫描）"}.get(
        r.mode, r.mode
    )
    head = (
        f"<h2>{escape(mode_name)}：{_badge(r.overall_severity.value)} 脏色风险 {r.overall_risk:.0f}/100</h2>"
        f"<p style='color:{TEXT_DIM}'>总分 = 最高分项 + 其余分项之和 × 0.2；结论是风险倾向而不是绝对事实。"
        f"整体置信度 {r.confidence:.2f}</p>"
    )
    comp_rows = "".join(
        f"<tr><td>{escape(c.name)}</td><td>{_badge(c.severity.value) if c.applicable else '不适用'}</td>"
        f"<td align=right>{c.score:.0f}</td><td>{escape(c.evidence[0]) if c.evidence else ''}"
        f"{('<br><i>' + escape(c.note) + '</i>') if c.note else ''}</td></tr>"
        for c in r.components
    )
    comps = f"<table cellpadding=3 width='100%'><tr><th align=left>分项</th><th>程度</th><th>分</th><th align=left>关键证据</th></tr>{comp_rows}</table>"
    body = head + "".join(_finding_html(f) for f in r.findings)
    if r.skin is not None:
        s = r.skin
        body += (
            f"<p><b>肤色概况：</b>Lab ({s.mean_lab[0]:.1f}, {s.mean_lab[1]:.1f}, {s.mean_lab[2]:.1f})，"
            f"C* {s.mean_lch[1]:.1f}，h {s.mean_lch[2]:.1f}°，深浅：{escape(s.depth_label)}（ITA {s.ita:.0f}°），"
            f"总体倾向：{escape(s.tendency)}</p>"
        )
    body += "<h3>风险分项</h3>" + comps
    if details:
        body += details_html(r)
    return body


def details_html(r: AnalysisReport) -> str:
    out = "<h3>详细数据</h3>"
    st = r.stats
    if st is not None:
        rows = [
            ("区域像素 / 参与计算", f"{st.total_pixels} / {st.sampled_pixels}"),
            ("RGB 平均 / 中位数", "(%.1f, %.1f, %.1f) / (%.0f, %.0f, %.0f)" % (*st.rgb_mean, *st.rgb_median)),
            ("Lab 平均 / 中位数", "(%.1f, %.1f, %.1f) / (%.1f, %.1f, %.1f)" % (*st.lab_mean, *st.lab_median)),
            ("LCh（平均色）", "L* %.1f  C* %.1f  h %.1f°" % st.lch_mean),
            ("OKLab / OKLCH 平均", "(%.3f, %.3f, %.3f) / (%.3f, %.3f, %.1f°)" % (*st.oklab_mean, *st.oklch_mean)),
            ("L* 标准差 / C* 标准差", f"{st.L_std:.2f} / {st.C_std:.2f}"),
            ("明度 P10 / P50 / P90", f"{st.L_p10:.1f} / {st.L_p50:.1f} / {st.L_p90:.1f}"),
            ("色度 P10 / P50 / P90", f"{st.C_p10:.1f} / {st.C_p50:.1f} / {st.C_p90:.1f}"),
            ("HSB 饱和度中位数", f"{st.sat_p50:.0f}%"),
            (
                "色相（按色度加权圆形平均）",
                "无（区域接近中性）" if st.hue_mean is None else f"{st.hue_mean:.1f}°，集中度 R {st.hue_concentration:.2f}，圆形标准差 {st.hue_spread:.1f}°",
            ),
            ("有彩像素比例", f"{st.chromatic_fraction:.1%}"),
            ("高光（L*≥90）/ 阴影（L*≤15）", f"{st.highlight_fraction:.1%} / {st.shadow_fraction:.1%}"),
            ("裁切像素（含 0 或 255）", f"{st.clipped_fraction:.1%}"),
        ]
        out += "<table cellpadding=2>" + "".join(
            f"<tr><td style='color:{TEXT_DIM}'>{escape(k)}</td><td>{escape(v)}</td></tr>" for k, v in rows
        ) + "</table>"
        if st.clusters:
            out += "<p><b>主色聚类</b></p><table cellpadding=2>" + "".join(
                f"<tr><td>{_swatch(c.rgb)}</td><td>{escape(c.hue_name)}</td><td>{c.share:.1%}</td>"
                f"<td>Lab ({c.lab[0]:.1f}, {c.lab[1]:.1f}, {c.lab[2]:.1f})</td><td>C* {c.chroma:.1f} h {c.hue:.0f}°</td></tr>"
                for c in st.clusters
            ) + "</table>"
    ct = r.contamination
    if ct is not None:
        out += (
            f"<p><b>脏色检测</b>：主色 {escape(ct.base_hue_name)}，对立色像素 {ct.opposite_fraction:.2%}"
            f"（阈值 C* ≥ {ct.opposite_threshold:.1f}），局部发灰 {ct.graying_fraction:.1%}，"
            f"标记位置 {len(ct.blobs)} 处</p>"
        )
    if r.comparison is not None:
        c = r.comparison
        out += (
            f"<p><b>参考对比</b>：ΔE00 {c.delta_e00:.2f}，ΔL* {c.dL:+.2f}，ΔC* {c.dC:+.2f}，ΔH* {c.dH:+.2f}，"
            f"Δh {c.dh:+.1f}°，Δa* {c.da:+.2f}，Δb* {c.db:+.2f}</p>"
        )
    out += "<h3>色彩管理</h3>" + context_html(r.context)
    return out
