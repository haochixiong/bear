"""多点取色对比的显示（取色数值页）。"""

from __future__ import annotations

from html import escape

from colordoctor.core.multipoint import MultiPointResult

# 差别等级 0–5 的颜色（绿 → 红），与文字一起出现，不单靠颜色表达
LEVEL_COLORS = ("#2e7d4f", "#4f8f3a", "#b08a1e", "#c8691c", "#c0392b", "#8e1b1b")
JUDGE_COLORS = {"基准": "#666", "一致": "#2e7d4f", "略有差别": "#b08a1e", "不均": "#c0392b"}


def multipoint_html(res: MultiPointResult) -> str:
    rows = []
    for r in res.rows:
        sw = f"<span style='background:rgb{r.rgb};'>&nbsp;&nbsp;&nbsp;&nbsp;</span>"
        if r.de is None:
            diff = "—"
        else:
            bar = "▮" * (r.level + 1) + "▯" * (5 - r.level)
            diff = (f"<span style='color:{LEVEL_COLORS[r.level]}'>{bar}</span> {escape(r.words)}"
                    f" <span style='color:#888'>({r.de:.1f})</span>")
        judge = f"<b style='color:{JUDGE_COLORS.get(r.judge, '#666')}'>{escape(r.judge)}</b>"
        lab = f"{r.lab[0]:.0f}, {r.lab[1]:.0f}, {r.lab[2]:.0f}"
        rows.append(f"<tr><td>{escape(r.name)}</td><td>{sw}</td><td>{lab}</td><td>{diff}</td>"
                    f"<td>{escape(r.direction)}</td><td>{judge}</td></tr>")
    return (
        "<h3>多点对比</h3>"
        f"<p>{escape(res.summary)}</p>"
        "<table cellspacing='0' cellpadding='4' border='1' style='border-collapse:collapse'>"
        "<tr><th>点</th><th>颜色</th><th>Lab</th><th>和基准的差别</th><th>偏向</th><th>判断</th></tr>"
        + "".join(rows) + "</table>"
        f"<p style='color:#888'>第一个点是基准。{escape(res.legend)}</p>"
    )
