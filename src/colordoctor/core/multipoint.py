"""多点取色对比：像 PS 的“颜色取样器”，每个点单独读数，以第一个点为基准比较。

用途：同一张脸是否均匀（额头 / 两颊 / 下巴 / 脖子），同一种颜色是否统一（衣服、背景几处），调整前后对比。
差别用人能读懂的话说（perception.de_words / direction_words），色差数字只作参考。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colordoctor.core import color_spaces as cs
from colordoctor.core import perception as pc

SAME_MAX = 2.0  # 小于它：仔细对比才看得出，算“一致”
SLIGHT_MAX = 3.5  # 2–3.5：能看出，算“略有差别”；更大算“不均”


@dataclass(frozen=True)
class PointRow:
    name: str
    lab: tuple[float, float, float]
    rgb: tuple[int, int, int]
    de: float | None  # 与基准的 ΔE00；基准自己为 None
    words: str  # 差别等级（几乎看不出 / 明显 …）
    level: int  # 0–5
    direction: str  # 相对基准偏向哪里（偏黄、偏暗 …）
    judge: str  # 基准 / 一致 / 略有差别 / 不均


@dataclass(frozen=True)
class MultiPointResult:
    rows: tuple[PointRow, ...]
    summary: str
    legend: str = pc.DE_LEGEND


def _judge(de: float) -> str:
    if de < SAME_MAX:
        return "一致"
    if de < SLIGHT_MAX:
        return "略有差别"
    return "不均"


def compare_points(labs, names=None) -> MultiPointResult:
    """labs：每个点的 Lab（D50）；names：点的名字（默认“点 1、点 2 …”）。第一个点是基准。"""
    labs = [np.asarray(x, float)[:3] for x in labs]
    if len(labs) < 2:
        raise ValueError("至少需要两个点才能对比")
    names = list(names) if names else [f"点 {i + 1}" for i in range(len(labs))]
    base = labs[0]
    rows = []
    for i, (lab, name) in enumerate(zip(labs, names, strict=True)):
        rgb = tuple(int(v) for v in cs.lab_to_srgb8(lab))
        lab_t = tuple(round(float(v), 1) for v in lab)
        if i == 0:
            rows.append(PointRow(name, lab_t, rgb, None, "—", 0, "—", "基准"))
            continue
        de = float(cs.delta_e_2000(base, lab))
        rows.append(PointRow(name, lab_t, rgb, round(de, 1), pc.de_words(de), pc.de_level(de),
                             pc.direction_words(base, lab) or "—", _judge(de)))
    return MultiPointResult(tuple(rows), _summary(rows))


def _summary(rows: list[PointRow]) -> str:
    base, others = rows[0], rows[1:]
    off = [r for r in others if r.judge == "不均"]
    slight = [r for r in others if r.judge == "略有差别"]
    n = len(rows)
    if not off and not slight:
        worst = max(others, key=lambda r: r.de or 0)
        return f"{n} 个点颜色一致：和「{base.name}」相比，最大的差别也只是{worst.words}。"
    parts = []
    def why(r):
        return f"（{r.direction}）" if r.direction and r.direction != "—" else ""

    if off:
        parts.append("、".join(f"「{r.name}」{r.words}{why(r)}" for r in off))
    if slight:
        parts.append("、".join(f"「{r.name}」略有差别{why(r)}" for r in slight))
    text = f"{n} 个点里，和「{base.name}」相比：" + "；".join(parts) + "。"
    dirs = {r.direction for r in off if r.direction != "—"}
    if len(off) >= 2 and len(dirs) == 1:
        text += f"这几处都{dirs.pop()}，可以用一个局部选区一起调。"
    elif off:
        text += "差别明显的位置建议单独建选区修正，或用它们重新做一次“人像肤色”分析。"
    return text
