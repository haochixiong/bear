"""把色差和颜色方向换成人能读懂的话（给修图师看，不用懂 ΔE00）。

色差等级按 CIEDE2000 的常用经验划分：
  < 1    几乎看不出
  1–2    仔细对比才看得出
  2–3.5  能看出
  3.5–5  明显
  5–10   很明显
  ≥ 10   差别很大
"""

from __future__ import annotations

import numpy as np

DE_LEVELS = (
    (1.0, "几乎看不出"),
    (2.0, "仔细对比才看得出"),
    (3.5, "能看出"),
    (5.0, "明显"),
    (10.0, "很明显"),
    (float("inf"), "差别很大"),
)
DE_LEGEND = "差别：几乎看不出 <1 · 仔细对比才看得出 1–2 · 能看出 2–3.5 · 明显 3.5–5 · 很明显 5–10 · 差别很大 ≥10（数字是 ΔE00 色差）"


def de_words(de: float) -> str:
    """色差（ΔE00）→ 等级文字。"""
    for top, words in DE_LEVELS:
        if de < top:
            return words
    return DE_LEVELS[-1][1]


def de_level(de: float) -> int:
    """0–5，对应 DE_LEVELS 的序号（用来画条形 / 选颜色）。"""
    for i, (top, _w) in enumerate(DE_LEVELS):
        if de < top:
            return i
    return len(DE_LEVELS) - 1


def direction_words(lab_from, lab_to, max_items: int = 2) -> str:
    """lab_to 相对 lab_from 偏向哪里，例如“偏黄、偏暗”。差得很少时返回“”。

    只说最主要的一两个方向：亮 / 暗（L*）、黄 / 蓝（b*）、红 / 绿（a*）、鲜艳 / 灰（C*）。
    """
    a0 = np.asarray(lab_from, float)[:3]
    a1 = np.asarray(lab_to, float)[:3]
    dL, da, db = a1 - a0
    dC = float(np.hypot(a1[1], a1[2]) - np.hypot(a0[1], a0[2]))
    cands = [
        (abs(dL), "偏亮" if dL > 0 else "偏暗", 1.5),
        (abs(db), "偏黄" if db > 0 else "偏蓝", 1.5),
        (abs(da), "偏红" if da > 0 else "偏绿", 1.5),
        (abs(dC), "更鲜艳" if dC > 0 else "更灰", 2.0),
    ]
    picked = [w for size, w, floor in sorted(cands, key=lambda t: -t[0]) if size >= floor]
    # “更灰 / 更鲜艳”只在没有更具体的色相方向时才说，避免“偏黄、更鲜艳”这种重复
    hue = [w for w in picked if w not in ("更鲜艳", "更灰")]
    out = hue[:max_items] if hue else picked[:1]
    return "、".join(out)
