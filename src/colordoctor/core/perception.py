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


# Lab 色相角上的几个“颜色名”位置（度）：色相往哪个方向转，就说偏向转过去的那个颜色
HUE_LANDMARKS = ((25.0, "红"), (90.0, "黄"), (160.0, "绿"), (210.0, "青"), (280.0, "蓝"), (330.0, "紫"))


def _hue_word(h0: float, turn: float) -> str:
    """从色相 h0 往 turn 的方向（>0 色相角增大）转，偏向哪个颜色。跳过离 h0 太近（15° 内）的那个名字。"""
    names = [(h, n) for h, n in HUE_LANDMARKS] + [(h + 360.0, n) for h, n in HUE_LANDMARKS] + [(h - 360.0, n) for h, n in HUE_LANDMARKS]
    if turn > 0:
        ahead = [(h, n) for h, n in names if h > h0 + 15.0]
        return min(ahead)[1]
    behind = [(h, n) for h, n in names if h < h0 - 15.0]
    return max(behind)[1]


def direction_words(lab_from, lab_to, max_items: int = 2) -> str:
    """lab_to 相对 lab_from 偏向哪里，例如“偏黄、偏暗”。差得很少时返回“”。

    a*/b* 的变化拆成两部分：沿原来颜色方向的（浓淡：更鲜艳 / 更灰）和横向的（色相转动：偏红 / 偏黄 …，
    按原来的色相往哪边转来命名，肤色往黄转叫“偏黄”、往红转叫“偏红”）。原来接近中性时，变化本身就是偏色。
    否则同一种肤色变灰（a*、b* 一起变小）会被说成“偏蓝、偏绿”。按大小只说最主要的一两个。
    """
    a0 = np.asarray(lab_from, float)[:3]
    a1 = np.asarray(lab_to, float)[:3]
    dL, da, db = a1 - a0
    c0 = float(np.hypot(a0[1], a0[2]))
    cands = [(abs(dL), "偏亮" if dL > 0 else "偏暗", 1.5)]
    if c0 >= 3.0:
        u = a0[1:] / c0
        radial = float(da * u[0] + db * u[1])  # 浓淡
        turn = float(u[0] * db - u[1] * da)  # 色相转动（>0 色相角增大）
        h0 = float(np.degrees(np.arctan2(a0[2], a0[1])) % 360.0)
        cands.append((abs(turn), "偏" + _hue_word(h0, turn), 1.5))
        cands.append((abs(radial), "更鲜艳" if radial > 0 else "更灰", 2.0))
    else:  # 原来接近中性：没有“原来的颜色”，按变化本身的方向说
        cands.append((abs(db), "偏黄" if db > 0 else "偏蓝", 1.5))
        cands.append((abs(da), "偏红" if da > 0 else "偏绿", 1.5))
    picked = [w for size, w, floor in sorted(cands, key=lambda t: -t[0]) if size >= floor]
    return "、".join(picked[:max_items])
