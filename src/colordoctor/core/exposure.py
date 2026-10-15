"""曝光 / 影调意图：画面很暗时，到底是刻意的暗调（情绪人像），还是欠曝？

单看像素分不出拍摄者的意图，所以由用户选择“影调”：
  auto   自动判断（默认）：不替用户判断拍摄意图。很暗 / 很亮的画面先不提亮、不压暗，
         出一张说明卡：列出主体（皮肤）亮度、明暗分布等线索，以及两种选择的做法和参考数值（不生成图层）。
  low    刻意暗调：按暗调处理，不提亮。
  high   刻意高调：按高调处理，不压暗。
  normal 正常曝光：画面偏暗就是欠曝、偏亮就是过曝，给出带曲线图层的修正。
"""

from __future__ import annotations

import numpy as np

from colordoctor.core import color_spaces as cs
from colordoctor.core import native_layers as nl
from colordoctor.core.color_models import Adjustment, Finding, Severity

TONE_INTENTS = {"auto": "自动判断", "low": "刻意暗调", "normal": "正常曝光", "high": "刻意高调"}

UNDER_MEDIAN_L = 40.0  # 正常曝光时，中位明度低于此值算欠曝
OVER_MEDIAN_L = 75.0  # 正常曝光时，中位明度高于此值算过曝
TARGET_MEDIAN_L = 50.0  # 欠曝 / 过曝修正后中位明度的目标
MAX_STEP_L = 20.0  # 一次最多改这么多 L*（其余靠重新分析后再调或回到 RAW 调曝光）
HIGHLIGHT_LIMIT = 250  # 提亮曲线让最亮处不超过这个值，避免把高光推到死白


def resolve_key(detected: str, intent: str) -> str:
    """实际按哪种基调处理：auto 用自动判断的结果；low / high 按用户选的；normal 不按暗调 / 高调处理。"""
    intent = intent if intent in TONE_INTENTS else "auto"
    if intent == "auto":
        return detected
    return "" if intent == "normal" else intent


def _gray8(L: float) -> int:
    return int(cs.lab_to_srgb8(np.array([float(np.clip(L, 0, 100)), 0.0, 0.0]))[0])


def _ev(L_from: float, L_to: float) -> float:
    """明度变化约等于 Camera Raw 曝光多少档（按亮度 Y 的比例）。"""
    y = lambda L: ((max(L, 1.0) + 16.0) / 116.0) ** 3  # noqa: E731
    return float(np.log2(y(L_to) / y(L_from)))


def _curve(L, target: float) -> tuple[list[list[int]], float, float]:
    """把中位明度移向 target（每次最多 MAX_STEP_L）的 RGB 复合曲线；提亮时最亮处不推到死白。"""
    p50, p99 = np.percentile(L, [50, 99])
    goal = float(np.clip(target, p50 - MAX_STEP_L, p50 + MAX_STEP_L))
    x, y = _gray8(p50), _gray8(goal)
    pts = [[0, 0], [x, y]]
    hi = _gray8(p99)
    if goal > p50 and hi > x + 8:
        # 最亮处只跟着提一点，留出余量（高光不过曝）
        lift = min(HIGHLIGHT_LIMIT, hi + (y - x) // 3)
        if lift > y:
            pts.append([hi, max(lift, y + 2)])
    pts.append([255, 255])
    pts = sorted({p[0]: p for p in pts}.values())
    return pts, float(p50), goal


SKIN_MIN_SHARE = 0.02  # 画面里像皮肤的像素至少占这么多，才用“主体（皮肤）亮度”做提示
SKIN_DARK_L = 45.0  # 浅中肤色正常曝光时一般在 L* 55–75；低于这个值提示“主体也偏暗”
SKIN_BRIGHT_L = 85.0


def _hints(L, skin_L: float | None, dark: bool) -> tuple[str, ...]:
    """给用户判断用的线索（不是结论）。

    摄影上区分暗调和欠曝看的是主体：暗调 = 主体曝光正常、周围压暗；欠曝 = 主体也暗。
    但深肤色正常曝光时本来就更暗、主体也不一定是人，所以这里只列出数字，由用户决定。
    """
    mid = float(((L >= 40) & (L <= 75)).mean())
    out = [f"中间调（L* 40–75）占 {mid:.0%}，" + (f"暗部（L* < 25）占 {float((L < 25).mean()):.0%}" if dark
                                              else f"亮部（L* > 85）占 {float((L > 85).mean()):.0%}")]
    if skin_L is None:
        out.append("画面里没有找到明显的皮肤，没法用主体亮度来参考")
    elif dark:
        out.append(f"画面里像皮肤的部分平均明度 L* {skin_L:.0f}：" + (
            "主体也偏暗（浅中肤色正常曝光时多在 L* 55–75），要考虑是否欠曝" if skin_L < SKIN_DARK_L
            else "主体亮度在正常范围，更接近“主体正常、周围压暗”的暗调")
            + "；深肤色本身就更暗，只作参考")
    else:
        out.append(f"画面里像皮肤的部分平均明度 L* {skin_L:.0f}：" + (
            "主体也很亮，皮肤细节可能已经损失，要考虑是否过曝" if skin_L > SKIN_BRIGHT_L
            else "主体亮度在正常范围，更接近“主体正常、周围提亮”的高调") + "，只作参考")
    return tuple(out)


def exposure_finding(L, detected: str, intent: str, confidence: float, skin_L: float | None = None) -> Finding | None:
    """根据影调意图给出曝光说明或修正。L：分析区域的明度（L*）；skin_L：画面里皮肤的平均明度（没有为 None）。

    自动判断时不替用户判断拍摄意图：很暗 / 很亮的画面先不提亮、不压暗（把刻意的暗调提亮会毁掉氛围，
    而欠曝的照片先保持原样只是等用户决定），卡片列出线索和两种选择。
    """
    L = np.asarray(L, dtype=float).ravel()
    if L.size == 0:
        return None
    intent = intent if intent in TONE_INTENTS else "auto"
    p1, p50, p99 = (float(v) for v in np.percentile(L, [1, 50, 99]))
    clip_hi = float((L >= 99.0).mean())
    ev_line = f"中位明度 L* {p50:.0f}，最暗 1% {p1:.0f}，最亮 1% {p99:.0f}"
    why_default = ("工具没法只凭像素判断拍摄意图，默认先不改明暗：刻意的暗调 / 高调被改了会毁掉氛围，"
                   "曝光问题先保持原样只是等你决定")

    if intent == "auto" and detected == "low":
        pts, cur, goal = _curve(L, TARGET_MEDIAN_L)
        return Finding(
            problem="画面整体很暗：没法只凭像素确定是刻意的暗调还是欠曝，暂不提亮",
            severity=Severity.NORMAL,
            evidence=(ev_line, *_hints(L, skin_L, dark=True)),
            causes=(),
            adjustments=(
                Adjustment("提示", f"如果是欠曝：「影调」选「正常曝光」再分析，会生成提亮曲线图层"
                                   f"（参考：RGB 曲线 输入 {pts[1][0]} → 输出 {pts[1][1]}，约等于 Camera Raw 曝光 +{_ev(cur, goal):.1f}）",
                           None, True),
                Adjustment("提示", "如果是刻意的暗调（情绪 / 低调人像）：「影调」选「刻意暗调」，这张说明就不再出现", None, False),
            ),
            confidence=confidence,
            notes=(why_default,),
        )
    if intent == "auto" and detected == "high":
        pts, cur, goal = _curve(L, TARGET_MEDIAN_L + 15)
        return Finding(
            problem="画面整体很亮：没法只凭像素确定是刻意的高调还是过曝，暂不压暗",
            severity=Severity.NORMAL,
            evidence=(ev_line, f"死白（L* ≥ 99）占 {clip_hi:.1%}", *_hints(L, skin_L, dark=False)),
            causes=(),
            adjustments=(
                Adjustment("提示", f"如果是过曝：「影调」选「正常曝光」再分析，会生成压暗曲线图层"
                                   f"（参考：RGB 曲线 输入 {pts[1][0]} → 输出 {pts[1][1]}）", None, True),
                Adjustment("提示", "如果是刻意的高调（日系 / 明亮）：「影调」选「刻意高调」，这张说明就不再出现", None, False),
            ),
            confidence=confidence,
            notes=(why_default,),
        )
    if intent != "normal":
        return None
    if p50 < UNDER_MEDIAN_L:
        pts, cur, goal = _curve(L, TARGET_MEDIAN_L)
        sev = Severity.NOTABLE if p50 < UNDER_MEDIAN_L - 12 else Severity.MILD
        text = ("RGB 曲线（提亮，最亮处少提、不推到死白）：" + "，".join(f"输入 {x} → 输出 {y}" for x, y in pts[1:-1])
                + f"（中位明度 L* {cur:.0f} → {goal:.0f}；约等于 Camera Raw 曝光 +{_ev(cur, goal):.1f}）"
                + ("。一次最多提 20 L*，调完重新分析再看" if goal < TARGET_MEDIAN_L else ""))
        return Finding(
            problem=f"整体欠曝：中位明度 L* {p50:.0f}（你选了「正常曝光」）",
            severity=sev,
            evidence=(ev_line,),
            causes=("曝光不足", "逆光 / 暗环境下没有补光"),
            adjustments=(
                nl.adjustment("curves", {"channels": {"composite": pts}}, text, "global", "整体欠曝 · 曲线（提亮）",
                              strength="中等" if goal - cur > 10 else "小幅"),
                Adjustment("Camera Raw", f"有 RAW 时优先在 Camera Raw 里加曝光（约 +{_ev(cur, goal):.1f}），暗部噪点更少", None, True),
            ),
            confidence=confidence,
        )
    if p50 > OVER_MEDIAN_L:
        pts, cur, goal = _curve(L, TARGET_MEDIAN_L + 15)
        text = ("RGB 曲线（压暗中间调）：" + "，".join(f"输入 {x} → 输出 {y}" for x, y in pts[1:-1])
                + f"（中位明度 L* {cur:.0f} → {goal:.0f}；约等于 Camera Raw 曝光 {_ev(cur, goal):.1f}）")
        return Finding(
            problem=f"整体过曝：中位明度 L* {p50:.0f}（你选了「正常曝光」）",
            severity=Severity.MILD,
            evidence=(ev_line, f"死白（L* ≥ 99）占 {clip_hi:.1%}：这部分细节已经没有了，压暗也找不回"),
            causes=("曝光过度", "强光直射 / 逆光测光"),
            adjustments=(nl.adjustment("curves", {"channels": {"composite": pts}}, text, "global", "整体过曝 · 曲线（压暗）"),),
            confidence=confidence,
        )
    return None
