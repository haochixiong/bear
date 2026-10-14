"""曝光 / 影调意图：画面很暗时，到底是刻意的暗调（情绪人像），还是欠曝？

单看像素分不出拍摄者的意图，所以由用户选择“影调”：
  auto   自动判断（默认）：很暗 / 很亮的画面按刻意的暗调 / 高调处理，不提亮也不压暗，
         但一定出一张说明卡；画面里几乎没有高光（像欠曝）时会明说，并给出提亮的数值供参考（不生成图层）。
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
NO_HIGHLIGHT_P99 = 60.0  # 最亮的 1% 像素都不到这个明度：几乎没有高光，更像欠曝而不是布光造成的暗调
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


def exposure_finding(L, detected: str, intent: str, confidence: float) -> Finding | None:
    """根据影调意图给出曝光说明或修正。L：分析区域的明度（L*）。"""
    L = np.asarray(L, dtype=float).ravel()
    if L.size == 0:
        return None
    intent = intent if intent in TONE_INTENTS else "auto"
    p1, p50, p99 = (float(v) for v in np.percentile(L, [1, 50, 99]))
    clip_hi = float((L >= 99.0).mean())
    ev_line = f"中位明度 L* {p50:.0f}，最暗 1% {p1:.0f}，最亮 1% {p99:.0f}"
    switch = "；如果判断不对，在「影调」里改选"

    if intent == "auto" and detected == "low":
        looks_under = p99 < NO_HIGHLIGHT_P99
        pts, cur, goal = _curve(L, TARGET_MEDIAN_L)
        hint = (f"如果其实是欠曝：把「影调」改为「正常曝光」再分析，会生成提亮曲线图层"
                f"（参考：RGB 曲线 输入 {pts[1][0]} → 输出 {pts[1][1]}，约等于 Camera Raw 曝光 +{_ev(cur, goal):.1f}）")
        return Finding(
            problem="画面整体很暗：" + ("看起来更像欠曝（几乎没有高光），暂按暗调处理、没有提亮" if looks_under
                                        else "按刻意的暗调（情绪 / 低调人像）处理，没有提亮"),
            severity=Severity.MILD if looks_under else Severity.NORMAL,
            evidence=(ev_line, "画面里有明显高光（轮廓光、眼神光等），更像刻意的暗调" if not looks_under
                      else f"最亮的 1% 像素也只有 L* {p99:.0f}：没有真正的高光，欠曝的可能性大"),
            causes=(),
            adjustments=(Adjustment("提示", hint, None, True),),
            confidence=confidence,
            notes=("单看像素分不出是刻意压暗还是欠曝" + switch + "「刻意暗调」或「正常曝光」",),
        )
    if intent == "auto" and detected == "high":
        pts, cur, goal = _curve(L, TARGET_MEDIAN_L + 10)
        return Finding(
            problem="画面整体很亮：按刻意的高调（日系 / 明亮）处理，没有压暗",
            severity=Severity.NORMAL,
            evidence=(ev_line, f"死白（L* ≥ 99）占 {clip_hi:.1%}"),
            causes=(),
            adjustments=(Adjustment("提示", f"如果其实是过曝：把「影调」改为「正常曝光」再分析，会生成压暗曲线图层"
                                          f"（参考：RGB 曲线 输入 {pts[1][0]} → 输出 {pts[1][1]}）", None, True),),
            confidence=confidence,
            notes=("单看像素分不出是刻意提亮还是过曝" + switch + "「刻意高调」或「正常曝光」",),
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
