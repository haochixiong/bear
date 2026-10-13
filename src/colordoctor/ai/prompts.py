"""AI 点评的提示词：把本工具的数值分析结果交给大模型，请它从修图师角度给意见。"""

from __future__ import annotations

import json

SYSTEM = (
    "你是一位资深的人像摄影后期调色师，熟悉 Camera Raw / Lightroom（基本面板、混色器 HSL、颜色分级、校准、曲线）"
    "和 Photoshop（曲线、色彩平衡、色相/饱和度、可选颜色）。用户会给你一个本地色彩分析工具的测量结果（JSON），"
    "其中 Lab 为 D50（与 Photoshop 信息面板一致），数值是真实测量值。请基于这些数值回答，不要编造数值；"
    "如果附带了图片，可以结合画面内容给出审美层面的判断。用中文回答，条理清楚，避免空话。"
)

DEFAULT_QUESTION = (
    "请按以下结构回答：\n"
    "1. 总体评价（一两句话）\n"
    "2. 主要问题（颜色哪里脏/发灰/不均匀/偏色，按重要性排序，引用数值）\n"
    "3. 调整步骤（按 ACR → PS 的顺序，给出具体面板、滑块方向和大致幅度，标明是估计）\n"
    "4. 与目标风格/参考的差距以及是否适合（如果数据里有）\n"
    "5. 需要注意不要过度调整的地方"
)


def build_prompt(summary: dict, question: str | None = None, image_note: str | None = None) -> str:
    data = json.dumps(summary, ensure_ascii=False, indent=1)
    parts = ["以下是色彩分析工具的测量结果（JSON）：", "```json", data, "```"]
    if image_note:
        parts.append(image_note)
    parts.append(question or DEFAULT_QUESTION)
    return "\n".join(parts)
