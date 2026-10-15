"""把分析结果转换为 JSON（给 AI / Agent / 其他程序使用）。不包含任何像素数据。"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum

import numpy as np

SKIP_FIELDS = {"mask", "valid_mask", "tile_map"}


def to_jsonable(obj, _depth: int = 0):
    if _depth > 12:
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_jsonable(getattr(obj, f.name), _depth + 1)
            for f in dataclasses.fields(obj)
            if f.name not in SKIP_FIELDS
        }
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        # 以下划线开头的键是内部提示（例如图层的蒙版位置），不导出
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items() if k not in SKIP_FIELDS and not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, _depth + 1) for v in obj]
    if isinstance(obj, np.ndarray):
        return None if obj.size > 64 else obj.tolist()
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 4)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


def report_to_json(report, indent: int | None = 2) -> str:
    return json.dumps(to_jsonable(report), ensure_ascii=False, indent=indent)


def summary(report) -> dict:
    """精简摘要（给 AI 的提示词用，控制长度）。"""
    from colordoctor.core.color_models import AnalysisReport, PointSample

    if isinstance(report, PointSample):
        v = report.values
        return {
            "类型": "单点取色",
            "取样": f"{report.size_label} 平均",
            "RGB": v.rgb,
            "HEX": v.hex,
            "Lab(D50)": v.lab,
            "LCh": v.lch,
            "HSB": v.hsb,
        }
    assert isinstance(report, AnalysisReport)
    out = {
        "模式": report.mode,
        "脏色风险": report.overall_risk,
        "严重程度": report.overall_severity.value,
        "置信度": report.confidence,
        "风险分项": [
            {"名称": c.name, "分数": c.score, "程度": c.severity.value, "证据": list(c.evidence[:2])}
            for c in report.components
            if c.applicable
        ],
        "结论": [
            {
                "问题": f.problem,
                "程度": f.severity.value,
                "证据": list(f.evidence[:6]),
                "建议": [f"{a.tool}：{a.instruction}" for a in f.adjustments[:6]],
            }
            for f in report.findings[:10]
        ],
    }
    if report.skin is not None:
        s = report.skin
        out["肤色"] = {"Lab": s.mean_lab, "LCh": s.mean_lch, "倾向": s.tendency, "深浅": s.depth_label, "ITA": s.ita}
    if report.part:
        out["部位"] = report.part
    if report.stats is not None:
        st = report.stats
        out["统计"] = {"Lab平均": st.lab_mean, "L分位P10/50/90": [st.L_p10, st.L_p50, st.L_p90], "色度P50": st.C_p50}
    if report.palette is not None:
        p = report.palette
        out["整图色彩"] = {
            "色彩丰富度": [p.colorfulness.value, p.colorfulness.label],
            "主辅点缀": [[c.family, c.share] for c in p.ratio.roles],
            "影调L分位": list(p.tone),
        }
        if p.tone_class is not None:
            out["整图色彩"]["影调"] = p.tone_class.name
            if p.tone_target:
                out["整图色彩"]["目标影调"] = [p.tone_label, p.tone_target]
    if report.reference is not None:
        out["仿色"] = {"相似度": report.reference.similarity, "合适性": list(report.reference.fit)}
    out["色彩管理"] = {
        "来源": report.context.source,
        "配置文件": report.context.source_profile,
        "假定sRGB": report.context.assumed_srgb,
    }
    return to_jsonable(out)
