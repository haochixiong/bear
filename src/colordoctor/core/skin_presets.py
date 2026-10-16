"""肤色风格预设（日系 / 法式 / 暗调 / Cos / 自然）。

内置数值见 data/skin_presets.json；用户可在配置目录放置同名文件覆盖或新增。
"""

from __future__ import annotations

import json
import os
import sys
from importlib import resources
from pathlib import Path

import numpy as np

from colordoctor.core import color_spaces as cs
from colordoctor.core.color_models import SkinPreset

DEFAULT_KEY = "natural"


def user_config_dir() -> Path:
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "ColorDoctor"
    if sys.platform == "darwin" and not os.environ.get("XDG_CONFIG_HOME"):
        return Path.home() / "Library" / "Application Support" / "ColorDoctor"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "colordoctor"


DEFAULT_QUICK_RULE = ((10.0, 20.0), (10.0, 20.0), 5.0)


def _range(v) -> tuple[float, float]:
    """[下限, 上限]：必须正好两个有限的数，否则当作文件写错。"""
    r = tuple(float(x) for x in v)
    if len(r) != 2 or not all(np.isfinite(r)):
        raise ValueError(f"范围要写成 [下限, 上限]：{v!r}")
    return r


def _rule(d) -> tuple | None:
    if not d:
        return None
    if not isinstance(d, dict):
        raise ValueError("quick_rule 格式不对")
    return (_range(d["a"]), _range(d["b"]), float(d["max_diff"]))


def _parse(data: dict) -> list[SkinPreset]:
    """解析预设文件；写错的单个预设跳过（其他照常用），整个文件不是对象时当作损坏。"""
    if not isinstance(data, dict) or not isinstance(data.get("presets", []), list):
        raise ValueError("预设文件格式不对")
    out = []
    for p in data.get("presets", []):
        try:
            out.append(
                SkinPreset(
                    key=str(p["key"]),
                    name=str(p["name"]),
                    description=str(p.get("description", "")),
                    L=_range(p["L"]),
                    a=_range(p["a"]),
                    b=_range(p["b"]),
                    hue=_range(p["hue"]),
                    chroma=_range(p["chroma"]),
                    notes=tuple(str(n) for n in p.get("notes", ())),
                    quick_rule=_rule(p.get("quick_rule")),
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return out


def global_quick_rule(user_file: Path | None = None) -> tuple:
    """经验快速判定规则：用户文件 > 内置文件 > 默认 (a 10–20, b 10–20, 差值 ≤ 5)。"""
    for loader in (
        lambda: json.loads((user_file or user_config_dir() / "skin_presets.json").read_text(encoding="utf-8")),
        lambda: json.loads(resources.files("colordoctor.data").joinpath("skin_presets.json").read_text(encoding="utf-8")),
    ):
        try:
            r = _rule(loader().get("quick_rule"))
            if r:
                return r
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return DEFAULT_QUICK_RULE


def builtin_presets() -> list[SkinPreset]:
    text = resources.files("colordoctor.data").joinpath("skin_presets.json").read_text(encoding="utf-8")
    return _parse(json.loads(text))


def load_presets(user_file: Path | None = None) -> list[SkinPreset]:
    """内置预设 + 用户覆盖（同 key 覆盖，新 key 追加）。用户文件损坏时忽略。"""
    presets = {p.key: p for p in builtin_presets()}
    path = user_file if user_file is not None else user_config_dir() / "skin_presets.json"
    try:
        if path.exists():
            for p in _parse(json.loads(path.read_text(encoding="utf-8"))):
                presets[p.key] = p
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return list(presets.values())


def get_preset(key: str, presets: list[SkinPreset] | None = None) -> SkinPreset:
    for p in presets or load_presets():
        if p.key == key:
            return p
    raise KeyError(key)


def preset_from_sample(lab_mean, name: str = "自定义（来自参考取样）", tolerance: float = 4.0) -> SkinPreset:
    """用参考作品中取样得到的肤色平均 Lab 生成目标预设。"""
    L, a, b = (float(v) for v in lab_mean)
    _, C, h = cs.lab_to_lch(np.array([L, a, b]))
    return SkinPreset(
        key="custom",
        name=name,
        description="根据你选取的参考肤色生成的目标",
        L=(L - 2 * tolerance, L + 2 * tolerance),
        a=(a - tolerance, a + tolerance),
        b=(b - tolerance, b + tolerance),
        hue=(float(h) - 7.0, float(h) + 7.0),
        chroma=(max(0.0, float(C) - tolerance), float(C) + tolerance),
        notes=("目标来自参考取样，区间为 ±容差",),
    )


def preset_swatch_rgb(p: SkinPreset) -> tuple[int, int, int]:
    return tuple(int(v) for v in cs.lab_to_srgb8(np.array(p.target_lab)))
