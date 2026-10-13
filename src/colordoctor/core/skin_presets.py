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
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "colordoctor"


DEFAULT_QUICK_RULE = ((10.0, 20.0), (10.0, 20.0), 5.0)


def _rule(d) -> tuple | None:
    if not d:
        return None
    return (tuple(float(v) for v in d["a"]), tuple(float(v) for v in d["b"]), float(d["max_diff"]))


def _parse(data: dict) -> list[SkinPreset]:
    out = []
    for p in data.get("presets", []):
        out.append(
            SkinPreset(
                key=str(p["key"]),
                name=str(p["name"]),
                description=str(p.get("description", "")),
                L=tuple(float(v) for v in p["L"]),
                a=tuple(float(v) for v in p["a"]),
                b=tuple(float(v) for v in p["b"]),
                hue=tuple(float(v) for v in p["hue"]),
                chroma=tuple(float(v) for v in p["chroma"]),
                notes=tuple(p.get("notes", ())),
                quick_rule=_rule(p.get("quick_rule")),
            )
        )
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
        except (OSError, ValueError, KeyError, TypeError):
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
    except (OSError, ValueError, KeyError, TypeError):
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
