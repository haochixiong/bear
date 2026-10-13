"""Camera Raw 参数 -> ACR 预设文件（.xmp，crs: 命名空间）。

只写入用到的设置：在 ACR 里应用这个预设时，没写到的滑块保持不变。
导出的是一个预设文件，不碰任何图片。

params 结构（全部可选）：
  hsl:         {"红色"|"Red": {"hue": -100..100, "sat": .., "lum": ..}, ...}
  grading:     {"shadow"|"midtone"|"highlight"|"global": {"hue": 0..359, "sat": 0..100, "lum": -100..100},
                "blending": 0..100, "balance": -100..100}
  calibration: {"red"|"green"|"blue": {"hue": .., "sat": ..}, "shadow_tint": ..}
  curve:       {"composite"|"red"|"green"|"blue": [[in, out], ...]}
  basic:       {"Exposure2012": .., "Contrast2012": .., "Highlights2012": .., "Shadows2012": .., "Whites2012": ..,
                "Blacks2012": .., "Vibrance": .., "Saturation": .., "Clarity2012": .., "Dehaze": .., "Texture": ..}
"""

from __future__ import annotations

import uuid
from xml.sax.saxutils import escape, quoteattr

HSL_CHANNELS = {
    "红色": "Red", "橙色": "Orange", "黄色": "Yellow", "绿色": "Green",
    "浅绿色": "Aqua", "蓝色": "Blue", "紫色": "Purple", "洋红": "Magenta",
}
_EN = {v.lower(): v for v in HSL_CHANNELS.values()}
BASIC_KEYS = ("Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012",
              "Vibrance", "Saturation", "Clarity2012", "Dehaze", "Texture", "Temperature", "Tint")
CURVE_KEYS = {"composite": "ToneCurvePV2012", "red": "ToneCurvePV2012Red", "green": "ToneCurvePV2012Green",
              "blue": "ToneCurvePV2012Blue"}


def _channel(name: str) -> str:
    if name in HSL_CHANNELS:
        return HSL_CHANNELS[name]
    if name.lower() in _EN:
        return _EN[name.lower()]
    raise ValueError(f"未知的混色器通道：{name}")


def _int(v, lo=-100, hi=100) -> int:
    return int(round(max(lo, min(hi, float(v)))))


def settings_from_params(params: dict) -> tuple[dict[str, str], dict[str, list[str]]]:
    """返回 (简单属性, 曲线序列)。属性值已格式化为 ACR 预设里的写法。"""
    attrs: dict[str, str] = {}
    seqs: dict[str, list[str]] = {}
    for ch, v in (params.get("hsl") or {}).items():
        en = _channel(ch)
        if "hue" in v:
            attrs[f"HueAdjustment{en}"] = f"{_int(v['hue']):+d}" if _int(v["hue"]) else "0"
        if "sat" in v:
            attrs[f"SaturationAdjustment{en}"] = f"{_int(v['sat']):+d}" if _int(v["sat"]) else "0"
        if "lum" in v:
            attrs[f"LuminanceAdjustment{en}"] = f"{_int(v['lum']):+d}" if _int(v["lum"]) else "0"
    g = params.get("grading") or {}
    if g:
        # ACR 13 起的颜色分级：阴影/高光沿用 SplitToning* 字段，中间调/全局为 ColorGrade*
        for zone, (hk, sk, lk) in {
            "shadow": ("SplitToningShadowHue", "SplitToningShadowSaturation", "ColorGradeShadowLum"),
            "highlight": ("SplitToningHighlightHue", "SplitToningHighlightSaturation", "ColorGradeHighlightLum"),
            "midtone": ("ColorGradeMidtoneHue", "ColorGradeMidtoneSat", "ColorGradeMidtoneLum"),
            "global": ("ColorGradeGlobalHue", "ColorGradeGlobalSat", "ColorGradeGlobalLum"),
        }.items():
            z = g.get(zone)
            if not z:
                continue
            if "hue" in z:
                attrs[hk] = str(_int(z["hue"], 0, 359) % 360)
            if "sat" in z:
                attrs[sk] = str(_int(z["sat"], 0, 100))
            if "lum" in z:
                attrs[lk] = str(_int(z["lum"]))
        if "blending" in g:
            attrs["ColorGradeBlending"] = str(_int(g["blending"], 0, 100))
        if "balance" in g:
            attrs["SplitToningBalance"] = str(_int(g["balance"]))
    cal = params.get("calibration") or {}
    for prim in ("red", "green", "blue"):
        z = cal.get(prim)
        if not z:
            continue
        cap = prim.capitalize()
        if "hue" in z:
            attrs[f"{cap}Hue"] = str(_int(z["hue"]))
        if "sat" in z:
            attrs[f"{cap}Saturation"] = str(_int(z["sat"]))
    if "shadow_tint" in cal:
        attrs["ShadowTint"] = str(_int(cal["shadow_tint"]))
    for k, v in (params.get("basic") or {}).items():
        if k not in BASIC_KEYS:
            raise ValueError(f"不支持的基本面板参数：{k}")
        if k == "Exposure2012":
            attrs[k] = f"{max(-5.0, min(5.0, float(v))):+.2f}"
        elif k == "Temperature":
            attrs[k] = str(_int(v, 2000, 50000))
        else:
            attrs[k] = str(_int(v))
    for ch, pts in (params.get("curve") or {}).items():
        if ch not in CURVE_KEYS:
            raise ValueError(f"未知的曲线通道：{ch}")
        by_x: dict[int, int] = {}
        for x, y in pts:  # 同一个输入值只能有一个点
            by_x.setdefault(_int(x, 0, 255), _int(y, 0, 255))
        clean = sorted(by_x.items())
        if clean[0][0] != 0:
            clean.insert(0, (0, 0))
        if clean[-1][0] != 255:
            clean.append((255, 255))
        seqs[CURVE_KEYS[ch]] = [f"{x}, {y}" for x, y in clean]
    if any(k.startswith(("HueAdj", "SaturationAdj", "LuminanceAdj")) for k in attrs) or seqs:
        attrs.setdefault("ConvertToGrayscale", "False")
    return attrs, seqs


def build_xmp(name: str, params: dict, group: str = "ColorDoctor") -> str:
    attrs, seqs = settings_from_params(params)
    head = {
        "PresetType": "Normal", "Cluster": "", "UUID": uuid.uuid4().hex.upper(), "SupportsAmount": "False",
        "SupportsColor": "True", "SupportsMonochrome": "True", "SupportsHighDynamicRange": "True",
        "SupportsNormalDynamicRange": "True", "SupportsSceneReferred": "True", "SupportsOutputReferred": "True",
        "CameraModelRestriction": "", "Copyright": "", "ContactInfo": "", "Version": "15.0",
        "ProcessVersion": "11.0",
    }
    lines = ['<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="ColorDoctor">',
             ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
             '  <rdf:Description rdf:about=""',
             '    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"']
    for k, v in {**head, **attrs, "HasSettings": "True"}.items():
        lines.append(f"   crs:{k}={quoteattr(v)}")
    lines[-1] += ">"
    for tag, text in (("Name", name), ("Group", group)):
        lines += [f"   <crs:{tag}>", "    <rdf:Alt>", f'     <rdf:li xml:lang="x-default">{escape(text)}</rdf:li>',
                  "    </rdf:Alt>", f"   </crs:{tag}>"]
    for tag, items in seqs.items():
        lines += [f"   <crs:{tag}>", "    <rdf:Seq>"] + [f"     <rdf:li>{p}</rdf:li>" for p in items] + ["    </rdf:Seq>", f"   </crs:{tag}>"]
    lines += ["  </rdf:Description>", " </rdf:RDF>", "</x:xmpmeta>", ""]
    return "\n".join(lines)


def params_from_adjustments(adjustments) -> dict:
    """从分析结果的建议（Adjustment 列表）里收集能直接写进 ACR 的数值。

    认得两类：tool == "Camera Raw 混色器"（instruction 以通道名开头，params 里有 色相/饱和度/明亮度），
    tool == "Camera Raw 颜色分级"（instruction 以 阴影/高光/中间调 开头，params 里有 色相/饱和度）。
    其余（文字说明、“常用做法”）不导出。
    """
    out: dict = {"hsl": {}, "grading": {}}
    keymap = {"色相": "hue", "饱和度": "sat", "明亮度": "lum"}
    zones = {"阴影": "shadow", "高光": "highlight", "中间调": "midtone"}
    for a in adjustments:
        if not a.params:
            continue
        if a.tool == "Camera Raw 混色器":
            ch = next((c for c in sorted(HSL_CHANNELS, key=len, reverse=True) if a.instruction.startswith(c)), None)
            if ch is None:
                continue
            slot = out["hsl"].setdefault(ch, {})
            for k, v in a.params:
                if k in keymap:
                    slot[keymap[k]] = int(round(slot.get(keymap[k], 0) + v))
        elif a.tool == "Camera Raw 颜色分级":
            zone = next((z for z in zones if a.instruction.startswith(z)), None)
            if zone is None:
                continue
            slot = out["grading"].setdefault(zones[zone], {})
            for k, v in a.params:
                if k in keymap:
                    slot[keymap[k]] = int(round(v))
    return {k: v for k, v in out.items() if v}


def params_for_report(report) -> dict:
    """一份分析结果能直接写进 ACR 的全部数值：混色器、颜色分级，以及仿色的 RGB 复合曲线。"""
    adjs = list(getattr(report, "adjustments", ()) or ())
    for f in getattr(report, "findings", ()) or ():
        adjs.extend(f.adjustments)
    pal = getattr(report, "palette", None)
    if pal is not None and getattr(pal, "style", None) is not None:
        adjs.extend(pal.style.adjustments)
    seen, uniq = set(), []
    for a in adjs:  # 同一条建议可能同时出现在结论和风格结果里
        key = (a.tool, a.instruction)
        if key not in seen:
            seen.add(key)
            uniq.append(a)
    out = params_from_adjustments(uniq)
    ref = getattr(report, "reference", None)
    if ref is not None and getattr(ref, "curve_points", None):
        out["curve"] = {"composite": [[int(a), int(b)] for a, b in ref.curve_points]}
    return out
