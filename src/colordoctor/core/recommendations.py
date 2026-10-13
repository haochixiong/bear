"""把诊断结果转换为可以在 Camera Raw / Photoshop 中执行的调整建议。

原则：
- 方向优先；强度用“轻微 / 小幅 / 中等”表示，并标注为估计值。
- 曲线对“取样平均色”给出精确的输入→输出点；色彩平衡、色相/饱和度为模型估算；
  可选颜色只给方向和起步值；Camera Raw 只给方向（无法从 JPG 反推滑块数值）。
- 应用后请重新取样，工具会显示剩余差异，再做第二轮微调。
"""

from __future__ import annotations

import numpy as np

from colordoctor.core import color_spaces as cs
from colordoctor.core import ps_models as pm
from colordoctor.core.color_models import Adjustment

# 本工具的色相名 -> (Camera Raw 混色器通道, Photoshop 色相/饱和度 & 可选颜色 通道)
CHANNELS = {
    "红": ("红色", "红色"),
    "橙": ("橙色", "红色"),
    "黄": ("黄色", "黄色"),
    "绿": ("绿色", "绿色"),
    "青": ("浅绿色", "青色"),
    "蓝": ("蓝色", "蓝色"),
    "紫/品红": ("紫色/洋红", "洋红"),
}


def strength_label(magnitude: float) -> str | None:
    m = abs(float(magnitude))
    if m < 2:
        return None
    if m < 4:
        return "轻微"
    if m < 8:
        return "小幅"
    return "中等"


def acr_directions(dL: float, da: float, db: float, dC: float, dh: float, channel: str = "橙色", skin: bool = True) -> list[Adjustment]:
    """把“需要的变化量”（目标 - 当前）映射为 Camera Raw 调整方向。"""
    out: list[Adjustment] = []
    s = strength_label(db)
    if s:
        out.append(
            Adjustment(
                "Camera Raw",
                f"色温向{'黄' if db > 0 else '蓝'}（{'提高' if db > 0 else '降低'}）——主要改变 b*（黄-蓝）",
                s,
            )
        )
    s = strength_label(da)
    if s:
        out.append(
            Adjustment("Camera Raw", f"色调(Tint)向{'洋红' if da > 0 else '绿'}——主要改变 a*（红-绿）", s)
        )
    s = strength_label(dh / 2.0)
    if s:
        out.append(
            Adjustment("Camera Raw", f"混色器 → 色相 → 橙色向{'黄' if dh > 0 else '红'}移动（肤色更{'黄' if dh > 0 else '红润'}）", s)
            if skin else
            Adjustment("Camera Raw", f"混色器 → 色相 → {channel}滑块往{'右（+）' if dh > 0 else '左（−）'}", s)
        )
    s = strength_label(dC)
    if s:
        out.append(Adjustment("Camera Raw", f"混色器 → 饱和度 → {channel}{'提高' if dC > 0 else '降低'}", s))
    s = strength_label(dL)
    if s:
        out.append(Adjustment("Camera Raw", f"混色器 → 明亮度 → {channel}{'提高' if dL > 0 else '降低'}"
                              + ("（只影响肤色亮度）" if skin else ""), s))
    if out:
        out.append(
            Adjustment(
                "Camera Raw",
                "提示：色温/色调是全局调整，会影响整张照片；只想改" + ("肤色" if skin else "这种颜色") + "时优先用混色器（HSL）或局部蒙版",
                None,
            )
        )
    return out


def _fmt(params) -> str:
    return "，".join(f"{n} {int(v):+d}" for n, v in params if int(v) != 0) or "无需调整"


def skin_adjustments(pixels_rgb8: np.ndarray, target_lab, keep_lightness: bool = False, skin: bool = True) -> list[Adjustment]:
    """当前像素 -> 目标 Lab 的 Photoshop/Camera Raw 参数建议。

    skin=True：肤色（色相/饱和度、可选颜色只在红色 / 黄色范围里选）；
    skin=False：任意颜色，范围按像素本身的色相从六个范围里选。
    """
    px = np.asarray(pixels_rgb8, dtype=float).reshape(-1, 3)
    if px.shape[0] > 200_000:  # 平均色和曲线点不需要全部像素
        px = px[:: int(np.ceil(px.shape[0] / 200_000))]
    ranges = pm.SKIN_RANGES if skin else tuple(pm.PS_RANGES)
    where = "皮肤" if skin else "这块区域"
    cur_lab = cs.srgb8_to_lab(px).mean(axis=0)
    tgt = np.array(target_lab, dtype=float)
    if keep_lightness:
        tgt[0] = cur_lab[0]
    de = float(cs.delta_e_2000(cur_lab, tgt))
    out: list[Adjustment] = []
    if de < 1.0:
        return [Adjustment("提示", f"当前肤色与目标已经非常接近（ΔE00 {de:.1f}），无需调整", None, False)]

    # 曲线：对取样平均色精确。明度差较大时，通道曲线只改颜色（保持当前明度），
    # 明度另用下面的 RGB 复合曲线处理，避免两步叠加把明度调两次。
    dL = float(tgt[0] - cur_lab[0])
    split_lightness = abs(dL) >= 1.5
    curve_tgt = np.array([cur_lab[0], tgt[1], tgt[2]]) if split_lightness else tgt
    mean_rgb = np.rint(px.mean(axis=0)).astype(int)
    tgt_rgb = cs.lab_to_srgb8(curve_tgt).astype(int)
    pts = pm.curves_points(mean_rgb, tgt_rgb)
    gamut_note = "" if cs.lab_in_srgb_gamut(curve_tgt) else "（目标超出 sRGB，已裁切）"
    out.append(
        Adjustment(
            "曲线",
            ("颜色：" if split_lightness else "") + "在 RGB 曲线的各通道上加一个点："
            + "；".join(f"{ch}通道 输入 {i} → 输出 {o}" for ch, (i, o) in pts.items())
            + gamut_note + f"。建议用蒙版只作用于{where}",
            strength_label(de),
            estimated=False,
            params=tuple((f"{ch}通道 {i}→", float(o)) for ch, (i, o) in pts.items()),
        )
    )

    # 明度单独用 RGB 复合曲线处理；颜色工具只负责颜色（目标保持当前明度）
    if split_lightness:
        g_in = int(cs.lab_to_srgb8(np.array([cur_lab[0], 0.0, 0.0]))[0])
        g_out = int(cs.lab_to_srgb8(np.array([tgt[0], 0.0, 0.0]))[0])
        out.append(
            Adjustment(
                "曲线",
                f"明度：RGB 复合曲线加一个点 输入 {g_in} → 输出 {g_out}（L* {cur_lab[0]:.1f} → {tgt[0]:.1f}）。"
                "放在调颜色的那条曲线上方（先调颜色、再调明度），两者不要合并到同一条曲线",
                strength_label(dL),
                estimated=True,
                params=(("RGB 输入", float(g_in)), ("RGB 输出", float(g_out))),
            )
        )
    color_tgt = np.array([cur_lab[0], tgt[1], tgt[2]])
    x = px / 255.0
    sol = pm.solve(x, color_tgt, "color_balance")
    out.append(
        Adjustment(
            "色彩平衡",
            f"色调：中间调，勾选“保留明度”：{_fmt(sol.params)}（模型估算，只改颜色，预计剩余色差 ΔE00 {sol.residual_de:.1f}）",
            strength_label(float(cs.delta_e_2000(cur_lab, color_tgt))),
            estimated=True,
            params=sol.params,
        )
    )
    sol = pm.solve(x, color_tgt, "hue_saturation", ranges=ranges)
    rng_name = pm.RANGE_NAMES.get(sol.extra.get("range"), "全图")
    out.append(
        Adjustment(
            "色相/饱和度",
            f"编辑范围选“{rng_name}”：{_fmt(sol.params)}（模型估算，只改颜色，预计剩余色差 ΔE00 {sol.residual_de:.1f}）",
            strength_label(float(cs.delta_e_2000(cur_lab, color_tgt))),
            estimated=True,
            params=sol.params,
        )
    )
    sol = pm.solve(x, color_tgt, "selective_color", ranges=ranges)
    rng_name = pm.RANGE_NAMES.get(sol.extra.get("range"), "红色")
    dirs = []
    for n, v in sol.params:
        if abs(v) >= 1:
            start = int(np.sign(v) * min(10, max(3, abs(v) // 4)))
            dirs.append(f"{n}{'增加' if v > 0 else '减少'}（起步 {start:+d}%）")
    out.append(
        Adjustment(
            "可选颜色",
            f"颜色选“{rng_name}”，方法“相对”：" + ("，".join(dirs) if dirs else "无需调整")
            + "。此工具只给方向和起步值，按起步值调整后重新取样再微调",
            strength_label(float(cs.delta_e_2000(cur_lab, color_tgt))),
            estimated=True,
            params=sol.params,
        )
    )
    cur_lch = cs.lab_to_lch(cur_lab)
    tgt_lch = cs.lab_to_lch(tgt)
    out += acr_directions(
        dL=float(tgt[0] - cur_lab[0]),
        da=float(tgt[1] - cur_lab[1]),
        db=float(tgt[2] - cur_lab[2]),
        dC=float(tgt_lch[1] - cur_lch[1]),
        dh=float(cs.hue_diff(cur_lch[2], tgt_lch[2])),
        channel="橙色" if skin else _acr_channel(cur_lch),
        skin=skin,
    )
    return out


_NEIGHBOURS = {  # ACR 混色器通道的左右邻居
    "红色": ("洋红", "橙色"), "橙色": ("红色", "黄色"), "黄色": ("橙色", "绿色"), "绿色": ("黄色", "浅绿色"),
    "浅绿色": ("绿色", "蓝色"), "蓝色": ("浅绿色", "紫色"), "紫色": ("蓝色", "洋红"), "洋红": ("紫色", "红色"),
}


def dispersion_adjustments(base_name: str, skin: bool) -> list[Adjustment]:
    """同一区域里色相分散：把两侧相邻通道的色相往主色收拢，而不是降低主色饱和度。"""
    acr, ps = ("橙色", "红色") if skin or base_name in ("肤色", "") else CHANNELS.get(base_name, (base_name, base_name))
    acr = acr.split("/")[0]
    left, right = _NEIGHBOURS.get(acr, ("相邻通道", "相邻通道"))
    return [
        Adjustment("Camera Raw 混色器",
                   f"色相：“{left}”往右、“{right}”往左（都往“{acr}”收拢），起步各 10–20；"
                   f"“{acr}”本身的色相和饱和度先不动。用蒙版只作用于这块区域", "小幅"),
        Adjustment("色相/饱和度",
                   f"编辑范围选“{ps}”，把范围两侧拉宽后整体色相小幅调整（±3–8），让偏离的部分并回主色；"
                   "或用“色彩平衡”分别修正偏离的那一部分", "轻微"),
    ]


def _acr_channel(lch) -> str:
    from colordoctor.core import acr_channels as ac

    return ac.channel_of_lab_hue(float(lch[2])) if lch[1] >= 4 else "橙色"


def contamination_adjustments(contaminant: str, base: str, skin: bool) -> list[Adjustment]:
    acr, ps = CHANNELS.get(contaminant, (contaminant, contaminant))
    out = [
        Adjustment(
            "Camera Raw 混色器",
            f"降低“{acr}”的饱和度；必要时把“{acr}”的色相往{_toward_base(contaminant, base)}方向拉，"
            "用画笔/蒙版只作用于问题区域",
            "小幅",
        ),
        Adjustment("色相/饱和度", f"编辑范围选“{ps}”，饱和度降低（起步 -20），用吸管确认范围只覆盖杂色", "小幅"),
        Adjustment("可选颜色", f"颜色选“{ps}”，减少该颜色自身的成分（方法“相对”，起步 10%）", "轻微"),
    ]
    if skin:
        out.append(
            Adjustment(
                "提示",
                "肤色里的杂色常来自环境反光（绿植、蓝天、彩色墙）或分离色调/颜色分级作用到了皮肤；"
                "也可以在中性灰/中性色图层上用画笔局部修正",
                None,
                False,
            )
        )
    return out


def graying_adjustments(base: str) -> list[Adjustment]:
    acr, ps = CHANNELS.get(base, (base, base))
    return [
        Adjustment("Camera Raw 混色器", f"局部提高“{acr}”的饱和度（只在发灰处用蒙版）", "小幅"),
        Adjustment("曲线", "发灰处的局部对比偏低：用曲线小幅加对比，或检查去朦胧/清晰度/降噪是否过度", "轻微"),
        Adjustment("色相/饱和度", f"编辑范围选“{ps}”，饱和度小幅提高（起步 +10），配合蒙版", "轻微"),
    ]


_CURVE_FIX = {
    "黄": "蓝通道上提",
    "青": "红通道上提",
    "绿": "绿通道下压",
    "洋红": "绿通道上提",
    "红": "红通道下压",
    "蓝": "蓝通道下压",
}


def cast_adjustments(where: str, shift_name: str) -> list[Adjustment]:
    """where: '高光' / '阴影'；shift_name: '黄' / '青' / '绿' / '洋红' / '红'。"""
    opposite = {"黄": "蓝", "青": "红", "绿": "洋红", "洋红": "绿", "红": "青", "蓝": "黄"}.get(shift_name, "")
    return [
        Adjustment("Camera Raw", f"颜色分级 → {where}：往“{opposite}”方向加一点（色相选 {opposite}，饱和度从 5 起）", "轻微"),
        Adjustment("色彩平衡", f"色调选“{where}”，往“{opposite}”方向小幅移动（起步 ±5）", "轻微"),
        Adjustment("曲线", f"{_CURVE_FIX.get(shift_name, '对应通道小幅修正')}：只在{'亮部' if where == '高光' else '暗部'}加一个点，幅度很小", "轻微"),
    ]


def _toward_base(contaminant: str, base: str) -> str:
    return f"“{base}”" if base else "相邻色"
