"""精准选中皮肤：在用户框选/套索的区域里，自动挑出属于皮肤的像素。

不使用 AI 分割，只用可解释的颜色规则：
1. 宽松的“肤色色域”：Lab 色相角 0°–100°、色度 5–65、明度 12–97
   （覆盖从深肤色到冷白皮，深肤色样例 Lab(27, 12, 12) 也在内）。
2. 以选区里最主要的肤色为中心（或用户点击的种子点），
   只保留色相相差 ≤ 22°、饱和度 s = C*/L* 在 0.45–2.2 倍之间的像素
   —— 去掉嘴唇（更红更艳）、眉毛头发（更暗、更灰）、眼白、背景、衣服。
3. 多数滤波去掉零散噪点。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from colordoctor.core import color_spaces as cs
from colordoctor.core.region_statistics import weighted_circular_hue

HUE_RANGE = (0.0, 100.0)
C_RANGE = (5.0, 65.0)
L_RANGE = (12.0, 97.0)
HUE_TOL = 22.0
SAT_RATIO = (0.45, 2.2)
DARK_RATIO = 0.5


def polygon_mask(points, width: int, height: int) -> np.ndarray:
    """套索多边形（相对选区左上角的像素坐标）-> 布尔蒙版。"""
    img = Image.new("1", (int(width), int(height)), 0)
    if len(points) >= 3:
        ImageDraw.Draw(img).polygon([(float(x), float(y)) for x, y in points], fill=1, outline=1)
    return np.asarray(img, dtype=bool)


def _majority(m: np.ndarray, passes: int = 2) -> np.ndarray:
    for _ in range(passes):
        p = np.pad(m.astype(np.int8), 1)
        s = sum(p[1 + dy : 1 + dy + m.shape[0], 1 + dx : 1 + dx + m.shape[1]] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        m = s >= 5
    return m


MAX_MASK_PIXELS = 1_500_000


def skin_mask(
    region: np.ndarray, within: np.ndarray | None = None, seed: tuple[int, int] | None = None
) -> tuple[np.ndarray, dict]:
    """返回 (与 region 同尺寸的布尔蒙版, 说明信息)。

    within：可选的套索蒙版；seed：可选的“这里是皮肤”种子点 (x, y)。
    大选区先降采样计算，再按最近邻放大回原尺寸。
    """
    H, W = region.shape[:2]
    step = int(np.ceil(np.sqrt(H * W / MAX_MASK_PIXELS))) if H * W > MAX_MASK_PIXELS else 1
    if step > 1:
        small = region[::step, ::step]
        w_small = within[::step, ::step] if within is not None else None
        sd = (seed[0] // step, seed[1] // step) if seed is not None else None
        m, info = _skin_mask(small, w_small, sd)
        big = np.repeat(np.repeat(m, step, axis=0), step, axis=1)[:H, :W]
        if within is not None:
            big &= within
        return big, info
    return _skin_mask(region, within, seed)


def _skin_mask(region, within, seed):
    lab = cs.srgb8_to_lab(region[..., :3])
    lch = cs.lab_to_lch(lab)
    L, C, h = lch[..., 0], lch[..., 1], lch[..., 2]
    cand = (h >= HUE_RANGE[0]) & (h <= HUE_RANGE[1]) & (C >= C_RANGE[0]) & (C <= C_RANGE[1])
    cand &= (L >= L_RANGE[0]) & (L <= L_RANGE[1])
    if within is not None:
        cand &= within
    info = {"candidate_share": float(cand.mean()) if cand.size else 0.0}
    if cand.sum() < 10:
        return cand, {**info, "skin_share": float(cand.mean()), "center_hue": None, "note": "选区中几乎没有肤色像素"}
    s = C / np.maximum(L, 1.0)
    if seed is not None and 0 <= seed[1] < region.shape[0] and 0 <= seed[0] < region.shape[1]:
        y0, x0 = seed[1], seed[0]
        sl = (slice(max(0, y0 - 3), y0 + 4), slice(max(0, x0 - 3), x0 + 4))
        hc, _, _ = weighted_circular_hue(h[sl].ravel(), C[sl].ravel())
        sc = float(np.median(s[sl]))
        lc = float(np.median(L[sl]))
    else:
        hc, _, _ = weighted_circular_hue(h[cand], C[cand])
        # 用色相中位附近的像素重新估计中心，避免被嘴唇/背景拉偏
        near = cand & (np.abs(cs.hue_diff(hc, h)) <= 30)
        hc, _, _ = weighted_circular_hue(h[near], C[near])
        sc = float(np.median(s[near]))
        lc = float(np.median(L[near]))
    m = cand & (np.abs(cs.hue_diff(hc, h)) <= HUE_TOL) & (s >= SAT_RATIO[0] * sc) & (s <= SAT_RATIO[1] * sc)
    m &= L >= DARK_RATIO * lc  # 头发、眉毛、瞳孔比皮肤暗得多
    m = _majority(m)
    if within is not None:
        m &= within
    base = within.sum() if within is not None else m.size
    info.update(
        {
            "skin_share": float(m.sum() / max(1, base)),
            "center_hue": round(float(hc), 2),
            "center_saturation": round(sc, 4),
            "note": "",
        }
    )
    return m, info
