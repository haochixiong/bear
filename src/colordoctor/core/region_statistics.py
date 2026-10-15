"""区域统计：均值/中位数、分位数、按色度加权的圆形色相统计、主色聚类。"""

from __future__ import annotations

import numpy as np

from colordoctor.core import clustering
from colordoctor.core import color_spaces as cs
from colordoctor.core.color_models import Cluster, RegionStats
from colordoctor.core.pixel_table import colour_counts
from colordoctor.core.sampling import FULL_RES_MAX, SampleError, grid_downsample

MAX_SAMPLES = FULL_RES_MAX  # 原图尺寸：每个像素都统计
CHROMA_MIN = 5.0  # 低于此色度的像素色相不稳定，不参与色相统计
HIGHLIGHT_L = 90.0
SHADOW_L = 15.0


def weighted_circular_hue(hue_deg: np.ndarray, weights: np.ndarray) -> tuple[float | None, float, float | None]:
    """返回 (加权圆形平均色相°, 集中度 R, 圆形标准差°)。权重和为 0 时返回 (None, 0, None)。"""
    w = np.asarray(weights, dtype=np.float64)
    tot = w.sum()
    if tot <= 0:
        return None, 0.0, None
    hr = np.radians(np.asarray(hue_deg, dtype=np.float64))
    c = float((w * np.cos(hr)).sum() / tot)
    s = float((w * np.sin(hr)).sum() / tot)
    R = float(np.hypot(c, s))
    if R < 1e-12:
        return None, 0.0, 180.0
    mean = float(np.degrees(np.arctan2(s, c)) % 360.0)
    spread = float(np.degrees(np.sqrt(-2.0 * np.log(min(R, 1.0))))) if R < 1 else 0.0
    return mean, min(R, 1.0), spread


def _t(a, nd=4):
    return tuple(round(float(v), nd) for v in np.ravel(a))


def compute_stats(
    region: np.ndarray, max_samples: int = MAX_SAMPLES, with_clusters: bool = True, valid: np.ndarray | None = None
) -> RegionStats:
    if region is None or region.ndim != 3 or region.shape[0] == 0 or region.shape[1] == 0:
        raise SampleError("选区为空")
    sub, step = grid_downsample(region, max_samples)
    if valid is not None:
        from colordoctor.core.contamination import downsample_mask

        vsub = downsample_mask(valid, step, sub.shape[:2])
        if not vsub.any():
            raise SampleError("蒙版内没有像素")
        total = int(np.asarray(valid, dtype=bool).sum())
    else:
        vsub = None
        total = int(region.shape[0] * region.shape[1])
    # 原图尺寸的每个像素都统计：按“不同的颜色 × 像素数”加权，和逐个像素算完全一样（快很多）
    rgb, wt = colour_counts(sub, vsub)
    n = int(wt.sum())
    w = wt.astype(np.float64)
    tot = float(w.sum())
    lab = cs.srgb8_to_lab(rgb).astype(np.float64)
    lch = cs.lab_to_lch(lab)
    ok = cs.srgb8_to_oklab(rgb).astype(np.float64)
    hsb = cs.srgb8_to_hsb(rgb).astype(np.float64)
    L, C, h = lch[:, 0], lch[:, 1], lch[:, 2]

    def mean(v):
        return (v * (w[:, None] if v.ndim == 2 else w)).sum(axis=0) / tot

    def frac(m):
        return float(w[m].sum() / tot)

    def pct(v, q):
        out = cs.weighted_percentile(v, w, q)  # q 可以是一组：同一组数只排序一次
        return float(out) if np.ndim(out) == 0 else [float(x) for x in out]

    def std(v):
        return float(np.sqrt(max(0.0, float(mean(v * v) - mean(v) ** 2))))

    chromatic = C >= CHROMA_MIN
    hue_mean, R, spread = weighted_circular_hue(h[chromatic], C[chromatic] * w[chromatic])

    clusters: tuple[Cluster, ...] = ()
    if with_clusters:
        cl = []
        for center, share in clustering.dominant_clusters(lab, weights=w):
            c_lch = cs.lab_to_lch(center)
            cl.append(
                Cluster(
                    lab=_t(center, 2),
                    rgb=tuple(int(v) for v in cs.lab_to_srgb8(center)),
                    share=round(share, 4),
                    chroma=round(float(c_lch[1]), 2),
                    hue=round(float(c_lch[2]), 1),
                    hue_name=cs.hue_name(float(c_lch[2]), float(c_lch[1])),
                )
            )
        clusters = tuple(cl)

    lab_mean = mean(lab)
    ok_mean = mean(ok)
    clipped = cs.clipped(rgb)  # 任一通道 ≥ 254 或三个通道都 ≤ 1；饱和的纯色（例如 R = 0 的蓝）不是裁切
    rgbf = rgb.astype(np.float64)
    Lq, Cq = pct(L, [10, 50, 90]), pct(C, [10, 50, 90])
    return RegionStats(
        total_pixels=total,
        sampled_pixels=n,
        rgb_mean=_t(mean(rgbf), 2),
        rgb_median=_t([float(cs.weighted_percentile_int(rgb[:, k], w, 50)) for k in range(3)], 2),
        lab_mean=_t(lab_mean),
        lab_median=_t([pct(lab[:, k], 50) for k in range(3)]),
        oklab_mean=_t(ok_mean, 5),
        oklch_mean=_t(cs.oklab_to_oklch(ok_mean), 5),
        lch_mean=_t(cs.lab_to_lch(lab_mean)),
        L_std=round(std(L), 4),
        C_std=round(std(C), 4),
        L_p10=round(Lq[0], 3),
        L_p50=round(Lq[1], 3),
        L_p90=round(Lq[2], 3),
        C_p10=round(Cq[0], 3),
        C_p50=round(Cq[1], 3),
        C_p90=round(Cq[2], 3),
        sat_p50=round(pct(hsb[:, 1], 50), 3),
        hue_mean=None if hue_mean is None or frac(chromatic) < 0.05 else round(hue_mean, 3),
        hue_concentration=round(R, 6),
        hue_spread=None if spread is None else round(spread, 4),
        chromatic_fraction=round(frac(chromatic), 6),
        highlight_fraction=round(frac(L >= HIGHLIGHT_L), 6),
        shadow_fraction=round(frac(L <= SHADOW_L), 6),
        clipped_fraction=round(frac(clipped), 6),
        clusters=clusters,
    )
