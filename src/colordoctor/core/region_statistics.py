"""区域统计：均值/中位数、分位数、按色度加权的圆形色相统计、主色聚类。"""

from __future__ import annotations

import numpy as np

from colordoctor.core import clustering
from colordoctor.core import color_spaces as cs
from colordoctor.core.color_models import Cluster, RegionStats
from colordoctor.core.sampling import SampleError, grid_downsample

MAX_SAMPLES = 262_144
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
        rgb = sub[vsub].reshape(-1, 3)
        total = int(np.asarray(valid, dtype=bool).sum())
    else:
        rgb = sub.reshape(-1, 3)
        total = int(region.shape[0] * region.shape[1])
    n = rgb.shape[0]
    lab = cs.srgb8_to_lab(rgb)
    lch = cs.lab_to_lch(lab)
    ok = cs.srgb8_to_oklab(rgb)
    hsb = cs.srgb8_to_hsb(rgb)
    L, C, h = lch[:, 0], lch[:, 1], lch[:, 2]

    chromatic = C >= CHROMA_MIN
    hue_mean, R, spread = weighted_circular_hue(h[chromatic], C[chromatic])

    clusters: tuple[Cluster, ...] = ()
    if with_clusters:
        cl = []
        for center, share in clustering.dominant_clusters(lab):
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

    lab_mean = lab.mean(axis=0)
    ok_mean = ok.mean(axis=0)
    clipped = np.any((rgb == 0) | (rgb == 255), axis=1)
    return RegionStats(
        total_pixels=total,
        sampled_pixels=n,
        rgb_mean=_t(rgb.mean(axis=0), 2),
        rgb_median=_t(np.median(rgb, axis=0), 2),
        lab_mean=_t(lab_mean),
        lab_median=_t(np.median(lab, axis=0)),
        oklab_mean=_t(ok_mean, 5),
        oklch_mean=_t(cs.oklab_to_oklch(ok_mean), 5),
        lch_mean=_t(cs.lab_to_lch(lab_mean)),
        L_std=round(float(L.std()), 4),
        C_std=round(float(C.std()), 4),
        L_p10=round(float(np.percentile(L, 10)), 3),
        L_p50=round(float(np.percentile(L, 50)), 3),
        L_p90=round(float(np.percentile(L, 90)), 3),
        C_p10=round(float(np.percentile(C, 10)), 3),
        C_p50=round(float(np.percentile(C, 50)), 3),
        C_p90=round(float(np.percentile(C, 90)), 3),
        sat_p50=round(float(np.median(hsb[:, 1])), 3),
        hue_mean=None if hue_mean is None or chromatic.mean() < 0.05 else round(hue_mean, 3),
        hue_concentration=round(R, 6),
        hue_spread=None if spread is None else round(spread, 4),
        chromatic_fraction=round(float(chromatic.mean()), 6),
        highlight_fraction=round(float((L >= HIGHLIGHT_L).mean()), 6),
        shadow_fraction=round(float((L <= SHADOW_L).mean()), 6),
        clipped_fraction=round(float(clipped.mean()), 6),
        clusters=clusters,
    )
