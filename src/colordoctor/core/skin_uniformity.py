"""肤色均匀度：整块皮肤是不是同一个色相、同样的饱和度。

把皮肤切成小块，每块计算色相角 h° 和与明度无关的饱和度 s = C*/L*，
与整块皮肤的中位值比较：
- 色相：比整体更偏黄（h 更大）/ 更偏红橙（h 更小）/ 偏洋红 / 偏青绿；
- 饱和度：与“同明度段”的皮肤比较，明显更灰（缺色）或更艳。
相邻的异常小块合并为一处“问题位置”，带坐标、面积和数值。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from colordoctor.core import color_spaces as cs
from colordoctor.core.contamination import _components, downsample_mask
from colordoctor.core.region_statistics import weighted_circular_hue
from colordoctor.core.pixel_table import par_ranges, table, wmedian
from colordoctor.core.sampling import FULL_RES_MAX, grid_downsample

HUE_MILD = 6.0  # 色相偏离（度）：轻微
HUE_NOTABLE = 10.0  # 明显（例如橙 55° 与黄 68° 相差约 13°）
SAT_REL = 0.22  # 饱和度相对偏离 22%
MIN_TILE_SKIN = 0.4
L_BAND = 10.0


@dataclass(frozen=True)
class UniformPatch:
    kind: str  # "偏黄" / "偏红橙" / "偏洋红" / "偏青绿" / "缺色发灰" / "饱和度偏高"
    bbox: tuple[int, int, int, int]
    area_fraction: float  # 占皮肤面积
    hue_shift: float  # 相对整体（度）
    sat_shift: float  # 相对同明度皮肤（比例）
    lab: tuple[float, float, float]
    severity: str
    # 真正偏离的那些小块（源图坐标的矩形）：外接框 bbox 里还有正常的皮肤，修正图层只能作用于这些小块，
    # 否则正常皮肤会被往反方向推（例如偏黄那块往红转时，框里本来正常的皮肤也被转红）
    cells: tuple[tuple[int, int, int, int], ...] = ()


@dataclass(frozen=True)
class UniformityResult:
    skin_pixels: int
    median_hue: float
    median_saturation: float
    hue_spread: float  # 各小块色相的加权标准差（度）
    sat_cv: float  # 各小块饱和度的变异系数
    uneven_hue_share: float  # 色相偏离 ≥ HUE_MILD 的皮肤面积比例
    uneven_sat_share: float
    score: float  # 0–100，越高越均匀
    patches: tuple[UniformPatch, ...] = ()
    tile_size: int = 0
    tile_map: object = field(default=None, repr=False)  # (th, tw, 3) 每块：色相偏移、饱和度偏移、有效
    step: int = 1


def _kind(dh: float, ds: float, a_abs: float) -> str | None:
    """a_abs：该小块的平均 a*。a* 接近 0 说明已经没有红色成分 -> 偏青绿。"""
    if abs(dh) >= HUE_MILD:
        if dh > 0:
            return "偏青绿" if (dh >= 25 or a_abs < 3) else "偏黄"
        return "偏洋红" if dh <= -25 else "偏红橙"
    if ds <= -SAT_REL:
        return "缺色发灰"
    if ds >= SAT_REL:
        return "饱和度偏高"
    return None


def analyze_uniformity(
    region: np.ndarray,
    mask: np.ndarray | None = None,
    offset: tuple[int, int] = (0, 0),
    max_samples: int = FULL_RES_MAX,
) -> UniformityResult:
    sub, step = grid_downsample(region, max_samples)
    hh, ww = sub.shape[:2]
    valid = downsample_mask(mask, step, (hh, ww))
    # 原图尺寸的每个像素：颜色只算一次（PixelTable），每一小块按“颜色 × 这块里的像素数”统计
    pt = table(sub)
    Lu, Cu, hu = pt.L, pt.C, pt.h
    ok_u = Cu >= 3  # 几乎无彩的颜色不算皮肤：按颜色排除（不建整幅的图）
    wv = pt.count(None if mask is None else valid) * ok_u
    n_skin = int(wv.sum())
    if n_skin < 20:
        return UniformityResult(n_skin, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, step=step)
    su = Cu / np.maximum(Lu, 1.0)
    nz = wv > 0
    h0, _, _ = weighted_circular_hue(hu[nz], Cu[nz] * wv[nz])
    s0 = wmedian(su[nz], wv[nz])

    ts = int(max(6, round(min(hh, ww) / 12)))
    th, tw = (hh + ts - 1) // ts, (ww + ts - 1) // ts
    tmap = np.zeros((th, tw, 3), dtype=float)
    cells = [(ty, tx) for ty in range(th) for tx in range(tw)]

    def do_tiles(a, b):  # 各小块互不相关：分给几个线程
        out = []
        for ty, tx in cells[a:b]:
            sl = (slice(ty * ts, (ty + 1) * ts), slice(tx * ts, (tx + 1) * ts))
            idx = pt.index[sl]
            cnt = np.bincount(idx.ravel() if mask is None else idx[valid[sl]], minlength=pt.n)
            cnt[~ok_u] = 0
            nv = int(cnt.sum())
            if idx.size == 0 or nv / idx.size < MIN_TILE_SKIN or nv < 6:
                continue
            k = np.flatnonzero(cnt)
            c = cnt[k].astype(np.float64)
            ht, _, _ = weighted_circular_hue(hu[k], Cu[k] * c)
            if ht is None:
                continue
            out.append((ty, tx, float(cs.hue_diff(h0, ht)), wmedian(su[k], c), wmedian(Lu[k], c), nv,
                        (pt.lab[k] * c[:, None]).sum(axis=0) / c.sum()))
        return out

    tiles = [t for part in par_ranges(do_tiles, len(cells), 1) for t in part]
    if not tiles:
        return UniformityResult(n_skin, float(h0), s0, 0.0, 0.0, 0.0, 0.0, 100.0, step=step)

    # 饱和度与“同明度段”的皮肤比较（阴影/高光本来就会改变饱和度）
    Ls = np.array([t[4] for t in tiles])
    Ss = np.array([t[3] for t in tiles])
    W = np.array([t[5] for t in tiles], dtype=float)
    bands = np.floor(Ls / L_BAND)
    ref_s = np.array([np.median(Ss[np.abs(bands - b) <= 1]) for b in bands])
    ds_all = Ss / np.maximum(ref_s, 1e-6) - 1.0
    dh_all = np.array([t[2] for t in tiles])

    flags = np.zeros((th, tw), dtype=object)
    for i, (ty, tx, dh, _s, _l, _n, mlab) in enumerate(tiles):
        k = _kind(dh, float(ds_all[i]), float(mlab[1]))
        tmap[ty, tx] = (dh, ds_all[i], 1.0)
        flags[ty, tx] = k

    patches = []
    tile_index = {(t[0], t[1]): i for i, t in enumerate(tiles)}
    for kind in ("偏黄", "偏红橙", "偏洋红", "偏青绿", "缺色发灰", "饱和度偏高"):
        fm = np.vectorize(lambda v, k=kind: v == k)(flags) if flags.size else np.zeros((th, tw), bool)
        for comp in _components(fm.astype(bool)):
            idx = [tile_index[c] for c in comp if c in tile_index]
            if not idx:
                continue
            w = W[idx]
            area = float(w.sum() / n_skin)
            if area < 0.01:
                continue
            ys = [c[0] for c in comp]
            xs = [c[1] for c in comp]
            y0, y1 = min(ys) * ts, min(hh, (max(ys) + 1) * ts)
            x0, x1 = min(xs) * ts, min(ww, (max(xs) + 1) * ts)
            # 用中位数：边缘小块只部分覆盖问题区域，会稀释平均值
            mdh = float(np.median(dh_all[idx]))
            mds = float(np.median(ds_all[idx]))
            mlab = np.average(np.array([tiles[i][6] for i in idx]), axis=0, weights=w)
            big = abs(mdh) >= HUE_NOTABLE or abs(mds) >= 1.6 * SAT_REL
            sev = ("明显" if area >= 0.03 else "轻微") if big else "轻微"
            patches.append(
                UniformPatch(
                    kind=kind,
                    bbox=(int(offset[0] + x0 * step), int(offset[1] + y0 * step), int((x1 - x0) * step), int((y1 - y0) * step)),
                    area_fraction=round(area, 4),
                    hue_shift=round(mdh, 2),
                    sat_shift=round(mds, 3),
                    lab=tuple(round(float(v), 2) for v in mlab),
                    severity=sev,
                    cells=tuple((int(offset[0] + c[1] * ts * step), int(offset[1] + c[0] * ts * step),
                                 int((min(ww, (c[1] + 1) * ts) - c[1] * ts) * step), int((min(hh, (c[0] + 1) * ts) - c[0] * ts) * step))
                                for c in comp if c in tile_index),
                )
            )
    patches.sort(key=lambda p: -(p.area_fraction * (abs(p.hue_shift) + 30 * abs(p.sat_shift))))

    hue_spread = float(np.sqrt(np.average(dh_all**2, weights=W)))
    sat_cv = float(np.sqrt(np.average(ds_all**2, weights=W)))
    uneven_h = float(W[np.abs(dh_all) >= HUE_MILD].sum() / W.sum())
    uneven_s = float(W[np.abs(ds_all) >= SAT_REL].sum() / W.sum())
    # 评分：色相/饱和度不均匀的皮肤面积是主要扣分项（10% 面积偏色约扣 18 分）
    score = float(
        np.clip(
            100 - 3.0 * max(0.0, hue_spread - 2.5) - 150 * max(0.0, sat_cv - 0.06) - 180 * uneven_h - 150 * uneven_s,
            0,
            100,
        )
    )
    return UniformityResult(
        skin_pixels=n_skin,
        median_hue=round(float(h0), 2),
        median_saturation=round(s0, 4),
        hue_spread=round(hue_spread, 3),
        sat_cv=round(sat_cv, 4),
        uneven_hue_share=round(uneven_h, 4),
        uneven_sat_share=round(uneven_s, 4),
        score=round(score, 1),
        patches=tuple(patches[:12]),
        tile_size=ts,
        tile_map=tmap,
        step=step,
    )
