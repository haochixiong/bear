"""脏色检测与定位。

“颜色脏”拆成两类可解释的现象：

1. **对立色混入**：主色调里夹杂着色相相反（相差 ≥ 90°，肤色为 ≥ 60°）、
   且色度明显高于噪声的像素。例：蓝色里混入黄色；橙色肤色里出现紫、青。
2. **局部发灰**：同一种颜色、在相近明度下，某处色度明显低于同色的干净部分
   （按 L* 分段比较，避免把“暗”或“粉彩/混白”误判为灰）。

防误报：
- 无彩色像素（C* < 5）色相不稳定，不参与判断；对立色阈值随噪声自适应。
- 两种颜色都占大比例（≥30%），或成片分布（混杂度低）时视为“配色”，不算污染。
- 整片统一的低饱和不算发灰（只比较同色区域内部的相对差异）。
"""

from __future__ import annotations

from collections import deque

import numpy as np

from colordoctor.core import color_spaces as cs
from colordoctor.core.color_models import Blob, ContaminationResult
from colordoctor.core.pixel_table import PixelTable, wmedian
from colordoctor.core.region_statistics import weighted_circular_hue
from colordoctor.core.sampling import FULL_RES_MAX, grid_downsample

MAX_SAMPLES = FULL_RES_MAX  # 原图尺寸：每个像素都检查
C_MIN = 5.0  # 有彩色像素的最低色度
OPP_C_MIN = 6.0  # 对立色像素的最低色度
FAMILY_HALF_WIDTH = 45.0
OPP_ANGLE = 90.0
SKIN_OPP_ANGLE = 60.0
BICOLOR_SHARE = 0.30
BICOLOR_SHARE_COHERENT = 0.15
GRAY_RATIO = 0.55  # 色度低于同明度同色参考的 55% 视为发灰
GRAY_REF_MIN = 12.0
L_BAND = 8.0
EDGE_GRAD = 8.0

MASK_NORMAL = 0
MASK_IGNORED = 1
MASK_OPPOSITE = 2
MASK_GRAYING = 3
MASK_OUTSIDE = 4  # 不在选区/皮肤蒙版内


def downsample_mask(valid: np.ndarray | None, step: int, shape: tuple[int, int]) -> np.ndarray:
    """把全分辨率布尔蒙版按 grid_downsample 的同一网格取样。"""
    if valid is None:
        return np.ones(shape, dtype=bool)
    off = step // 2 if step > 1 else 0
    v = np.asarray(valid, dtype=bool)[off::step, off::step]
    return v[: shape[0], : shape[1]]


def _smoothed_hist(h: np.ndarray, w: np.ndarray, bins: int = 72) -> np.ndarray:
    hist = np.bincount((h / (360.0 / bins)).astype(int) % bins, weights=w, minlength=bins)
    k = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0
    for _ in range(2):
        hist = sum(k[i] * np.roll(hist, i - 2) for i in range(5))
    return hist


def _peak_hue(h: np.ndarray, C: np.ndarray, allowed: tuple[float, float] | None = None) -> float | None:
    if h.size == 0:
        return None
    hist = _smoothed_hist(h, C)
    centers = (np.arange(hist.size) + 0.5) * (360.0 / hist.size)
    if allowed is not None:
        lo, hi = allowed
        ok = (centers >= lo) & (centers <= hi)
        if not np.any(hist[ok] > 0):
            return None
        hist = np.where(ok, hist, -1)
    peak = centers[int(np.argmax(hist))]
    near = np.abs(cs.hue_diff(peak, h)) <= 40.0
    mean, _, _ = weighted_circular_hue(h[near], C[near])
    return peak if mean is None else mean


def _hue_peaks(h: np.ndarray, C: np.ndarray, min_share: float = 0.05) -> list[float]:
    """全图扫描用：找出所有主要色相族（平滑直方图的局部极大值）。"""
    if h.size == 0:
        return []
    hist = _smoothed_hist(h, C)
    tot = hist.sum()
    centers = (np.arange(hist.size) + 0.5) * (360.0 / hist.size)
    peaks = []
    for i in range(hist.size):
        if hist[i] >= hist[i - 1] and hist[i] > hist[(i + 1) % hist.size]:
            window = hist[np.arange(i - 6, i + 7) % hist.size].sum()
            if window / tot >= min_share:
                peaks.append(float(centers[i]))
    return peaks


def _pairs(a: np.ndarray, b: np.ndarray) -> int:
    """4 邻域内 a、b 相邻的像素对数。"""
    n = np.sum(a[:, 1:] & b[:, :-1]) + np.sum(a[:, :-1] & b[:, 1:])
    n += np.sum(a[1:, :] & b[:-1, :]) + np.sum(a[:-1, :] & b[1:, :])
    return int(n)


def _isolated_fraction(m: np.ndarray) -> float:
    if not m.any():
        return 0.0
    p = np.pad(m, 1)
    neigh = sum(
        p[1 + dy : 1 + dy + m.shape[0], 1 + dx : 1 + dx + m.shape[1]]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy or dx
    )
    return float(np.sum(m & (neigh == 0)) / m.sum())


def _graying(L, C, family: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """family 内、按 L* 分段：色度低于同段干净参考的 GRAY_RATIO 视为发灰。"""
    gray = np.zeros_like(family)
    if family.sum() < 30:
        return gray
    bands = np.floor(L / L_BAND).astype(int)
    for band in np.unique(bands[family]):
        sel = family & (bands == band)
        if sel.sum() < 30:
            continue
        c_ref = np.percentile(C[sel], 85)
        if c_ref < GRAY_REF_MIN:
            continue
        gray |= sel & (C < GRAY_RATIO * c_ref) & ~edge
    return gray


def _components(flags: np.ndarray) -> list[list[tuple[int, int]]]:
    """瓦片网格上的连通区域（先 3×3 膨胀以连接相邻零散瓦片）。"""
    if not flags.any():
        return []
    th, tw = flags.shape
    dil = flags.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            dil |= np.roll(np.roll(flags, dy, 0), dx, 1) & _shift_valid(th, tw, dy, dx)
    seen = np.zeros_like(dil)
    out = []
    for sy in range(th):
        for sx in range(tw):
            if dil[sy, sx] and not seen[sy, sx]:
                comp = []
                q = deque([(sy, sx)])
                seen[sy, sx] = True
                while q:
                    y, x = q.popleft()
                    if flags[y, x]:
                        comp.append((y, x))
                    for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= ny < th and 0 <= nx < tw and dil[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                if comp:
                    out.append(comp)
    return out


def _shift_valid(th: int, tw: int, dy: int, dx: int) -> np.ndarray:
    v = np.ones((th, tw), dtype=bool)
    if dy > 0:
        v[:dy] = False
    elif dy < 0:
        v[dy:] = False
    if dx > 0:
        v[:, :dx] = False
    elif dx < 0:
        v[:, dx:] = False
    return v


def _blobs(mask, kind_value, kind, L, C, h, step, offset, base_name, min_count, min_density, max_blobs=12):
    """base_name 为 None 时（整图扫描）按每一处自己的主色命名。"""
    hh, ww = mask.shape
    bt = int(max(4, round(min(hh, ww) / 16)))
    th, tw = (hh + bt - 1) // bt, (ww + bt - 1) // bt
    m = mask == kind_value
    counts = np.zeros((th, tw))
    sizes = np.zeros((th, tw))
    for ty in range(th):
        for tx in range(tw):
            sl = (slice(ty * bt, (ty + 1) * bt), slice(tx * bt, (tx + 1) * bt))
            counts[ty, tx] = m[sl].sum()
            sizes[ty, tx] = m[sl].size
    flags = (counts >= min_count) & (counts / np.maximum(sizes, 1) >= min_density)
    total = hh * ww
    blobs = []
    for comp in _components(flags):
        ys = [t[0] for t in comp]
        xs = [t[1] for t in comp]
        y0, y1 = min(ys) * bt, min(hh, (max(ys) + 1) * bt)
        x0, x1 = min(xs) * bt, min(ww, (max(xs) + 1) * bt)
        region = np.zeros_like(m)
        for ty, tx in comp:
            region[ty * bt : (ty + 1) * bt, tx * bt : (tx + 1) * bt] = True
        sel = m & region
        n = int(sel.sum())
        share = n / max(1, int(region.sum()))
        c_mean = float(C[sel].mean()) if n else 0.0
        hue, _, _ = weighted_circular_hue(h[sel], C[sel])
        cname = cs.hue_name(hue, c_mean) if (hue is not None and kind == "opposite") else None
        bname = base_name
        if bname is None:
            own = region & ((mask == MASK_NORMAL) | (mask == MASK_GRAYING))
            bh, _, _ = weighted_circular_hue(h[own], C[own]) if own.any() else (None, 0, None)
            bname = cs.hue_name(bh, float(np.median(C[own]))) if bh is not None else "局部主色"
        ev = [f"该处 {share:.1%} 的像素为{('“' + cname + '”') if cname else '发灰'}像素，平均色度 C* {c_mean:.1f}"]
        if kind == "graying":
            ev = [f"该处 {share:.1%} 的像素色度明显低于同明度的同色区域（平均 C* {c_mean:.1f}）"]
        blobs.append(
            Blob(
                kind=kind,
                bbox=(
                    int(offset[0] + x0 * step),
                    int(offset[1] + y0 * step),
                    int((x1 - x0) * step),
                    int((y1 - y0) * step),
                ),
                area_fraction=round(int(region.sum()) / total, 5),
                base_hue_name=bname,
                contaminant_hue_name=cname,
                score=round(share * min(1.0, c_mean / 15.0 if kind == "opposite" else 1.0), 5),
                evidence=tuple(ev),
                contaminant_hue=None if cname is None else round(float(hue), 2),
            )
        )
    blobs.sort(key=lambda b: -(b.score * b.area_fraction))
    return blobs[:max_blobs]


def _graying_colours(L, C, cand, w, ratio_of=None):
    """按颜色判断发灰候选：cand 内、按 L* 分段，色度低于同段干净参考（85 分位）的 GRAY_RATIO。

    w：每种颜色计入参考的像素数（和逐个像素统计一样）。返回 (每种颜色是否发灰, {分段: 参考色度})。
    """
    gray = np.zeros(L.shape, dtype=bool)
    refs: dict[int, float] = {}
    if w[cand].sum() < 30:
        return gray, refs
    bands = np.floor(L / L_BAND).astype(int)
    for band in np.unique(bands[cand & (w > 0)]):
        sel = cand & (bands == band)
        if w[sel].sum() < 30:
            continue
        c_ref = float(cs.weighted_percentile(C[sel], w[sel], 85))
        refs[int(band)] = c_ref
        if c_ref < GRAY_REF_MIN:
            continue
        gray |= sel & (C < GRAY_RATIO * c_ref)
    return gray, refs


def analyze(
    region: np.ndarray,
    mode: str = "region",
    base_hue: float | None = None,
    max_samples: int = MAX_SAMPLES,
    offset: tuple[int, int] = (0, 0),
    valid: np.ndarray | None = None,
) -> ContaminationResult:
    """mode: "region"（用户框选的单一主体）、"skin"（肤色）、"scan"（整图扫描）。

    valid：与 region 同尺寸的布尔数组（例如精准抠出的皮肤或套索选区），
    只有 True 的像素参与判断，其余标记为“忽略”。

    按原图尺寸检查每一个像素：只和颜色有关的判断按“不同的颜色”算一次（PixelTable），
    和位置有关的（边缘、是否孤立、成片还是散点、在哪一块）逐个像素做。
    """
    if region is None or region.ndim != 3 or region.shape[0] == 0 or region.shape[1] == 0:
        raise ValueError("选区为空，无法分析")
    sub, step = grid_downsample(region, max_samples)
    hh, ww = sub.shape[:2]
    vsub = downsample_mask(valid, step, (hh, ww))
    pt = PixelTable(sub)
    Lu, Cu, hu = pt.L, pt.C, pt.h
    wv = pt.count(vsub)  # 每种颜色在有效范围内的像素数
    chrom_u = Cu >= C_MIN
    chromatic = chrom_u[pt.index] & vsub
    n_total = max(1, int(vsub.sum()))
    n_chrom = int(wv[chrom_u].sum())
    chrom_frac = float(n_chrom / n_total)
    Lp = pt.pix(Lu.astype(np.float32))
    gy, gx = np.gradient(Lp)
    edge = np.hypot(gx, gy) > EDGE_GRAD
    edge |= pt.pix(pt.clipped)  # 被裁切的亮部 / 死黑颜色被压淡，不是“发灰”：和边缘一样不参与发灰判断
    del gy, gx

    mask = np.where(chromatic, MASK_NORMAL, MASK_IGNORED).astype(np.uint8)
    mask[~vsub] = MASK_OUTSIDE

    if mode == "scan":
        return _scan(pt, wv, step, offset, chromatic, edge, mask, chrom_frac, vsub)

    opp_angle = SKIN_OPP_ANGLE if mode == "skin" else OPP_ANGLE
    if base_hue is None:
        if chrom_frac < 0.10:
            return _neutral_result(mask, step, chrom_frac)
        allowed = (0.0, 110.0) if mode == "skin" else None
        sel = chrom_u & (wv > 0)
        base_hue = _peak_hue(hu[sel], Cu[sel] * wv[sel], allowed)
        if base_hue is None:
            base_hue = _peak_hue(hu[sel], Cu[sel] * wv[sel])
    base_hue = float(base_hue) % 360.0
    hd_u = np.abs(cs.hue_diff(base_hue, hu))
    family_u = chrom_u & (hd_u <= FAMILY_HALF_WIDTH)
    n_fam = int(wv[family_u].sum())
    base_share = float(n_fam / max(1, n_chrom))
    base_C = wmedian(Cu[family_u], wv[family_u]) if n_fam else 0.0

    # 噪声地板：主色族在“垂直于主色相方向”上的离散度
    th = np.radians(base_hue)
    perp_u = -pt.lab[:, 1] * np.sin(th) + pt.lab[:, 2] * np.cos(th)
    if n_fam >= 10:
        pf, wf = perp_u[family_u], wv[family_u]
        med = wmedian(pf, wf)
        sigma = 1.4826 * wmedian(np.abs(pf - med), wf)
    else:
        sigma = 0.0
    thr = max(OPP_C_MIN, 3.0 * sigma)
    opp_u = chrom_u & (hd_u >= opp_angle) & (Cu >= thr)
    opp = opp_u[pt.index] & vsub
    family = family_u[pt.index] & vsub

    n_opp = int(wv[opp_u].sum())
    opp_share_chrom = n_opp / max(1, n_chrom)
    pairs = _pairs(opp, family)
    interspersion = pairs / n_opp if n_opp else 0.0
    bicolor = opp_share_chrom >= BICOLOR_SHARE or (
        opp_share_chrom >= BICOLOR_SHARE_COHERENT and interspersion < 1.0
    )
    opp_hue, _, _ = weighted_circular_hue(hu[opp_u], Cu[opp_u] * wv[opp_u]) if n_opp else (None, 0, None)
    opp_C = float((Cu[opp_u] * wv[opp_u]).sum() / n_opp) if n_opp else 0.0
    iso = _isolated_fraction(opp)
    noise_like = n_opp > 0 and iso > 0.8 and opp_C < 1.5 * thr and n_opp / n_total < 0.01

    anomalies: dict[str, float] = {}
    if n_opp and not bicolor:
        names, cnt = cs.hue_names(hu[opp_u]), wv[opp_u]
        for name in np.unique(names):
            anomalies[str(name)] = round(float(cnt[names == name].sum()) / n_total, 5)
        mask[opp] = MASK_OPPOSITE

    gray_u, _refs = _graying_colours(Lu, Cu, family_u & ~opp_u, wv)
    gray = gray_u[pt.index] & vsub & ~edge
    mask[gray] = MASK_GRAYING
    n_gray = int(gray.sum())
    gray_frac = float(n_gray / max(1, n_fam))
    depth = 0.0
    if n_gray:
        gw = pt.count(gray)  # 每种颜色有多少个像素被判成发灰
        bands = np.floor(Lu / L_BAND).astype(int)
        vals, wts = [], []
        for band in np.unique(bands[gw > 0]):
            sel = family_u & (bands == band)
            ref = float(cs.weighted_percentile(Cu[sel], wv[sel], 85))
            g = (gw > 0) & (bands == band)
            vals.append(Cu[g] / ref)
            wts.append(gw[g])
        depth = float(1.0 - wmedian(np.concatenate(vals), np.concatenate(wts)))

    rest_u = family_u if bicolor else (chrom_u & ~opp_u)
    rw = wv * rest_u
    _, _, spread = weighted_circular_hue(hu[rw > 0], (Cu * rw)[rw > 0]) if rw.any() else (None, 0, None)

    base_name = "肤色" if mode == "skin" else cs.hue_name(base_hue, base_C)
    blobs: list[Blob] = []
    if (n_opp and not bicolor) or n_gray:
        Cp, hp = pt.pix(Cu.astype(np.float32)), pt.pix(hu.astype(np.float32))
        if n_opp and not bicolor:
            blobs += _blobs(mask, MASK_OPPOSITE, "opposite", Lp, Cp, hp, step, offset, base_name, 2, 0.004)
        if n_gray:
            blobs += _blobs(mask, MASK_GRAYING, "graying", Lp, Cp, hp, step, offset, base_name, 4, 0.4)

    return ContaminationResult(
        base_hue=round(base_hue, 3),
        base_hue_name=base_name,
        base_share=round(base_share, 5),
        opposite_fraction=0.0 if bicolor else round(n_opp / n_total, 6),
        opposite_hue=None if opp_hue is None else round(opp_hue, 2),
        opposite_hue_name=None if opp_hue is None else cs.hue_name(opp_hue, opp_C),
        opposite_chroma=round(opp_C, 3),
        base_chroma=round(base_C, 3),
        interspersion=round(interspersion, 4),
        bicolor=bool(bicolor),
        graying_fraction=round(gray_frac, 5),
        graying_depth=round(depth, 4),
        chromatic_fraction=round(chrom_frac, 5),
        noise_like=bool(noise_like),
        hue_spread=None if spread is None else round(spread, 3),
        opposite_threshold=round(thr, 3),
        anomaly_fractions=tuple(sorted(anomalies.items(), key=lambda t: -t[1])),
        mask=mask,
        mask_scale=float(step),
        blobs=tuple(blobs),
    )


def _neutral_result(mask, step, chrom_frac) -> ContaminationResult:
    return ContaminationResult(
        base_hue=None,
        base_hue_name="中性灰",
        base_share=0.0,
        opposite_fraction=0.0,
        opposite_hue=None,
        opposite_hue_name=None,
        opposite_chroma=0.0,
        base_chroma=0.0,
        interspersion=0.0,
        bicolor=False,
        graying_fraction=0.0,
        graying_depth=0.0,
        chromatic_fraction=round(chrom_frac, 5),
        noise_like=False,
        mask=mask,
        mask_scale=float(step),
    )


def _scan(pt, wv, step, offset, chromatic, edge, mask, chrom_frac, vsub=None) -> ContaminationResult:
    """整图扫描：按瓦片判断“局部主色里的散点对立色”，按色相族判断局部发灰。

    整图没有语义（不知道哪里是一件衣服），因此只在“散点状混杂”时报告对立色；
    成片的不同颜色视为画面中本来就有的不同物体或配色。
    """
    hh, ww = mask.shape
    n_total = max(1, int(vsub.sum())) if vsub is not None else hh * ww  # 透明 / 蒙版外的像素不算进分母
    Lu, Cu, hu = pt.L, pt.C, pt.h
    Lp, Cp, hp = (pt.pix(v.astype(np.float32)) for v in (Lu, Cu, hu))
    ts = int(max(8, round(min(hh, ww) / 12)))
    opp_all = np.zeros((hh, ww), dtype=bool)
    for y0 in range(0, hh, ts):
        for x0 in range(0, ww, ts):
            sl = (slice(y0, y0 + ts), slice(x0, x0 + ts))
            ch = chromatic[sl]
            if ch.sum() < 0.3 * ch.size:
                continue
            tb = _peak_hue(hp[sl][ch], Cp[sl][ch])
            if tb is None:
                continue
            hd = np.abs(cs.hue_diff(tb, hp[sl]))
            fam = ch & (hd <= FAMILY_HALF_WIDTH)
            opp = ch & (hd >= OPP_ANGLE) & (Cp[sl] >= OPP_C_MIN)
            n = int(opp.sum())
            if n < 2 or n / max(1, ch.sum()) >= 0.2:
                continue
            if _pairs(opp, fam) / n < 2.0:  # 成片 -> 物体边界/小物件，不算混杂
                continue
            opp_all[sl] |= opp

    chrom_u = Cu >= C_MIN
    sel = chrom_u & (wv > 0)
    peaks = _hue_peaks(hu[sel], Cu[sel] * wv[sel])
    gray = np.zeros((hh, ww), dtype=bool)
    if peaks:
        dist = np.stack([np.abs(cs.hue_diff(p, hu)) for p in peaks])
        nearest = dist.argmin(axis=0)
        w_ok = pt.count(vsub & ~opp_all) if opp_all.any() else wv  # 参考色度不算散点对立色
        for i, _p in enumerate(peaks):
            fam_u = chrom_u & (nearest == i) & (dist[i] <= FAMILY_HALF_WIDTH)
            g_u, _ = _graying_colours(Lu, Cu, fam_u, w_ok)
            gray |= g_u[pt.index]
        gray &= chromatic & ~opp_all & ~edge

    mask[opp_all] = MASK_OPPOSITE
    mask[gray] = MASK_GRAYING
    n_opp = int(opp_all.sum())
    ow = pt.count(opp_all) if n_opp else None
    opp_hue, _, _ = weighted_circular_hue(hu, Cu * ow) if n_opp else (None, 0, None)
    wc = wv * chrom_u
    base_hue, _, _ = weighted_circular_hue(hu, Cu * wc) if wc.any() else (None, 0, None)
    blobs = []
    if n_opp:
        blobs += _blobs(mask, MASK_OPPOSITE, "opposite", Lp, Cp, hp, step, offset, None, 2, 0.004)
    if gray.any():
        blobs += _blobs(mask, MASK_GRAYING, "graying", Lp, Cp, hp, step, offset, None, 4, 0.4)
    n_chrom = int(wc.sum())
    return ContaminationResult(
        base_hue=None if base_hue is None else round(base_hue, 3),
        base_hue_name="多种颜色（整图）",
        base_share=0.0,
        opposite_fraction=round(n_opp / n_total, 6),
        opposite_hue=None if opp_hue is None else round(opp_hue, 2),
        opposite_hue_name=None if opp_hue is None else cs.hue_name(opp_hue),
        opposite_chroma=round(float((Cu * ow).sum() / n_opp), 3) if n_opp else 0.0,
        base_chroma=round(wmedian(Cu[wc > 0], wc[wc > 0]), 3) if n_chrom else 0.0,
        interspersion=0.0,
        bicolor=False,
        graying_fraction=round(float(gray.sum() / max(1, n_chrom)), 5),
        graying_depth=0.0,
        chromatic_fraction=round(chrom_frac, 5),
        noise_like=False,
        opposite_threshold=OPP_C_MIN,
        mask=mask,
        mask_scale=float(step),
        blobs=tuple(blobs),
    )
