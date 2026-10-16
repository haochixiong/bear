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
from colordoctor.core.pixel_table import par_ranges, par_rows, table, wmedian
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


HIST_BINS = 72
# _smoothed_hist 的平滑（[1 4 6 4 1]/16 循环卷积两次）写成一个矩阵，每块只做一次矩阵乘法
_SMOOTH = np.linalg.matrix_power(
    sum(w / 16.0 * np.roll(np.eye(HIST_BINS), d, axis=0) for d, w in zip(range(-2, 3), (1, 4, 6, 4, 1), strict=True)), 2)


def _tile_peak(u, w_u, h32, c32, hb, ccos, csin) -> float:
    """和 _peak_hue(这一块有彩色像素的色相, 色度) 一样（平滑色相直方图的峰，再在峰两侧 40° 内按色度加权求平均色相），
    按颜色算：u 是这一块里出现的颜色编号，w_u 是每种颜色的像素数；直方图格、色度 × cos / sin 事先按颜色算好。"""
    cw = c32[u] * w_u
    hist = _SMOOTH @ np.bincount(hb[u], weights=cw, minlength=HIST_BINS)
    peak = (int(np.argmax(hist)) + 0.5) * (360.0 / HIST_BINS)
    near = np.abs((h32[u] - np.float32(peak) + 180) % 360 - 180) <= 40.0
    c = float((ccos[u][near] * w_u[near]).sum(dtype=np.float64))
    s = float((csin[u][near] * w_u[near]).sum(dtype=np.float64))
    tot = float(cw[near].sum(dtype=np.float64))
    if tot <= 0 or np.hypot(c, s) < 1e-12 * tot:
        return peak
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


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


SPARSE_FRACTION = 0.002  # 比这还少时按坐标只看这些像素的邻居，否则按行分块整幅比较（多线程）


def _neighbour(m: np.ndarray, ys: np.ndarray, xs: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """m 在 (ys+dy, xs+dx) 处的值；出界算 False。"""
    y, x = ys + dy, xs + dx
    ok = (y >= 0) & (y < m.shape[0]) & (x >= 0) & (x < m.shape[1])
    out = np.zeros(ys.shape, dtype=bool)
    out[ok] = m[y[ok], x[ok]]
    return out


def _padded_rows(m: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """m[y0:y1] 四周各补一圈：上下用相邻的行（图外为 False），左右为 False。"""
    hh, ww = m.shape
    ext = np.zeros((y1 - y0 + 2, ww + 2), dtype=bool)
    ext[1:-1, 1:-1] = m[y0:y1]
    if y0 > 0:
        ext[0, 1:-1] = m[y0 - 1]
    if y1 < hh:
        ext[-1, 1:-1] = m[y1]
    return ext


def _pairs(a: np.ndarray, b: np.ndarray) -> int:
    """4 邻域内 a、b 相邻的像素对数（= 每个 a 像素上下左右有几个 b 像素，加起来）。"""
    n_a = int(np.count_nonzero(a))
    if n_a == 0:
        return 0
    if n_a < SPARSE_FRACTION * a.size:  # a 很少（一般如此）：只看 a 像素的邻居
        ys, xs = np.nonzero(a)
        return sum(int(np.count_nonzero(_neighbour(b, ys, xs, dy, dx))) for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)))
    hh, ww = a.shape

    def part(y0, y1):
        e, c = _padded_rows(b, y0, y1), a[y0:y1]
        return sum(int(np.count_nonzero(c & e[1 + dy : 1 + dy + (y1 - y0), 1 + dx : 1 + dx + ww]))
                   for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)))

    return int(sum(par_rows(part, hh, ww)))


def _isolated_fraction(m: np.ndarray) -> float:
    """m 里 8 邻域内没有别的 m 像素的比例。"""
    n_m = int(np.count_nonzero(m))
    if n_m == 0:
        return 0.0
    if n_m < SPARSE_FRACTION * m.size:
        ys, xs = np.nonzero(m)
        has = np.zeros(n_m, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    has |= _neighbour(m, ys, xs, dy, dx)
        return float(np.count_nonzero(~has) / n_m)
    hh, ww = m.shape

    def part(y0, y1):
        e, n = _padded_rows(m, y0, y1), y1 - y0
        has = np.zeros((n, ww), dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    has |= e[1 + dy : 1 + dy + n, 1 + dx : 1 + dx + ww]
        return int(np.count_nonzero(m[y0:y1] & ~has))

    return float(sum(par_rows(part, hh, ww)) / n_m)


def _base_mask(chromatic, vsub, all_valid) -> np.ndarray:
    # 有彩色 = MASK_NORMAL(0)，其余 = MASK_IGNORED(1)：直接把“不是有彩色”当 0/1 字节用（不经过 64 位的中间数组）
    mask = np.logical_not(chromatic).view(np.uint8)
    if not all_valid:
        mask[~vsub] = MASK_OUTSIDE
    return mask


def _drop_edges(pt, cand: np.ndarray) -> None:
    """从发灰候选 cand 里去掉边缘（明度梯度大）和被裁切的颜色（就地修改）。

    只在候选像素上算（和对整幅图 np.gradient 再取这些位置完全一样）：候选少时按坐标取上下左右的明度，
    多时只算含有候选的那几段行。被裁切的亮部 / 死黑颜色被压淡，不是“发灰”：和边缘一样不参与发灰判断。
    """
    hh, ww = cand.shape
    n = int(np.count_nonzero(cand))
    if n == 0:
        return
    L32 = pt.L.astype(np.float32)
    lim = np.float32(EDGE_GRAD) ** 2
    if n < 0.05 * cand.size:
        ys, xs = np.nonzero(cand)
        drop = pt.clipped[pt.index[ys, xs]] if pt.clipped.any() else np.zeros(n, dtype=bool)
        if hh >= 2 and ww >= 2:
            drop |= _grad2_at(pt.index, L32, ys, xs, 0) + _grad2_at(pt.index, L32, ys, xs, 1) > lim
        cand[ys[drop], xs[drop]] = False
        return
    clipped = pt.clipped if pt.clipped.any() else None

    def part(y0, y1):
        c = cand[y0:y1]
        if not c.any():
            return
        if clipped is not None:
            c &= ~clipped[pt.index[y0:y1]]
        if hh >= 2 and ww >= 2:
            a, b = max(0, y0 - 1), min(hh, y1 + 1)
            g = np.gradient(L32[pt.index[a:b]])
            gy, gx = g[0][y0 - a : y0 - a + (y1 - y0)], g[1][y0 - a : y0 - a + (y1 - y0)]
            c &= ~(gx * gx + gy * gy > lim)

    par_rows(part, hh, ww)


def _grad2_at(index, L32, ys, xs, axis) -> np.ndarray:
    """(ys, xs) 处沿 axis 的 np.gradient 分量的平方（内部中心差分，两端单侧差分；float32，与整幅计算一致）。"""
    size = index.shape[axis]
    pos = ys if axis == 0 else xs
    lo, hi = np.maximum(pos - 1, 0), np.minimum(pos + 1, size - 1)

    def val(p):
        return L32[index[p, xs]] if axis == 0 else L32[index[ys, p]]

    d = val(hi) - val(lo)
    inner = (pos > 0) & (pos < size - 1)
    g = np.where(inner, d / np.float32(2.0), d).astype(np.float32)
    return g * g


def _edges(Lp: np.ndarray) -> np.ndarray:
    """明度梯度大于 EDGE_GRAD 的像素（和 np.gradient 的中心差分一样，按行分块多线程，不建整幅的梯度数组）。"""
    hh, ww = Lp.shape
    out = np.zeros((hh, ww), dtype=bool)
    if hh < 2 or ww < 2:
        return out
    lim = np.float32(EDGE_GRAD) ** 2

    def part(y0, y1):
        a, b = max(0, y0 - 1), min(hh, y1 + 1)
        g = np.gradient(Lp[a:b])
        gy, gx = g[0][y0 - a : y0 - a + (y1 - y0)], g[1][y0 - a : y0 - a + (y1 - y0)]
        out[y0:y1] = gx * gx + gy * gy > lim

    par_rows(part, hh, ww)
    return out


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


def _blobs(mask, kind_value, kind, pt, step, offset, base_name, min_count, min_density, max_blobs=12):
    """base_name 为 None 时（整图扫描）按每一处自己的主色命名。色度、色相按像素的颜色编号查 pt。"""
    def ch_of(box, sel):
        ci = pt.index[box][sel]
        return pt.C[ci], pt.h[ci]

    hh, ww = mask.shape
    bt = int(max(4, round(min(hh, ww) / 16)))
    th, tw = (hh + bt - 1) // bt, (ww + bt - 1) // bt
    m = mask == kind_value
    rows, cols = np.arange(0, hh, bt), np.arange(0, ww, bt)
    counts = np.zeros((th, tw))

    def count_rows(t0, t1):  # 每一行小块：先按列数，再把列合并成块（不列出几千万个坐标）
        for ty in range(t0, t1):
            colsum = np.count_nonzero(m[ty * bt : (ty + 1) * bt], axis=0)
            counts[ty] = np.add.reduceat(colsum, cols)

    par_ranges(count_rows, th, 1)
    sizes = np.outer(np.minimum(bt, hh - rows), np.minimum(bt, ww - cols)).astype(float)
    flags = (counts >= min_count) & (counts / np.maximum(sizes, 1) >= min_density)
    total = hh * ww
    blobs = []
    for comp in _components(flags):
        ys = [t[0] for t in comp]
        xs = [t[1] for t in comp]
        y0, y1 = min(ys) * bt, min(hh, (max(ys) + 1) * bt)
        x0, x1 = min(xs) * bt, min(ww, (max(xs) + 1) * bt)
        # 只在这一处的外框里算（不建整幅的图）
        box = (slice(y0, y1), slice(x0, x1))
        region = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for ty, tx in comp:
            region[ty * bt - y0 : (ty + 1) * bt - y0, tx * bt - x0 : (tx + 1) * bt - x0] = True
        sel = m[box] & region
        n = int(sel.sum())
        share = n / max(1, int(region.sum()))
        c_sel, h_sel = ch_of(box, sel)
        c_mean = float(c_sel.mean()) if n else 0.0
        hue, _, _ = weighted_circular_hue(h_sel, c_sel)
        cname = cs.hue_name(hue, c_mean) if (hue is not None and kind == "opposite") else None
        bname = base_name
        if bname is None:
            mb = mask[box]
            own = region & ((mb == MASK_NORMAL) | (mb == MASK_GRAYING))
            c_own, h_own = ch_of(box, own)
            bh, _, _ = weighted_circular_hue(h_own, c_own) if c_own.size else (None, 0, None)
            bname = cs.hue_name(bh, float(np.median(c_own))) if bh is not None else "局部主色"
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
    pt = table(sub)  # 一次分析内同一张图只编号一次
    Lu, Cu, hu = pt.L, pt.C, pt.h
    wv = pt.count(vsub)  # 每种颜色在有效范围内的像素数
    chrom_u = Cu >= C_MIN
    n_total = max(1, int(np.count_nonzero(vsub)))
    n_chrom = int(wv[chrom_u].sum())
    chrom_frac = float(n_chrom / n_total)
    all_valid = n_total == hh * ww

    if mode == "scan":
        chromatic = pt.pix(chrom_u)
        if not all_valid:
            chromatic &= vsub
        mask = _base_mask(chromatic, vsub, all_valid)
        return _scan(pt, wv, step, offset, chromatic, mask, chrom_frac, vsub)

    opp_angle = SKIN_OPP_ANGLE if mode == "skin" else OPP_ANGLE
    if base_hue is None:
        if chrom_frac < 0.10:
            chromatic = pt.pix(chrom_u)
            if not all_valid:
                chromatic &= vsub
            return _neutral_result(_base_mask(chromatic, vsub, all_valid), step, chrom_frac)
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
    n_opp = int(wv[opp_u].sum())
    gray_u, _refs = _graying_colours(Lu, Cu, family_u & ~opp_u, wv)
    # 每个像素属于哪几类（有彩色 / 对立色 / 主色族 / 发灰候选）：按颜色算好，一次查回（1 个字节），
    # 不再一类建一张整幅的图
    codes = pt.pix((chrom_u * 1 | opp_u * 2 | family_u * 4 | gray_u * 8).astype(np.uint8))
    if not all_valid:
        codes[~vsub] = 0
    if n_opp:
        opp = (codes & 2).astype(bool)
        family = (codes & 4).astype(bool)
    else:
        opp = family = None
    opp_share_chrom = n_opp / max(1, n_chrom)
    pairs = _pairs(opp, family) if n_opp else 0
    interspersion = pairs / n_opp if n_opp else 0.0
    bicolor = opp_share_chrom >= BICOLOR_SHARE or (
        opp_share_chrom >= BICOLOR_SHARE_COHERENT and interspersion < 1.0
    )
    opp_hue, _, _ = weighted_circular_hue(hu[opp_u], Cu[opp_u] * wv[opp_u]) if n_opp else (None, 0, None)
    opp_C = float((Cu[opp_u] * wv[opp_u]).sum() / n_opp) if n_opp else 0.0
    iso = _isolated_fraction(opp) if n_opp else 0.0
    noise_like = n_opp > 0 and iso > 0.8 and opp_C < 1.5 * thr and n_opp / n_total < 0.01

    anomalies: dict[str, float] = {}
    lut = np.full(16, MASK_IGNORED, dtype=np.uint8)
    lut[1::2] = MASK_NORMAL  # 有彩色
    if n_opp and not bicolor:
        names, cnt = cs.hue_names(hu[opp_u]), wv[opp_u]
        for name in np.unique(names):
            anomalies[str(name)] = round(float(cnt[names == name].sum()) / n_total, 5)
        lut[[c for c in range(16) if c & 2]] = MASK_OPPOSITE
    mask = lut[codes]
    if not all_valid:
        mask[~vsub] = MASK_OUTSIDE

    if (gray_u & (wv > 0)).any():
        gray = (codes & 8).astype(bool)
        _drop_edges(pt, gray)  # 边缘只在发灰候选像素上算
        mask[gray] = MASK_GRAYING
        n_gray = int(np.count_nonzero(gray))
    else:
        gray, n_gray = None, 0
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
        if n_opp and not bicolor:
            blobs += _blobs(mask, MASK_OPPOSITE, "opposite", pt, step, offset, base_name, 2, 0.004)
        if n_gray:
            blobs += _blobs(mask, MASK_GRAYING, "graying", pt, step, offset, base_name, 4, 0.4)

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


def _scan(pt, wv, step, offset, chromatic, mask, chrom_frac, vsub=None) -> ContaminationResult:
    """整图扫描：按瓦片判断“局部主色里的散点对立色”，按色相族判断局部发灰。

    整图没有语义（不知道哪里是一件衣服），因此只在“散点状混杂”时报告对立色；
    成片的不同颜色视为画面中本来就有的不同物体或配色。
    """
    hh, ww = mask.shape
    n_total = max(1, int(vsub.sum())) if vsub is not None else hh * ww  # 透明 / 蒙版外的像素不算进分母
    Lu, Cu, hu = pt.L, pt.C, pt.h
    # 按颜色的小表（几十万种颜色，放得进缓存）：每块按颜色编号查，不建整幅的逐像素数组
    h32, c32 = hu.astype(np.float32), Cu.astype(np.float32)
    hr = np.radians(hu)
    hb = ((hu / (360.0 / HIST_BINS)).astype(int) % HIST_BINS).astype(np.uint8)
    ccos, csin = (Cu * np.cos(hr)).astype(np.float32), (Cu * np.sin(hr)).astype(np.float32)
    chrom_u = Cu >= C_MIN
    ts = int(max(8, round(min(hh, ww) / 12)))
    opp_all = np.zeros((hh, ww), dtype=bool)
    tiles = [(y0, x0) for y0 in range(0, hh, ts) for x0 in range(0, ww, ts)]

    all_valid = vsub is None or bool(vsub.all())

    def scan_tiles(a, b):  # 各块互不重叠：多个线程同时写 opp_all 的不同块
        for y0, x0 in tiles[a:b]:
            sl = (slice(y0, y0 + ts), slice(x0, x0 + ts))
            idx_t = pt.index[sl]
            # 先数这一块里每种颜色有几个有彩色像素，主色相、对立色有多少都按颜色算（和逐像素算一样）；
            # 大多数块到这里就知道没有散点，不用逐像素建图
            if all_valid:
                cnt = np.bincount(idx_t.ravel(), minlength=pt.n)
                cnt[~chrom_u] = 0
            else:
                cnt = np.bincount(idx_t[chromatic[sl]], minlength=pt.n)
            u = np.flatnonzero(cnt)
            w_u = cnt[u]
            n_ch = int(w_u.sum())
            if n_ch < 0.3 * idx_t.size:
                continue
            tb = _tile_peak(u, w_u, h32, c32, hb, ccos, csin)
            hd = np.abs((h32[u] - np.float32(tb) + 180) % 360 - 180)
            opp_c = (hd >= OPP_ANGLE) & (c32[u] >= OPP_C_MIN)
            n = int(w_u[opp_c].sum())
            if n < 2 or n / max(1, n_ch) >= 0.2:
                continue
            ch = chromatic[sl]
            lut = np.zeros(pt.n, dtype=np.uint8)
            lut[u[opp_c]] = 1
            lut[u[hd <= FAMILY_HALF_WIDTH]] |= 2
            code = lut[idx_t]
            opp = ch & (code & 1).astype(bool)
            fam = ch & (code & 2).astype(bool)
            if _pairs(opp, fam) / n < 2.0:  # 成片 -> 物体边界/小物件，不算混杂
                continue
            opp_all[sl] |= opp

    par_ranges(scan_tiles, len(tiles), 1)

    chrom_u = Cu >= C_MIN
    sel = chrom_u & (wv > 0)
    peaks = _hue_peaks(hu[sel], Cu[sel] * wv[sel])
    gray_any = np.zeros(pt.n, dtype=bool)
    if peaks:
        dist = np.stack([np.abs(cs.hue_diff(p, hu)) for p in peaks])
        nearest = dist.argmin(axis=0)
        w_ok = pt.count(vsub & ~opp_all) if opp_all.any() else wv  # 参考色度不算散点对立色
        for i, _p in enumerate(peaks):
            fam_u = chrom_u & (nearest == i) & (dist[i] <= FAMILY_HALF_WIDTH)
            g_u, _ = _graying_colours(Lu, Cu, fam_u, w_ok)
            gray_any |= g_u  # 每种颜色只属于一个色相族：先按颜色合并，再查回像素一次
    if (gray_any & (wv > 0)).any():
        gray = pt.pix(gray_any)
        gray &= chromatic
        gray &= ~opp_all
        _drop_edges(pt, gray)
    else:
        gray = np.zeros((hh, ww), dtype=bool)

    mask[opp_all] = MASK_OPPOSITE
    mask[gray] = MASK_GRAYING
    n_opp = int(opp_all.sum())
    ow = pt.count(opp_all) if n_opp else None
    opp_hue, _, _ = weighted_circular_hue(hu, Cu * ow) if n_opp else (None, 0, None)
    wc = wv * chrom_u
    base_hue, _, _ = weighted_circular_hue(hu, Cu * wc) if wc.any() else (None, 0, None)
    blobs = []
    if n_opp:
        blobs += _blobs(mask, MASK_OPPOSITE, "opposite", pt, step, offset, None, 2, 0.004)
    if gray.any():
        blobs += _blobs(mask, MASK_GRAYING, "graying", pt, step, offset, None, 4, 0.4)
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
