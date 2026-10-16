"""确定性 k-means（纯 NumPy，不依赖 scikit-learn）与主色提取。"""

from __future__ import annotations

import numpy as np

from colordoctor.core import color_spaces as cs


def kmeans(points: np.ndarray, k: int, iters: int = 40, seed: int = 0, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """weights：每个点代表多少个像素（按不同颜色统计时传像素数）：和把每个点重复这么多次再聚类一样
    （只有 k-means++ 抽初始中心时按权重抽）。"""
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0, pts.shape[1] if pts.ndim == 2 else 3)), np.zeros(0, dtype=int)
    k = max(1, min(int(k), n))
    w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    # k-means++ 初始化（固定随机种子 -> 结果可复现）
    if w is None:
        centers = [pts[rng.integers(n)]]
    else:
        cw = np.cumsum(w)
        centers = [pts[min(int(np.searchsorted(cw, rng.random() * cw[-1], side="right")), n - 1)]]
    d2 = np.sum((pts - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        p = d2 if w is None else d2 * w
        total = p.sum()
        if total <= 0:
            break
        idx = int(np.searchsorted(np.cumsum(p), rng.random() * total))
        centers.append(pts[min(idx, n - 1)])
        d2 = np.minimum(d2, np.sum((pts - centers[-1]) ** 2, axis=1))
    C = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        new_labels = nearest(pts, C)
        cnt = np.bincount(new_labels, weights=w, minlength=len(C))
        wp = pts if w is None else pts * w[:, None]
        sums = np.stack([np.bincount(new_labels, weights=wp[:, d], minlength=len(C)) for d in range(pts.shape[1])], axis=1)
        newC = np.where(cnt[:, None] > 0, sums / np.where(cnt > 0, cnt, 1)[:, None], C)  # 权重可以是小数
        if np.array_equal(new_labels, labels) and np.allclose(newC, C):
            break
        labels, C = new_labels, newC
    return C, labels


def nearest(pts: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """每个点最近的中心（|p|² - 2p·c + |c|²，用矩阵乘法，比逐点相减快很多）。"""
    d = (centers ** 2).sum(axis=1)[None, :] - 2.0 * (pts @ centers.T)
    return d.argmin(axis=1)


def dominant_clusters(
    lab: np.ndarray, max_k: int = 6, merge_de: float = 4.0, max_points: int = 20_000, weights=None
) -> list[tuple[np.ndarray, float]]:
    """返回 [(Lab 中心, 占比)]，按占比降序；ΔE00 < merge_de 的簇会合并。

    weights：每个点代表多少个像素（按不同颜色统计时传像素数），占比按像素算，找中心时也按像素数
    （颜色不多时按权重聚类，多时按像素数抽样），和对每个像素聚类是同一件事。
    """
    pts = np.asarray(lab, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n == 0:
        return []
    w_all = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    fit, fit_w = pts, None
    if weights is not None and n > max_points:
        # 按像素数抽样找中心（固定随机种子，结果可复现）：像素多的颜色被抽到的机会大
        rng = np.random.default_rng(0)
        fit = pts[rng.choice(n, max_points, replace=True, p=w_all / w_all.sum())]
    elif weights is not None:
        fit_w = w_all  # 颜色不多：全部颜色按像素数加权聚类（不按“一种颜色一票”，否则零星的颜色被放大）
    elif n > max_points:
        step = int(np.ceil(n / max_points))
        fit = pts[step // 2 :: step]
    centers, _ = kmeans(fit, max_k, weights=fit_w)
    # 用全部采样点重新分配，得到准确占比
    labels = np.empty(n, dtype=int)
    for s in range(0, n, 200_000):
        labels[s : s + 200_000] = nearest(pts[s : s + 200_000], centers)
    counts = np.bincount(labels, weights=w_all, minlength=len(centers)).astype(float)
    groups = [[i] for i in range(len(centers)) if counts[i] > 0]

    def center_of(g):
        w = counts[g]
        return (centers[g] * w[:, None]).sum(axis=0) / w.sum()

    merged = True
    while merged and len(groups) > 1:
        merged = False
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                de = float(cs.delta_e_2000(center_of(groups[i]), center_of(groups[j])))
                if de < merge_de and (best is None or de < best[0]):
                    best = (de, i, j)
        if best is not None:
            _, i, j = best
            groups[i] = groups[i] + groups[j]
            del groups[j]
            merged = True
    out = [(center_of(g), float(counts[g].sum() / w_all.sum())) for g in groups]
    out.sort(key=lambda t: -t[1])
    return out
