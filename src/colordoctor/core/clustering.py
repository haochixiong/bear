"""确定性 k-means（纯 NumPy，不依赖 scikit-learn）与主色提取。"""

from __future__ import annotations

import numpy as np

from colordoctor.core import color_spaces as cs


def kmeans(points: np.ndarray, k: int, iters: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0, pts.shape[1] if pts.ndim == 2 else 3)), np.zeros(0, dtype=int)
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    # k-means++ 初始化（固定随机种子 -> 结果可复现）
    centers = [pts[rng.integers(n)]]
    d2 = np.sum((pts - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        total = d2.sum()
        if total <= 0:
            break
        idx = int(np.searchsorted(np.cumsum(d2), rng.random() * total))
        centers.append(pts[min(idx, n - 1)])
        d2 = np.minimum(d2, np.sum((pts - centers[-1]) ** 2, axis=1))
    C = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = ((pts[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        newC = np.array(
            [pts[new_labels == j].mean(axis=0) if np.any(new_labels == j) else C[j] for j in range(len(C))]
        )
        if np.array_equal(new_labels, labels) and np.allclose(newC, C):
            break
        labels, C = new_labels, newC
    return C, labels


def dominant_clusters(
    lab: np.ndarray, max_k: int = 6, merge_de: float = 4.0, max_points: int = 20_000
) -> list[tuple[np.ndarray, float]]:
    """返回 [(Lab 中心, 占比)]，按占比降序；ΔE00 < merge_de 的簇会合并。"""
    pts = np.asarray(lab, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n == 0:
        return []
    fit = pts
    if n > max_points:
        step = int(np.ceil(n / max_points))
        fit = pts[step // 2 :: step]
    centers, _ = kmeans(fit, max_k)
    # 用全部采样点重新分配，得到准确占比
    labels = np.empty(n, dtype=int)
    for s in range(0, n, 200_000):
        chunk = pts[s : s + 200_000]
        labels[s : s + 200_000] = ((chunk[:, None, :] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
    counts = np.bincount(labels, minlength=len(centers)).astype(float)
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
    out = [(center_of(g), float(counts[g].sum() / n)) for g in groups]
    out.sort(key=lambda t: -t[1])
    return out
