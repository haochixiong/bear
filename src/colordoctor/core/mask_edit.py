"""皮肤蒙版的手动修正：画笔添加/擦除、魔棒（相连同色区域）添加/删除。纯 NumPy，可测试。"""

from __future__ import annotations

import numpy as np

from colordoctor.core import color_spaces as cs

MAX_WAND_PIXELS = 1_000_000


def paint(mask: np.ndarray, x: float, y: float, radius: float, value: bool) -> np.ndarray:
    """在 (x, y) 画一个半径 radius 的圆（原地修改并返回）。"""
    h, w = mask.shape
    r = max(0.5, float(radius))
    x0, x1 = max(0, int(np.floor(x - r))), min(w, int(np.ceil(x + r)) + 1)
    y0, y1 = max(0, int(np.floor(y - r))), min(h, int(np.ceil(y + r)) + 1)
    if x0 >= x1 or y0 >= y1:
        return mask
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    mask[y0:y1, x0:x1][disk] = value
    return mask


def stroke(mask: np.ndarray, points, radius: float, value: bool) -> np.ndarray:
    """沿折线连续涂抹（点间按半径的一半插值，避免断笔）。"""
    pts = [(float(px), float(py)) for px, py in points]
    if not pts:
        return mask
    paint(mask, *pts[0], radius, value)
    step = max(1.0, radius / 2)
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        n = int(np.ceil(np.hypot(bx - ax, by - ay) / step))
        for t in np.linspace(0, 1, n + 1)[1:]:
            paint(mask, ax + (bx - ax) * t, ay + (by - ay) * t, radius, value)
    return mask


def magic_wand(region: np.ndarray, x: int, y: int, tolerance: float = 8.0) -> np.ndarray:
    """返回与 (x, y) 相连、颜色相近（Lab 距离 < tolerance）的区域蒙版（与 region 同尺寸）。"""
    H, W = region.shape[:2]
    if not (0 <= x < W and 0 <= y < H):
        return np.zeros((H, W), dtype=bool)
    step = int(np.ceil(np.sqrt(H * W / MAX_WAND_PIXELS))) if H * W > MAX_WAND_PIXELS else 1
    small = region[::step, ::step, :3]
    lab = cs.srgb8_to_lab(small)
    sy, sx = min(y // step, small.shape[0] - 1), min(x // step, small.shape[1] - 1)
    seed_lab = lab[sy, sx]
    similar = np.linalg.norm(lab - seed_lab, axis=-1) < tolerance  # 种子像素自身必然在内
    grow = connected_component(similar, sy, sx)
    if step > 1:
        grow = np.repeat(np.repeat(grow, step, axis=0), step, axis=1)[:H, :W]
    return grow


def connected_component(mask: np.ndarray, y: int, x: int) -> np.ndarray:
    """mask 里与 (y, x) 四连通的部分。

    按行把 True 切成“段”，相邻两行有重叠的段相连，再在段上做标签传播。
    段的数量远少于像素，大块平坦区域或细长蜿蜒的区域都不会逐像素迭代。
    """
    h, w = mask.shape
    if not mask[y, x]:
        return np.zeros((h, w), dtype=bool)
    pad = np.zeros((h, w + 2), np.int8)
    pad[:, 1:-1] = mask
    d = np.diff(pad, axis=1)
    rows, starts = np.nonzero(d == 1)  # 段的起点（含）
    _, ends = np.nonzero(d == -1)  # 段的终点（不含）；按行优先顺序与起点一一对应
    n = len(rows)
    W2 = w + 2
    start_key = rows * W2 + starts
    end_key = rows * W2 + ends
    # 段 a（第 r 行）与第 r+1 行的段 b 相连：b.start < a.end 且 b.end > a.start
    nxt = (rows + 1) * W2
    lo = np.searchsorted(end_key, nxt + starts, side="right")
    hi = np.searchsorted(start_key, nxt + ends, side="left")
    cnt = np.maximum(hi - lo, 0)
    a = np.repeat(np.arange(n), cnt)
    b = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    labels = np.arange(n)
    while True:
        m = np.minimum(labels[a], labels[b])
        new = labels.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]  # 指针跳跃，加快收敛
        if np.array_equal(new, labels):
            break
        labels = new
    seed = int(np.nonzero((rows == y) & (starts <= x) & (ends > x))[0][0])
    keep = labels == labels[seed]
    acc = np.zeros((h, w + 1), np.int32)
    np.add.at(acc, (rows[keep], starts[keep]), 1)
    np.add.at(acc, (rows[keep], ends[keep]), -1)
    return np.cumsum(acc, axis=1)[:, :w] > 0


class MaskHistory:
    """简单的撤销栈。"""

    def __init__(self, mask: np.ndarray, limit: int = 30):
        self.mask = mask.copy()
        self._undo: list[np.ndarray] = []
        self.limit = limit

    def push(self) -> None:
        self._undo.append(self.mask.copy())
        self._undo = self._undo[-self.limit :]

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.mask = self._undo.pop()
        return True
