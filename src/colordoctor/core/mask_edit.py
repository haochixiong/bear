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


def _dilate(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:] |= m[:-1]
    out[:-1] |= m[1:]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


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
    grow = np.zeros_like(similar)
    grow[sy, sx] = True
    while True:
        nxt = _dilate(grow) & similar
        if nxt.sum() == grow.sum():
            break
        grow = nxt
    if step > 1:
        grow = np.repeat(np.repeat(grow, step, axis=0), step, axis=1)[:H, :W]
    return grow


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
