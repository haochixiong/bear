"""原图尺寸的像素表：每个像素记下它是第几种颜色，每种颜色只换算一次。

一张照片有几千万个像素，但不同的颜色一般只有十几万到一两百万种。只和颜色有关的判断（色相、色度、
是不是对立色、是不是发灰候选……）按颜色算一次，再按编号查回每个像素；只和颜色有关的统计
（占比、平均、中位数、分位数）按“颜色 × 这种颜色的像素数”加权。结果和逐个像素算完全一样，
不降采样、每个像素都算进去，只是快很多。和位置有关的判断（边缘、是否孤立、在哪一块）仍然逐个像素做。

速度：
- 编号、查回、计数按行分块多线程做（numpy 在这些操作里会释放 GIL）；
- 一次分析里同一张图只编号一次（`scope()` 内按数组缓存），脏色、统计、分段、肤色、影调共用；
- 互不依赖的几项分析用 `together()` 同时做。
"""

from __future__ import annotations

import contextvars
import os
import threading
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import cached_property

import numpy as np

from colordoctor.core import color_spaces as cs

THREADS = max(1, min(8, os.cpu_count() or 1))
CHUNK = 1 << 20  # 每块至少这么多像素，再小线程开销不划算
_pool: ThreadPoolExecutor | None = None
_tasks: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(THREADS, thread_name_prefix="cd-px")
        return _pool


# ---------------------------------------------------------------- 取消
class Cancelled(Exception):
    """这次分析已经被新的分析取代（例如原图精查还没算完，用户又分析了一次）。"""


_cancel: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar("colordoctor_cancel", default=None)


@contextmanager
def cancellable(event: threading.Event):
    """在这段代码里，event 一被设置，分析就在下一个检查点停下（抛 Cancelled），不再白占 CPU。"""
    token = _cancel.set(event)
    try:
        yield
    finally:
        _cancel.reset(token)


def check_cancel() -> None:
    ev = _cancel.get()
    if ev is not None and ev.is_set():
        raise Cancelled("这次分析已被新的分析取代")


def par_ranges(fn, n: int, min_chunk: int = CHUNK) -> list:
    """把 [0, n) 分成几段并行执行 fn(start, stop)，按顺序返回结果。

    fn 内只做释放 GIL 的 numpy 操作。已经在这个线程池里时直接顺序执行（不嵌套等待）。
    每次调用前后检查是否已被取消（一次分析里有几十次，取消后很快停下）。
    """
    check_cancel()
    k = min(THREADS * 2, max(1, n // max(1, min_chunk)))
    if k <= 1 or threading.current_thread().name.startswith("cd-px"):
        return [fn(0, n)]
    edges = np.linspace(0, n, k + 1).astype(int)
    futs = [_executor().submit(fn, int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    out = [f.result() for f in futs]
    check_cancel()
    return out


def par_rows(fn, h: int, w: int) -> list:
    """按行分块并行执行 fn(y0, y1)。"""
    return par_ranges(fn, h, max(1, CHUNK // max(1, w)))


def together(*fns) -> list:
    """同时执行几项互不依赖的分析（各自内部也会多线程），按顺序返回结果；共用 scope() 的缓存。"""
    global _tasks
    check_cancel()
    if len(fns) <= 1 or threading.current_thread().name.startswith(("cd-task", "cd-px")):
        return [f() for f in fns]
    with _pool_lock:
        if _tasks is None:
            _tasks = ThreadPoolExecutor(4, thread_name_prefix="cd-task")
    futs = [_tasks.submit(contextvars.copy_context().run, f) for f in fns[1:]]
    first = fns[0]()  # 第一项在本线程做
    return [first, *(f.result() for f in futs)]


class PixelTable:
    def __init__(self, rgb: np.ndarray):
        px = np.asarray(rgb)[..., :3]
        if px.dtype != np.uint8:
            px = np.clip(np.rint(px), 0, 255).astype(np.uint8)
        self.shape = px.shape[:2]
        h, w = self.shape
        n = h * w
        packed = np.empty(n, dtype=np.uint32)
        present = np.zeros(1 << 24, dtype=bool)

        def pack(y0, y1):
            # 每个像素编成 R | G << 8 | B << 16：从每个像素的起点按小端读 4 个字节（多读到的是下一个像素的 R），
            # 去掉最高字节即可，不用补字节、不用逐通道移位（快 4 倍）。这一段最后一个像素后面没有字节了，单独算
            a, b = y0 * w, y1 * w
            c = px[y0:y1]
            if not c.flags.c_contiguous:  # 裁剪出来的视图、RGBA 的前三个通道：先复制这一段
                c = np.ascontiguousarray(c)
            cf = c.reshape(-1)
            m = b - a - 1
            if m > 0:
                win = np.ndarray(shape=(m,), dtype="<u4", buffer=cf, offset=0, strides=(3,))
                np.bitwise_and(win, 0xFFFFFF, out=packed[a : a + m])
            r, g, bl = (int(v) for v in cf[-3:])
            packed[b - 1] = r | (g << 8) | (bl << 16)
            present[packed[a:b]] = True  # 多个线程同时写 True：结果一样

        par_rows(pack, h, w)
        uniq = np.flatnonzero(present).astype(np.uint32)
        lut = np.empty(1 << 24, dtype=np.int32)  # 只会读到出现过的颜色
        lut[uniq] = np.arange(uniq.size, dtype=np.int32)
        index = packed.view(np.int32)  # 颜色编号直接写回同一块内存（不再多占一份整幅数组）
        m = int(uniq.size)
        small = m <= 2_000_000  # 颜色不太多：顺便按块数好每种颜色的像素数（数据还在缓存里，不用再扫一遍）

        def number(a, b):
            np.take(lut, packed[a:b], out=index[a:b], mode="clip")
            return np.bincount(index[a:b], minlength=m) if small else None

        parts = par_ranges(number, n)
        self.index = index.reshape(self.shape)
        self.colours = np.stack([uniq & 255, (uniq >> 8) & 255, (uniq >> 16) & 255], axis=-1).astype(np.uint8)
        self.n = m
        if small:
            self.__dict__["_all_counts"] = np.sum(parts, axis=0)
        self._mask_counts: dict = {}  # 同一个蒙版只数一次（见 count）
        self._mask_lock = threading.Lock()

    @cached_property
    def lab(self) -> np.ndarray:
        return np.asarray(cs.srgb8_to_lab(self.colours), dtype=np.float64)

    @cached_property
    def _lch(self) -> np.ndarray:
        return cs.lab_to_lch(self.lab)

    @property
    def L(self) -> np.ndarray:
        return self._lch[:, 0]

    @property
    def C(self) -> np.ndarray:
        return self._lch[:, 1]

    @property
    def h(self) -> np.ndarray:
        return self._lch[:, 2]

    @cached_property
    def clipped(self) -> np.ndarray:
        return cs.clipped(self.colours)

    @cached_property
    def _all_counts(self) -> np.ndarray:
        return self._count(None)

    def count(self, mask: np.ndarray | None = None) -> np.ndarray:
        """每种颜色有多少个像素（只数 mask 为 True 的）。

        蒙版全是 True 时直接用总数；同一个蒙版（同一块内存、内容没变）在统计、脏色、肤色、均匀度里
        各数一次很浪费：记住最近几个蒙版的结果。"""
        if mask is None:
            return self._all_counts.copy()
        mask = np.asarray(mask, dtype=bool)
        if mask.all():
            return self._all_counts.copy()
        k = (_key(mask), _fingerprint_mask(mask))
        with self._mask_lock:  # 几项分析同时要同一个蒙版的计数：后来的等前面的数完直接用（不重复数）
            hit = self._mask_counts.get(k)
            if hit is None:
                out = self._count(mask.reshape(-1))
                hit = self._mask_counts[k] = (mask, out)  # 连蒙版一起存着：地址不会被别的数组复用
                while len(self._mask_counts) > 4:
                    self._mask_counts.pop(next(iter(self._mask_counts)))
        return hit[1].copy()

    def _count(self, mask) -> np.ndarray:
        idx = self.index.reshape(-1)
        if self.n > 2_000_000:  # 颜色极多（例如纯噪声）：每块一份计数表太占内存，不分块
            return np.bincount(idx if mask is None else idx[mask], minlength=self.n)

        def part(a, b):
            v = idx[a:b] if mask is None else idx[a:b][mask[a:b]]
            return np.bincount(v, minlength=self.n)

        return np.sum(par_ranges(part, idx.size), axis=0)

    def pix(self, values: np.ndarray) -> np.ndarray:
        """按颜色给的值 -> 每个像素的值（与原图同尺寸）。"""
        values = np.asarray(values)
        idx = self.index.reshape(-1)
        out = np.empty((idx.size, *values.shape[1:]), dtype=values.dtype)
        par_ranges(lambda a, b: np.take(values, idx[a:b], axis=0, out=out[a:b], mode="clip"), idx.size)
        return out.reshape(*self.shape, *values.shape[1:])

    def colour_counts(self, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(出现过的颜色, 每种的像素数)：只保留 mask 内至少有一个像素的颜色。"""
        cnt = self.count(mask)
        keep = cnt > 0
        return self.colours[keep], cnt[keep]


# ---------------------------------------------------------------- 一次分析内共用
# 用 contextvar 而不是线程局部变量：together() 把同一次分析拆到几个线程里时，它们看到的是同一份缓存
_cache: contextvars.ContextVar[dict | None] = contextvars.ContextVar("colordoctor_pixel_tables", default=None)


@contextmanager
def scope():
    """在这段代码里，同一张图（同一个数组）只编号一次。可以嵌套（沿用外层的）；离开最外层时释放。"""
    if _cache.get() is not None:
        yield
        return
    token = _cache.set({"lock": threading.Lock()})
    try:
        yield
    finally:
        _cache.reset(token)


def _key(a: np.ndarray):
    return (a.__array_interface__["data"][0], a.shape, a.strides, a.dtype.str)


def table(rgb: np.ndarray) -> PixelTable:
    """rgb 的像素表；在 scope() 内同一个数组只建一次，分析之间也复用（见 _recent）。"""
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
        return PixelTable(rgb)
    return cached("table", rgb, lambda: PixelTable(rgb))


# ---------------------------------------------------------------- 分析之间复用
# 同一张图换个影调风格 / 颜色模板 / 肤色风格再分析、或者先出快速结果再精查时，不必重新给颜色编号。
# 按数组本身（地址、形状）记住最近两张图的结果；连数组一起存着，地址就不会被别的图复用。
# 每次取用前核对一遍内容指纹（抽样），数组被就地改写过就不用旧结果。
RECENT_IMAGES = 2
_recent: OrderedDict = OrderedDict()
_recent_lock = threading.Lock()


def _fingerprint(a: np.ndarray) -> int:
    h, w = a.shape[:2]
    sy, sx = max(1, h // 256), max(1, w // 256)
    return zlib.crc32(np.ascontiguousarray(a[::sy, ::sx]).tobytes())


def _fingerprint_mask(m: np.ndarray) -> int:
    """蒙版内容的指纹：True 的个数 + 抽样（蒙版被就地改写过就认不出来，不会用旧的计数）。"""
    h = m.shape[0] if m.ndim else 1
    step = max(1, h // 512)
    return hash((int(np.count_nonzero(m)), zlib.crc32(np.ascontiguousarray(m[::step]).tobytes())))


def forget_all() -> None:
    with _recent_lock:
        _recent.clear()


def _recent_get(tag, arr):
    k = _key(arr)
    with _recent_lock:
        e = _recent.get(k)
        if e is None:
            return None
        _recent.move_to_end(k)
    if e["fp"] != _fingerprint(arr):
        with _recent_lock:
            _recent.pop(k, None)
        return None
    return e["items"].get(tag)


def _recent_put(tag, arr, value) -> None:
    if arr.ndim != 3:
        return
    k = _key(arr)
    fp = _fingerprint(arr)
    with _recent_lock:
        e = _recent.get(k)
        if e is None or e["fp"] != fp:
            e = _recent[k] = {"arr": arr, "fp": fp, "items": {}}
        e["items"][tag] = value
        _recent.move_to_end(k)
        while len(_recent) > RECENT_IMAGES * 3:  # 一张图的缩小图也算一项
            _recent.popitem(last=False)


def content_key(a: np.ndarray) -> tuple:
    """整块像素内容的指纹（形状 + 分 4 段的 CRC32，多线程）：插件每次重新传来的同一张图认得出来。"""
    flat = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
    edges = np.linspace(0, flat.size, 5).astype(int)
    crcs = par_ranges(lambda i, j: [zlib.crc32(memoryview(flat[edges[k]:edges[k + 1]])) for k in range(i, j)], 4, 1)
    return (a.shape, a.dtype.str, *(c for part in crcs for c in part))


def cached(tag, arr: np.ndarray, make):
    """按 (tag, 数组) 缓存 make() 的结果（像素表、同一张图的缩小图）：scope() 内只算一次，分析之间也复用。"""
    if not isinstance(arr, np.ndarray):
        return make()
    cache = _cache.get()
    if cache is None:
        return make()
    k = (tag, _key(arr))
    with cache["lock"]:  # 两项并行的分析同时要同一张图时，后来的等前面的建完直接用
        hit = cache.get(k)
        if hit is None:
            value = _recent_get(tag, arr)
            if value is None:
                value = make()
                _recent_put(tag, arr, value)
            # 连数组一起存着：数组活着，同一个地址就不会被别的图复用
            hit = cache[k] = (arr, value)
    return hit[1]


def colour_counts(rgb: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """图像（mask 范围内）-> (不同的颜色, 每种的像素数)。像素不多时直接返回每个像素（权重 1）。"""
    px = np.asarray(rgb)[..., :3]
    if px.shape[0] * px.shape[1] < cs.PER_COLOUR_MIN or px.dtype != np.uint8:
        flat = px.reshape(-1, 3) if mask is None else px[np.asarray(mask, dtype=bool)]
        return cs.unique_colours(flat)
    return table(rgb).colour_counts(mask)


def wmedian(values, weights) -> float:
    return float(cs.weighted_percentile(values, weights, 50))
