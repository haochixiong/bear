"""测速：在这台电脑上按原图尺寸分析一张大图要多久（colordoctor bench）。

默认用一张合成的人像（背景渐变、皮肤、噪声、零星杂色点），也可以给一张自己的 JPG / RAW。
每种分析算两次取较快的一次；每次前清掉“同一张图的颜色编号”缓存，所以是第一次分析这张图的时间。
"""

from __future__ import annotations

import os
import time

import numpy as np

MODES = ("脏色 / 区域", "整图扫描", "人像肤色", "整图色彩")


def synthetic(mp: float, seed: int = 0) -> np.ndarray:
    """mp 百万像素（3:2）的合成人像：蓝绿渐变背景 + 肤色椭圆 + 噪声 + 背景里零星的黄色杂点。"""
    h = int((mp * 1e6 / 1.5) ** 0.5)
    w = int(h * 1.5)
    rng = np.random.default_rng(seed)
    out = np.empty((h, w, 3), np.uint8)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    rows = max(1, (1 << 22) // w)
    for y0 in range(0, h, rows):  # 按行分块生成：1 亿像素也不占太多内存
        y = yy[y0 : y0 + rows]
        bg = np.stack(np.broadcast_arrays(60 + 40 * xx + 0 * y, 110 + 50 * y + 0 * xx, 170 - 30 * xx + 0 * y), axis=-1)
        face = ((xx - 0.5) / 0.22) ** 2 + ((y - 0.5) / 0.33) ** 2 <= 1
        blk = np.where(face[..., None], np.array([200, 156, 132], np.float32) - 25 * (y - 0.5)[..., None], bg)
        blk += rng.normal(0, 2.5, blk.shape).astype(np.float32)
        speck = (~face) & (rng.random(face.shape) < 0.002)
        blk[speck] = (225, 200, 60)
        out[y0 : y0 + rows] = np.clip(blk, 0, 255).astype(np.uint8)
    return out


def run(img: np.ndarray, repeats: int = 2) -> list[tuple[str, float, float]]:
    """[(分析, 第一次分析这张图的秒数, 同一张图换设置重算的秒数)]"""
    from colordoctor.core import analysis as an
    from colordoctor.core import pixel_table as ptm
    from colordoctor.core.color_management import synthetic_context

    ctx = synthetic_context()
    jobs = {
        "脏色 / 区域": lambda: an.analyze_region(img, ctx),
        "整图扫描": lambda: an.scan_image(img, ctx),
        "人像肤色": lambda: an.analyze_skin_region(img, ctx),
        "整图色彩": lambda: an.analyze_palette_report(img, ctx, tone_style="japanese", style_key="teal_orange"),
    }
    jobs["整图色彩"]()  # 预热（载入模板、第一次分配内存）
    out = []
    for name, fn in jobs.items():
        cold, warm = [], []
        for _ in range(repeats):
            ptm.forget_all()
            t = time.perf_counter()
            fn()
            cold.append(time.perf_counter() - t)
            t = time.perf_counter()
            fn()
            warm.append(time.perf_counter() - t)
        out.append((name, min(cold), min(warm)))
    return out


def report(sizes: list[float], image_path: str | None = None) -> str:
    from colordoctor.core.pixel_table import THREADS

    lines = [f"CPU 逻辑核数 {os.cpu_count()}，分析用 {THREADS} 个线程。每种分析都按原图尺寸检查每一个像素。", ""]
    cases = []
    if image_path:
        from colordoctor.core.color_management import load_image

        img = load_image(image_path).pixels
        cases.append((f"{image_path}（{img.shape[1]}×{img.shape[0]}，{img.shape[0] * img.shape[1] / 1e6:.0f} MP）", img))
    else:
        cases += [(f"合成人像 {mp:g} MP", None) for mp in sizes]
    for i, (title, img) in enumerate(cases):
        if img is None:
            img = synthetic(sizes[i])
        lines.append(title)
        lines.append(f"  {'分析':<10s}{'第一次':>8s}{'换设置重算':>12s}")
        for name, cold, warm in run(img):
            lines.append(f"  {name:<10s}{cold:>7.2f}s{warm:>11.2f}s")
        lines.append("")
        del img
    return "\n".join(lines)
