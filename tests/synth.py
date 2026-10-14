"""合成测试样本（全部确定性生成）。"""

from __future__ import annotations

import numpy as np

from colordoctor.core import color_spaces as cs

BLUE = (40, 80, 200)
YELLOW = (230, 200, 40)
SKIN = (214, 165, 135)


def flat(color, h=120, w=160) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = color
    return img


def noisy(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.astype(float) + rng.normal(0, sigma, img.shape)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def sprinkle(img: np.ndarray, color, fraction: float, seed: int = 0) -> np.ndarray:
    """随机把 fraction 比例的像素替换为 color（散点混入）。"""
    out = img.copy()
    rng = np.random.default_rng(seed)
    n = int(round(fraction * img.shape[0] * img.shape[1]))
    idx = rng.choice(img.shape[0] * img.shape[1], size=n, replace=False)
    out.reshape(-1, 3)[idx] = color
    return out


def patch(img: np.ndarray, color, rect) -> np.ndarray:
    x, y, w, h = rect
    out = img.copy()
    out[y : y + h, x : x + w] = color
    return out


def lab_color(L, a, b) -> tuple[int, int, int]:
    return tuple(int(v) for v in cs.lab_to_srgb8(np.array([L, a, b])))


def lch_color(L, C, h) -> tuple[int, int, int]:
    return tuple(int(v) for v in cs.lab_to_srgb8(cs.lch_to_lab(np.array([L, C, h]))))


def skin_region(L=68.0, a=15.0, b=20.0, h=120, w=120, sigma_L=6.0, sigma_ab=1.2, seed=0) -> np.ndarray:
    """带明暗起伏（高光到阴影）的自然肤色区域：色相恒定，C* 随 L* 线性变化。"""
    rng = np.random.default_rng(seed)
    yy = np.linspace(-1.0, 1.0, h)[:, None]
    xx = np.linspace(-1.0, 1.0, w)[None, :]
    shade = (0.6 * yy + 0.4 * xx)  # 从左上到右下的明暗梯度
    Lmap = L + sigma_L * 2.0 * shade
    scale = Lmap / L
    amap = a * scale
    bmap = b * scale
    lab = np.stack([Lmap, amap, bmap], axis=-1)
    lab[..., 1:] += rng.normal(0, sigma_ab, lab[..., 1:].shape)
    return cs.lab_to_srgb8(lab)


def face_scene(h=200, w=200, skin_lab=(68.0, 14.0, 19.0), seed=0) -> tuple[np.ndarray, np.ndarray]:
    """合成“人脸”场景：青色背景 + 皮肤椭圆 + 深色头发 + 红色嘴唇 + 深色眼睛。

    返回 (图像, 真实皮肤蒙版)。
    """
    img = flat(lch_color(45, 28, 205), h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    face = ((xx - w / 2) / (w * 0.32)) ** 2 + ((yy - h * 0.55) / (h * 0.40)) ** 2 <= 1
    skin = skin_region(*skin_lab, h=h, w=w, seed=seed)
    img[face] = skin[face]
    truth = face.copy()
    hair = (yy < h * 0.25) & face
    img[hair] = lch_color(18, 6, 60)
    truth &= ~hair
    lips = (((xx - w / 2) / (w * 0.09)) ** 2 + ((yy - h * 0.78) / (h * 0.03)) ** 2) <= 1
    img[lips] = lch_color(48, 42, 28)
    truth &= ~lips
    for ex in (w * 0.40, w * 0.60):
        eye = (((xx - ex) / (w * 0.04)) ** 2 + ((yy - h * 0.48) / (h * 0.02)) ** 2) <= 1
        img[eye] = lch_color(15, 3, 60)
        truth &= ~eye
    return img, truth


def lowkey_face_scene(h=220, w=200, skin_lab=(45.0, 12.0, 16.0), bg_lab=(14.0, 1.0, -3.0), seed=0) -> tuple[np.ndarray, np.ndarray]:
    """暗调（低调）人像：face_scene 的人脸放在很暗、接近中性的背景上（背景带一点左右明暗变化和噪点）。"""
    img, truth = face_scene(h=h, w=w, skin_lab=skin_lab, seed=seed)
    yy, xx = np.mgrid[0:h, 0:w]
    face = ((xx - w / 2) / (w * 0.32)) ** 2 + ((yy - h * 0.55) / (h * 0.40)) ** 2 <= 1
    rng = np.random.default_rng(seed)
    lab = np.empty((h, w, 3))
    lab[:] = bg_lab
    lab[..., 0] += np.linspace(-6, 6, w)[None, :]
    lab[..., 1:] += rng.normal(0, 0.6, (h, w, 2))
    img[~face] = cs.lab_to_srgb8(lab)[~face]
    return img, truth
