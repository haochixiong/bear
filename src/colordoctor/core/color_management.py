"""色彩管理：JPG 读取与屏幕像素还原到统一的 sRGB 分析空间。

- JPG：按 EXIF 方向旋转；带 ICC 时用 LittleCMS（Pillow ImageCms）转换到 sRGB；
  没有 ICC 时按 sRGB 处理并给出提示。只读取文件，绝不写回。
- 屏幕：Photoshop 等色彩管理软件会把文档颜色按“显示器配置文件”转换后
  再显示，所以屏幕像素 ≠ 文档数值。用同一个显示器配置文件反向转换，
  即可得到与设备无关的 sRGB 数值（广色域/校色过的显示器差异最明显）。
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from colordoctor.core.color_models import ColorContext, SourceImage

ANALYSIS_SPACE = "sRGB IEC61966-2.1"
MAX_PIXELS = 120_000_000

_SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
_INTENT = ImageCms.Intent.RELATIVE_COLORIMETRIC
try:  # Pillow >= 10.3
    _FLAGS = ImageCms.Flags.BLACKPOINTCOMPENSATION
except AttributeError:  # pragma: no cover
    _FLAGS = 0x2000

# sRGB 原色（D50 适应后的 ICC 原色 XYZ），用于判断配置文件是否“接近 sRGB”
_SRGB_COLORANTS = np.array(
    [
        [0.4360747, 0.2225045, 0.0139322],
        [0.3850649, 0.7168786, 0.0971045],
        [0.1430804, 0.0606169, 0.7141733],
    ]
)


class ImageLoadError(Exception):
    """图像无法读取（中文信息可直接展示给用户）。"""


def _open_profile(icc: bytes) -> ImageCms.ImageCmsProfile | None:
    try:
        return ImageCms.ImageCmsProfile(io.BytesIO(icc))
    except Exception:
        return None


def profile_description(icc: bytes | None) -> str | None:
    if not icc:
        return None
    prof = _open_profile(icc)
    if prof is None:
        return None
    try:
        desc = ImageCms.getProfileDescription(prof).strip()
    except Exception:
        desc = ""
    return desc or "未命名配置文件"


def profile_is_near_srgb(icc: bytes | None, tol: float = 0.01) -> bool | None:
    prof = _open_profile(icc) if icc else None
    if prof is None:
        return None
    p = prof.profile
    try:
        cols = [p.red_colorant, p.green_colorant, p.blue_colorant]
        if any(c is None for c in cols):
            return None
        xyz = np.array([c[0] for c in cols], dtype=float)
    except Exception:
        return None
    if str(getattr(p, "xcolor_space", "")).strip() not in ("RGB", ""):
        return None
    return bool(np.max(np.abs(xyz - _SRGB_COLORANTS)) < tol)


def _transform_to_srgb(img: Image.Image, prof: ImageCms.ImageCmsProfile, in_mode: str) -> Image.Image:
    t = ImageCms.buildTransform(prof, _SRGB_PROFILE, in_mode, "RGB", renderingIntent=_INTENT, flags=_FLAGS)
    return ImageCms.applyTransform(img, t)


def _estimate_clipping(src_rgb: np.ndarray | None, out: np.ndarray) -> float:
    """估算转换造成的色域裁切：输出出现 0/255 而原值不在端点的像素比例。"""
    at_edge = np.any((out == 0) | (out == 255), axis=-1)
    if src_rgb is not None:
        src_edge = np.any((src_rgb == 0) | (src_rgb == 255), axis=-1)
        at_edge &= ~src_edge | np.any(src_rgb != out, axis=-1)
    return float(at_edge.mean()) if at_edge.size else 0.0


def load_jpeg(path: str | os.PathLike) -> SourceImage:
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"找不到文件：{path}")
    if path.stat().st_size == 0:
        raise ImageLoadError("文件为空，无法读取")
    try:
        with Image.open(path) as im:
            if im.format != "JPEG":
                raise ImageLoadError(f"只支持 JPG 图片（当前为 {im.format or '未知格式'}）")
            w, h = im.size
            if w * h > MAX_PIXELS:
                raise ImageLoadError(f"图片过大（{w}×{h}），超过 {MAX_PIXELS // 1_000_000} MP 上限")
            icc = im.info.get("icc_profile")
            im.load()
            im = ImageOps.exif_transpose(im)
    except ImageLoadError:
        raise
    except UnidentifiedImageError as e:
        raise ImageLoadError("无法识别的图片文件（不是有效的 JPG）") from e
    except (OSError, SyntaxError, ValueError) as e:
        raise ImageLoadError(f"JPG 文件已损坏或不完整：{e}") from e

    warnings: list[str] = []
    mode = im.mode
    desc = profile_description(icc)
    prof = _open_profile(icc) if icc else None
    near = profile_is_near_srgb(icc) if icc else None
    converted = False
    assumed = False
    clipped = 0.0

    if prof is not None and mode in ("RGB", "L", "CMYK"):
        try:
            src_arr = np.asarray(im) if mode == "RGB" else None
            out_img = _transform_to_srgb(im, prof, mode)
            out = np.asarray(out_img.convert("RGB"))
            converted = True
            if near is False:
                clipped = _estimate_clipping(src_arr, out)
                if clipped > 0.005:
                    warnings.append(
                        f"源色域比 sRGB 宽，约 {clipped:.1%} 像素在转换时被裁切，高饱和区域的数值偏保守"
                    )
        except Exception:
            warnings.append(f"ICC 配置文件（{desc}）无法使用，已假定为sRGB")
            out = np.asarray(im.convert("RGB"))
            assumed = True
    else:
        if icc and prof is None:
            warnings.append("ICC 配置文件损坏，已假定为sRGB")
        else:
            warnings.append("未发现ICC，已假定为sRGB")
        if mode == "CMYK":
            warnings.append("CMYK 图片没有配置文件，颜色换算不可靠")
        out = np.asarray(im.convert("RGB"))
        assumed = True

    ctx = ColorContext(
        source="file",
        analysis_space=ANALYSIS_SPACE,
        source_profile=desc,
        converted=converted,
        assumed_srgb=assumed,
        profile_near_srgb=near,
        device=path.name,
        clipped_fraction=clipped,
        warnings=tuple(warnings),
    )
    pixels = np.ascontiguousarray(out, dtype=np.uint8)
    pixels.setflags(write=False)
    return SourceImage(pixels=pixels, context=ctx, original_size=(im.size[0], im.size[1]))


def screen_pixels_to_srgb(
    pixels: np.ndarray,
    monitor_icc: bytes | None,
    managed_source: bool = True,
    device: str | None = None,
    scale: float | None = None,
) -> tuple[np.ndarray, ColorContext]:
    """把屏幕截取的像素还原为 sRGB 分析数值。

    managed_source=True：画面来自 Photoshop/ACR 等做了色彩管理的软件，
    用显示器配置文件反向转换；False：画面来自未做色彩管理的软件，
    屏幕像素就是文件里的数值，原样使用。
    """
    px = np.ascontiguousarray(pixels[..., :3], dtype=np.uint8)
    warnings: list[str] = []
    desc = profile_description(monitor_icc)
    near = profile_is_near_srgb(monitor_icc) if monitor_icc else None
    converted = False
    assumed = False
    out = px

    if not managed_source:
        warnings.append("按“未做色彩管理的软件”处理：屏幕数值即文件数值，未做显示器配置文件反转换")
    elif not monitor_icc:
        assumed = True
        warnings.append("未找到显示器ICC，已假定为sRGB")
    else:
        prof = _open_profile(monitor_icc)
        if prof is None:
            assumed = True
            warnings.append("无法读取显示器配置文件，已假定为sRGB")
        else:
            try:
                img = Image.fromarray(px, "RGB")
                out = np.asarray(_transform_to_srgb(img, prof, "RGB"))
                converted = True
            except Exception:
                assumed = True
                warnings.append(f"显示器配置文件（{desc}）无法使用，已假定为sRGB")

    ctx = ColorContext(
        source="screen",
        analysis_space=ANALYSIS_SPACE,
        source_profile=desc,
        converted=converted,
        assumed_srgb=assumed,
        managed_source=managed_source,
        profile_near_srgb=near,
        device=device,
        scale=scale,
        warnings=tuple(warnings),
    )
    return np.ascontiguousarray(out, dtype=np.uint8), ctx


def synthetic_context() -> ColorContext:
    return ColorContext(source="synthetic")
