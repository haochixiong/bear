"""色彩管理：JPG 读取与屏幕像素还原到统一的 sRGB 分析空间。

- JPG：按 EXIF 方向旋转；带 ICC 时用 LittleCMS（Pillow ImageCms）转换到 sRGB；
  没有 ICC 时按 sRGB 处理并给出提示。只读取文件，绝不写回。
- 屏幕：Photoshop 等色彩管理软件会把文档颜色按“显示器配置文件”转换后
  再显示，所以屏幕像素 ≠ 文档数值。用同一个显示器配置文件反向转换，
  即可得到与设备无关的 sRGB 数值（广色域/校色过的显示器差异最明显）。
"""

from __future__ import annotations

import dataclasses
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
    # 扩展名是 .jpg 但内容是 PNG 等（常见于网上下载的图）：按实际格式读
    return _decode_jpeg(lambda: Image.open(path), path.name, IMAGE_FORMATS)


# 除 JPG 外还能读的普通图片（参考图常是 PNG / TIFF / WebP）
OTHER_EXTS = {".png", ".tif", ".tiff", ".webp"}
IMAGE_FORMATS = ("JPEG", "PNG", "TIFF", "WEBP")


def load_picture(path: str | os.PathLike) -> SourceImage:
    """JPG / PNG / TIFF / WebP（8 位或 16 位；16 位按 8 位分析）。"""
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"找不到文件：{path}")
    if path.stat().st_size == 0:
        raise ImageLoadError("文件为空，无法读取")
    return _decode_jpeg(lambda: Image.open(path), path.name, IMAGE_FORMATS)


def load_image_bytes(data: bytes, name: str, use_raw_preview: bool = False, max_side: int | None = None) -> SourceImage:
    """从文件内容读取（PS 插件选的参考图）。name 只用来判断格式和显示。

    max_side：只需要这么大时，JPG 直接按缩小的尺寸解码（快很多；结果至少这么大，调用方再缩到准确尺寸）。
    """
    if not data:
        raise ImageLoadError("文件为空，无法读取")
    suffix = Path(name).suffix.lower()
    if suffix in RAW_EXTS:
        import tempfile

        # rawpy 只能从文件读：写到临时文件，读完删除
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            src = load_raw(tmp, use_raw_preview)
        finally:
            _remove_later(tmp)
        return SourceImage(src.pixels, dataclasses.replace(src.context, device=name), src.original_size)
    return _decode_jpeg(lambda: Image.open(io.BytesIO(data)), name, IMAGE_FORMATS, max_side=max_side)


_pending_removal: list[str] = []


def _remove_later(path: str) -> None:
    """删除临时文件；Windows 上杀毒软件 / 索引可能暂时占用，删不掉就在程序退出时再删一次。"""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        if not _pending_removal:
            import atexit

            atexit.register(lambda: [_try_remove(p) for p in list(_pending_removal)])
        _pending_removal.append(path)


def _try_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def alpha_of_bytes(data: bytes) -> np.ndarray | None:
    """图片文件的透明度（与解码后的像素同方向、同尺寸）；没有透明通道返回 None。"""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.mode not in ("RGBA", "LA", "PA") and not (im.mode == "P" and "transparency" in im.info):
                return None
            im = ImageOps.exif_transpose(im)
            return np.asarray(im.convert("RGBA").getchannel("A"))
    except Exception:
        return None


def _to_8bit(im: Image.Image) -> Image.Image:
    """16 / 32 位灰度（PNG / TIFF 常见）转 8 位；Pillow 的 convert("RGB") 会把它们直接截断成白色。"""
    if im.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I"):
        a = np.asarray(im)
        if im.mode != "I" or int(a.max()) > 255:
            a = np.right_shift(a.astype(np.uint16, copy=False), 8)  # 0–65535 → 0–255，不做大块浮点拷贝
        return Image.fromarray(np.asarray(a, dtype=np.uint8), mode="L")
    if im.mode == "F":
        a = np.asarray(im, dtype=np.float32)
        return Image.fromarray(np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8), mode="L")
    if im.mode == "LA":
        return im.convert("L")  # 保留灰度，灰度 ICC 才能用；透明度另外读（alpha_of_bytes）
    if im.mode in ("RGBA", "P", "PA", "1"):
        return im.convert("RGB")
    return im


def _decode_jpeg(opener, name: str, formats: tuple[str, ...] = ("JPEG",), max_side: int | None = None) -> SourceImage:
    kinds = "/".join("JPG" if f == "JPEG" else f for f in formats)
    try:
        with opener() as im:
            if im.format not in formats:
                raise ImageLoadError(f"只支持 {kinds} 图片（当前为 {im.format or '未知格式'}）")
            w, h = im.size
            if w * h > MAX_PIXELS:
                raise ImageLoadError(f"图片过大（{w}×{h}），超过 {MAX_PIXELS // 1_000_000} MP 上限")
            icc = im.info.get("icc_profile")
            if max_side and im.format == "JPEG" and max(w, h) > 2 * max_side:
                im.draft("RGB", (max_side, max_side))  # 按 1/2、1/4、1/8 解码，尺寸仍不小于 max_side
            im.load()
            im = ImageOps.exif_transpose(im)
            im = _to_8bit(im)
    except ImageLoadError:
        raise
    except Image.DecompressionBombError as e:
        raise ImageLoadError(f"图片过大，超过 {MAX_PIXELS // 1_000_000} MP 上限") from e
    except UnidentifiedImageError as e:
        raise ImageLoadError(f"无法识别的图片文件（不是有效的 {kinds}）") from e
    except (OSError, SyntaxError, ValueError) as e:
        raise ImageLoadError(f"图片文件已损坏、不完整，或是不支持的格式（详细：{e}）") from e

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
        device=name,
        clipped_fraction=clipped,
        warnings=tuple(warnings),
    )
    pixels = np.ascontiguousarray(out, dtype=np.uint8)
    pixels.setflags(write=False)
    return SourceImage(pixels=pixels, context=ctx, original_size=(im.size[0], im.size[1]))


RAW_EXTS = {".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf", ".orf", ".rw2", ".dng",
            ".pef", ".srw", ".3fr", ".iiq", ".erf", ".kdc", ".mos", ".mrw", ".x3f"}
RAW_WARNING = ("RAW 由 LibRaw 解码（相机白平衡、sRGB、自动亮度）：颜色和明暗与 ACR 的相机配置文件（如“Adobe 颜色”）不同，"
               "数值只作参考。要与 ACR 一致，请在 PS 插件里分析 ACR 打开后的画面")


def is_raw(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in RAW_EXTS


def load_raw(path: str | os.PathLike, use_preview: bool = False) -> SourceImage:
    """读取相机 RAW。use_preview=True 时改用文件里内嵌的 JPG 预览（相机直出的颜色）。"""
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"找不到文件：{path}")
    try:
        import rawpy
    except ImportError as e:
        raise ImageLoadError("读取 RAW 需要安装 rawpy：pip install rawpy") from e
    try:
        with rawpy.imread(str(path)) as raw:
            if use_preview:
                thumb = raw.extract_thumb()
                if thumb.format != rawpy.ThumbFormat.JPEG:
                    raise ImageLoadError("这个 RAW 文件没有内嵌的 JPG 预览，请取消“使用内嵌预览”")
                data = bytes(thumb.data)
                src = _decode_jpeg(lambda: Image.open(io.BytesIO(data)), path.name)
                note = "使用 RAW 内嵌的 JPG 预览（相机直出的颜色，尺寸可能较小）"
                ctx = dataclasses.replace(src.context, warnings=(*src.context.warnings, note))
                return SourceImage(src.pixels, ctx, src.original_size)
            sizes = raw.sizes
            if sizes.width * sizes.height > MAX_PIXELS:
                raise ImageLoadError(f"图片过大（{sizes.width}×{sizes.height}），超过 {MAX_PIXELS // 1_000_000} MP 上限")
            # gamma=(2.4, 12.92)：sRGB 传递曲线（LibRaw 默认是 BT.709 的 2.222/4.5，中灰会暗约 13 级）
            rgb = raw.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB, output_bps=8,
                                  gamma=(2.4, 12.92))
    except ImageLoadError:
        raise
    except Exception as e:  # rawpy 的错误类型较多（LibRawFileUnsupportedError 等）
        detail = e.args[0] if e.args else e
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        raise ImageLoadError(f"无法读取 RAW 文件（不是相机 RAW，或相机型号还不支持）：{detail}") from e
    ctx = ColorContext(source="file", analysis_space=ANALYSIS_SPACE, source_profile="相机 RAW（LibRaw 解码）",
                       converted=True, device=path.name, warnings=(RAW_WARNING,))
    pixels = np.ascontiguousarray(rgb, dtype=np.uint8)
    pixels.setflags(write=False)
    return SourceImage(pixels=pixels, context=ctx, original_size=(pixels.shape[1], pixels.shape[0]))


def load_image(path: str | os.PathLike, use_raw_preview: bool = False) -> SourceImage:
    """按扩展名选择：RAW 用 load_raw，PNG / TIFF / WebP 用 load_picture，其余按 JPG 读取。"""
    if is_raw(path):
        return load_raw(path, use_raw_preview)
    return load_picture(path)


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
