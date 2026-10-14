import io

import numpy as np
import pytest
from PIL import Image, ImageCms

from colordoctor.core import color_management as cm
from icc_util import PRIMARIES, make_rgb_profile, profile_rgb8_to_srgb8, srgb8_to_profile_rgb8

SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
ADOBE_ICC = make_rgb_profile("Adobe RGB (1998) test", PRIMARIES["adobe_rgb"], trc="gamma", gamma=563 / 256)
P3_ICC = make_rgb_profile("Display P3 test", PRIMARIES["display_p3"], trc="srgb")


def _save(tmp_path, name, img, **kw):
    p = tmp_path / name
    img.save(p, **kw)
    return p


def _flat(color, size=(40, 30)):
    return Image.new("RGB", size, color)


# ------------------------------------------------------------------ JPEG 文件
def test_jpeg_with_srgb_icc_is_not_altered(tmp_path):
    p = _save(tmp_path, "a.jpg", _flat((180, 120, 90)), quality=100, icc_profile=SRGB_ICC)
    before = p.read_bytes()
    src = cm.load_jpeg(p)
    assert src.pixels.shape == (30, 40, 3)
    assert src.context.source_profile and "sRGB" in src.context.source_profile
    assert not src.context.assumed_srgb
    assert src.context.profile_near_srgb is True
    px = src.pixels[15, 20].astype(int)
    assert np.all(np.abs(px - [180, 120, 90]) <= 2)
    assert p.read_bytes() == before  # 原图未被改写


def test_jpeg_without_icc_is_assumed_srgb_with_warning(tmp_path):
    p = _save(tmp_path, "b.jpg", _flat((50, 100, 200)), quality=100)
    src = cm.load_jpeg(p)
    assert src.context.assumed_srgb
    assert not src.context.converted
    assert any("未发现ICC" in w for w in src.context.warnings)


def test_adobe_rgb_jpeg_is_converted_to_srgb(tmp_path):
    srgb_color = np.array([150, 90, 60], dtype=np.uint8)
    adobe = srgb8_to_profile_rgb8(srgb_color, PRIMARIES["adobe_rgb"], trc="gamma", gamma=563 / 256)
    p = _save(tmp_path, "c.jpg", _flat(tuple(int(v) for v in adobe)), quality=100, icc_profile=ADOBE_ICC)
    src = cm.load_jpeg(p)
    assert src.context.converted
    assert src.context.profile_near_srgb is False
    assert "Adobe" in src.context.source_profile
    got = src.pixels[15, 20].astype(int)
    assert np.all(np.abs(got - srgb_color.astype(int)) <= 3), got


def test_adobe_rgb_saturated_green_reports_clipping(tmp_path):
    p = _save(tmp_path, "d.jpg", _flat((0, 255, 0)), quality=100, icc_profile=ADOBE_ICC)
    src = cm.load_jpeg(p)
    assert src.context.clipped_fraction > 0.5
    assert any("色域" in w for w in src.context.warnings)


def test_exif_orientation_is_applied(tmp_path):
    img = _flat((10, 20, 30), size=(40, 30))
    exif = img.getexif()
    exif[0x0112] = 6
    p = _save(tmp_path, "e.jpg", img, quality=95, exif=exif)
    src = cm.load_jpeg(p)
    assert src.pixels.shape[:2] == (40, 30)
    assert src.original_size == (30, 40)


def test_png_named_jpg_is_read_and_other_formats_rejected(tmp_path):
    # 扩展名是 .jpg、内容是 PNG：按实际格式读（网上下载的图常这样）
    p = _save(tmp_path, "f.png", _flat((1, 2, 3)))
    renamed = tmp_path / "renamed.jpg"
    renamed.write_bytes(p.read_bytes())
    assert cm.load_jpeg(renamed).pixels.shape[2] == 3
    g = _save(tmp_path, "f.gif", _flat((1, 2, 3)))
    with pytest.raises(cm.ImageLoadError, match="JPG"):
        cm.load_jpeg(g)


def test_corrupt_jpeg_raises_friendly_error(tmp_path):
    good = _save(tmp_path, "g.jpg", _flat((1, 2, 3), size=(300, 300)), quality=95)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(good.read_bytes()[:300])
    with pytest.raises(cm.ImageLoadError, match="损坏"):
        cm.load_jpeg(bad)


def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.jpg"
    p.write_bytes(b"")
    with pytest.raises(cm.ImageLoadError):
        cm.load_jpeg(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(cm.ImageLoadError, match="找不到"):
        cm.load_jpeg(tmp_path / "nope.jpg")


def test_grayscale_jpeg_becomes_rgb(tmp_path):
    p = _save(tmp_path, "h.jpg", Image.new("L", (10, 10), 128), quality=100)
    src = cm.load_jpeg(p)
    assert src.pixels.shape == (10, 10, 3)
    assert abs(int(src.pixels[5, 5, 0]) - 128) <= 1


def test_cmyk_jpeg_without_profile_warns(tmp_path):
    p = _save(tmp_path, "i.jpg", Image.new("CMYK", (10, 10), (0, 50, 50, 0)), quality=100)
    src = cm.load_jpeg(p)
    assert src.pixels.shape == (10, 10, 3)
    assert any("CMYK" in w for w in src.context.warnings)


def test_too_many_pixels_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "MAX_PIXELS", 100)
    p = _save(tmp_path, "j.jpg", _flat((1, 2, 3), size=(20, 20)), quality=90)
    with pytest.raises(cm.ImageLoadError, match="过大"):
        cm.load_jpeg(p)


# ------------------------------------------------------------------ 屏幕像素
def _simulate_photoshop_display(doc_srgb: np.ndarray, monitor_icc: bytes) -> np.ndarray:
    """模拟 Photoshop 把 sRGB 文档按显示器配置文件显示到屏幕上。"""
    img = Image.fromarray(doc_srgb, "RGB")
    t = ImageCms.buildTransform(
        ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")),
        ImageCms.ImageCmsProfile(io.BytesIO(monitor_icc)),
        "RGB",
        "RGB",
    )
    return np.asarray(ImageCms.applyTransform(img, t))


def test_screen_pixels_from_wide_gamut_monitor_are_restored_to_document_srgb():
    rng = np.random.default_rng(3)
    doc = rng.integers(20, 236, (16, 16, 3)).astype(np.uint8)
    screen = _simulate_photoshop_display(doc, P3_ICC)
    assert np.abs(screen.astype(int) - doc.astype(int)).max() > 10  # 屏幕数值确实不同
    out, ctx = cm.screen_pixels_to_srgb(screen, P3_ICC, managed_source=True, device="DISPLAY1")
    assert ctx.converted and not ctx.assumed_srgb
    assert ctx.profile_near_srgb is False
    # 两次 8 位量化（文档->显示器->还原）会有极小误差：用 ΔE00 衡量
    from colordoctor.core import color_spaces as cs

    de = cs.delta_e_2000(cs.srgb8_to_lab(out), cs.srgb8_to_lab(doc))
    assert de.max() < 1.5 and de.mean() < 0.3
    naive = cs.delta_e_2000(cs.srgb8_to_lab(screen), cs.srgb8_to_lab(doc))
    assert naive.mean() > 3 * de.mean()  # 不做反转换误差明显更大


def test_screen_pixels_unmanaged_source_are_used_as_is():
    px = np.full((4, 4, 3), 100, np.uint8)
    out, ctx = cm.screen_pixels_to_srgb(px, P3_ICC, managed_source=False)
    assert np.array_equal(out, px)
    assert not ctx.converted
    assert any("未做色彩管理" in w for w in ctx.warnings)


def test_screen_pixels_without_profile_are_assumed_srgb():
    px = np.full((4, 4, 3), 100, np.uint8)
    out, ctx = cm.screen_pixels_to_srgb(px, None, managed_source=True)
    assert np.array_equal(out, px)
    assert ctx.assumed_srgb
    assert any("已假定为sRGB" in w for w in ctx.warnings)


def test_srgb_monitor_profile_is_near_identity():
    rng = np.random.default_rng(4)
    px = rng.integers(0, 256, (8, 8, 3)).astype(np.uint8)
    out, ctx = cm.screen_pixels_to_srgb(px, SRGB_ICC, managed_source=True)
    assert ctx.profile_near_srgb is True
    assert np.abs(out.astype(int) - px.astype(int)).max() <= 2


def test_broken_monitor_profile_falls_back_to_srgb():
    px = np.full((2, 2, 3), 50, np.uint8)
    out, ctx = cm.screen_pixels_to_srgb(px, b"not an icc profile", managed_source=True)
    assert np.array_equal(out, px)
    assert ctx.assumed_srgb
    assert any("无法读取" in w for w in ctx.warnings)


def test_profile_description():
    assert "Display P3" in cm.profile_description(P3_ICC)
    assert cm.profile_description(b"junk") is None


def test_profile_helper_self_check():
    c = np.array([120, 60, 200], np.uint8)
    enc = srgb8_to_profile_rgb8(c, PRIMARIES["display_p3"])
    np.testing.assert_allclose(profile_rgb8_to_srgb8(enc, PRIMARIES["display_p3"]), c, atol=2)
