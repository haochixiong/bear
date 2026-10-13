"""相机 RAW 读取（用假的 rawpy 模块测试流程；真实 RAW 解码由 LibRaw 负责）。"""

import io
import sys
import types

import numpy as np
import pytest
from PIL import Image

from colordoctor.core import color_management as cm


class _FakeRaw:
    def __init__(self, rgb, thumb=None):
        self.rgb = rgb
        self.thumb = thumb
        self.sizes = types.SimpleNamespace(width=rgb.shape[1], height=rgb.shape[0])
        self.kwargs = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def postprocess(self, **kw):
        self.kwargs = kw
        return self.rgb

    def extract_thumb(self):
        return self.thumb


def _fake_rawpy(monkeypatch, raw):
    mod = types.ModuleType("rawpy")
    mod.ColorSpace = types.SimpleNamespace(sRGB="sRGB")
    mod.ThumbFormat = types.SimpleNamespace(JPEG="jpeg", BITMAP="bitmap")
    mod.imread = lambda path: raw
    monkeypatch.setitem(sys.modules, "rawpy", mod)
    return mod


def test_is_raw():
    assert cm.is_raw("a.CR3") and cm.is_raw("b.nef") and cm.is_raw("c.dng") and not cm.is_raw("d.jpg")


def test_load_raw_decodes_with_camera_wb_to_srgb(monkeypatch, tmp_path):
    rgb = np.zeros((30, 40, 3), np.uint8)
    rgb[..., 0] = 200
    raw = _FakeRaw(rgb)
    _fake_rawpy(monkeypatch, raw)
    f = tmp_path / "x.CR3"
    f.write_bytes(b"fake")
    src = cm.load_image(f)
    assert src.pixels.shape == (30, 40, 3) and src.pixels[0, 0, 0] == 200
    assert raw.kwargs["use_camera_wb"] is True and raw.kwargs["output_color"] == "sRGB" and raw.kwargs["output_bps"] == 8
    assert any("ACR" in w for w in src.context.warnings)
    assert src.context.source == "file" and src.context.device == "x.CR3"


def test_load_raw_embedded_preview(monkeypatch, tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (20, 10), (10, 120, 200)).save(buf, "JPEG", quality=95)
    raw = _FakeRaw(np.zeros((4, 4, 3), np.uint8), types.SimpleNamespace(format="jpeg", data=buf.getvalue()))
    _fake_rawpy(monkeypatch, raw)
    f = tmp_path / "y.nef"
    f.write_bytes(b"fake")
    src = cm.load_raw(f, use_preview=True)
    assert src.pixels.shape == (10, 20, 3)
    assert abs(int(src.pixels[5, 10, 2]) - 200) <= 3
    assert any("内嵌" in w for w in src.context.warnings)
    raw.thumb = types.SimpleNamespace(format="bitmap", data=b"")
    with pytest.raises(cm.ImageLoadError, match="内嵌"):
        cm.load_raw(f, use_preview=True)


def test_load_raw_errors(monkeypatch, tmp_path):
    with pytest.raises(cm.ImageLoadError, match="找不到"):
        cm.load_raw(tmp_path / "none.arw")
    f = tmp_path / "z.arw"
    f.write_bytes(b"fake")
    mod = _fake_rawpy(monkeypatch, None)

    def boom(path):
        raise RuntimeError("unsupported file")

    mod.imread = boom
    with pytest.raises(cm.ImageLoadError, match="无法读取 RAW"):
        cm.load_raw(f)
    monkeypatch.setitem(sys.modules, "rawpy", None)
    with pytest.raises(cm.ImageLoadError, match="rawpy"):
        cm.load_raw(f)
