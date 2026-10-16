import numpy as np
import pytest

from colordoctor.capture import geometry as geo
from colordoctor.capture import hotkeys as hk
from colordoctor.capture import windows_display as wd

S1 = geo.ScreenInfo("\\\\.\\DISPLAY1", 0, 0, 1920, 1080, 1.0)
S2 = geo.ScreenInfo("\\\\.\\DISPLAY2", 1920, 0, 1707, 960, 1.5)  # 2560×1440 @150%
S3 = geo.ScreenInfo("\\\\.\\DISPLAY3", -1280, 200, 1280, 1024, 1.25)


def test_screen_lookup_with_negative_coordinates():
    screens = [S1, S2, S3]
    assert geo.screen_index_at(screens, 10, 10) == 0
    assert geo.screen_index_at(screens, 2000, 100) == 1
    assert geo.screen_index_at(screens, -100, 300) == 2
    assert geo.screen_index_at(screens, -100, 100) is None


def test_logical_to_device_respects_scaling_and_offset():
    assert geo.logical_to_device(S1, 100, 50) == (100, 50)
    assert geo.logical_to_device(S2, 1920 + 100, 50) == (150, 75)
    assert geo.logical_to_device(S3, -1280 + 10, 210) == (12, 12)
    assert S2.device_size == (2560, 1440)


def test_logical_to_device_is_clamped():
    assert geo.logical_to_device(S2, 1920 + 1707, 960) == (2559, 1439)


def test_device_to_logical_roundtrip():
    gx, gy = geo.device_to_logical(S2, 150, 75)
    assert (gx, gy) == (2020.0, 50.0)
    assert geo.logical_to_device(S2, gx, gy) == (150, 75)


def test_rect_device_clamps_and_normalizes():
    assert geo.rect_device(S1, 1900, 1000, 100, 100) == (1900, 1000, 20, 80)
    assert geo.rect_device(S1, 50, 50, -20, -10) == (30, 40, 20, 10)
    assert geo.rect_device(S1, 3000, 0, 10, 10) is None


def test_virtual_bounds():
    assert geo.virtual_bounds([S1, S2, S3]) == (-1280, 0, 1280 + 1920 + 1707, 1224)
    assert geo.virtual_bounds([]) == (0, 0, 0, 0)


def test_view_to_image_fit():
    scale, ox, oy = geo.view_to_image(0, 0, 400, 300, 800, 400)
    assert scale == 0.5 and ox == 0 and oy == 50


@pytest.mark.parametrize(
    "text,mods,vk",
    [
        ("Ctrl+Alt+Shift+Z", hk.MOD_CONTROL | hk.MOD_ALT | hk.MOD_SHIFT, ord("Z")),
        ("ctrl + shift + f9", hk.MOD_CONTROL | hk.MOD_SHIFT, 0x78),
        ("Alt+1", hk.MOD_ALT, ord("1")),
    ],
)
def test_parse_hotkey(text, mods, vk):
    assert hk.parse_hotkey(text) == (mods, vk)


@pytest.mark.parametrize("text", ["", "Z", "Ctrl+", "Hyper+Z", "Ctrl+Alt", "Ctrl+Èè"])
def test_parse_hotkey_errors(text):
    with pytest.raises(hk.HotkeyParseError):
        hk.parse_hotkey(text)


def test_default_hotkeys_parse_and_avoid_known_photoshop_combos():
    taken = set("CENSWKMBLTOP")
    for text in hk.DEFAULT_HOTKEYS.values():
        mods, vk = hk.parse_hotkey(text)
        assert mods == hk.MOD_CONTROL | hk.MOD_ALT | hk.MOD_SHIFT
        assert chr(vk) not in taken


def test_windows_display_returns_none_off_windows():
    import sys

    if not sys.platform.startswith("win"):
        assert wd.monitor_icc_bytes("\\\\.\\DISPLAY1") is None


def test_hotkey_manager_without_windows(qapp):
    import sys

    m = hk.make_manager()
    ok, msg = m.register("capture", "Ctrl+Alt+Shift+X")
    if not sys.platform.startswith("win"):
        assert not ok and "不支持" in msg
    bad_ok, bad_msg = m.register("x", "Z")
    assert not bad_ok and bad_msg


def test_qimage_numpy_roundtrip(qapp):
    from colordoctor.capture import screen_capture as sc

    arr = np.random.default_rng(0).integers(0, 256, (7, 13, 3)).astype(np.uint8)  # 奇数宽度：bytesPerLine 有填充
    back = sc.qimage_to_array(sc.array_to_qimage(arr))
    assert np.array_equal(arr, back)


def test_freeze_all_screens_offscreen(qapp):
    from colordoctor.capture import screen_capture as sc

    caps = sc.freeze_all_screens()
    assert caps  # offscreen 平台也有一块虚拟屏幕
    assert caps[0].pixels.ndim == 3


def test_hotkey_managers_close_cleanly(qapp):
    """多个窗口各有一个快捷键管理器；关闭后不再接收，过滤器全程序只有一个。"""
    a, b = hk.make_manager(), hk.make_manager()
    a.close()
    b.close()
    a.close()  # 重复关闭无害
    if hk.is_supported():  # pragma: no cover - Windows only
        assert hk._FILTER is not None and len(hk._FILTER.owners) == 0
