"""全局快捷键（Windows：RegisterHotKey；其他平台不支持，界面按钮仍可用）。

默认组合的选择理由（Photoshop 默认快捷键里 Ctrl+Alt+Shift 组合已经占用了
C 内容识别缩放、E 盖印、N 新建图层、S 存储为 Web、W 导出为、K 键盘快捷键、
M 菜单、B 黑白、L 自动对比度、T 再次变换、O/P 等），选用左手可单手按到、
据 Photoshop 默认快捷键表未被占用的 Z / X：
    Ctrl+Alt+Shift+Z  锁定当前实时取色读数
    Ctrl+Alt+Shift+X  冻结屏幕并框选
全局快捷键会优先于 Photoshop 本身的快捷键，因此只在本工具运行时注册，可在设置中修改。
"""

from __future__ import annotations

import sys

DEFAULT_HOTKEYS = {
    "lock_sample": "Ctrl+Alt+Shift+Z",
    "capture": "Ctrl+Alt+Shift+X",
}

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312

_MODS = {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT, "win": MOD_WIN}
_NAMED_VK = {
    **{f"f{i}": 0x6F + i for i in range(1, 25)},
    "space": 0x20,
    "pause": 0x13,
    "insert": 0x2D,
    "home": 0x24,
    "end": 0x23,
    "pageup": 0x21,
    "pagedown": 0x22,
}


class HotkeyParseError(ValueError):
    pass


def parse_hotkey(text: str) -> tuple[int, int]:
    """'Ctrl+Alt+Shift+Z' -> (modifiers, virtual-key)。至少需要一个修饰键。"""
    parts = [p.strip().lower() for p in text.replace(" ", "").split("+") if p.strip()]
    if not parts:
        raise HotkeyParseError("快捷键为空")
    mods = 0
    for p in parts[:-1]:
        if p not in _MODS:
            raise HotkeyParseError(f"未知修饰键：{p}")
        mods |= _MODS[p]
    key = parts[-1]
    if key in _MODS:
        raise HotkeyParseError("缺少主键")
    if len(key) == 1 and (key.isalpha() or key.isdigit()):
        vk = ord(key.upper())
    elif key in _NAMED_VK:
        vk = _NAMED_VK[key]
    else:
        raise HotkeyParseError(f"不支持的按键：{key}")
    if mods == 0:
        raise HotkeyParseError("全局快捷键至少需要一个修饰键（Ctrl/Alt/Shift/Win）")
    return mods, vk


def is_supported() -> bool:
    return sys.platform.startswith("win")


def make_manager(parent=None):
    """返回一个 GlobalHotkeys 对象（Qt）；非 Windows 平台返回的对象注册总是失败。"""
    from PyQt6.QtCore import QAbstractNativeEventFilter, QCoreApplication, QObject, pyqtSignal

    class _Filter(QAbstractNativeEventFilter):
        def __init__(self, owner):
            super().__init__()
            self.owner = owner

        def nativeEventFilter(self, event_type, message):  # pragma: no cover - Windows only
            try:
                if bytes(event_type) not in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
                    return False, 0
                from ctypes import wintypes

                msg = wintypes.MSG.from_address(int(message))
                if msg.message == WM_HOTKEY:
                    name = self.owner._ids.get(int(msg.wParam))
                    if name:
                        self.owner.triggered.emit(name)
                        return True, 0
            except Exception:
                pass
            return False, 0

    class GlobalHotkeys(QObject):
        triggered = pyqtSignal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._ids: dict[int, str] = {}
            self._next = 0xB000
            self._filter = None
            if is_supported():
                self._filter = _Filter(self)
                QCoreApplication.instance().installNativeEventFilter(self._filter)

        def register(self, name: str, text: str) -> tuple[bool, str]:
            try:
                mods, vk = parse_hotkey(text)
            except HotkeyParseError as e:
                return False, str(e)
            if not is_supported():
                return False, "当前系统不支持全局快捷键，请使用界面按钮"
            import ctypes  # pragma: no cover - Windows only

            self.unregister(name)
            hid = self._next
            self._next += 1
            ok = ctypes.windll.user32.RegisterHotKey(None, hid, mods | MOD_NOREPEAT, vk)
            if not ok:
                return False, f"{text} 已被其他程序占用"
            self._ids[hid] = name
            return True, ""

        def unregister(self, name: str) -> None:
            for hid, n in list(self._ids.items()):
                if n == name:
                    if is_supported():  # pragma: no cover - Windows only
                        import ctypes

                        ctypes.windll.user32.UnregisterHotKey(None, hid)
                    del self._ids[hid]

        def unregister_all(self) -> None:
            for name in set(self._ids.values()):
                self.unregister(name)

    return GlobalHotkeys(parent)
