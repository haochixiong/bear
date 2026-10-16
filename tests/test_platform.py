"""跨平台（Windows / macOS / Linux）：Mac 的显示器配置文件、设置文件夹、从 Finder 启动、插件安装包。"""

import json
import sys
import zipfile
from pathlib import Path

import pytest

from colordoctor.capture import mac_display
from colordoctor.core import skin_presets as sp

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform == "darwin", reason="只在非 Mac 上检查“读不到时返回 None”")
def test_mac_display_returns_none_elsewhere():
    assert mac_display.icc_at(10, 10) is None
    assert mac_display.display_icc_bytes(1) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="只在 Mac 上能读显示器配置文件")
def test_mac_display_reads_an_icc_profile():  # pragma: no cover - macOS only
    data = mac_display.icc_at(10, 10)
    assert data is None or (len(data) > 128 and data[36:40] == b"acsp")  # 无显示器（CI）时可以是 None


def test_config_dir_on_mac(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert sp.user_config_dir() == tmp_path / "Library" / "Application Support" / "ColorDoctor"


def test_finder_process_serial_argument_is_ignored(capsys):
    from colordoctor.app import main

    assert main(["ColorDoctor", "-psn_0_12345", "presets"]) == 0  # 旧版 macOS 从 Finder 打开时多传的参数
    assert "natural" in capsys.readouterr().out


def test_plugin_package_is_a_flat_ccx(tmp_path):
    sys.path.insert(0, str(ROOT / "packaging"))
    try:
        import build_plugin
    finally:
        sys.path.pop(0)
    out = build_plugin.build(tmp_path)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        manifest = json.loads(z.read("manifest.json"))
    assert "manifest.json" in names and "index.html" in names and "js/main.js" in names
    assert not any(n.startswith("test/") or n.endswith(".md") for n in names)
    assert manifest["host"][0]["app"] == "PS"  # 不限定操作系统：Windows / Mac 的 PS 都能装
    assert out.name == f"ColorDoctor-plugin-{manifest['version']}.ccx"


def test_key_names_follow_platform(monkeypatch):
    import importlib

    import colordoctor.ui.keys as keys

    monkeypatch.setattr(sys, "platform", "darwin")
    assert importlib.reload(keys).CMD == "⌘"
    monkeypatch.setattr(sys, "platform", "win32")
    assert importlib.reload(keys).CMD == "Ctrl+"
    monkeypatch.undo()
    importlib.reload(keys)


def test_gc_guard_collects_on_main_thread(qapp):
    import gc

    from colordoctor.ui import gc_guard

    g = gc_guard.MainThreadGC(interval_ms=60000)
    assert not gc.isenabled() and g.timer.isActive()
    g.threshold = (0, 0, 0)
    g.check()  # 三代都过阈值：全部回收，不报错
    assert gc.get_count()[0] <= 1
    g.timer.stop()
