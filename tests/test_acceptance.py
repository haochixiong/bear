"""验收相关的横向检查：离线、不写原图、大图性能。"""

import ast
import time
from pathlib import Path

import numpy as np

from colordoctor.core import analysis as an
from colordoctor.core.color_management import synthetic_context
from synth import BLUE, YELLOW, flat, noisy, sprinkle

SRC = Path(__file__).resolve().parents[1] / "src" / "colordoctor"
FORBIDDEN = {"socket", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib", "smtplib"}


def test_no_network_modules_outside_optional_ai_package():
    """联网代码只允许出现在可选的 colordoctor/ai 包里（默认关闭，用户点击“AI 点评”才会调用），
    以及 colordoctor/bridge（只监听 127.0.0.1 的本机服务，给 PS 插件用，见下一个测试）。"""
    for py in SRC.rglob("*.py"):
        if py.relative_to(SRC).parts[:1] in (("ai",), ("bridge",)):
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert n.split(".")[0] not in FORBIDDEN, f"{py.name} imports {n}"
                assert "QtNetwork" not in n and "QtWebEngine" not in n, f"{py.name} imports {n}"


def test_bridge_only_listens_on_loopback():
    from colordoctor.bridge import server

    assert server.HOST == "127.0.0.1"
    text = (SRC / "bridge" / "server.py").read_text(encoding="utf-8")
    assert "0.0.0.0" not in text and "urlopen" not in text and "Access-Control-Allow-Origin" not in text


def test_core_does_not_depend_on_qt():
    for py in (SRC / "core").rglob("*.py"):
        assert "PyQt6" not in py.read_text(encoding="utf-8"), py.name


def test_no_code_writes_image_files():
    """唯一例外：ai/thumbnail.py 在系统临时目录写给 AI 用的缩略图（用户主动点击 AI 点评且勾选发送图片时）。"""
    for py in SRC.rglob("*.py"):
        if py.relative_to(SRC).as_posix() == "ai/thumbnail.py":
            continue
        text = py.read_text(encoding="utf-8")
        assert ".save(" not in text, f"{py.name} 不应保存图像"


def test_24mp_region_analysis_is_fast_enough():
    img = sprinkle(noisy(flat(BLUE, 4000, 6000), 2), YELLOW, 0.02)  # 单个像素的杂色点：按原图尺寸检查，一个都不漏
    t = time.perf_counter()
    r = an.analyze_region(img, synthetic_context())
    dt = time.perf_counter() - t
    assert dt < 10.0, dt
    assert r.stats.total_pixels == 24_000_000
    assert r.stats.sampled_pixels == 24_000_000  # 每个像素都统计（原图尺寸）
    assert np.isclose(r.contamination.opposite_fraction, 0.02, rtol=0.25)


def test_selftest_entry_point(qapp, tmp_path, monkeypatch):
    from colordoctor import app

    log = tmp_path / "selftest.log"
    monkeypatch.setenv("COLORDOCTOR_SELFTEST_LOG", str(log))
    assert app.selftest() == 0
    assert "SELFTEST OK" in log.read_text(encoding="utf-8")


def test_ai_package_is_not_imported_by_default():
    import subprocess
    import sys

    code = (
        "import os; os.environ['QT_QPA_PLATFORM']='offscreen'\n"
        "import colordoctor.ui.main_window, colordoctor.cli, sys\n"
        "print([m for m in sys.modules if m.startswith('colordoctor.ai') or m == 'anthropic'])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out.endswith("[]"), out
