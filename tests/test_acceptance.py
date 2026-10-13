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


def test_no_network_modules_are_imported_anywhere():
    for py in SRC.rglob("*.py"):
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


def test_core_does_not_depend_on_qt():
    for py in (SRC / "core").rglob("*.py"):
        assert "PyQt6" not in py.read_text(encoding="utf-8"), py.name


def test_no_code_writes_image_files():
    for py in SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert ".save(" not in text, f"{py.name} 不应保存图像"


def test_24mp_region_analysis_is_fast_enough():
    img = sprinkle(noisy(flat(BLUE, 4000, 6000), 2), YELLOW, 0.02)
    t = time.perf_counter()
    r = an.analyze_region(img, synthetic_context())
    dt = time.perf_counter() - t
    assert dt < 10.0, dt
    assert r.stats.total_pixels == 24_000_000
    assert r.stats.sampled_pixels <= 262_144
    assert np.isclose(r.contamination.opposite_fraction, 0.02, rtol=0.25)
