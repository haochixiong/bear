# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置：pyinstaller --noconfirm --clean packaging/colordoctor.spec
# Windows：生成 dist/ColorDoctor/ColorDoctor.exe（onedir：启动快，杀毒误报少）
# macOS：另外生成 dist/ColorDoctor.app（PyInstaller 不能跨平台打包：Mac 版要在 Mac 上打，见 build_mac.sh）

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(ROOT, "src")

# AI 抠皮肤模型（可选）：build_windows.bat 会下载到 packaging/models/；没有就不打包，程序自动退回肤色规则
MODEL = os.path.join(SPECPATH, "models", "selfie_multiclass_256x256.onnx")
EXTRA = [(MODEL, "models")] if os.path.isfile(MODEL) else []

a = Analysis(
    [os.path.join(SPECPATH, "run_colordoctor.py")],
    pathex=[SRC],
    binaries=[],
    datas=[
        (os.path.join(SRC, "colordoctor", "data", "skin_presets.json"), "colordoctor/data"),
        (os.path.join(SRC, "colordoctor", "data", "grading_styles.json"), "colordoctor/data"),
    ] + EXTRA,
    hiddenimports=["colordoctor.data"],
    excludes=[
        "tkinter",
        "matplotlib",
        "scipy",
        "sklearn",
        "pandas",
        "PyQt6.QtNetwork",
        "PyQt6.QtWebEngineCore",
        "PyQt6.QtWebEngineWidgets",
        "PyQt6.QtQml",
        "PyQt6.QtQuick",
        "PyQt6.QtMultimedia",
        "PyQt6.QtBluetooth",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ColorDoctor",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ColorDoctor")

if sys.platform == "darwin":
    with open(os.path.join(SRC, "colordoctor", "__init__.py"), encoding="utf-8") as f:
        VERSION = re.search(r'__version__ = "([^"]+)"', f.read()).group(1)
    app = BUNDLE(
        coll,
        name="ColorDoctor.app",
        icon=None,
        bundle_identifier="com.colordoctor.app",
        info_plist={
            "CFBundleName": "ColorDoctor",
            "CFBundleDisplayName": "ColorDoctor",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSMinimumSystemVersion": "11.0",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
        },
    )
