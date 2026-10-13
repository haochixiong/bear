# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置：pyinstaller --noconfirm --clean packaging/colordoctor.spec
# 生成 dist/ColorDoctor/ColorDoctor(.exe)（onedir：启动快，杀毒误报少）

import os

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
