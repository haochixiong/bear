#!/usr/bin/env bash
# 在 Mac 上打包 ColorDoctor.app（PyInstaller 不能跨平台：必须在 Mac 上运行）。
# 用法：bash packaging/build_mac.sh
# 输出：dist/ColorDoctor.app 和 dist/ColorDoctor-mac-<arm64|x86_64>.zip（Apple 芯片打出 arm64，Intel 芯片打出 x86_64）
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements-dev.txt "pyinstaller>=6.10"
pip install -e .
# 可选功能：Claude API、RAW 读取、AI 抠皮肤（Mac 用 CPU 版 onnxruntime）
pip install "anthropic>=1.0" "rawpy>=0.19" "onnxruntime>=1.17"

# AI 抠皮肤模型：下载后校验 SHA-256，不一致就删除并停止
MODEL=packaging/models/selfie_multiclass_256x256.onnx
MODEL_SHA=13961350bb2099688eb96182ffe26f24eb12c77b1df278a2ec95bbdc923136e1
mkdir -p packaging/models
if [ ! -f "$MODEL" ]; then
  curl -fL -o "$MODEL" https://huggingface.co/FreeHugsForRobots/ps-skin-seg/resolve/main/selfie_multiclass_256x256.onnx
fi
if [ "$(shasum -a 256 "$MODEL" | cut -d' ' -f1)" != "$MODEL_SHA" ]; then
  rm -f "$MODEL"; echo "模型文件校验失败，已删除"; exit 1
fi

echo "[1/3] Running tests..."
QT_QPA_PLATFORM=offscreen python -m pytest -q

echo "[2/3] Building..."
pyinstaller --noconfirm --clean packaging/colordoctor.spec

echo "[3/3] Self test of the packaged app..."
rm -f selftest.log
COLORDOCTOR_SELFTEST_LOG="$PWD/selftest.log" QT_QPA_PLATFORM=offscreen dist/ColorDoctor.app/Contents/MacOS/ColorDoctor --selftest || true
grep -q "SELFTEST OK" selftest.log || { cat selftest.log; exit 1; }

ARCH=$(uname -m)
rm -f "dist/ColorDoctor-mac-$ARCH.zip"
ditto -c -k --sequesterRsrc --keepParent dist/ColorDoctor.app "dist/ColorDoctor-mac-$ARCH.zip"
echo "Done: dist/ColorDoctor.app, dist/ColorDoctor-mac-$ARCH.zip"
