@echo off
REM Build ColorDoctor for Windows (run from any folder).
REM Output: dist\ColorDoctor\ColorDoctor.exe
setlocal
cd /d "%~dp0\.."

if not exist .venv (
    py -3.13 -m venv .venv 2>nul || python -m venv .venv
)
call .venv\Scripts\activate.bat || exit /b 1

python -m pip install --upgrade pip || exit /b 1
pip install -r requirements-dev.txt "pyinstaller>=6.10" || exit /b 1
pip install -e . || exit /b 1
rem 可选功能：Claude API、RAW 读取、AI 抠皮肤（显卡加速用 DirectML 版 onnxruntime）
pip install "anthropic>=1.0" "rawpy>=0.19" "onnxruntime-directml>=1.17" || exit /b 1
rem AI 抠皮肤模型：下载后校验 SHA-256，不一致就删除并停止（不会把错误页面打包进去）
set MODEL=packaging\models\selfie_multiclass_256x256.onnx
set MODEL_SHA=13961350bb2099688eb96182ffe26f24eb12c77b1df278a2ec95bbdc923136e1
if not exist packaging\models mkdir packaging\models
if not exist %MODEL% (
  curl -fL -o %MODEL% https://huggingface.co/FreeHugsForRobots/ps-skin-seg/resolve/main/selfie_multiclass_256x256.onnx || (del /q %MODEL% 2>nul & echo 模型下载失败 & exit /b 1)
)
powershell -NoProfile -Command "if ((Get-FileHash -Algorithm SHA256 '%MODEL%').Hash.ToLower() -ne '%MODEL_SHA%') { Remove-Item '%MODEL%'; exit 1 }" || (echo 模型文件校验失败，已删除 & exit /b 1)

echo [1/3] Running tests...
python -m pytest -q || exit /b 1

echo [2/3] Building...
pyinstaller --noconfirm --clean packaging\colordoctor.spec || exit /b 1

echo [3/3] Self test of the packaged app...
if exist selftest.log del selftest.log
start "" /wait dist\ColorDoctor\ColorDoctor.exe --selftest
findstr /c:"SELFTEST OK" selftest.log >nul || (type selftest.log & exit /b 1)

echo Done: dist\ColorDoctor\ColorDoctor.exe
endlocal
