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
rem 可选：RAW 读取与 AI 抠皮肤（显卡加速用 DirectML 版 onnxruntime）
pip install "rawpy>=0.19" "onnxruntime-directml>=1.17"
if not exist packaging\models mkdir packaging\models
if not exist packaging\models\selfie_multiclass_256x256.onnx (
  curl -L -o packaging\models\selfie_multiclass_256x256.onnx https://huggingface.co/FreeHugsForRobots/ps-skin-seg/resolve/main/selfie_multiclass_256x256.onnx
)

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
