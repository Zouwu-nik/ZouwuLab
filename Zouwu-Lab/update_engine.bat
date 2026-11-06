@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "UV_CACHE_DIR=%~dp0cache\uv"
set "UV_PYTHON_INSTALL_DIR=%~dp0python"
set "UV_LINK_MODE=copy"
set "PYTHONUTF8=1"
echo Close Zouwu Lab (the black window) before updating the engine.
pause
rem Re-downloads the engine version set in app\setup.py (ENGINE_VERSION). Models are not touched.
"venv\Scripts\python.exe" "app\setup.py" --update-engine %*
pause
