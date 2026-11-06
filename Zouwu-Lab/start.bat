@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
if not exist "venv\Scripts\python.exe" (
  echo Run install.bat first.
  pause
  exit /b 1
)
set "PYTHONUTF8=1"
set "HF_HOME=%~dp0cache\huggingface"
set "TRITON_CACHE_DIR=%~dp0cache\triton"
set "TORCHINDUCTOR_CACHE_DIR=%~dp0cache\inductor"
title Zouwu Lab
"venv\Scripts\python.exe" "app\server.py" %*
pause
