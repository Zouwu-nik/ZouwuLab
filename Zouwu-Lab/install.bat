@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "ROOT=%~dp0"
rem ---- everything stays inside this folder ----
set "UV_CACHE_DIR=%ROOT%cache\uv"
set "UV_PYTHON_INSTALL_DIR=%ROOT%python"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_LINK_MODE=copy"
set "PYTHONUTF8=1"
set "PIP_CACHE_DIR=%ROOT%cache\pip"

chcp 65001 >nul

echo.
echo ==== Zouwu Lab - installation into: %ROOT%
echo      Any disk works (C:, D:, external SSD). Everything stays inside this folder.
echo.

rem ---- can we write here? (Program Files and some system folders need admin rights) ----
echo test>"%ROOT%.write_test" 2>nul
if not exist "%ROOT%.write_test" goto noaccess
del "%ROOT%.write_test" >nul 2>&1

if not exist "tools\uv.exe" (
  echo [1/4] Downloading uv package manager...
  if not exist tools mkdir tools
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile 'tools\uv.zip'; Expand-Archive 'tools\uv.zip' 'tools\uvtmp' -Force; Get-ChildItem 'tools\uvtmp' -Recurse -Filter '*.exe' | Move-Item -Destination 'tools' -Force; Remove-Item 'tools\uv.zip','tools\uvtmp' -Recurse -Force"
  if errorlevel 1 goto fail
)

echo [2/4] Local Python 3.12...
"tools\uv.exe" python install 3.12
if errorlevel 1 goto fail

if not exist "venv\Scripts\python.exe" (
  "tools\uv.exe" venv "venv" --python 3.12
  if errorlevel 1 goto fail
)

echo [3/4] PyTorch, engine and libraries...
"venv\Scripts\python.exe" "app\setup.py" %*
if errorlevel 1 goto fail

rem ---- downloaded package archives are no longer needed (saves ~3 GB) ----
"tools\uv.exe" cache clean >nul 2>&1

echo.
echo ==== Done. Run "start.bat"
echo.
pause
exit /b 0

:noaccess
echo ERROR: no permission to write into this folder:
echo        "%ROOT%"
echo Move the ZouwuLab folder to a place you own, for example C:\ZouwuLab
echo or run install.bat as administrator.
:fail
echo.
echo ==== INSTALLATION FAILED. See messages above.
echo.
pause
exit /b 1
