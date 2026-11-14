@echo off
rem Author: bd78
rem Creates the virtual environment (.venv) and installs the dependencies.
rem Usage:  setup.bat
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PYTHON="
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON=python"
) else (
    where py >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "PYTHON=py"
    )
)

if "%PYTHON%"=="" (
    echo Python 3.10+ was not found on PATH. Install it from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PYTHON% -m venv .venv
    if errorlevel 1 exit /b %errorlevel%
)

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

echo Installing dependencies from requirements.txt ...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 exit /b %errorlevel%

"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 exit /b %errorlevel%

"%VENV_PYTHON%" -c "import cv2, numpy, PIL, tkinter; print('OK: OpenCV', cv2.__version__, '- NumPy', numpy.__version__, '- Pillow', PIL.__version__)"
echo Setup complete. Start the application with run.bat or .\run.ps1
