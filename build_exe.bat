@echo off
setlocal EnableExtensions
title Krea2 SVD LoRA Merger - EXE Builder

rem Build a standalone Windows EXE from the current source.
rem Run this file from the repository root, using a Python environment that has
rem torch, safetensors, and ttkbootstrap installed.

set "SCRIPT=Krea2_SVD_LoRA_Merger.py"
set "APP_NAME=Krea2_SVD_LoRA_Merger"

if not exist "%SCRIPT%" (
    echo ERROR: %SCRIPT% was not found.
    echo Please run build_exe.bat from the repository root folder.
    pause
    exit /b 1
)

python --version
if errorlevel 1 (
    echo ERROR: Python is not available on PATH.
    pause
    exit /b 1
)

echo.
echo Checking PyInstaller...
python -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    python -m pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Could not install PyInstaller.
        pause
        exit /b 1
    )
)

echo.
echo Building one-file Windows application...
echo This can take several minutes. The EXE may be large because PyTorch is bundled.
echo.

python -m PyInstaller --noconfirm --clean --onefile --windowed ^
    --name "%APP_NAME%" ^
    --collect-all torch ^
    --collect-all safetensors ^
    --collect-all ttkbootstrap ^
    --collect-all numpy ^
    --hidden-import tkinter ^
    --hidden-import tkinter.ttk ^
    --hidden-import tkinter.filedialog ^
    --hidden-import tkinter.messagebox ^
    "%SCRIPT%"

if errorlevel 1 (
    echo.
    echo BUILD FAILED. See the messages above.
    pause
    exit /b 1
)

echo.
echo BUILD SUCCESSFUL:
echo dist\%APP_NAME%.exe
echo.
echo Test the EXE on this PC first. Keep the original Python script and source LoRAs.
pause
endlocal
