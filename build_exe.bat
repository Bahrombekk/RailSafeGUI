@echo off
setlocal
chcp 65001 >nul
title RailSafe AI - EXE yaratish

cd /d "%~dp0"

echo.
echo   RailSafe AI - Mustaqil EXE yaratish (PyInstaller)
echo   =================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo   [XATO] Avval install.bat ni ishga tushiring.
    pause
    exit /b 1
)

echo   PyInstaller o'rnatilmoqda (agar yo'q bo'lsa)...
".venv\Scripts\python.exe" -m pip install --upgrade pyinstaller -q

echo   Build boshlandi (bir necha daqiqa)...
".venv\Scripts\pyinstaller.exe" RailSafeAI.spec --noconfirm --clean
if %errorlevel% neq 0 (
    echo   [XATO] Build muvaffaqiyatsiz.
    pause
    exit /b 1
)

echo.
echo   [OK] Tayyor:  dist\RailSafeAI\RailSafeAI.exe
echo   Windows setup.exe yaratish uchun:  installer.iss (Inno Setup)
echo.
pause
