@echo off
setlocal
chcp 65001 >nul
title RailSafe AI

cd /d "%~dp0"

echo.
echo   RailSafe AI - Aqilliy Temir Yo'l Kesishmalari
echo   ============================================
echo.

REM Virtual muhit tekshirish
if not exist ".venv\Scripts\python.exe" (
    echo   [XATO] Virtual muhit topilmadi.
    echo          Avval o'rnatuvchini ishga tushiring:  install.bat
    echo.
    pause
    exit /b 1
)

echo   Dastur ishga tushirilmoqda...
echo.

REM Dasturni ishga tushirish (paket sifatida - import yo'llari to'g'ri bo'lishi uchun)
".venv\Scripts\python.exe" -m app.main
set "EXITCODE=%errorlevel%"

if not "%EXITCODE%"=="0" (
    echo.
    echo   [!] Dastur xatolik bilan yakunlandi (kod: %EXITCODE%).
    echo       Batafsil: app\data\railsafe.log
    echo.
    pause
)

exit /b %EXITCODE%
