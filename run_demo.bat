@echo off
setlocal
chcp 65001 >nul
title RailSafe AI - NAMOYISH (demo baza)

cd /d "%~dp0"

echo.
echo   RailSafe AI - NAMOYISH REJIMI
echo   =============================
echo.
echo   Statistika NAMOYISH bazasidan olinadi:
echo     build\video\demo_stats.db
echo.
echo   - Haqiqiy app\data\stats.db ga TEGILMAYDI
echo   - Tashqi tizimga (aisup.das-uty.uz) hech narsa YUBORILMAYDI
echo   - Kameralar va aniqlash HAQIQIY ishlaydi
echo.

if not exist ".venv\Scripts\python.exe" (
    echo   [XATO] Virtual muhit topilmadi. Avval: install.bat
    echo.
    pause
    exit /b 1
)

if not exist "build\video\demo_stats.db" (
    echo   [!] Namoyish bazasi topilmadi, yaratilmoqda...
    echo.
    ".venv\Scripts\python.exe" tools\make_demo_data.py --hybrid
    echo.
)

set "RAILSAFE_DB=%~dp0build\video\demo_stats.db"

echo   Dastur ishga tushirilmoqda...  (yozish uchun: Alt+F9)
echo.

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
