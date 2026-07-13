@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title RailSafe AI - O'rnatish

echo.
echo ============================================================
echo   RailSafe AI - Aqilliy Temir Yo'l Kesishmalari Monitoring
echo   O'rnatuvchi (Installer)
echo ============================================================
echo.

cd /d "%~dp0"

REM ── 1. Python 3.10+ tekshirish ──────────────────────────────
echo [1/5] Python tekshirilmoqda...
set "PYEXE="
for %%P in (py python) do (
    if not defined PYEXE (
        %%P -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,10) else 1)" >nul 2>&1
        if !errorlevel! equ 0 set "PYEXE=%%P"
    )
)
if not defined PYEXE (
    echo   [XATO] Python 3.10 yoki yuqorisi topilmadi.
    echo          https://www.python.org/downloads/ dan o'rnating
    echo          va o'rnatishda "Add Python to PATH" ni belgilang.
    pause
    exit /b 1
)
for /f "delims=" %%V in ('%PYEXE% -c "import sys;print('.'.join(map(str,sys.version_info[:3])))"') do set "PYVER=%%V"
echo   [OK] Python !PYVER! topildi (%PYEXE%)

REM ── 2. Virtual muhit (.venv) yaratish ──────────────────────
echo [2/5] Virtual muhit (.venv) tayyorlanmoqda...
if not exist ".venv\Scripts\python.exe" (
    %PYEXE% -m venv .venv
    if !errorlevel! neq 0 (
        echo   [XATO] .venv yaratib bo'lmadi.
        pause
        exit /b 1
    )
    echo   [OK] .venv yaratildi
) else (
    echo   [OK] .venv allaqachon mavjud
)
set "VPY=.venv\Scripts\python.exe"

REM ── 3. pip yangilash ───────────────────────────────────────
echo [3/5] pip yangilanmoqda...
"%VPY%" -m pip install --upgrade pip setuptools wheel -q

REM ── 4. GPU aniqlash va kutubxonalarni o'rnatish ────────────
echo [4/5] GPU aniqlanmoqda...
where nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    nvidia-smi >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [OK] NVIDIA GPU topildi - GPU versiyasi o'rnatiladi
        set "REQ=requirements-gpu.txt"
    ) else (
        echo   [!] nvidia-smi ishlamadi - CPU versiyasi o'rnatiladi
        set "REQ=requirements-cpu.txt"
    )
) else (
    echo   [!] NVIDIA GPU topilmadi - CPU versiyasi o'rnatiladi
    set "REQ=requirements-cpu.txt"
)

echo   Kutubxonalar o'rnatilmoqda (!REQ!) - bu bir necha daqiqa olishi mumkin...
"%VPY%" -m pip install -r !REQ!
if !errorlevel! neq 0 (
    echo   [XATO] Kutubxonalarni o'rnatishda xatolik.
    pause
    exit /b 1
)
echo   [OK] Kutubxonalar o'rnatildi

REM ── 5. Import tekshiruvi ───────────────────────────────────
echo [5/5] O'rnatish tekshirilmoqda...
"%VPY%" -c "import PyQt6, cv2, numpy, docx, yaml, snap7, av; from PyQt6.QtWebEngineWidgets import QWebEngineView; print('  [OK] Barcha asosiy kutubxonalar ishlayapti')"
if !errorlevel! neq 0 (
    echo   [OGOHLANTIRISH] Ba'zi kutubxonalar import bo'lmadi - yuqoridagi xatoga qarang.
)

echo.
echo ============================================================
echo   O'rnatish tugadi!
echo   Dasturni ishga tushirish uchun:  run_gui.bat
echo ============================================================
echo.

REM Ish stoliga yorliq yaratishni taklif qilish
set /p MAKESHORTCUT="Ish stoliga yorliq yaratilsinmi? (H/y): "
if /i "!MAKESHORTCUT!"=="y" call :make_shortcut
if /i "!MAKESHORTCUT!"=="H" call :make_shortcut
if /i "!MAKESHORTCUT!"=="" call :make_shortcut

pause
exit /b 0

:make_shortcut
powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$sc = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\RailSafe AI.lnk');" ^
  "$sc.TargetPath = '%~dp0run_gui.bat';" ^
  "$sc.WorkingDirectory = '%~dp0';" ^
  "$sc.IconLocation = '%~dp0app\assets\icons\train.svg';" ^
  "$sc.Description = 'RailSafe AI Monitoring';" ^
  "$sc.Save()" >nul 2>&1
echo   [OK] Ish stolida "RailSafe AI" yorlig'i yaratildi
exit /b 0
