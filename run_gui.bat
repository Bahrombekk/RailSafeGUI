@echo off
REM RailSafe AI - Desktop GUI Launcher for Windows

echo.
echo 🚉 RailSafe AI - Aqilliy Temir Yo'l Kesishmalari
echo ================================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo ⚠️  Virtual environment topilmadi. Yaratilmoqda...
    python -m venv venv
    echo ✅ Virtual environment yaratildi!
)

REM Activate virtual environment
echo 🔧 Virtual environment yoqilmoqda...
call venv\Scripts\activate.bat

REM Install requirements
echo 📦 Kutubxonalar tekshirilmoqda...
pip install -q -r requirements_gui.txt

echo.
echo ▶️  Dastur ishga tushirilmoqda...
echo.

REM Run the application
cd app
python main.py

REM Deactivate virtual environment
call deactivate

pause
