#!/bin/bash
# RailSafe AI - Desktop GUI Launcher

echo "🚉 RailSafe AI - Aqilliy Temir Yo'l Kesishmalari"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment topilmadi. Yaratilmoqda..."
    python3 -m venv venv
    echo "✅ Virtual environment yaratildi!"
fi

# Activate virtual environment
echo "🔧 Virtual environment yoqilmoqda..."
source venv/bin/activate

# Check if requirements are installed
echo "📦 Kutubxonalar tekshirilmoqda..."
pip install -q -r requirements_gui.txt

echo ""
echo "▶️  Dastur ishga tushirilmoqda..."
echo ""

# Run the application
cd gui
python3 main.py

# Deactivate virtual environment on exit
deactivate
