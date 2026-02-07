#!/bin/bash
# RailSafe AI - Desktop GUI Launcher

echo "RailSafe AI - Aqilliy Temir Yo'l Kesishmalari"
echo "================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Virtual environment topilmadi. Yaratilmoqda..."
    python3 -m venv "$SCRIPT_DIR/venv"
    echo "Virtual environment yaratildi!"
fi

echo "Python: $($VENV_PYTHON --version)"
echo ""
echo "Dastur ishga tushirilmoqda..."
echo ""

# Run from gui/ directory (gui_config.json is there)
cd "$SCRIPT_DIR/gui"
exec "$VENV_PYTHON" main.py
