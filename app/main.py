"""
RailSafe AI - Main Entry Point
Aqilliy Temir Yo'l Kesishmalari Monitoring Tizimi
"""

import sys
import os
import logging
import logging.handlers
import threading
import traceback
import faulthandler
from pathlib import Path

# Windowed (console=False) EXE build'da sys.stdout/stderr = None bo'ladi.
# Bu holda faulthandler.enable() va har qanday print() crash qiladi
# ("RuntimeError: sys.stderr is None"). Shuning uchun None oqimlarni
# eng birinchi bo'lib yo'naltiramiz.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

# C-darajadagi crash (segfault) ni stderr ga yozish (stderr endi mavjud)
try:
    faulthandler.enable()
except Exception:
    pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Konsol chiqishini UTF-8 ga o'tkazish — Windows cp1251 konsolida '→', '—',
# emoji kabi belgilar UnicodeEncodeError bermasligi uchun (print xatolari).
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream is not None:
            _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Suppress HEVC/ffmpeg codec warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

# ─── Frozen (.exe) da torch DLL yuklanishini ta'minlash ───────────────
# WinError 1114 (c10.dll init) frozen build'da torch/lib bog'liq DLL'lari
# qidiruv yo'lida bo'lmagani + OpenMP konflikti sabab yuz beradi. Bularni
# torch import'idan OLDIN to'g'irlaymiz.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # ikki OpenMP konflikti
if getattr(sys, "frozen", False):
    _bundle = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    for _sub in ("torch/lib", "_internal/torch/lib", "torch\\lib"):
        _dll_dir = _bundle / _sub
        if _dll_dir.is_dir():
            try:
                os.add_dll_directory(str(_dll_dir))
            except Exception:
                pass

# PyQt6 dan OLDIN torch yuklanishi SHART - Windows da Qt DLL lari
# c10.dll ni ishga tushirishiga to'sqinlik qiladi
try:
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.zeros(1, device='cuda')  # CUDA kontekstini main threadda yaratish
        _torch.cuda.synchronize()
except Exception as _e:
    print(f"[Warning] torch pre-init: {_e}")

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
# QWebEngineWidgets QApplication dan OLDIN import bo'lishi shart
from PyQt6.QtWebEngineWidgets import QWebEngineView as _QWEView  # noqa: F401

from app.pages.main_window import MainWindow

# ─── Logging ──────────────────────────────────────────────
# WARNING darajasi: PLC uzilishi, RTSP qayta ulanishi, stats-push xatolari
# kabi ogohlantirishlar log'ga tushishi uchun (ERROR ularni yashirar edi).
# RotatingFileHandler: log fayli cheksiz o'smasin (5 MB × 3 zaxira).
_log_dir = Path(__file__).parent / "data"
_log_dir.mkdir(exist_ok=True)
_log_handler = logging.handlers.RotatingFileHandler(
    str(_log_dir / "railsafe.log"),
    maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
_log_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.basicConfig(level=logging.WARNING, handlers=[_log_handler])
_logger = logging.getLogger("RailSafe")

# snap7 TCP xatolarini log faylidan yashirish (PLCManager o'z print() dan foydalanadi)
logging.getLogger("snap7").setLevel(logging.CRITICAL)
logging.getLogger("snap7.client").setLevel(logging.CRITICAL)


# ─── Global exception handlers ───────────────────────────
def _global_exception_handler(exc_type, exc_value, exc_tb):
    """Dastur qotib qolmasdan xatolikni log qilish"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    _logger.error(f"Unhandled exception:\n{msg}")
    print(f"[CRITICAL] Unhandled exception:\n{msg}", file=sys.stderr)
    sys.stderr.flush()


def _thread_exception_handler(args):
    """Background thread lardagi xatoliklar"""
    if args.exc_type is SystemExit:
        return
    msg = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    _logger.error(f"Thread '{args.thread.name}' exception:\n{msg}")
    print(f"[THREAD-ERROR] {args.thread.name}:\n{msg}", file=sys.stderr)


sys.excepthook = _global_exception_handler
threading.excepthook = _thread_exception_handler


def main():
    """Main application entry point"""
    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("RailSafe AI")
    app.setOrganizationName("RailSafe AI Team")
    app.setApplicationVersion("1.1.0")

    # Set application style
    app.setStyle("Fusion")

    # Oyna/taskbar ikonasi (logo)
    _icon_path = Path(__file__).parent / "assets" / "images" / "icon_desktop.png"
    if _icon_path.exists():
        app.setWindowIcon(QIcon(str(_icon_path)))

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    # MUHIM (PyInstaller .exe): torch/ultralytics ichki multiprocessing
    # child jarayonlari frozen exe'ni QAYTA ishga tushirib GUI'ni cheksiz
    # ochib yuborishining oldini oladi. Har qanday boshqa importdan oldin.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
