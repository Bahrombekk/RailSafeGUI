"""
RailSafe AI - Main Entry Point
Aqilliy Temir Yo'l Kesishmalari Monitoring Tizimi
"""

import sys
import os
import logging
import threading
import traceback
import faulthandler
from pathlib import Path

# C-darajadagi crash (segfault) ni stderr ga yozish
faulthandler.enable()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress HEVC/ffmpeg codec warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

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

from gui.ui.main_window import MainWindow

# ─── Logging ──────────────────────────────────────────────
_log_dir = Path(__file__).parent / "data"
_log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=str(_log_dir / "railsafe.log"),
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_logger = logging.getLogger("RailSafe")


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
    app.setApplicationVersion("1.0.0")

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
