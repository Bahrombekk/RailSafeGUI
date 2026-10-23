"""
RailSafe AI — tanishtiruv roliki uchun dastur ekranini yozish.

Dasturni ishga tushiradi, sahifalar bo'ylab avtomatik "ekskursiya" qiladi va
oynani ICHKARIDAN (QWidget.grab) 1920x1080 video qilib yozadi. Ish stolidagi
boshqa oynalar kadrga tushmaydi, sichqoncha kerak emas.

    .venv/Scripts/python.exe tools/demo_record.py

Natija:
    build/video/raw_app.mp4     — xom yozuv
    build/video/scenes.json     — qaysi soniyada qaysi sahifa (montaj uchun)
"""

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# app.main importi torch/OpenCV muhitini to'g'ri sozlaydi (GUI ochmaydi)
import app.main  # noqa: F401,E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtGui import QIcon, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.pages.main_window import MainWindow  # noqa: E402

OUT_DIR = ROOT / "build" / "video"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H, FPS = 1920, 1080, 15

# Ekskursiya: (soniya, sahifa nomi)
# Birinchi 20 soniya — kameralar ulanishi va detektor yuklanishi (montajda kesiladi)
TOUR = [
    (20, "dashboard"),
    (38, "detail"),
    (56, "analytics"),
    (86, "about"),
    (96, "dashboard_end"),
]
TOTAL = 102

# --dashboard: faqat boshqaruv paneli (kameralar isib ulgursin), qayta dubl uchun
if "--dashboard" in sys.argv:
    TOUR = []
    TOTAL = 75
    OUT_PATH = OUT_DIR / "raw_dash.mp4"
    SCENES_PATH = OUT_DIR / "scenes_dash.json"
else:
    OUT_PATH = OUT_DIR / "raw_app.mp4"
    SCENES_PATH = OUT_DIR / "scenes.json"


class Recorder:
    """Kadrlar GUI oqimida FAQAT olinadi, navbatga (bufer) tushadi va alohida
    oqimda kodlanadi — shu bois dastur yozuv paytida qotib qolmaydi."""

    def __init__(self, win):
        self.win = win
        self.writer = cv2.VideoWriter(
            str(OUT_PATH), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (W, H))
        if not self.writer.isOpened():
            raise RuntimeError("VideoWriter ochilmadi")
        self.t0 = time.monotonic()
        self.written = 0
        self.dropped = 0
        self.scenes = []
        # ~240 kadr (16 s) bufer: qisqa sekinlashuvlar kadr yo'qotmaydi
        self.q = queue.Queue(maxsize=240)
        self.thread = threading.Thread(target=self._encode_loop,
                                       name="DemoEncoder", daemon=True)
        self.thread.start()

    def _encode_loop(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            raw, w, h, stride, repeat = item
            arr = np.frombuffer(raw, np.uint8).reshape(h, stride)
            arr = arr[:, : w * 3].reshape(h, w, 3)
            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if (w, h) != (W, H):
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            for _ in range(repeat):
                self.writer.write(frame)

    def mark(self, name):
        t = time.monotonic() - self.t0
        self.scenes.append({"t": round(t, 2), "scene": name})
        print(f"[{t:6.1f}s] {name}")

    def tick(self):
        """Kadrni olib navbatga qo'yadi. Haqiqiy vaqtga moslash uchun
        timer kechikkan bo'lsa kadr necha marta takrorlanishi ham uzatiladi."""
        target = int((time.monotonic() - self.t0) * FPS)
        repeat = target - self.written + 1
        if repeat <= 0:
            return
        img = self.win.grab().toImage().convertToFormat(
            QImage.Format.Format_RGB888)
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        payload = (bytes(ptr), img.width(), img.height(),
                   img.bytesPerLine(), repeat)
        try:
            self.q.put_nowait(payload)
            self.written += repeat
        except queue.Full:
            self.dropped += repeat          # bufer to'lgan — kadr tashlanadi

    def close(self):
        self.q.put(None)
        self.thread.join()
        self.writer.release()
        SCENES_PATH.write_text(
            json.dumps({"fps": FPS, "size": [W, H],
                        "duration": round(self.written / FPS, 2),
                        "scenes": self.scenes},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nYozildi: {OUT_PATH}  ({self.written} kadr, "
              f"{self.written / FPS:.1f} s, tashlangan kadr: {self.dropped})")


def smooth_scroll(scroll_area, duration_ms=22000):
    """Analitika sahifasini sekin pastga aylantirish (grafiklar ko'rinsin)."""
    bar = scroll_area.verticalScrollBar()
    steps = max(1, duration_ms // 40)
    state = {"i": 0}

    def step():
        bar_max = bar.maximum()
        state["i"] += 1
        bar.setValue(int(bar_max * min(1.0, state["i"] / steps)))
        if state["i"] < steps:
            QTimer.singleShot(40, step)

    QTimer.singleShot(40, step)


def main():
    qapp = QApplication(sys.argv)
    qapp.setApplicationName("RailSafe AI")
    qapp.setStyle("Fusion")
    icon = ROOT / "app" / "assets" / "images" / "icon_desktop.png"
    if icon.exists():
        qapp.setWindowIcon(QIcon(str(icon)))

    win = MainWindow()

    # MUHIM: yozuv (ayniqsa demo baza) paytida tashqi tizimga statistika
    # YUBORILMASIN — aks holda hamkor serveriga namuna raqamlar tushadi.
    for attr in ("stats_push", "stats_api"):
        obj = getattr(win, attr, None)
        if obj is not None:
            try:
                obj.stop()
                print(f"[yozuv] {attr} to'xtatildi")
            except Exception as e:
                print(f"[yozuv] {attr} to'xtatilmadi: {e}")

    # Rolik uchun aniq 16:9 — maximized (2752x1152) emas
    win._set_maximized(False)
    screen = qapp.primaryScreen().availableGeometry()
    win.setGeometry(screen.x() + max(0, (screen.width() - W) // 2),
                    screen.y() + max(0, (screen.height() - H) // 2), W, H)
    win.setFixedSize(W, H)
    win.show()
    win.raise_()

    rec = Recorder(win)
    rec.mark("start")

    timer = QTimer()
    timer.timeout.connect(rec.tick)
    timer.start(int(1000 / FPS))

    def go(name):
        if name == "dashboard" or name == "dashboard_end":
            win._show_dashboard()
        elif name == "detail":
            crossings = win.config_manager.get_crossings()
            if crossings:
                win._show_crossing_detail(crossings[0]["id"])
        elif name == "analytics":
            win._show_analytics()
            # Sahifa _show_analytics ichida yaratiladi — stackdan olamiz
            page = win.stacked_widget.currentWidget()
            if hasattr(page, "scroll"):
                QTimer.singleShot(2500, lambda p=page: smooth_scroll(p.scroll))
        elif name == "about":
            win._show_about()
        rec.mark(name)

    for secs, name in TOUR:
        QTimer.singleShot(secs * 1000, lambda n=name: go(n))

    def finish():
        timer.stop()
        rec.close()
        win.close()
        qapp.quit()

    QTimer.singleShot(TOTAL * 1000, finish)
    qapp.exec()


if __name__ == "__main__":
    main()
