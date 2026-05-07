"""
VideoRecorder - lightweight background video writer.
Non-blocking: frames are queued and written in a separate thread.
VideoWriter lazy init: actual resolution detected from first frame.
"""

import queue
import threading
import re
from datetime import datetime
from pathlib import Path

import cv2


def _safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', '_', name).strip()


def get_record_dir() -> Path:
    return Path.home() / "Desktop" / "RailSafe_Yozuvlar"


def _open_writer(filepath: str, fps: float, w: int, h: int) -> tuple[cv2.VideoWriter, str]:
    """VideoWriter ochish: H264 → mp4v → XVID fallback zanjiri."""
    size = (w, h)

    # H.264 — eng yaxshi sifat/hajm nisbati
    for fourcc_str, ext in [("avc1", ".mp4"), ("X264", ".mp4"), ("mp4v", ".mp4"), ("XVID", ".avi")]:
        path = filepath if filepath.endswith(ext) else (filepath.rsplit(".", 1)[0] + ext)
        writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*fourcc_str), fps, size)
        if writer.isOpened():
            return writer, path
        writer.release()

    return None, filepath


class VideoRecorder:
    """
    Thread-safe, non-blocking video recorder.
    VideoWriter ochilishi birinchi frame kelguncha kechiktiriladi —
    shu tarzda haqiqiy frame o'lchami avtomatik aniqlanadi.
    """

    _SENTINEL = object()

    def __init__(self, queue_size: int = 120):
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self.is_recording: bool = False
        self._filepath_base: str = ""
        self._filepath: str = ""
        self._fps: float = 25.0

    # ── Public API ─────────────────────────────────────────────────────

    def start(self, crossing_name: str, camera_name: str, fps: float = 25.0) -> None:
        """Yozishni tayyorlaydi. VideoWriter birinchi frame da ochiladi."""
        if self.is_recording:
            return

        now = datetime.now()
        safe_cross = _safe_name(crossing_name)
        safe_cam = _safe_name(camera_name)

        folder = get_record_dir() / safe_cross / now.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)

        base = str(folder / f"{now.strftime('%H-%M-%S')}_{safe_cross}_{safe_cam}.mp4")
        self._filepath_base = base
        self._filepath = base
        self._fps = max(1.0, float(fps))
        self.is_recording = True

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def write(self, frame) -> None:
        """Frame ni navbatga qo'shish (non-blocking, drop if full)."""
        if not self.is_recording:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass

    def stop(self) -> None:
        """Yozishni to'xtatish va faylni yopish."""
        if not self.is_recording:
            return
        self.is_recording = False
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            self._queue.put(self._SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=8.0)
            self._thread = None
        print(f"[VideoRecorder] Saqlandi: {self._filepath}")

    # ── Internal ────────────────────────────────────────────────────────

    def _worker(self) -> None:
        """Background writer thread. VideoWriter birinchi frame da ochiladi."""
        writer: cv2.VideoWriter | None = None
        fps = self._fps

        try:
            while True:
                try:
                    frame = self._queue.get(timeout=1.0)
                except queue.Empty:
                    if not self.is_recording:
                        break
                    continue

                if frame is self._SENTINEL:
                    break

                # Lazy init — birinchi frame dan haqiqiy o'lchamni ol
                if writer is None:
                    h, w = frame.shape[:2]
                    writer, self._filepath = _open_writer(self._filepath_base, fps, w, h)
                    if writer is None:
                        print(f"[VideoRecorder] VideoWriter ochmadi: {self._filepath_base}")
                        break
                    print(f"[VideoRecorder] Boshlandi ({w}x{h} @ {fps:.0f}fps): {self._filepath}")

                writer.write(frame)

        finally:
            if writer:
                writer.release()
