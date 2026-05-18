"""
VideoRecorder - production-grade background video writer.

Features:
- Non-blocking frame submission (queue + background thread).
- Format selection: mp4 / avi / mkv (codec chosen per format).
- Quality selection: 720p / 1080p / original (resize on the fly).
- Auto file rotation when segment exceeds max_duration_sec.
- Disk space check before start and periodically during recording.
- Frame drop counter + codec fallback warning.
- VideoWriter init is lazy — actual size detected from first frame.
"""

import os
import queue
import shutil
import sys
import threading
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import cv2


# Codec preference per output format. First entry that opens wins.
# Windows-da FFmpeg ko'pincha X264 bilan kelmaydi, shuning uchun MKV
# uchun ham mp4v ni birinchi sinab ko'ramiz (universal ishlaydi).
_CODEC_CHAIN = {
    "mp4": [("avc1", ".mp4"), ("H264", ".mp4"), ("mp4v", ".mp4")],
    "avi": [("XVID", ".avi"), ("MJPG", ".avi")],
    "mkv": [("mp4v", ".mkv"), ("avc1", ".mkv"), ("H264", ".mkv"), ("XVID", ".mkv")],
}


@contextmanager
def _suppress_native_stderr():
    """OpenCV/FFmpeg ning stderr ga yozadigan xatolarini vaqtincha jimlatamiz.
    Codec mavjud emasligi haqida shovqin yozmasligi uchun."""
    try:
        sys.stderr.flush()
    except Exception:
        pass
    saved_fd = -1
    devnull_fd = -1
    try:
        saved_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        yield
    except OSError:
        # fd manipulation ishlamasa, original stderr da qoldiramiz
        yield
    finally:
        if saved_fd >= 0:
            try:
                os.dup2(saved_fd, 2)
                os.close(saved_fd)
            except OSError:
                pass
        if devnull_fd >= 0:
            try:
                os.close(devnull_fd)
            except OSError:
                pass

# Quality → max height (None means keep original resolution)
_QUALITY_HEIGHT = {
    "720p": 720,
    "1080p": 1080,
    "original": None,
}


def _safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', '_', name).strip()


def get_record_dir() -> Path:
    return Path.home() / "Desktop" / "RailSafe_Yozuvlar"


def _free_disk_mb(path: Path) -> int:
    """Return free disk space in MB at path (or its parent). -1 if unknown."""
    try:
        check = path if path.exists() else path.parent
        usage = shutil.disk_usage(check)
        return usage.free // (1024 * 1024)
    except Exception:
        return -1


class VideoRecorder:
    """Thread-safe, non-blocking video recorder with rotation & disk safety."""

    _SENTINEL = object()

    def __init__(
        self,
        fmt: str = "mp4",
        quality: str = "original",
        max_duration_sec: int = 1800,        # 30 minutes per segment
        min_free_disk_mb: int = 500,          # require >=500 MB free
        queue_size: int = 120,
    ):
        self.fmt = fmt if fmt in _CODEC_CHAIN else "mp4"
        self.quality = quality if quality in _QUALITY_HEIGHT else "original"
        self.max_duration_sec = max(60, int(max_duration_sec))
        self.min_free_disk_mb = max(100, int(min_free_disk_mb))

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self.is_recording: bool = False

        self._folder: Path = get_record_dir()
        self._base_name: str = ""
        self._fps: float = 25.0
        self._segment_idx: int = 0

        # Telemetry
        self._dropped_frames: int = 0
        self._written_frames: int = 0
        self.last_error: str = ""
        self.last_codec_fell_back: bool = False
        self.last_path: str = ""

    # ── Public API ─────────────────────────────────────────────────────

    def start(self, crossing_name: str, camera_name: str, fps: float = 25.0) -> bool:
        """Yozishni boshlash. False qaytsa — disk to'la yoki boshqa xato."""
        if self.is_recording:
            return False

        now = datetime.now()
        safe_cross = _safe_name(crossing_name) or "crossing"
        safe_cam = _safe_name(camera_name) or "camera"

        folder = get_record_dir() / safe_cross / now.strftime("%Y-%m-%d")
        free = _free_disk_mb(folder)
        if 0 <= free < self.min_free_disk_mb:
            self.last_error = (
                f"Disk to'la: {free}MB qoldi, "
                f"minimum {self.min_free_disk_mb}MB kerak")
            print(f"[VideoRecorder] {self.last_error}")
            return False

        self._folder = folder
        ts = now.strftime("%H-%M-%S")
        self._base_name = f"{ts}_{safe_cross}_{safe_cam}"
        # Clamp to sane FPS range
        self._fps = max(5.0, min(60.0, float(fps)))
        self._segment_idx = 0
        self._dropped_frames = 0
        self._written_frames = 0
        self.last_error = ""
        self.last_codec_fell_back = False
        self.is_recording = True

        self._thread = threading.Thread(
            target=self._worker, daemon=True, name=f"vid-rec-{safe_cam}")
        self._thread.start()
        return True

    def write(self, frame) -> None:
        """Non-blocking frame submission. Drops on backpressure (queue full)."""
        if not self.is_recording:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._dropped_frames += 1

    def stop(self) -> None:
        """Yozishni to'xtatish va background thread ni kutish."""
        if not self.is_recording:
            return
        self.is_recording = False
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            try:
                self._queue.put(self._SENTINEL, timeout=2.0)
            except queue.Full:
                pass
        if self._thread is not None:
            self._thread.join(timeout=8.0)
            self._thread = None
        if self._dropped_frames > 0:
            print(f"[VideoRecorder] {self._base_name}: "
                  f"yozildi={self._written_frames}, "
                  f"tushib qoldi={self._dropped_frames}")

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def written_frames(self) -> int:
        return self._written_frames

    # ── Internal ────────────────────────────────────────────────────────

    def _resize_frame(self, frame):
        """Quality sozlamasiga ko'ra frame ni kichraytirish."""
        max_h = _QUALITY_HEIGHT.get(self.quality)
        if max_h is None:
            return frame
        h, w = frame.shape[:2]
        if h <= max_h:
            return frame
        scale = max_h / h
        # Round to even — ko'p codec lar juft son talab qiladi
        new_w = (int(round(w * scale)) // 2) * 2
        new_h = (int(round(h * scale)) // 2) * 2
        if new_w <= 0 or new_h <= 0:
            return frame
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _open_writer(self, w: int, h: int):
        """Returns (writer, actual_path, fourcc_str, fell_back)."""
        requested = self.fmt
        order = [requested]
        for fb in ("mp4", "avi"):
            if fb not in order:
                order.append(fb)

        if self._segment_idx == 0:
            name = self._base_name
        else:
            name = f"{self._base_name}_part{self._segment_idx + 1}"
        base_path = str(self._folder / name)
        size = (w, h)

        # Codec probing — har bir variant uchun stderr-ni jimlatamiz.
        # FFmpeg "Failed to initialize VideoWriter" shovqini chiqarmasligi uchun.
        with _suppress_native_stderr():
            for fmt_name in order:
                for fourcc_str, ext in _CODEC_CHAIN[fmt_name]:
                    path = base_path + ext
                    writer = cv2.VideoWriter(
                        path, cv2.VideoWriter.fourcc(*fourcc_str),
                        self._fps, size)
                    if writer.isOpened():
                        return writer, path, fourcc_str, (fmt_name != requested)
                    writer.release()
        return None, base_path, "", True

    def _worker(self) -> None:
        writer: cv2.VideoWriter | None = None
        actual_path = ""
        segment_start = time.monotonic()
        last_disk_check = time.monotonic()

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

                frame = self._resize_frame(frame)
                now_t = time.monotonic()

                # Rotation: max duration tugadi → segmentni yopish
                if writer is not None and (now_t - segment_start) >= self.max_duration_sec:
                    writer.release()
                    print(f"[VideoRecorder] Segment yakunlandi: {actual_path}")
                    writer = None
                    self._segment_idx += 1

                # Lazy init (birinchi frame da yoki rotation dan keyin)
                if writer is None:
                    h, w = frame.shape[:2]
                    self._folder.mkdir(parents=True, exist_ok=True)
                    writer, actual_path, fourcc, fell_back = self._open_writer(w, h)
                    if writer is None:
                        self.last_error = (
                            f"VideoWriter ochilmadi ({w}x{h}, fmt={self.fmt})")
                        print(f"[VideoRecorder] {self.last_error}")
                        break
                    self.last_codec_fell_back = fell_back
                    self.last_path = actual_path
                    msg = (f"[VideoRecorder] Boshlandi "
                           f"({w}x{h} @ {self._fps:.1f}fps, codec={fourcc}): "
                           f"{actual_path}")
                    if fell_back:
                        msg += (f"  ⚠️ '{self.fmt}' codec topilmadi, "
                                f"fallback ishlatildi")
                    print(msg)
                    segment_start = now_t

                # Periodic disk check (~ har 30 sek)
                if now_t - last_disk_check > 30.0:
                    last_disk_check = now_t
                    free = _free_disk_mb(self._folder)
                    if 0 <= free < self.min_free_disk_mb:
                        self.last_error = (
                            f"Disk to'lib qoldi: {free}MB qoldi — yozish to'xtatildi")
                        print(f"[VideoRecorder] {self.last_error}")
                        break

                try:
                    writer.write(frame)
                    self._written_frames += 1
                except Exception as e:
                    self.last_error = f"Yozishda xato: {e}"
                    print(f"[VideoRecorder] {self.last_error}")
                    break

        finally:
            if writer:
                writer.release()
                print(f"[VideoRecorder] Yopildi: {actual_path}  "
                      f"(yozildi={self._written_frames}, "
                      f"tushib qoldi={self._dropped_frames})")
            # Loop abnormal tugagan bo'lsa ham state ni to'g'rilash
            self.is_recording = False
