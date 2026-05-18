"""
RecordingController — owns the video recorders for a single crossing.

Extracted from CrossingCard to keep GUI code thin. Handles:
  - Reading fresh settings (format, quality) on each start (hot-reload)
  - Starting a VideoRecorder per camera worker
  - Real FPS detection from worker.current_fps
  - Clean stop (releases all writers, joins threads)
"""

from typing import List, Optional

from app.utils.video_recorder import VideoRecorder


class RecordingController:
    """Per-crossing video recording orchestrator."""

    def __init__(self, config_manager, crossing_data: dict):
        self.config_manager = config_manager
        self.crossing_data = crossing_data
        self._active: List[VideoRecorder] = []

    # ── State ────────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return len(self._active) > 0

    @property
    def active_count(self) -> int:
        return len(self._active)

    # ── Public control ───────────────────────────────────────────────

    def start(self, workers: list, crossing_name: Optional[str] = None) -> bool:
        """Yozishni boshlash. Settings dan format/sifat ni fresh o'qiydi.

        IDEMPOTENT: agar allaqachon yozilayotgan bo'lsa — qaytadan ochmaydi.
        Pereezd ochilgandan yopilgunicha BIR video bo'lib qoladi.

        Returns True if at least one recorder is now active.
        """
        if not workers:
            return False

        # Allaqachon yozayotgan bo'lsa hech narsa qilmaymiz
        if self.is_recording:
            return True

        # Hot-reload settings
        settings = (self.config_manager.get_settings()
                    if self.config_manager else {})
        if not settings.get("record_enabled", False):
            return False

        fmt = settings.get("record_format", "mp4")
        quality = settings.get("record_quality", "1080p")

        if crossing_name is None:
            crossing_name = self.crossing_data.get("name", "crossing")

        any_started = False
        for worker in workers:
            fps = float(getattr(worker, 'current_fps', 0) or 0)
            if fps < 5.0 or fps > 60.0:
                fps = 25.0
            rec = VideoRecorder(fmt=fmt, quality=quality)
            if rec.start(crossing_name, worker.camera_name, fps=fps):
                worker.video_recorder = rec
                self._active.append(rec)
                any_started = True
            else:
                print(f"[RecordingController] {worker.camera_name}: "
                      f"{rec.last_error or 'unknown error'}")
        return any_started

    def stop(self):
        """Barcha aktiv recorder-larni to'xtatish va worker dan uzish."""
        for rec in self._active:
            try:
                rec.stop()
            except Exception as e:
                print(f"[RecordingController] stop error: {e}")
        self._active.clear()

    def detach_workers(self, workers: list):
        """Worker-lardan video_recorder reference-ni olib tashlash."""
        for w in workers:
            try:
                w.video_recorder = None
            except AttributeError:
                pass
