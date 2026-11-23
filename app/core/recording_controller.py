"""
RecordingController — owns the video recorders for a single crossing.

Extracted from CrossingCard to keep GUI code thin. Handles:
  - Reading fresh settings (format, quality) on each start (hot-reload)
  - Starting a VideoRecorder per camera worker
  - Real FPS detection from worker.current_fps
  - Clean stop (releases all writers, joins threads)
"""

import logging
from typing import Optional

from app.utils.video_recorder import VideoRecorder

logger = logging.getLogger("RailSafe.recording")


class RecordingController:
    """Per-crossing video recording orchestrator."""

    def __init__(self, config_manager, crossing_data: dict):
        self.config_manager = config_manager
        self.crossing_data = crossing_data
        self._active: dict[int, VideoRecorder] = {}

    # ── State ────────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return len(self._active) > 0

    @property
    def active_count(self) -> int:
        return len(self._active)

    # ── Public control ───────────────────────────────────────────────

    def start(self, workers: dict, crossing_name: Optional[str] = None) -> bool:
        """Start missing recorders and reattach existing ones by camera ID."""
        if not workers:
            return False

        # Hot-reload settings
        settings = (self.config_manager.get_settings()
                    if self.config_manager else {})
        fmt = settings.get("record_format", "mp4")
        quality = settings.get("record_quality", "1080p")

        if crossing_name is None:
            crossing_name = self.crossing_data.get("name", "crossing")

        any_started = False
        for camera_id, worker in workers.items():
            existing = self._active.get(camera_id)
            if existing is not None and existing.is_recording:
                worker.video_recorder = existing
                any_started = True
                continue
            if not settings.get("record_enabled", False):
                continue
            fps = float(getattr(worker, 'current_fps', 0) or 0)
            if fps < 5.0 or fps > 60.0:
                fps = 25.0
            rec = VideoRecorder(fmt=fmt, quality=quality)
            if rec.start(crossing_name, worker.camera_name, fps=fps):
                worker.video_recorder = rec
                self._active[camera_id] = rec
                any_started = True
            else:
                logger.warning("%s: %s", worker.camera_name, rec.last_error or "unknown error")
        return any_started

    def stop(self, wait: bool = True) -> list:
        """Barcha aktiv recorder-larni to'xtatish. Saqlangan fayl yo'llarini qaytaradi."""
        paths = []
        recorders = list(self._active.values())
        self._active.clear()
        for rec in recorders:
            try:
                rec.stop(wait=wait)
                if rec.last_path:
                    paths.append(rec.last_path)
            except Exception as e:
                logger.exception("Video yozuvni to'xtatishda xato")
        return paths

    def detach_workers(self, workers: list):
        """Worker-lardan video_recorder reference-ni olib tashlash."""
        for w in workers:
            try:
                w.video_recorder = None
            except AttributeError:
                pass
