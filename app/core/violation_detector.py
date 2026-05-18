"""
ViolationDetector — radar-camera-like violation logger.

Algorithm:
  1. PLC train signal arrives  →  on_plc_signal() (arm)
  2. Configurable warning delay (e.g. 5 sec) passes
  3. Any car that ENTERS the polygon after the delay is a violation
  4. Crop the car, run plate detection + OCR
  5. Save evidence: annotated full frame + plate crop + CSV log row
  6. Each track ID processed ONCE (no duplicate violations per car)
  7. PLC signal clears  →  on_plc_clear() (disarm)

Output structure:
  Desktop/RailSafe_Yozuvlar/_violations/<crossing>/<date>/
    HH-MM-SS_<camera>_<plate>.jpg       # full annotated frame
    HH-MM-SS_<camera>_<plate>_crop.jpg  # plate close-up
    violations.csv                       # append-only log
"""

import csv
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from app.core.plate_recognizer import PlateRecognizer
from app.utils.video_recorder import get_record_dir


def _safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', '_', name).strip() or "unknown"


def get_violations_dir() -> Path:
    return get_record_dir() / "_violations"


class ViolationDetector:
    """
    Radar-camera algorithm. Listens to PLC events + per-frame tracker data.
    Saves violation evidence when a NEW car enters polygon AFTER the
    configured delay following PLC train-arriving signal.
    """

    def __init__(
        self,
        recognizer: PlateRecognizer,
        delay_seconds: float = 5.0,
        output_dir: Optional[Path] = None,
        min_track_stable_sec: float = 0.5,
    ):
        self.recognizer = recognizer
        self.delay_seconds = max(0.0, float(delay_seconds))
        self.output_dir = output_dir or get_violations_dir()
        self.min_track_stable_sec = min_track_stable_sec

        self._armed: bool = False
        self._arm_time: float = 0.0
        self._seen_tracks: set = set()
        self._lock = threading.Lock()

        # Telemetry
        self.violations_saved: int = 0
        self.last_error: str = ""

    # ── PLC integration ──────────────────────────────────────────────

    def set_delay(self, seconds: float):
        """Runtime da delay-ni yangilash."""
        self.delay_seconds = max(0.0, float(seconds))

    def on_plc_signal(self):
        """PLC: poyezd kelmoqda signali keldi → kuzatishni armlash."""
        with self._lock:
            self._armed = True
            self._arm_time = time.monotonic()
            self._seen_tracks.clear()
        print(f"[ViolationDetector] ARMED — {self.delay_seconds}s grace period")

    def on_plc_clear(self):
        """PLC: signal tugadi → kuzatishni o'chirish."""
        with self._lock:
            self._armed = False
        if self.violations_saved > 0:
            print(f"[ViolationDetector] DISARMED — {self.violations_saved} violation saved this train")
        self.violations_saved = 0

    # ── Per-frame processing ─────────────────────────────────────────

    def process_frame(self, frame, in_poly_tracks: list,
                      crossing_name: str, camera_name: str):
        """Vorker har frame uchun chaqiradi.

        Args:
            frame: BGR numpy frame (full camera frame)
            in_poly_tracks: PolygonTracker.get_in_polygon_tracks() ro'yxati
            crossing_name: pereezd nomi
            camera_name: kamera nomi
        """
        if not self._armed or not in_poly_tracks:
            return

        now = time.monotonic()
        if (now - self._arm_time) < self.delay_seconds:
            return  # Hali grace period — ogohlantirish vaqti

        with self._lock:
            seen_snapshot = set(self._seen_tracks)

        new_violations = []
        for trk in in_poly_tracks:
            tid = trk['id']
            if tid in seen_snapshot:
                continue
            # Track yaqindagina paydo bo'lgan bo'lsa, bbox barqaror bo'lguncha kutish
            enter_t = trk.get('enter_time')
            if enter_t is None:
                continue
            if (now - enter_t) < self.min_track_stable_sec:
                continue
            new_violations.append(trk)

        if not new_violations:
            return

        # Mark all as seen first, so re-entrant calls don't double-process
        with self._lock:
            for trk in new_violations:
                self._seen_tracks.add(trk['id'])

        for trk in new_violations:
            try:
                self._save_violation(frame, trk, crossing_name, camera_name)
            except Exception as e:
                self.last_error = f"_save_violation xato: {e}"
                print(f"[ViolationDetector] {self.last_error}")

    # ── Evidence saving ──────────────────────────────────────────────

    def _save_violation(self, frame, track, crossing_name, camera_name):
        x1, y1, x2, y2 = track['bbox']
        h, w = frame.shape[:2]
        # Padding car crop uchun
        pad = max(20, (x2 - x1) // 10)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        car_crop = frame[cy1:cy2, cx1:cx2].copy()

        # Raqamni o'qish
        result = self.recognizer.detect_and_read(car_crop)
        if result is None:
            plate_text = "UNKNOWN"
            plate_conf = 0.0
            plate_crop = None
        else:
            plate_text = _safe_name(result['text']) or "UNKNOWN"
            plate_conf = result['conf']
            plate_crop = result['plate_crop']

        now_dt = datetime.now()
        date_str = now_dt.strftime("%Y-%m-%d")
        time_str = now_dt.strftime("%H-%M-%S")
        safe_cross = _safe_name(crossing_name)
        safe_cam = _safe_name(camera_name)

        folder = self.output_dir / safe_cross / date_str
        folder.mkdir(parents=True, exist_ok=True)

        # 1) Annotated full frame
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"{plate_text}  ({plate_conf:.0%})"
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ly = max(th + 10, y1 - 5)
        cv2.rectangle(annotated, (x1, ly - th - 8),
                      (x1 + tw + 12, ly + 4), (0, 0, 255), -1)
        cv2.putText(annotated, label, (x1 + 6, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        # Banner top: timestamp + crossing + camera
        banner = f"{now_dt.strftime('%Y-%m-%d %H:%M:%S')}  |  {crossing_name}  |  {camera_name}"
        cv2.rectangle(annotated, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(annotated, banner, (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        full_path = folder / f"{time_str}_{safe_cam}_{plate_text}.jpg"
        cv2.imwrite(str(full_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # 2) Plate close-up (faqat o'qilgan bo'lsa)
        if plate_crop is not None and plate_crop.size > 0:
            crop_path = folder / f"{time_str}_{safe_cam}_{plate_text}_crop.jpg"
            cv2.imwrite(str(crop_path), plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 3) CSV log
        csv_path = folder / "violations.csv"
        is_new = not csv_path.exists()
        try:
            with open(csv_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(["timestamp", "crossing", "camera",
                                      "plate", "confidence", "photo"])
                writer.writerow([now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                                  crossing_name, camera_name,
                                  plate_text, f"{plate_conf:.2f}",
                                  full_path.name])
        except Exception as e:
            print(f"[ViolationDetector] CSV yozishda xato: {e}")

        self.violations_saved += 1
        print(f"[ViolationDetector] 🚨 {plate_text} ({plate_conf:.0%}) "
              f"@ {safe_cross}/{safe_cam} → {full_path.name}")
