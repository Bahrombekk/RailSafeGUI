"""
PolygonTracker - IoU-based vehicle tracking with polygon time and counting.
Works with Detection(bbox, confidence, class_id, class_name) from RealtimeMultiCameraDetector.

Har kamera uchun alohida instance yaratiladi. Thread-safe (bitta thread ichida ishlatiladi).
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """IoU between two (x1, y1, x2, y2) boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """Single tracked vehicle"""
    track_id: int
    class_id: int
    bbox: Tuple[int, int, int, int]
    in_polygon: bool = False
    polygon_enter_time: Optional[float] = None
    time_in_polygon: float = 0.0
    last_seen: float = 0.0
    counted: bool = False


class PolygonTracker:
    """
    Lightweight IoU-based tracker with polygon time tracking and vehicle counting.

    Usage:
        tracker = PolygonTracker(poly_mask, frame_width=w, frame_height=h)
        # Har kadr:
        tracks = tracker.process_detections(detections)
        print(tracker.light_count, tracker.heavy_count)
    """

    LIGHT_CLASSES = {2, 3}    # car, motorcycle
    HEAVY_CLASSES = {5, 7}    # bus, truck

    def __init__(self, poly_mask: np.ndarray,
                 iou_threshold: float = 0.3,
                 max_age: float = 2.0,
                 frame_width: int = 1920,
                 frame_height: int = 1080):
        self._poly_mask = poly_mask
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._frame_w = frame_width
        self._frame_h = frame_height

        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 1

        self.light_count: int = 0
        self.heavy_count: int = 0

    @property
    def total_count(self) -> int:
        return self.light_count + self.heavy_count

    def _in_polygon(self, cx: int, cy: int) -> bool:
        if 0 <= cy < self._frame_h and 0 <= cx < self._frame_w:
            return self._poly_mask[cy, cx] > 0
        return False

    def _match_detections(self, detections):
        """Greedy IoU matching: existing tracks <-> new detections.
        Returns list of (track_id_or_None, detection)."""
        if not self._tracks:
            return [(None, det) for det in detections]
        if not detections:
            return []

        track_ids = list(self._tracks.keys())
        track_boxes = [self._tracks[tid].bbox for tid in track_ids]
        det_boxes = [det.bbox for det in detections]

        n_tracks = len(track_ids)
        n_dets = len(detections)

        # IoU matrix
        iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for i in range(n_tracks):
            for j in range(n_dets):
                iou_matrix[i, j] = _iou(track_boxes[i], det_boxes[j])

        # Greedy matching - highest IoU first
        pairs = []
        for i in range(n_tracks):
            for j in range(n_dets):
                if iou_matrix[i, j] >= self._iou_threshold:
                    pairs.append((iou_matrix[i, j], i, j))
        pairs.sort(key=lambda x: x[0], reverse=True)

        matched_tracks = set()
        matched_dets = set()
        results = [None] * n_dets

        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            results[di] = track_ids[ti]
            matched_tracks.add(ti)
            matched_dets.add(di)

        return [(results[di], det) for di, det in enumerate(detections)]

    def process_detections(self, detections) -> Dict[int, Track]:
        """Har kadr chaqiriladi. Kumulyativ light_count/heavy_count yangilanadi.
        Returns: active tracks dict (visualization uchun)."""
        now = time.time()

        matches = self._match_detections(detections)

        for track_id, det in matches:
            cx = int((det.bbox[0] + det.bbox[2]) / 2)
            cy = int((det.bbox[1] + det.bbox[3]) / 2)
            inside = self._in_polygon(cx, cy)

            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = Track(
                    track_id=track_id,
                    class_id=det.class_id,
                    bbox=det.bbox,
                    last_seen=now,
                )

            tr = self._tracks[track_id]
            tr.bbox = det.bbox
            tr.last_seen = now
            tr.class_id = det.class_id

            if inside:
                if not tr.in_polygon:
                    tr.in_polygon = True
                    tr.polygon_enter_time = now
                    tr.time_in_polygon = 0.0

                    if not tr.counted:
                        tr.counted = True
                        if det.class_id in self.LIGHT_CLASSES:
                            self.light_count += 1
                        elif det.class_id in self.HEAVY_CLASSES:
                            self.heavy_count += 1

                if tr.polygon_enter_time is not None:
                    tr.time_in_polygon = now - tr.polygon_enter_time
            else:
                if tr.in_polygon:
                    tr.in_polygon = False

        # Eski track-larni o'chirish
        expired = [tid for tid, tr in self._tracks.items()
                   if now - tr.last_seen > self._max_age]
        for tid in expired:
            del self._tracks[tid]

        return self._tracks

    def get_inside_count(self) -> int:
        return sum(1 for tr in self._tracks.values() if tr.in_polygon)

    def get_max_time(self) -> float:
        times = [tr.time_in_polygon for tr in self._tracks.values() if tr.in_polygon]
        return max(times) if times else 0.0

    def reset_counts(self):
        self.light_count = 0
        self.heavy_count = 0
        self._tracks.clear()
        self._next_id = 1
