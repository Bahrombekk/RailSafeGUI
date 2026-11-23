"""Estimate train movement from fresh detections and configured rails."""

from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from app.core.tracker import _iou

AB = "ab"
BA = "ba"

TRAIN_CLASS_NAMES = {"train", "poyezd", "poezd", "lokomotiv", "locomotive"}
ROAD_VEHICLE_NAMES = {
    "truck", "bus", "lorry", "semitruck", "trailer", "fura", "heavy",
    "ogir", "og'ir", "yuk mashinasi", "грузовик", "автобус",
}


def train_class_ids(class_names: Dict) -> Set[int]:
    """Model sinf nomlaridan poyezd sinflarining id larini topish."""
    ids = set()
    for cid, name in (class_names or {}).items():
        try:
            if str(name).strip().lower() in TRAIN_CLASS_NAMES:
                ids.add(int(cid))
        except (TypeError, ValueError):
            continue
    return ids


def split_train(detections: Iterable, train_ids: Set[int]) -> Tuple[List, List]:
    """Deteksiyalarni (mashinalar, poyezdlar) ga ajratish."""
    if not train_ids:
        return list(detections), []
    others, trains = [], []
    for d in detections:
        (trains if d.class_id in train_ids else others).append(d)
    return others, trains


def valid_rail(rail) -> bool:
    """Check the normalized polyline from camera settings."""
    if not isinstance(rail, dict):
        return False
    try:
        points = np.asarray(rail.get("points"), dtype=float)
        return (points.ndim == 2 and points.shape[1] == 2 and len(points) >= 2
                and bool(np.isfinite(points).all())
                and bool(((points >= 0) & (points <= 1)).all())
                and float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) >= 0.02)
    except (TypeError, ValueError):
        return False


def direction_text(code: Optional[str], rail: Optional[dict], default_ab: str,
                   default_ba: str) -> str:
    """Yo'nalish kodi → operatorga ko'rsatiladigan matn (strelkasiz).

    Uchlarga nom berilgan bo'lsa "Toshkent → Chuqursoy", chiziq bor-u nom
    yo'q bo'lsa "A → B", chiziq yo'q bo'lsa default (chapdan o'ngga ...).
    """
    if code not in (AB, BA):
        return ""
    if valid_rail(rail):
        a = (rail.get("a_name") or "").strip() or "A"
        b = (rail.get("b_name") or "").strip() or "B"
        return f"{a} → {b}" if code == AB else f"{b} → {a}"
    return default_ab if code == AB else default_ba


class TrainDirectionEstimator:
    """Confirm one moving train candidate per camera."""

    EDGE = 0.03
    HYST = 0.03
    SAMPLES = 120
    CONFIRM_GAP_SEC = 1.0
    PRESENCE_GAP_SEC = 3.0
    MIN_CONFIDENCE = 0.55
    MIN_WIDTH = 0.05
    MIN_TRAVEL = 0.12

    def __init__(self, frame_width: int, frame_height: int, rail: dict = None):
        self.frame_size = (frame_width, frame_height)
        self._checkpoints = (0.25, 0.5, 0.75)
        self._rail_px = None
        self._rail_s = None
        self._last_observation = None
        if valid_rail(rail):
            points = np.asarray(rail["points"], dtype=float) * self.frame_size
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            # np.interp needs unique distances for consecutive vertices.
            distances = np.concatenate(([0.0], np.cumsum(lengths)))
            keep = np.concatenate(([True], lengths > 0))
            self._rail_s = np.linspace(0.0, 1.0, self.SAMPLES)
            if distances[-1] >= 10.0:
                positions = distances[keep] / distances[-1]
                xs = np.interp(self._rail_s, positions, points[keep, 0])
                ys = np.interp(self._rail_s, positions, points[keep, 1])
                self._rail_px = np.column_stack((xs, ys))
        self._reset()

    def _reset(self):
        self._box = None
        self._first_t = 0.0
        self._last_t = 0.0
        self._first_position = 0.0
        self._first_box = None
        self._samples = 0
        self._checkpoint_states = ({}, {})
        self._passed_ab = set()
        self._passed_ba = set()
        self.direction: Optional[str] = None
        self.present = False

    def _span_of(self, box) -> Optional[Tuple[float, float]]:
        x1, y1, x2, y2 = box
        if self._rail_px is None:
            width = self.frame_size[0]
            return x1 / width, x2 / width
        px, py = self._rail_px.T
        # The wheels and rails meet in the lower part of a train box.
        lower_edge = y1 + (y2 - y1) * 0.5
        inside = (px >= x1) & (px <= x2) & (py >= lower_edge) & (py <= y2)
        if inside.sum() < 2:
            return None
        positions = self._rail_s[inside]
        # A short touch or a corner of the box overlapping the rails is insufficient.
        if positions[-1] - positions[0] < 0.04:
            return None
        return float(positions[0]), float(positions[-1])

    def expire(self, now: float) -> Optional[str]:
        """Clear stale evidence even when the camera or inference stops delivering frames."""
        gap = self.PRESENCE_GAP_SEC if self.direction else self.CONFIRM_GAP_SEC
        if self._box is None or now - self._last_t < gap:
            return None
        direction = self.direction
        self._reset()
        return direction

    def update(self, trains: Iterable, now: Optional[float] = None,
               road_detections: Iterable = ()) -> None:
        """Process a result at its frame timestamp once."""
        now = time.monotonic() if now is None else now
        if self._last_observation is not None and now <= self._last_observation:
            return
        self._last_observation = now
        self.expire(now)
        road_boxes = [
            d.bbox for d in road_detections
            if d.confidence >= 0.35
            and d.class_name.strip().lower() in ROAD_VEHICLE_NAMES
        ]
        if self._box is not None and not self.direction:
            # A road vehicle may replace the train label in the next frame.
            # Allow for motion when comparing with the previous train box.
            if any(_iou(self._box, box) >= 0.25 for box in road_boxes):
                self._reset()

        candidates = []
        for detection in trains:
            box = detection.bbox
            if (detection.confidence < self.MIN_CONFIDENCE
                    or box[2] - box[0] < self.MIN_WIDTH * self.frame_size[0]
                    or box[3] <= box[1]
                    or any(_iou(box, road_box) >= 0.45 for road_box in road_boxes)):
                continue
            span = self._span_of(box)
            if span is not None:
                candidates.append((box, span, detection.confidence))
        if not candidates:
            return

        if self._box is not None:
            best = max(candidates, key=lambda candidate: _iou(self._box, candidate[0]))
            if _iou(self._box, best[0]) < 0.1:
                if self.direction:
                    return
                self._reset()
        if self._box is None:
            best = max(candidates, key=lambda candidate: candidate[2])
            self._first_box = best[0]
            self._first_position = sum(best[1]) / 2
            self._first_t = now

        self._box, (lo, hi), _ = best
        self._last_t = now
        self._samples += 1
        if self.direction:
            return

        if lo > self.EDGE:
            self._step_edge(self._checkpoint_states[0], lo)
        if hi < 1.0 - self.EDGE:
            self._step_edge(self._checkpoint_states[1], hi)

        if self._samples < 5 or now - self._first_t < 1.0:
            return
        position = (lo + hi) / 2
        travel = position - self._first_position
        if abs(travel) < self.MIN_TRAVEL or not self._moves_along_rail(position):
            return
        if travel > 0 and len(self._passed_ab - self._passed_ba) >= 2:
            self.direction = AB
        elif travel < 0 and len(self._passed_ba - self._passed_ab) >= 2:
            self.direction = BA
        self.present = self.direction is not None and self._rail_px is not None

    def _moves_along_rail(self, position: float) -> bool:
        if self._rail_px is None:
            return True
        # A truck crossing the rails can move the projected overlap without
        # moving along the track. Compare its box's lower midpoint to the rail.
        first = self._first_box
        current = self._box
        movement = np.array(((current[0] + current[2] - first[0] - first[2]) / 2,
                             current[3] - first[3]))
        start = round(self._first_position * (self.SAMPLES - 1))
        end = round(position * (self.SAMPLES - 1))
        along = self._rail_px[end] - self._rail_px[start]
        scale = np.linalg.norm(movement) * np.linalg.norm(along)
        return bool(scale > 0 and movement @ along >= 0.65 * scale)

    def _step_edge(self, states: dict, position: float):
        for checkpoint in self._checkpoints:
            if position > checkpoint + self.HYST:
                side = 1
            elif position < checkpoint - self.HYST:
                side = -1
            else:
                continue
            previous = states.get(checkpoint)
            states[checkpoint] = side
            if previous is not None and previous != side:
                passed = self._passed_ab if side > previous else self._passed_ba
                passed.add(checkpoint)
