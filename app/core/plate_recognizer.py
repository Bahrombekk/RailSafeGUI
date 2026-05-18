"""
PlateRecognizer — wraps two YOLO models:
  1. car_number_lines.pt — detects license plate bounding box on car
  2. car_number ocr.pt  — detects individual characters (0-9, A-Z)

Pipeline:
  car_crop → detect_plate() → plate_bbox → read_plate() → text
"""

import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DETECTOR = _PROJECT_ROOT / "models" / "car_number_lines.pt"
_DEFAULT_OCR = _PROJECT_ROOT / "models" / "car_number ocr.pt"


class PlateRecognizer:
    """Two-stage plate recognition: detect → OCR.

    Models loaded LAZILY on first use (saves ~2 sec startup time).
    Thread-safe — internal lock protects YOLO inference calls.
    """

    def __init__(
        self,
        detector_path: str | Path = _DEFAULT_DETECTOR,
        ocr_path: str | Path = _DEFAULT_OCR,
        device: str = "cuda",
        detector_imgsz: int = 1024,
        ocr_imgsz: int = 800,
        detector_conf: float = 0.40,
        ocr_conf: float = 0.30,
    ):
        self.detector_path = str(detector_path)
        self.ocr_path = str(ocr_path)
        self.device = device
        self.detector_imgsz = detector_imgsz
        self.ocr_imgsz = ocr_imgsz
        self.detector_conf = detector_conf
        self.ocr_conf = ocr_conf

        self._detector = None
        self._ocr = None
        self._lock = threading.Lock()

    # ── Lazy model loading ────────────────────────────────────────────

    def _load_detector(self):
        if self._detector is None:
            from ultralytics import YOLO
            print(f"[PlateRecognizer] Loading detector: {self.detector_path}")
            self._detector = YOLO(self.detector_path)
        return self._detector

    def _load_ocr(self):
        if self._ocr is None:
            from ultralytics import YOLO
            print(f"[PlateRecognizer] Loading OCR: {self.ocr_path}")
            self._ocr = YOLO(self.ocr_path)
        return self._ocr

    def preload(self) -> bool:
        """Eagerly load both models. Returns True on success."""
        try:
            self._load_detector()
            self._load_ocr()
            return True
        except Exception as e:
            print(f"[PlateRecognizer] Preload xato: {e}")
            return False

    # ── Public API ────────────────────────────────────────────────────

    def detect_plate(self, car_crop) -> Optional[Tuple[int, int, int, int]]:
        """Avtomobil rasmida raqam joyini topish.
        Returns (x1, y1, x2, y2) — eng yuqori ishonchli plate, yoki None."""
        if car_crop is None or car_crop.size == 0:
            return None
        try:
            with self._lock:
                det = self._load_detector()
                results = det.predict(
                    car_crop, imgsz=self.detector_imgsz,
                    conf=self.detector_conf, verbose=False, device=self.device)
            if not results or len(results[0].boxes) == 0:
                return None
            boxes = results[0].boxes
            # Eng ishonchli plate
            best = int(boxes.conf.argmax().item())
            xyxy = boxes.xyxy[best].cpu().numpy().astype(int)
            return tuple(int(v) for v in xyxy)
        except Exception as e:
            print(f"[PlateRecognizer] detect_plate xato: {e}")
            return None

    def read_plate(self, plate_crop) -> Tuple[str, float]:
        """Raqam plitasidan matnni o'qish.
        Returns (text, avg_confidence). Bo'sh stringga (text='', conf=0.0)."""
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0
        try:
            with self._lock:
                ocr = self._load_ocr()
                results = ocr.predict(
                    plate_crop, imgsz=self.ocr_imgsz,
                    conf=self.ocr_conf, verbose=False, device=self.device)
            if not results or len(results[0].boxes) == 0:
                return "", 0.0
            boxes = results[0].boxes
            names = ocr.names if hasattr(ocr, 'names') else self._ocr.names

            chars = []
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x_center = float((xyxy[0] + xyxy[2]) / 2.0)
                cls = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                chars.append((x_center, names.get(cls, '?'), conf))

            # Chap-dan-o'ngga tartiblash
            chars.sort(key=lambda c: c[0])
            text = "".join(c[1] for c in chars)
            avg_conf = sum(c[2] for c in chars) / len(chars) if chars else 0.0
            return text, avg_conf
        except Exception as e:
            print(f"[PlateRecognizer] read_plate xato: {e}")
            return "", 0.0

    def detect_and_read(self, car_crop) -> Optional[dict]:
        """To'liq pipeline: avtomobil rasmidan raqam matnini qaytarish.
        Returns {'plate_bbox': (x1,y1,x2,y2), 'plate_crop': ndarray,
                 'text': str, 'conf': float} yoki None."""
        plate_bbox = self.detect_plate(car_crop)
        if plate_bbox is None:
            return None
        x1, y1, x2, y2 = plate_bbox
        # Kichik padding qo'shish — character-larni butun ushlash uchun
        h, w = car_crop.shape[:2]
        pad = max(2, (x2 - x1) // 30)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)
        plate_crop = car_crop[cy1:cy2, cx1:cx2]

        # Plate kichik bo'lsa OCR uchun upsample (qisqa tomon 80px)
        ph, pw = plate_crop.shape[:2]
        if ph > 0 and pw > 0 and min(ph, pw) < 80:
            scale = 80.0 / min(ph, pw)
            new_w = int(pw * scale)
            new_h = int(ph * scale)
            plate_crop = cv2.resize(plate_crop, (new_w, new_h),
                                    interpolation=cv2.INTER_CUBIC)

        text, conf = self.read_plate(plate_crop)
        if not text:
            return None
        return {
            'plate_bbox': plate_bbox,
            'plate_crop': plate_crop,
            'text': text,
            'conf': conf,
        }


# ── Singleton helper ─────────────────────────────────────────────────

_INSTANCE: PlateRecognizer | None = None
_INSTANCE_LOCK = threading.Lock()


def get_plate_recognizer(device: str = "cuda") -> PlateRecognizer:
    """Module-level singleton — bir marta yuklanadi, hamma joydan ishlatiladi."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = PlateRecognizer(device=device)
    return _INSTANCE
