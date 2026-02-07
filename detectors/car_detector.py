"""
Car Detector - Avtomobillarni aniqlash moduli
YOLO modelidan foydalangan holda avtomobillarni aniqlaydi
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class Detection:
    """Bitta aniqlangan obyekt"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[int, int]:
        """Bounding box markazi"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Bounding box maydoni"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class CarDetector:
    """
    Avtomobillarni aniqlash klassi

    Misol:
        detector = CarDetector("models/car_detect.pt")
        detections = detector.detect(frame)
        frame_with_boxes = detector.draw_detections(frame, detections)
    """

    # Default rang palitralari
    COLORS = {
        0: (0, 255, 0),    # Yashil - avtomobil
        1: (255, 165, 0),  # Apelsin - yuk mashinasi
        2: (0, 0, 255),    # Qizil - avtobus
        3: (255, 255, 0),  # Sariq - mototsikl
    }

    DEFAULT_COLOR = (128, 128, 128)  # Kulrang - noma'lum

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        imgsz: int = 640,
        half: bool = True,
        stream: bool = True
    ):
        """
        CarDetector yaratish

        Args:
            model_path: YOLO model fayli yo'li (.pt yoki .onnx)
            confidence_threshold: Ishonch chegarasi (0-1)
            iou_threshold: NMS IOU chegarasi
            device: "auto", "cuda", "cpu"
            imgsz: Inference rasm o'lchami
            half: FP16 rejimini yoqish (faqat CUDA uchun) - 2x tezroq
            stream: Streaming mode - real-time uchun optimizatsiya
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.half = half
        self.stream = stream

        # Model yuklanganmi
        self._model = None
        self._device = device
        self._class_names = {}
        self._is_loaded = False

        # Performance tracking
        self._inference_times = []
        self._max_times = 30

        # Last detections cache (for frame skip optimization)
        self._last_detections = []

        # Per-camera detection cache
        self._camera_detections = {}  # camera_id -> detections

        # Thread-safe lock for non-blocking detection
        import threading
        self._detect_lock = threading.Lock()

    def load(self) -> bool:
        """
        Modelni yuklash

        Returns:
            True agar muvaffaqiyatli yuklansa
        """
        if self._is_loaded:
            return True

        if not self.model_path.exists():
            print(f"[CarDetector] Model topilmadi: {self.model_path}")
            return False

        try:
            from ultralytics import YOLO

            print(f"[CarDetector] Model yuklanmoqda: {self.model_path}")
            self._model = YOLO(str(self.model_path))

            # Device aniqlash
            if self._device == "auto":
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Class nomlarini olish
            if hasattr(self._model, 'names'):
                self._class_names = self._model.names

            self._is_loaded = True
            print(f"[CarDetector] Model yuklandi! Device: {self._device}")
            print(f"[CarDetector] Klasslar: {self._class_names}")
            return True

        except ImportError:
            print("[CarDetector] Ultralytics kutubxonasi topilmadi!")
            print("[CarDetector] Buyruq: pip install ultralytics")
            return False
        except Exception as e:
            print(f"[CarDetector] Model yuklashda xato: {e}")
            return False

    def detect(self, frame: np.ndarray, verbose: bool = False, blocking: bool = False,
                camera_id: str = None) -> List[Detection]:
        """
        Kadrda avtomobillarni aniqlash (real-time optimized, non-blocking)

        Args:
            frame: BGR formatidagi rasm (numpy array)
            verbose: Batafsil ma'lumotlarni chiqarish
            blocking: True = kutish, False = band bo'lsa skip
            camera_id: Kamera identifikatori (per-camera caching uchun)

        Returns:
            Detection obyektlari ro'yxati
        """
        if not self._is_loaded:
            if not self.load():
                return []

        if frame is None or frame.size == 0:
            if camera_id and camera_id in self._camera_detections:
                return self._camera_detections[camera_id]
            return self._last_detections

        # Non-blocking: agar detector band bo'lsa, oxirgi natijani qaytar
        if not blocking:
            if not self._detect_lock.acquire(blocking=False):
                # Return cached detections for this camera
                if camera_id and camera_id in self._camera_detections:
                    return self._camera_detections[camera_id]
                return self._last_detections
        else:
            self._detect_lock.acquire()

        detections = []

        try:
            start_time = time.perf_counter()

            # YOLO inference - real-time optimized
            results = self._model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self._device,
                half=self.half,
                verbose=verbose,
                stream=self.stream,
                max_det=50,
                agnostic_nms=True,
            )

            # Inference vaqtini saqlash
            inference_time = (time.perf_counter() - start_time) * 1000
            self._inference_times.append(inference_time)
            if len(self._inference_times) > self._max_times:
                self._inference_times.pop(0)

            # Natijalarni qayta ishlash
            for result in results:
                if result.boxes is None:
                    continue

                boxes = result.boxes
                if len(boxes) == 0:
                    continue

                # GPU da to'g'ridan-to'g'ri ishlash
                xyxy_all = boxes.xyxy.int().cpu().numpy()
                conf_all = boxes.conf.cpu().numpy()
                cls_all = boxes.cls.int().cpu().numpy()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = xyxy_all[i]
                    conf = float(conf_all[i])
                    cls_id = int(cls_all[i])
                    cls_name = self._class_names.get(cls_id, f"class_{cls_id}")

                    detection = Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name
                    )
                    detections.append(detection)

            # Cache last detections
            self._last_detections = detections
            if camera_id:
                self._camera_detections[camera_id] = detections

        except Exception as e:
            print(f"[CarDetector] Aniqlashda xato: {e}")
            if camera_id and camera_id in self._camera_detections:
                detections = self._camera_detections[camera_id]
            else:
                detections = self._last_detections
        finally:
            self._detect_lock.release()

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_labels: bool = True,
        show_confidence: bool = True,
        thickness: int = 2,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        Aniqlangan obyektlarni kadrga chizish

        Args:
            frame: Asl rasm
            detections: Aniqlangan obyektlar
            show_labels: Klass nomini ko'rsatish
            show_confidence: Ishonch foizini ko'rsatish
            thickness: Chiziq qalinligi
            font_scale: Shrift o'lchami

        Returns:
            Chizilgan rasm
        """
        result = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.COLORS.get(det.class_id, self.DEFAULT_COLOR)

            # Bounding box chizish
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            # Label tayyorlash
            if show_labels or show_confidence:
                parts = []
                if show_labels:
                    parts.append(det.class_name)
                if show_confidence:
                    parts.append(f"{det.confidence:.0%}")
                label = " ".join(parts)

                # Label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Label joylashuvi
                label_y = y1 - 10 if y1 > 30 else y2 + text_h + 10

                cv2.rectangle(
                    result,
                    (x1, label_y - text_h - 5),
                    (x1 + text_w + 5, label_y + 5),
                    color,
                    -1
                )

                cv2.putText(
                    result,
                    label,
                    (x1 + 2, label_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )

        return result

    def get_fps(self) -> float:
        """O'rtacha inference FPS qaytarish"""
        if not self._inference_times:
            return 0.0
        avg_ms = sum(self._inference_times) / len(self._inference_times)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    def get_inference_time_ms(self) -> float:
        """O'rtacha inference vaqti (ms)"""
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    @property
    def is_loaded(self) -> bool:
        """Model yuklanganligi"""
        return self._is_loaded

    @property
    def class_names(self) -> Dict[int, str]:
        """Klass nomlari lug'ati"""
        return self._class_names.copy()

    @property
    def device(self) -> str:
        """Joriy device"""
        return self._device

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"CarDetector(model={self.model_path.name}, {status}, device={self._device})"


# Sinov uchun
if __name__ == "__main__":
    import sys

    # Test model
    model_path = "/home/bahrombek/Desktop/RailSafeGUI/models/car_detect.pt"

    detector = CarDetector(model_path, confidence_threshold=0.5)

    if detector.load():
        print(f"Model yuklandi: {detector}")
        print(f"Klasslar: {detector.class_names}")

        # Test rasm yaratish
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:] = (50, 50, 50)

        # Detect
        detections = detector.detect(test_frame)
        print(f"Aniqlandi: {len(detections)} ta obyekt")

        if detections:
            frame_with_boxes = detector.draw_detections(test_frame, detections)
            cv2.imshow("Test", frame_with_boxes)
            cv2.waitKey(0)
    else:
        print("Model yuklanmadi!")
        sys.exit(1)
