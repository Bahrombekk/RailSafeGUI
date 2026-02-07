"""
Batch Car Detector - GPU ni maksimal ishlatish uchun
Bir vaqtda ko'p kameradan kadrlarni batch qilib process qiladi
"""

import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Detection:
    """Bitta aniqlangan obyekt"""
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DetectionRequest:
    """Kameradan kelgan detection so'rovi"""
    camera_id: str
    frame: np.ndarray
    timestamp: float


@dataclass
class DetectionResult:
    """Detection natijasi"""
    camera_id: str
    detections: List[Detection]
    inference_time: float


class BatchCarDetector:
    """
    Batch processing bilan real-time car detection
    GPU ni maksimal ishlatadi - bir vaqtda 4-8 kadrni process qiladi
    """

    # COCO class colors
    COLORS = {
        2: (0, 255, 0),    # car - Yashil
        3: (255, 165, 0),  # motorcycle - Apelsin
        5: (255, 255, 0),  # bus - Sariq
        7: (0, 0, 255),    # truck - Qizil
    }
    DEFAULT_COLOR = (0, 255, 0)

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cuda",
        half: bool = True,
        batch_size: int = 8,  # Bir vaqtda nechta kadr
        max_queue_size: int = 32,
        filter_classes: List[int] = None  # Faqat bu classlarni detect qilish
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.batch_size = batch_size
        self.filter_classes = set(filter_classes) if filter_classes else None

        # Model
        self._model = None
        self._class_names = {}
        self._is_loaded = False

        # Request/Result queues
        self._request_queue = Queue(maxsize=max_queue_size)
        self._results: Dict[str, DetectionResult] = {}
        self._results_lock = threading.Lock()

        # Per-camera latest detections (for immediate access)
        self._camera_cache: Dict[str, List[Detection]] = {}
        self._cache_lock = threading.Lock()

        # Worker thread
        self._running = False
        self._worker_thread = None

        # Stats
        self._inference_times = []
        self._processed_count = 0

        # Sync detection lock
        self._sync_lock = threading.Lock()

    def load(self) -> bool:
        """Model yuklash va worker thread boshlash"""
        if self._is_loaded:
            return True

        try:
            from ultralytics import YOLO
            import torch

            print(f"[BatchDetector] Model yuklanmoqda: {self.model_path}")
            self._model = YOLO(str(self.model_path))

            # GPU warmup with batch
            if torch.cuda.is_available():
                print(f"[BatchDetector] GPU warmup (batch={self.batch_size})...")
                dummy = [np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)] * self.batch_size
                self._model.predict(
                    dummy,
                    conf=self.confidence_threshold,
                    device=self.device,
                    half=self.half,
                    verbose=False
                )
                torch.cuda.synchronize()

            self._class_names = self._model.names if hasattr(self._model, 'names') else {}
            self._is_loaded = True

            # Start worker thread
            self._running = True
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

            print(f"[BatchDetector] Yuklandi! Device: {self.device}, Batch: {self.batch_size}")
            print(f"[BatchDetector] Klasslar: {self._class_names}")
            return True

        except Exception as e:
            print(f"[BatchDetector] Yuklash xatosi: {e}")
            return False

    def _worker_loop(self):
        """Background worker - tezkor batch processing (lag kamaytirish)"""
        while self._running:
            batch_requests = []

            # Collect batch - TEZKOR rejim
            try:
                # Birinchi requestni kut (qisqa timeout)
                req = self._request_queue.get(timeout=0.02)  # 20ms
                batch_requests.append(req)

                # Qo'shimcha requestlarni ol (juda qisqa vaqt)
                deadline = time.perf_counter() + 0.005  # 5ms ichida batch yig'
                while len(batch_requests) < self.batch_size:
                    if time.perf_counter() > deadline:
                        break
                    try:
                        req = self._request_queue.get_nowait()
                        batch_requests.append(req)
                    except Empty:
                        break

            except Empty:
                continue

            if not batch_requests:
                continue

            # Process batch
            try:
                start_time = time.perf_counter()

                # Prepare batch frames
                frames = [req.frame for req in batch_requests]

                # Batch inference
                results = self._model.predict(
                    frames,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                    max_det=50,
                    agnostic_nms=True,
                    classes=list(self.filter_classes) if self.filter_classes else None,
                )

                inference_time = (time.perf_counter() - start_time) * 1000
                per_frame_time = inference_time / len(batch_requests)

                # Track stats
                self._inference_times.append(per_frame_time)
                if len(self._inference_times) > 100:
                    self._inference_times.pop(0)
                self._processed_count += len(batch_requests)

                # Parse results for each camera
                for i, (req, result) in enumerate(zip(batch_requests, results)):
                    detections = self._parse_result(result)

                    # Update cache
                    with self._cache_lock:
                        self._camera_cache[req.camera_id] = detections

            except Exception as e:
                print(f"[BatchDetector] Worker error: {e}")

    def _parse_result(self, result) -> List[Detection]:
        """YOLO natijasini Detection listga aylantirish"""
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        xyxy_all = boxes.xyxy.int().cpu().numpy()
        conf_all = boxes.conf.cpu().numpy()
        cls_all = boxes.cls.int().cpu().numpy()

        for i in range(len(boxes)):
            cls_id = int(cls_all[i])

            # Filter classes (agar berilgan bo'lsa)
            if self.filter_classes and cls_id not in self.filter_classes:
                continue

            x1, y1, x2, y2 = xyxy_all[i]
            conf = float(conf_all[i])
            cls_name = self._class_names.get(cls_id, f"class_{cls_id}")

            detections.append(Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name
            ))

        return detections

    def detect_sync(self, frame: np.ndarray, camera_id: str = None) -> List[Detection]:
        """
        Sinxron detection - LAG YO'Q, thread-safe

        Args:
            frame: BGR rasm
            camera_id: Kamera identifikatori

        Returns:
            Hozirgi kadrning detections (real-time)
        """
        if not self._is_loaded:
            return []

        # Thread-safe: bir vaqtda faqat bitta inference
        with self._sync_lock:
            try:
                start = time.perf_counter()

                # To'g'ridan-to'g'ri inference
                results = self._model.predict(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                    max_det=50,
                    classes=list(self.filter_classes) if self.filter_classes else None,
                )

                # Inference vaqtini track qilish
                inference_ms = (time.perf_counter() - start) * 1000
                self._inference_times.append(inference_ms)
                if len(self._inference_times) > 100:
                    self._inference_times.pop(0)

                detections = []
                if results and len(results) > 0:
                    detections = self._parse_result(results[0])

                # Cache yangilash
                if camera_id:
                    with self._cache_lock:
                        self._camera_cache[camera_id] = detections

                return detections

            except Exception as e:
                print(f"[BatchDetector] Sync detect error: {e}")
                return []

    def detect_async(self, frame: np.ndarray, camera_id: str) -> List[Detection]:
        """
        Async detection - tezkor, lekin biroz lag bo'lishi mumkin
        Agar real-time kerak bo'lsa detect_sync() ishlating

        Args:
            frame: BGR rasm
            camera_id: Kamera identifikatori

        Returns:
            Cached detections
        """
        if not self._is_loaded:
            return []

        # Add to queue (non-blocking)
        try:
            self._request_queue.put_nowait(DetectionRequest(
                camera_id=camera_id,
                frame=frame,
                timestamp=time.time()
            ))
        except:
            pass  # Queue to'lgan bo'lsa skip

        # Return cached result
        with self._cache_lock:
            return self._camera_cache.get(camera_id, [])

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        thickness: int = 2,
        font_scale: float = 0.5
    ) -> np.ndarray:
        """Detection boxlarni chizish - qora box muammosi to'g'irlangan"""
        result = frame.copy()
        h, w = result.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.COLORS.get(det.class_id, self.DEFAULT_COLOR)

            # Bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            # Label text
            label = f"{det.class_name} {det.confidence:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

            # Label pozitsiyasi - box ichida yuqori chap burchak
            pad = 2
            lx = x1 + pad
            ly = y1 + th + pad

            # Agar box juda yuqorida bo'lsa, pastga tushir
            if y1 < 5:
                ly = y1 + th + pad + 5

            # Background koordinatalari
            bg_x1 = x1
            bg_y1 = ly - th - pad
            bg_x2 = min(x1 + tw + pad * 2 + 2, w)
            bg_y2 = ly + pad

            # Rangli background
            cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

            # Oq text
            cv2.putText(result, label, (lx, ly), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        return result

    def get_fps(self) -> float:
        """O'rtacha inference FPS"""
        if not self._inference_times:
            return 0.0
        avg_ms = sum(self._inference_times) / len(self._inference_times)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    def get_stats(self) -> Dict:
        """Statistics"""
        return {
            "processed": self._processed_count,
            "avg_ms": sum(self._inference_times) / len(self._inference_times) if self._inference_times else 0,
            "fps": self.get_fps(),
            "queue_size": self._request_queue.qsize(),
            "batch_size": self.batch_size
        }

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def class_names(self) -> Dict:
        return self._class_names.copy()

    def stop(self):
        """Worker thread to'xtatish"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

    def __del__(self):
        self.stop()

    def __repr__(self):
        return f"BatchCarDetector(batch={self.batch_size}, fps={self.get_fps():.1f})"


# Test
if __name__ == "__main__":
    import time

    detector = BatchCarDetector(
        "models/car_detect.pt",
        batch_size=8,
        device="cuda",
        half=True
    )

    if detector.load():
        print(f"Loaded: {detector}")

        # Simulate 8 cameras
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(8)]

        print("\nSimulating 8 cameras for 5 seconds...")
        start = time.time()
        frame_count = 0

        while time.time() - start < 5:
            for i, frame in enumerate(frames):
                dets = detector.detect_async(frame, f"cam_{i}")
                frame_count += 1
            time.sleep(0.033)  # ~30 FPS per camera

        elapsed = time.time() - start
        print(f"\nResults:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total FPS: {frame_count / elapsed:.1f}")
        print(f"  Per-camera FPS: {frame_count / elapsed / 8:.1f}")
        print(f"  Stats: {detector.get_stats()}")

        detector.stop()
