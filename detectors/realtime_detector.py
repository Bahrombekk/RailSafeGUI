"""
Real-Time Multi-Camera Detector
8 ta kamerani BITTA batch qilib process qiladi
GPU 100% ishlatiladi, LAG YO'Q

TensorRT Native Mode (208 FPS):
- .engine faylni to'g'ridan-to'g'ri TensorRT API bilan yuklaydi
- PyTorch faqat CUDA memory uchun ishlatiladi
- Ultralytics kerak EMAS (inference uchun)
- 3-5x tezroq PyTorch/Ultralytics dan

Fallback: ONNX/PyTorch (Ultralytics) agar engine yo'q bo'lsa
"""

import cv2
import numpy as np
import threading
import time
import os
import glob
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


# COCO class names (80 classes)
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


@dataclass
class Detection:
    """Bitta aniqlangan obyekt"""
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


def find_engine(model_path: str) -> Optional[str]:
    """TensorRT engine faylni topish (model_path asosida)"""
    base = os.path.splitext(model_path)[0]

    # Exact match: yolo26m.engine
    exact = f"{base}.engine"
    if os.path.exists(exact):
        return exact

    # TRT10 specific: yolo26m_trt10.engine
    trt10 = f"{base}_trt10.engine"
    if os.path.exists(trt10):
        return trt10

    # Any engine with same base: yolo26m*.engine
    pattern = f"{base}*.engine"
    engines = glob.glob(pattern)
    if engines:
        return max(engines, key=os.path.getmtime)

    return None


class TensorRTBackend:
    """
    Native TensorRT inference - 208 FPS (batch=8)
    PyTorch faqat CUDA memory management uchun ishlatiladi
    """

    def __init__(self, engine_path: str, max_batch: int = 8):
        import tensorrt as trt
        import torch
        from concurrent.futures import ThreadPoolExecutor
        self._torch = torch

        self.device = torch.device("cuda:0")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if not self.engine:
            raise RuntimeError(f"Engine yuklanmadi: {engine_path}")

        self.context = self.engine.create_execution_context()

        # IO tensor info
        self._input_name = None
        self._output_name = None
        self._input_shape = None
        self._output_shape = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
                self._input_shape = tuple(shape)
            else:
                self._output_name = name
                self._output_shape = tuple(shape)

        self.max_batch = self._input_shape[0]
        self._imgsz = self._input_shape[2]  # H = W

        # Pre-allocate GPU buffers (max batch)
        self._input_buf = torch.zeros(self._input_shape, dtype=torch.float32, device=self.device)
        self._output_buf = torch.zeros(self._output_shape, dtype=torch.float32, device=self.device)
        self._stream = torch.cuda.Stream()

        # Parallel preprocess pool (cv2/numpy release GIL - real parallelism)
        self._preprocess_pool = ThreadPoolExecutor(max_workers=min(max_batch, 4))

        print(f"[TensorRT] Engine: {engine_path}")
        print(f"[TensorRT] Input: {self._input_name} {self._input_shape}")
        print(f"[TensorRT] Output: {self._output_name} {self._output_shape}")
        print(f"[TensorRT] Max batch: {self.max_batch}, ImgSz: {self._imgsz}")

    @property
    def imgsz(self) -> int:
        return self._imgsz

    def _preprocess_one(self, frame: np.ndarray) -> np.ndarray:
        """Single frame: letterbox + BGR→RGB + HWC→CHW + normalize (CPU, GIL-free)"""
        img = self._letterbox(frame, self._imgsz)
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(img)

    def preprocess(self, frames: List[np.ndarray]) -> None:
        """BGR frames -> normalized NCHW tensor (PARALLEL preprocess + batch GPU copy)"""
        batch_size = len(frames)

        # Parallel CPU preprocess (cv2.resize + numpy release GIL → real parallelism)
        # 8 frames: ~80ms serial → ~20ms parallel (4 threads)
        processed = list(self._preprocess_pool.map(self._preprocess_one, frames))

        # Batch stack + single GPU copy (faster than N individual copies)
        batch_np = np.stack(processed)  # (N, 3, H, W)
        self._input_buf[:batch_size].copy_(self._torch.from_numpy(batch_np))

        # Zero out unused batch slots
        if batch_size < self.max_batch:
            self._input_buf[batch_size:].zero_()

    def infer(self) -> np.ndarray:
        """Run TensorRT inference, return output as numpy"""
        self.context.set_tensor_address(self._input_name, self._input_buf.data_ptr())
        self.context.set_tensor_address(self._output_name, self._output_buf.data_ptr())

        with self._torch.cuda.stream(self._stream):
            self.context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        return self._output_buf.cpu().numpy()

    @staticmethod
    def _letterbox(img: np.ndarray, new_shape: int) -> np.ndarray:
        """Resize with letterbox (aspect ratio preserved, gray padding)"""
        h, w = img.shape[:2]
        r = min(new_shape / h, new_shape / w)
        new_w, new_h = int(w * r), int(h * r)

        if (new_w, new_h) != (w, h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        dw = (new_shape - new_w) // 2
        dh = (new_shape - new_h) // 2
        if dw > 0 or dh > 0 or new_w != new_shape or new_h != new_shape:
            padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
            padded[dh:dh + new_h, dw:dw + new_w] = img
            img = padded

        return img


class UltralyticsBackend:
    """Fallback: Ultralytics YOLO (PyTorch/ONNX)"""

    def __init__(self, model_path: str, conf: float, iou: float,
                 imgsz: int, device: str, half: bool, classes: Optional[List[int]]):
        from ultralytics import YOLO
        import torch

        self._model = YOLO(model_path, task='detect')
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._device = device
        self._half = half
        self._classes = classes

        # Warmup
        if torch.cuda.is_available():
            dummy = [np.zeros((imgsz, imgsz, 3), dtype=np.uint8)] * 4
            self._model.predict(dummy, conf=conf, device=device, half=half, verbose=False)
            torch.cuda.synchronize()

        self.class_names = self._model.names if hasattr(self._model, 'names') else {}

    @property
    def imgsz(self) -> int:
        return self._imgsz

    def predict(self, frames: List[np.ndarray]) -> list:
        return self._model.predict(
            frames,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            half=self._half,
            verbose=False,
            max_det=50,
            agnostic_nms=True,
            classes=self._classes,
        )


class RealtimeMultiCameraDetector:
    """
    8+ kamera uchun REAL-TIME detection

    TensorRT mode: 208 FPS (native TRT API + torch CUDA)
    Fallback mode: Ultralytics YOLO (PyTorch/ONNX)

    Arxitektura:
    - Har bir kamera o'z kadrini submit qiladi
    - Har 15ms da BARCHA kadrlar BITTA batch qilib process qilinadi
    - Natijalar darhol qaytariladi
    """

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
        filter_classes: List[int] = None,
        batch_interval_ms: float = 15.0,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.filter_classes = set(filter_classes) if filter_classes else None
        self.batch_interval = batch_interval_ms / 1000.0

        # Backend (TensorRT or Ultralytics)
        self._trt: Optional[TensorRTBackend] = None
        self._ultralytics: Optional[UltralyticsBackend] = None
        self._class_names: Dict = {}
        self._is_loaded = False
        self._model_type = "unknown"

        # Frame buffer: camera_id -> (frame, timestamp, event)
        self._pending_frames: Dict[str, Tuple[np.ndarray, float, threading.Event]] = {}
        self._frames_lock = threading.Lock()

        # Results cache
        self._results: Dict[str, List[Detection]] = {}
        self._results_lock = threading.Lock()

        # Original frame sizes for bbox scaling
        self._frame_sizes: Dict[str, Tuple[int, int]] = {}

        # Worker
        self._running = False
        self._worker_thread = None

        # Stats
        self._inference_times = []
        self._batch_sizes = []
        self._processed_count = 0

    def load(self) -> bool:
        """Model yuklash - TensorRT native > Ultralytics fallback"""
        if self._is_loaded:
            return True

        # 1. TensorRT native (eng tez - 208 FPS)
        engine_path = find_engine(str(self.model_path))
        if engine_path:
            try:
                self._trt = TensorRTBackend(engine_path)
                self._model_type = "tensorrt"
                self._class_names = COCO_NAMES.copy()
                self._is_loaded = True
                self._start_worker()
                print(f"[RealtimeDetector] TensorRT NATIVE mode! Engine: {engine_path}")
                return True
            except Exception as e:
                print(f"[RealtimeDetector] TensorRT xato: {e}")
                self._trt = None

        # 2. Ultralytics fallback (ONNX > PyTorch)
        candidates = []
        onnx_path = os.path.splitext(str(self.model_path))[0] + ".onnx"
        if os.path.exists(onnx_path):
            candidates.append((onnx_path, "onnx"))
        candidates.append((str(self.model_path), "pytorch"))

        for model_path, mtype in candidates:
            try:
                print(f"[RealtimeDetector] {mtype.upper()} yuklanmoqda: {model_path}")
                self._ultralytics = UltralyticsBackend(
                    model_path, self.confidence_threshold, self.iou_threshold,
                    self.imgsz, self.device, self.half,
                    list(self.filter_classes) if self.filter_classes else None,
                )
                self._model_type = mtype
                self._class_names = self._ultralytics.class_names
                self._is_loaded = True
                self._start_worker()
                print(f"[RealtimeDetector] Yuklandi! Mode: {mtype.upper()}")
                return True
            except Exception as e:
                print(f"[RealtimeDetector] {mtype.upper()} xato: {e}")
                self._ultralytics = None

        print("[RealtimeDetector] Hech qaysi model yuklanmadi!")
        return False

    def _start_worker(self):
        self._running = True
        self._worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self._worker_thread.start()

    def _batch_worker(self):
        """ASOSIY WORKER - GPU 100% ishlatish"""
        while self._running:
            time.sleep(self.batch_interval)

            with self._frames_lock:
                if not self._pending_frames:
                    continue
                batch_data = []
                for cam_id, (frame, ts, event) in self._pending_frames.items():
                    batch_data.append((cam_id, frame, event))
                self._pending_frames.clear()

            if not batch_data:
                continue

            try:
                start = time.perf_counter()
                frames = [item[1] for item in batch_data]
                batch_size = len(frames)

                if self._trt:
                    detections_list = self._infer_tensorrt(frames)
                else:
                    detections_list = self._infer_ultralytics(frames)

                inference_ms = (time.perf_counter() - start) * 1000

                # Stats
                self._inference_times.append(inference_ms)
                self._batch_sizes.append(batch_size)
                if len(self._inference_times) > 50:
                    self._inference_times.pop(0)
                    self._batch_sizes.pop(0)
                self._processed_count += batch_size

                # Signal results (detections + original frame for aligned drawing)
                with self._results_lock:
                    for (cam_id, frame, event), dets in zip(batch_data, detections_list):
                        self._results[cam_id] = (dets, frame)
                        event.set()

            except Exception as e:
                print(f"[RealtimeDetector] Batch error: {e}")
                for cam_id, frame, event in batch_data:
                    event.set()

    def _infer_tensorrt(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """TensorRT native inference + NMS"""
        batch_size = len(frames)
        orig_sizes = [(f.shape[0], f.shape[1]) for f in frames]

        # Pad to max_batch if needed (engine has fixed batch)
        if batch_size < self._trt.max_batch:
            padded = frames + [np.zeros((self._trt.imgsz, self._trt.imgsz, 3), dtype=np.uint8)] * (self._trt.max_batch - batch_size)
        else:
            padded = frames[:self._trt.max_batch]

        self._trt.preprocess(padded)
        raw_output = self._trt.infer()  # (max_batch, 300, 6)

        # Parse output for each real frame
        results = []
        for i in range(batch_size):
            dets = self._parse_trt_output(raw_output[i], orig_sizes[i])
            results.append(dets)

        return results

    def _parse_trt_output(self, output: np.ndarray, orig_size: Tuple[int, int]) -> List[Detection]:
        """
        Parse TensorRT YOLO output (300, 6) -> List[Detection]
        Format: [x1, y1, x2, y2, confidence, class_id]
        Coordinates are in model input space (640x640 letterboxed)
        """
        detections = []
        orig_h, orig_w = orig_size
        imgsz = self._trt.imgsz

        # Scale factor from letterbox
        r = min(imgsz / orig_h, imgsz / orig_w)
        new_w, new_h = int(orig_w * r), int(orig_h * r)
        dw = (imgsz - new_w) / 2
        dh = (imgsz - new_h) / 2

        for det in output:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)

            if conf < self.confidence_threshold:
                continue
            if self.filter_classes and cls_id not in self.filter_classes:
                continue

            # Unscale from letterbox to original
            x1 = int((x1 - dw) / r)
            y1 = int((y1 - dh) / r)
            x2 = int((x2 - dw) / r)
            y2 = int((y2 - dh) / r)

            # Clamp
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            if x2 <= x1 or y2 <= y1:
                continue

            cls_name = self._class_names.get(cls_id, f"class_{cls_id}")
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                class_id=cls_id,
                class_name=cls_name,
            ))

        # NMS (simple IoU-based)
        detections = self._nms(detections)
        return detections

    @staticmethod
    def _nms(detections: List[Detection], iou_thresh: float = 0.45) -> List[Detection]:
        """Simple NMS"""
        if len(detections) <= 1:
            return detections

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [
                d for d in detections
                if _iou(best.bbox, d.bbox) < iou_thresh
            ]

        return keep

    def _infer_ultralytics(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Ultralytics YOLO inference"""
        import torch
        results = self._ultralytics.predict(frames)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        detections_list = []
        for result in results:
            dets = self._parse_ultralytics_result(result)
            detections_list.append(dets)
        return detections_list

    def _parse_ultralytics_result(self, result) -> List[Detection]:
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
            if self.filter_classes and cls_id not in self.filter_classes:
                continue

            x1, y1, x2, y2 = xyxy_all[i]
            conf = float(conf_all[i])
            cls_name = self._class_names.get(cls_id, f"class_{cls_id}")

            detections.append(Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name,
            ))

        return detections

    def detect(self, frame: np.ndarray, camera_id: str, timeout: float = 0.05) -> List[Detection]:
        """BLOCKING detection - frame submit qiladi, natijani kutadi"""
        if not self._is_loaded:
            return []

        event = threading.Event()

        with self._frames_lock:
            self._pending_frames[camera_id] = (frame, time.time(), event)

        event.wait(timeout=timeout)
        with self._results_lock:
            result = self._results.get(camera_id)
            if result is None:
                return []
            return result[0]

    def detect_async(self, frame: np.ndarray, camera_id: str) -> tuple:
        """NON-BLOCKING detection - (detections, det_frame) qaytaradi.

        Frame batch processingga qo'yiladi. Oldingi batch natijalarini
        va ularning ORIGINAL frameini darhol qaytaradi.
        Boxlar det_frame ga chizilsa - objectlar bilan 100% mos bo'ladi.
        """
        if not self._is_loaded:
            return ([], None)

        event = threading.Event()
        with self._frames_lock:
            self._pending_frames[camera_id] = (frame, time.time(), event)

        # Cached natijalarni DARHOL qaytarish - BLOKLAMASDAN
        with self._results_lock:
            result = self._results.get(camera_id)
            if result is None:
                return ([], None)
            return result  # (detections, original_frame)

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        thickness: int = 2,
        font_scale: float = 0.5
    ) -> np.ndarray:
        """Detection boxlarni chizish"""
        result = frame.copy()
        h, w = result.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.COLORS.get(det.class_id, self.DEFAULT_COLOR)

            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            label = f"{det.class_name} {det.confidence:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

            pad = 2
            lx = x1 + pad
            ly = y1 + th + pad
            if y1 < 5:
                ly = y1 + th + pad + 5

            bg_x1 = x1
            bg_y1 = ly - th - pad
            bg_x2 = min(x1 + tw + pad * 2 + 2, w)
            bg_y2 = ly + pad

            cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(result, label, (lx, ly), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        return result

    def get_fps(self) -> float:
        if not self._inference_times:
            return 0.0
        avg_ms = sum(self._inference_times) / len(self._inference_times)
        avg_batch = sum(self._batch_sizes) / len(self._batch_sizes) if self._batch_sizes else 1
        return (1000.0 / avg_ms) * avg_batch if avg_ms > 0 else 0.0

    def get_stats(self) -> Dict:
        avg_ms = sum(self._inference_times) / len(self._inference_times) if self._inference_times else 0
        avg_batch = sum(self._batch_sizes) / len(self._batch_sizes) if self._batch_sizes else 0
        return {
            "processed": self._processed_count,
            "avg_batch_ms": avg_ms,
            "avg_batch_size": avg_batch,
            "per_frame_ms": avg_ms / avg_batch if avg_batch > 0 else 0,
            "fps": self.get_fps(),
            "model_type": self._model_type,
        }

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def class_names(self) -> Dict:
        return self._class_names.copy()

    def stop(self):
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

    def __del__(self):
        self.stop()

    def __repr__(self):
        return f"RealtimeMultiCameraDetector(fps={self.get_fps():.1f}, mode={self._model_type.upper()})"


def _iou(box1: Tuple, box2: Tuple) -> float:
    """IoU hisoblash"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0
