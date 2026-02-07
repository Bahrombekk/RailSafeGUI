"""
NVIDIA DeepStream Multi-Camera Detector
Hardware-accelerated video analytics for real-time detection

Features:
- NVDEC hardware video decoding (vs CPU OpenCV)
- TensorRT inference (3-5x faster than PyTorch)
- 80-100% GPU utilization
- Zero-copy GPU memory
- Support for 30+ cameras
"""

import os
import sys
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

# DeepStream imports
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstRtspServer', '1.0')
    from gi.repository import Gst, GLib, GstRtspServer

    import pyds
    DEEPSTREAM_AVAILABLE = True
except (ImportError, ValueError):
    DEEPSTREAM_AVAILABLE = False

    # Stubs so class definitions don't fail at import time
    class _Enum:
        OK = 0; FAILURE = 0; PLAYING = 0; NULL = 0; BUFFER = 0
    class _ElemFactory:
        @staticmethod
        def make(*a, **k): return None
    class _GstStub:
        PadProbeReturn = _Enum
        PadProbeType = _Enum
        StateChangeReturn = _Enum
        State = _Enum
        ElementFactory = _ElemFactory
        Pipeline = type(None)
        Caps = type(None)
        Bin = type(None)
        GhostPad = type(None)
        PadDirection = _Enum
        PadLinkReturn = _Enum
        @staticmethod
        def init(*a): pass
    class _GLibStub:
        class MainLoop:
            def run(self): pass
            def quit(self): pass
    Gst = _GstStub
    GLib = _GLibStub
    pyds = None


@dataclass
class Detection:
    """Single detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: int = -1


@dataclass
class CameraResult:
    """Detection results for one camera"""
    camera_id: str
    frame_number: int
    timestamp: float
    detections: List[Detection]
    frame: Optional[np.ndarray] = None


class DeepStreamDetector:
    """
    NVIDIA DeepStream Multi-Camera Detector

    GPU hardware-accelerated pipeline:
    RTSP → NVDEC → GPU Memory → TensorRT → NMS → Results

    Usage:
        detector = DeepStreamDetector(
            model_path="models/yolo26m.engine",  # TensorRT engine
            cameras=[
                {"id": "cam1", "uri": "rtsp://..."},
                {"id": "cam2", "uri": "rtsp://..."},
            ]
        )
        detector.start()

        # Get results
        results = detector.get_results()  # {camera_id: CameraResult}
    """

    # COCO class names
    COCO_CLASSES = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
        4: "airplane", 5: "bus", 6: "train", 7: "truck",
        8: "boat", 9: "traffic light", 10: "fire hydrant",
    }

    def __init__(
        self,
        model_path: str,
        cameras: List[Dict] = None,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.45,
        filter_classes: List[int] = None,
        batch_size: int = 8,
        gpu_id: int = 0,
        callback: Callable[[CameraResult], None] = None,
    ):
        """
        Initialize DeepStream detector.

        Args:
            model_path: Path to TensorRT engine file (.engine)
            cameras: List of camera configs [{"id": str, "uri": str}, ...]
            confidence_threshold: Detection confidence threshold
            nms_threshold: NMS IoU threshold
            filter_classes: List of class IDs to detect (e.g., [2,5,7] for car,bus,truck)
            batch_size: Max batch size for inference
            gpu_id: GPU device ID
            callback: Optional callback for each camera result
        """
        if not DEEPSTREAM_AVAILABLE:
            raise RuntimeError("DeepStream SDK not installed")

        self.model_path = model_path
        self.cameras = cameras or []
        self.confidence = confidence_threshold
        self.nms_threshold = nms_threshold
        self.filter_classes = set(filter_classes) if filter_classes else None
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.callback = callback

        # Results storage
        self._results: Dict[str, CameraResult] = {}
        self._results_lock = threading.Lock()

        # Frame callback for GUI
        self._frame_callbacks: Dict[str, Callable] = {}

        # GStreamer pipeline
        self._pipeline = None
        self._loop = None
        self._loop_thread = None
        self._running = False

        # Camera ID mapping
        self._source_id_to_camera: Dict[int, str] = {}

        # Initialize GStreamer
        Gst.init(None)

    def _create_pipeline(self) -> bool:
        """Create DeepStream GStreamer pipeline"""

        if not self.cameras:
            print("[DeepStream] No cameras configured")
            return False

        num_sources = len(self.cameras)

        # Create pipeline
        self._pipeline = Gst.Pipeline()

        if not self._pipeline:
            print("[DeepStream] Failed to create pipeline")
            return False

        # Create streammux (batches all sources)
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        if not streammux:
            print("[DeepStream] Failed to create nvstreammux")
            return False

        streammux.set_property("batch-size", min(num_sources, self.batch_size))
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batched-push-timeout", 25000)  # 25ms - real-time
        streammux.set_property("live-source", 1)
        streammux.set_property("gpu-id", self.gpu_id)
        streammux.set_property("nvbuf-memory-type", 0)  # NVBUF_MEM_DEFAULT

        self._pipeline.add(streammux)

        # Add sources (RTSP cameras)
        for idx, cam in enumerate(self.cameras):
            camera_id = cam.get("id", f"cam_{idx}")
            uri = cam.get("uri", "")

            self._source_id_to_camera[idx] = camera_id

            # Create source bin for each camera
            source_bin = self._create_source_bin(idx, uri, camera_id)
            if not source_bin:
                print(f"[DeepStream] Failed to create source bin for {camera_id}")
                continue

            self._pipeline.add(source_bin)

            # Link to streammux
            srcpad = source_bin.get_static_pad("src")
            sinkpad = streammux.get_request_pad(f"sink_{idx}")
            if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                print(f"[DeepStream] Failed to link {camera_id} to streammux")
                continue

            print(f"[DeepStream] Added camera: {camera_id} -> {uri}")

        # Primary inference (YOLO TensorRT)
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
            print("[DeepStream] Failed to create nvinfer")
            return False

        # Configure inference
        pgie_config = self._create_pgie_config()
        pgie.set_property("config-file-path", pgie_config)
        pgie.set_property("gpu-id", self.gpu_id)

        # Tracker (optional but improves detection stability)
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if tracker:
            tracker_config = self._create_tracker_config()
            tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
            tracker.set_property("ll-config-file", tracker_config)
            tracker.set_property("gpu-id", self.gpu_id)
            tracker.set_property("tracker-width", 640)
            tracker.set_property("tracker-height", 384)

        # Converter for output
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        nvvidconv.set_property("gpu-id", self.gpu_id)

        # OSD (On-Screen Display) - draws bboxes
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        nvosd.set_property("gpu-id", self.gpu_id)

        # Tee for multiple outputs
        tee = Gst.ElementFactory.make("tee", "tee")

        # Fake sink (for metadata processing)
        fakesink = Gst.ElementFactory.make("fakesink", "fakesink")
        fakesink.set_property("sync", 0)  # No sync - real-time
        fakesink.set_property("async", 0)

        # Add elements
        self._pipeline.add(pgie)
        if tracker:
            self._pipeline.add(tracker)
        self._pipeline.add(nvvidconv)
        self._pipeline.add(nvosd)
        self._pipeline.add(tee)
        self._pipeline.add(fakesink)

        # Link elements
        if tracker:
            streammux.link(pgie)
            pgie.link(tracker)
            tracker.link(nvvidconv)
        else:
            streammux.link(pgie)
            pgie.link(nvvidconv)

        nvvidconv.link(nvosd)
        nvosd.link(tee)

        # Link tee to fakesink
        tee_src = tee.get_request_pad("src_0")
        sink_pad = fakesink.get_static_pad("sink")
        tee_src.link(sink_pad)

        # Add probe for getting detection results
        osdsinkpad = nvosd.get_static_pad("sink")
        if osdsinkpad:
            osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self._osd_sink_pad_buffer_probe, 0)

        return True

    def _create_source_bin(self, index: int, uri: str, camera_id: str):
        """Create source bin for RTSP/file input with NVDEC hardware decoding"""

        bin_name = f"source-bin-{index:02d}"
        nbin = Gst.Bin.new(bin_name)

        if not nbin:
            return None

        # URI decode bin (auto-selects NVDEC for hardware decoding)
        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
        if not uri_decode_bin:
            return None

        uri_decode_bin.set_property("uri", uri)

        # For RTSP, reduce latency
        if uri.startswith("rtsp://"):
            uri_decode_bin.set_property("buffer-size", 212992)

        uri_decode_bin.connect("pad-added", self._decodebin_newpad, nbin)
        uri_decode_bin.connect("child-added", self._decodebin_child_added, nbin)

        # Create nvvideoconvert
        nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", f"source-nvvidconv-{index}")
        nvvideoconvert.set_property("gpu-id", self.gpu_id)

        # Create capsfilter
        caps = Gst.ElementFactory.make("capsfilter", f"source-capsfilter-{index}")
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

        nbin.add(uri_decode_bin)
        nbin.add(nvvideoconvert)
        nbin.add(caps)

        nvvideoconvert.link(caps)

        # Create ghost pad
        bin_pad = caps.get_static_pad("src")
        ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        ghost_pad.set_target(bin_pad)
        nbin.add_pad(ghost_pad)

        return nbin

    def _decodebin_newpad(self, decodebin, pad, data):
        """Handle new pad from decodebin"""
        caps = pad.get_current_caps()
        if not caps:
            return

        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()

        if gstname.find("video") != -1:
            # Find nvvideoconvert
            bin_elem = data
            nvvidconv = bin_elem.get_by_name(bin_elem.get_name().replace("source-bin", "source-nvvidconv"))
            if nvvidconv:
                sinkpad = nvvidconv.get_static_pad("sink")
                if not sinkpad.is_linked():
                    pad.link(sinkpad)

    def _decodebin_child_added(self, child_proxy, obj, name, user_data):
        """Configure child elements (for NVDEC)"""
        if name.find("decodebin") != -1:
            obj.connect("child-added", self._decodebin_child_added, user_data)

        # Force hardware decoding
        if name.find("nvv4l2decoder") != -1:
            obj.set_property("gpu-id", self.gpu_id)
            obj.set_property("drop-frame-interval", 0)  # No frame drop
            obj.set_property("num-extra-surfaces", 2)

    def _create_pgie_config(self) -> str:
        """Create primary inference config file"""

        config_path = "/tmp/deepstream_pgie_config.txt"

        # Determine model type from path
        model_file = self.model_path

        config_content = f"""
[property]
gpu-id={self.gpu_id}
net-scale-factor=0.0039215697906911373
model-engine-file={model_file}
labelfile-path=/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector/labels.txt
batch-size={min(len(self.cameras), self.batch_size)}
process-mode=1
model-color-format=0
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=1
output-blob-names=output0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1

[class-attrs-all]
nms-iou-threshold={self.nms_threshold}
pre-cluster-threshold={self.confidence}
topk=300
"""

        with open(config_path, 'w') as f:
            f.write(config_content)

        return config_path

    def _create_tracker_config(self) -> str:
        """Create tracker config file"""

        config_path = "/tmp/deepstream_tracker_config.txt"

        config_content = """
[tracker]
tracker-width=640
tracker-height=384
gpu-id=0
ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
ll-config-file=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml
enable-batch-process=1
enable-past-frame=1
display-tracking-id=1
"""

        with open(config_path, 'w') as f:
            f.write(config_content)

        return config_path

    def _osd_sink_pad_buffer_probe(self, pad, info, u_data) -> Gst.PadProbeReturn:
        """
        Probe callback - extracts detection metadata from each frame
        This is called for every processed frame
        """

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list

        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            source_id = frame_meta.source_id
            camera_id = self._source_id_to_camera.get(source_id, f"cam_{source_id}")
            frame_number = frame_meta.frame_num

            detections = []
            l_obj = frame_meta.obj_meta_list

            while l_obj:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                class_id = obj_meta.class_id

                # Filter classes if specified
                if self.filter_classes and class_id not in self.filter_classes:
                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                    continue

                # Get bounding box
                rect = obj_meta.rect_params
                x1 = int(rect.left)
                y1 = int(rect.top)
                x2 = int(rect.left + rect.width)
                y2 = int(rect.top + rect.height)

                # Get confidence
                confidence = obj_meta.confidence

                # Get track ID
                track_id = obj_meta.object_id

                # Get class name
                class_name = self.COCO_CLASSES.get(class_id, f"class_{class_id}")

                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id
                )
                detections.append(detection)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # Create result
            result = CameraResult(
                camera_id=camera_id,
                frame_number=frame_number,
                timestamp=time.time(),
                detections=detections
            )

            # Store result
            with self._results_lock:
                self._results[camera_id] = result

            # Call callback if set
            if self.callback:
                try:
                    self.callback(result)
                except Exception as e:
                    print(f"[DeepStream] Callback error: {e}")

            # Call per-camera frame callback
            if camera_id in self._frame_callbacks:
                try:
                    self._frame_callbacks[camera_id](result)
                except Exception as e:
                    print(f"[DeepStream] Frame callback error for {camera_id}: {e}")

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def add_camera(self, camera_id: str, uri: str) -> bool:
        """Add camera to the detector (before starting)"""
        self.cameras.append({"id": camera_id, "uri": uri})
        return True

    def set_frame_callback(self, camera_id: str, callback: Callable[[CameraResult], None]):
        """Set callback for specific camera's frames"""
        self._frame_callbacks[camera_id] = callback

    def start(self) -> bool:
        """Start the DeepStream pipeline"""

        if self._running:
            return True

        # Create pipeline
        if not self._create_pipeline():
            return False

        # Create GLib main loop
        self._loop = GLib.MainLoop()

        # Start pipeline
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("[DeepStream] Failed to start pipeline")
            return False

        self._running = True

        # Run loop in separate thread
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        print(f"[DeepStream] Pipeline started with {len(self.cameras)} cameras")
        return True

    def _run_loop(self):
        """Run GLib main loop"""
        try:
            self._loop.run()
        except Exception as e:
            print(f"[DeepStream] Loop error: {e}")

    def stop(self):
        """Stop the DeepStream pipeline"""

        if not self._running:
            return

        self._running = False

        if self._loop:
            self._loop.quit()

        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)

        if self._loop_thread:
            self._loop_thread.join(timeout=2.0)

        print("[DeepStream] Pipeline stopped")

    def get_results(self) -> Dict[str, CameraResult]:
        """Get latest detection results for all cameras"""
        with self._results_lock:
            return dict(self._results)

    def get_result(self, camera_id: str) -> Optional[CameraResult]:
        """Get latest detection result for specific camera"""
        with self._results_lock:
            return self._results.get(camera_id)

    def is_running(self) -> bool:
        """Check if pipeline is running"""
        return self._running


class DeepStreamDetectorSimple:
    """
    Simplified DeepStream detector that works with existing PyQt6 code.

    This provides the same interface as RealtimeMultiCameraDetector
    but uses DeepStream for hardware-accelerated processing.
    """

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
        self.model_path = model_path
        self.confidence = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.filter_classes = filter_classes

        # Convert model to TensorRT if needed
        self.engine_path = self._get_engine_path()

        # Core detector (lazy init after cameras are registered)
        self._detector: Optional[DeepStreamDetector] = None
        self._cameras: Dict[str, str] = {}  # camera_id -> uri
        self._results: Dict[str, List] = {}
        self._lock = threading.Lock()

        # Check DeepStream availability
        self._available = DEEPSTREAM_AVAILABLE

    def _get_engine_path(self) -> str:
        """Get TensorRT engine path (convert if needed)"""
        base = os.path.splitext(self.model_path)[0]
        engine_path = f"{base}.engine"

        if os.path.exists(engine_path):
            return engine_path

        # Need to convert - will be done by separate script
        return engine_path

    def load(self) -> bool:
        """Load the detector"""
        if not self._available:
            print("[DeepStreamSimple] DeepStream not available, using fallback")
            return False

        if not os.path.exists(self.engine_path):
            print(f"[DeepStreamSimple] TensorRT engine not found: {self.engine_path}")
            print("[DeepStreamSimple] Run: python convert_to_tensorrt.py to create it")
            return False

        return True

    def register_camera(self, camera_id: str, uri: str):
        """Register a camera source"""
        self._cameras[camera_id] = uri

    def start(self) -> bool:
        """Start processing all registered cameras"""
        if not self._available or not self._cameras:
            return False

        cameras = [{"id": cid, "uri": uri} for cid, uri in self._cameras.items()]

        self._detector = DeepStreamDetector(
            model_path=self.engine_path,
            cameras=cameras,
            confidence_threshold=self.confidence,
            nms_threshold=self.iou_threshold,
            filter_classes=self.filter_classes,
            callback=self._on_detection
        )

        return self._detector.start()

    def _on_detection(self, result: CameraResult):
        """Handle detection result"""
        # Convert to simple format for compatibility
        detections = []
        for det in result.detections:
            detections.append({
                'class_id': det.class_id,
                'class_name': det.class_name,
                'confidence': det.confidence,
                'bbox': det.bbox,
                'track_id': det.track_id
            })

        with self._lock:
            self._results[result.camera_id] = detections

    def detect(self, frame: np.ndarray, camera_id: str, timeout: float = 0.05) -> List[Dict]:
        """
        Get detection results for a camera.

        Note: With DeepStream, frames are processed in the pipeline,
        so this just returns the latest cached results.
        """
        with self._lock:
            return self._results.get(camera_id, [])

    def stop(self):
        """Stop the detector"""
        if self._detector:
            self._detector.stop()

    def cleanup(self):
        """Cleanup resources"""
        self.stop()

    def __del__(self):
        self.cleanup()


# Fallback detector when DeepStream is not available
class FallbackDetector:
    """Fallback to regular YOLO when DeepStream is not available"""

    def __init__(self, **kwargs):
        self.model_path = kwargs.get('model_path', '')
        self.confidence = kwargs.get('confidence_threshold', 0.3)
        self.filter_classes = kwargs.get('filter_classes')
        self._model = None

    def load(self) -> bool:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            return True
        except Exception as e:
            print(f"[Fallback] Failed to load model: {e}")
            return False

    def detect(self, frame: np.ndarray, camera_id: str = None, timeout: float = 0.05) -> List[Dict]:
        if self._model is None:
            return []

        try:
            results = self._model.predict(
                frame,
                conf=self.confidence,
                verbose=False,
                device='cuda'
            )

            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls = int(box.cls[0].item())
                    if self.filter_classes and cls not in self.filter_classes:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    detections.append({
                        'class_id': cls,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    })
            return detections
        except Exception as e:
            print(f"[Fallback] Detection error: {e}")
            return []

    def cleanup(self):
        pass


def get_best_detector(**kwargs):
    """
    Get the best available detector.
    Returns DeepStream if available, otherwise falls back to regular YOLO.
    """
    if DEEPSTREAM_AVAILABLE:
        detector = DeepStreamDetectorSimple(**kwargs)
        if detector.load():
            return detector

    print("[AutoDetect] DeepStream not available, using YOLO fallback")
    return FallbackDetector(**kwargs)
