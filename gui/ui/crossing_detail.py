"""
Crossing Detail View - Responsive cameras with grid layout, auto-reconnect
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QScrollArea, QFrame,
                              QSizePolicy, QApplication,
                              QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QMutex
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np
import os
import time
import json
import threading

from gui.utils.theme_colors import C
from gui.utils.language import t, LM
from gui.utils.polygon_tracker import PolygonTracker
from gui.widgets.hourly_chart import HourlyChartPanel

# RTSP ultra-low-latency: UDP transport (TCP dan tezroq), minimal buffer
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;udp|stimeout;2000000|'
    'fflags;nobuffer+discardcorrupt|flags;low_delay|'
    'analyzeduration;0|probesize;32|'
    'max_delay;0|reorder_queue_size;0|'
    'thread_queue_size;1'
)
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # AV_LOG_QUIET

# Intel GPU OpenCL — resize va color conversion uchun
def _init_opencl() -> bool:
    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            dev = cv2.ocl.Device.getDefault()
            print(f"[OpenCL] Intel GPU: {dev.name()} — resize/cvtColor GPU da")
            return True
    except Exception:
        pass
    print("[OpenCL] Mavjud emas — CPU ishlatiladi")
    return False

_USE_OPENCL = _init_opencl()

# GStreamer mavjudligini bir marta tekshirish (har kamera uchun 8 urinish oldini oladi)
def _check_gstreamer() -> bool:
    try:
        info = cv2.getBuildInformation()
        idx = info.find("GStreamer")
        return idx != -1 and "YES" in info[idx:idx + 40]
    except Exception:
        return False

_HAS_GSTREAMER = _check_gstreamer()


def _open_camera(source: str, camera_name: str = "") -> tuple:
    """RTSP kamerani ochish: D3D11 → QSV → NVDEC → CPU → FFmpeg.
    Returns (cap, backend_name) or (None, None)."""
    is_rtsp = source.lower().startswith("rtsp://")

    if is_rtsp and _HAS_GSTREAMER:
        _appsink = "appsink drop=true max-buffers=1 sync=false"

        # 1) Intel/AMD/NVIDIA Direct3D11 H.265 (Windows GPU universal)
        cap = cv2.VideoCapture(
            f"rtspsrc location={source} latency=0 protocols=tcp ! "
            f"rtph265depay ! h265parse ! d3d11h265dec ! "
            f"d3d11convert ! video/x-raw(memory:SystemMemory),format=BGR ! {_appsink}",
            cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[{camera_name}] D3D11 GPU H.265")
            return cap, "d3d11-h265"

        # 2) Intel/AMD/NVIDIA Direct3D11 H.264
        cap = cv2.VideoCapture(
            f"rtspsrc location={source} latency=0 protocols=tcp ! "
            f"rtph264depay ! h264parse ! d3d11h264dec ! "
            f"d3d11convert ! video/x-raw(memory:SystemMemory),format=BGR ! {_appsink}",
            cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[{camera_name}] D3D11 GPU H.264")
            return cap, "d3d11-h264"

        # 3) Intel Quick Sync H.265 (iGPU / Arc)
        cap = cv2.VideoCapture(
            f"rtspsrc location={source} latency=0 protocols=tcp ! "
            f"rtph265depay ! h265parse ! msdkh265dec ! "
            f"videoconvert ! video/x-raw,format=BGR ! {_appsink}",
            cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[{camera_name}] Intel QSV H.265")
            return cap, "qsv-h265"

        # 4) Intel Quick Sync H.264
        cap = cv2.VideoCapture(
            f"rtspsrc location={source} latency=0 protocols=tcp ! "
            f"rtph264depay ! h264parse ! msdkh264dec ! "
            f"videoconvert ! video/x-raw,format=BGR ! {_appsink}",
            cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[{camera_name}] Intel QSV H.264")
            return cap, "qsv-h264"

        # 5) NVIDIA NVDEC H.265
        cap = cv2.VideoCapture(
            f"rtspsrc location={source} latency=0 protocols=tcp ! "
            f"rtph265depay ! h265parse ! nvh265dec ! "
            f"videoconvert ! video/x-raw,format=BGR ! {_appsink}",
            cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[{camera_name}] NVDEC GPU H.265")
            return cap, "nvdec-h265"

        # 6) NVIDIA NVDEC H.264
        cap = cv2.VideoCapture(
            f"rtspsrc location={source} latency=0 protocols=tcp ! "
            f"rtph264depay ! h264parse ! nvh264dec ! "
            f"videoconvert ! video/x-raw,format=BGR ! {_appsink}",
            cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[{camera_name}] NVDEC GPU H.264")
            return cap, "nvdec-h264"

        # 7) CPU software H.265
        cap = cv2.VideoCapture(
            f"rtspsrc location={source} latency=0 protocols=tcp ! "
            f"rtph265depay ! h265parse ! avdec_h265 ! "
            f"videoconvert ! video/x-raw,format=BGR ! {_appsink}",
            cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[{camera_name}] CPU H.265")
            return cap, "cpu-h265"

        # 8) CPU software H.264
        cap = cv2.VideoCapture(
            f"rtspsrc location={source} latency=0 protocols=tcp ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! video/x-raw,format=BGR ! {_appsink}",
            cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[{camera_name}] CPU H.264")
            return cap, "cpu-h264"

    # 9) FFmpeg fallback (har doim ishlaydi)
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[{camera_name}] FFmpeg fallback (CPU)")
        return cap, "ffmpeg"

    return None, None


def _load_polygon(polygon_file: str, frame_w: int, frame_h: int):
    """Polygon JSON yuklash va frame o'lchamiga scale qilish.
    Returns (poly_pts, poly_mask) or (None, None)."""
    if not polygon_file or not os.path.isfile(polygon_file):
        return None, None
    try:
        with open(polygon_file, 'r') as f:
            data = json.load(f)
        orig_w = data['images'][0]['width']
        orig_h = data['images'][0]['height']
        scale_x = frame_w / orig_w
        scale_y = frame_h / orig_h
        pts = np.array(data['annotations'][0]['segmentation'][0]).reshape(-1, 2)
        poly_pts = (pts * [scale_x, scale_y]).astype(np.int32)
        poly_mask = np.zeros((frame_h, frame_w), np.uint8)
        cv2.fillPoly(poly_mask, [poly_pts], 255)
        return poly_pts, poly_mask
    except Exception as e:
        print(f"[Polygon] Yuklab bo'lmadi: {polygon_file}: {e}")
        return None, None


class DetailCameraWorker(QThread):
    """Worker thread - all heavy work here, GUI only does setPixmap, auto-reconnect"""
    frame_ready = pyqtSignal()  # Lightweight - payload yo'q (queue backup bo'lmaydi)
    status_changed = pyqtSignal(str)
    stats_updated = pyqtSignal(int, int, int, float, float)  # light_count, heavy_count, in_poly_count, max_time, fps

    def __init__(self, source: str, camera_name: str = "Camera", display_width: int = 1920,
                 car_detector: 'CarDetector' = None, detection_enabled: bool = True,
                 polygon_file: str = None, warning_threshold: float = 10.0,
                 violation_threshold: float = 15.0, is_custom_model: bool = False):
        super().__init__()
        self.source = source
        self.camera_name = camera_name
        self.display_width = display_width
        self._running = True
        self._mutex = QMutex()
        self._retry_delay = 3
        self._latest_qimg = None  # Atomic latest frame (GIL-safe)

        # Car detector - non-blocking real-time mode
        self.car_detector = car_detector
        self.detection_enabled = detection_enabled and car_detector is not None

        # Polygon
        self.polygon_file = polygon_file
        self._poly_pts = None
        self._poly_mask = None
        self.warning_threshold = warning_threshold
        self.violation_threshold = violation_threshold
        self.is_custom_model = is_custom_model

    def take_frame(self):
        """Eng oxirgi kadrni olish - eski framelar avtomatik tashlanadi"""
        qimg = self._latest_qimg
        self._latest_qimg = None
        return qimg

    def run(self):
        retry_count = 0
        while self._is_running():
            cap = None
            gt = None
            _grab_running = [True]

            try:
                cap, backend = _open_camera(self.source, self.camera_name)
                if cap is not None:
                    # RTSP buffer tozalash — vaqt bo'yicha (300ms), tez flush
                    _flush_end = time.perf_counter() + 0.3
                    while time.perf_counter() < _flush_end:
                        cap.grab()

                    self.status_changed.emit("online")
                    retry_count = 0

                    # --- Dedicated grab thread ---
                    # grab() har doim tight loop — network buffer doim bo'sh
                    # retrieve() (decode) faqat main thread so'raganda — minimum latency
                    _frame_lock = threading.Lock()
                    _grab_error = [False]
                    _want_frame = [False]   # main thread: "menga frame kerak"
                    _latest_frame = [None]  # grab thread: dekod qilingan frame

                    def _grab_loop():
                        fails = 0
                        while _grab_running[0]:
                            ret = cap.grab()  # tez: faqat compressed packet o'qiydi
                            if not ret:
                                fails += 1
                                if fails > 30:
                                    _grab_error[0] = True
                                    break
                                continue
                            fails = 0
                            # Faqat main thread so'raganda decode — har doim eng yangi packet
                            if _want_frame[0]:
                                _want_frame[0] = False
                                ret2, frm = cap.retrieve()
                                if ret2:
                                    with _frame_lock:
                                        _latest_frame[0] = frm

                    gt = threading.Thread(target=_grab_loop, daemon=True)
                    gt.start()

                    # --- Main loop: eng oxirgi kadrni process qilish ---
                    _fc = 0
                    _fps_t = time.perf_counter()
                    _cam_fps = 0.0
                    _poly_loaded = False
                    _tracker = None  # PolygonTracker (polygon yuklangandan keyin yaratiladi)
                    _last_stats_emit = 0.0      # Stats throttle: max 4/sec
                    _last_in_poly = -1          # Polygon holatini kuzatish (darhol emit)

                    _target_interval = 1.0 / 30.0  # 30 FPS cap
                    _last_frame_t = 0.0

                    while self._is_running() and not _grab_error[0]:
                        # 30 FPS limitiga yetmagan bo'lsa grab threadga signal ber
                        _now_t = time.perf_counter()
                        if _now_t - _last_frame_t >= _target_interval:
                            _want_frame[0] = True

                        with _frame_lock:
                            frame = _latest_frame[0]
                            _latest_frame[0] = None

                        if frame is None:
                            time.sleep(0.002)
                            continue

                        _last_frame_t = time.perf_counter()
                        h, w = frame.shape[:2]

                        # Resize — Intel GPU (OpenCL UMat) yoki CPU
                        if w > self.display_width:
                            scale = self.display_width / w
                            new_w = self.display_width
                            new_h = int(h * scale)
                            if _USE_OPENCL:
                                umat = cv2.UMat(frame)
                                umat = cv2.resize(umat, (new_w, new_h),
                                                  interpolation=cv2.INTER_LINEAR)
                                frame = umat.get()
                            else:
                                frame = cv2.resize(frame, (new_w, new_h),
                                                   interpolation=cv2.INTER_LINEAR)
                            h, w = new_h, new_w

                        # Polygon yuklash (birinchi frame kelganda)
                        if not _poly_loaded:
                            _poly_loaded = True
                            if self.polygon_file:
                                self._poly_pts, self._poly_mask = _load_polygon(
                                    self.polygon_file, w, h)
                            if self._poly_mask is not None:
                                light_cls = PolygonTracker.CUSTOM_LIGHT if self.is_custom_model else None
                                heavy_cls = PolygonTracker.CUSTOM_HEAVY if self.is_custom_model else None
                                _tracker = PolygonTracker(
                                    poly_mask=self._poly_mask,
                                    iou_threshold=0.3,
                                    max_age=2.0,
                                    frame_width=w,
                                    frame_height=h,
                                    light_classes=light_cls,
                                    heavy_classes=heavy_cls,
                                )

                        # Car detection - NON-BLOCKING
                        detection_count = 0
                        in_poly_count = 0
                        max_time = 0.0
                        if self.detection_enabled and self.car_detector is not None:
                            try:
                                detections, det_frame = self.car_detector.detect_async(
                                    frame, camera_id=f"detail_{self.camera_name}")
                                detection_count = len(detections)
                                # Tracking + counting (har doim chaqiriladi — eski tracklar expire bo'lishi uchun)
                                in_poly_bboxes = None
                                if _tracker is not None:
                                    _tracker.process_detections(detections)
                                    in_poly_count = _tracker.get_inside_count()
                                    max_time = _tracker.get_max_time()
                                    if detections:
                                        in_poly_bboxes = _tracker.get_in_polygon_bboxes()
                                if detections:
                                    # det_frame = aniqlashan kadr, uning ustiga chizish (100% mos)
                                    draw_on = det_frame if det_frame is not None else frame
                                    frame = self.car_detector.draw_detections(
                                        draw_on, detections,
                                        thickness=2, font_scale=0.6,
                                        in_polygon_bboxes=in_poly_bboxes)
                                    h, w = frame.shape[:2]
                            except Exception as e:
                                print(f"[{self.camera_name}] Detection error: {e}")

                        # Polygon chizish (yashil→sariq→apelsin→qizil)
                        if self._poly_pts is not None:
                            if in_poly_count == 0:
                                color = (0, 255, 0)    # YASHIL — bo'sh
                            elif max_time < self.warning_threshold:
                                color = (0, 255, 255)  # SARIQ — mashina bor
                            elif max_time < self.violation_threshold:
                                color = (0, 165, 255)  # APELSIN — ogohlantirish
                            else:
                                color = (0, 0, 255)    # QIZIL — buzilish!
                            cv2.polylines(frame, [self._poly_pts], True, color, 2)

                        # BGR→RGB — Intel GPU (OpenCL) yoki CPU
                        if _USE_OPENCL:
                            umat = cv2.UMat(frame)
                            umat = cv2.cvtColor(umat, cv2.COLOR_BGR2RGB)
                            rgb = umat.get()
                        else:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        qimg = QImage(rgb.data, w, h, w * 3,
                                      QImage.Format.Format_RGB888).copy()

                        self._latest_qimg = qimg
                        self.frame_ready.emit()

                        # FPS hisoblash
                        _fc += 1
                        _now = time.time()
                        _el = _now - _fps_t
                        if _el >= 1.0:
                            _cam_fps = _fc / _el
                            _fc = 0
                            _fps_t = _now
                        # Stats throttle: polygon holat o'zgarse darhol, aks holda 4/sec
                        _now_stats = time.time()
                        _poly_changed = in_poly_count != _last_in_poly
                        if _poly_changed or (_now_stats - _last_stats_emit) >= 0.25:
                            _last_stats_emit = _now_stats
                            _last_in_poly = in_poly_count
                            if _tracker is not None:
                                self.stats_updated.emit(
                                    _tracker.light_count,
                                    _tracker.heavy_count,
                                    in_poly_count,
                                    _tracker.get_max_time(),
                                    _cam_fps
                                )
                            else:
                                self.stats_updated.emit(0, 0, detection_count, 0.0, _cam_fps)
                else:
                    if self._is_running():
                        self.status_changed.emit("error")
            except Exception as e:
                if self._is_running():
                    print(f"[Detail-{self.camera_name}] Error: {e}")
            finally:
                _grab_running[0] = False
                if gt is not None:
                    gt.join(timeout=3.0)
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

            if not self._is_running():
                break

            # Reconnect with backoff
            retry_count += 1
            if self._is_running():
                self.status_changed.emit("reconnecting")
                delay = min(self._retry_delay * retry_count, 15)
                for _ in range(int(delay * 10)):
                    if not self._is_running():
                        return
                    self.msleep(100)

    def _is_running(self):
        try:
            self._mutex.lock()
            r = self._running
            self._mutex.unlock()
            return r
        except Exception:
            return False

    def stop(self):
        # tryLock — deadlock dan himoya
        try:
            if self._mutex.tryLock(1000):
                self._running = False
                self._mutex.unlock()
            else:
                self._running = False
        except Exception:
            self._running = False
        try:
            if self.isRunning():
                self.quit()
                if not self.wait(5000):
                    print(f"[Detail-{self.camera_name}] Worker thread did not stop in 5s")
        except (RuntimeError, Exception):
            pass


class CrossingDetail(QWidget):
    """Detailed view - responsive cameras with grid, auto-reconnect"""

    back_clicked = pyqtSignal()
    add_camera_clicked = pyqtSignal(int)
    edit_crossing_clicked = pyqtSignal(int)
    delete_crossing_clicked = pyqtSignal(int)

    def __init__(self, config_manager, crossing_id: int, car_detector=None,
                 stats_db=None, is_custom_model=False, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.crossing_id = crossing_id
        self.crossing_data = config_manager.get_crossing(crossing_id)
        self.camera_workers = []
        self.camera_workers_dict = {}   # cam_id -> worker
        self.camera_toggle_btns = {}    # cam_id -> pause/resume QPushButton
        self.camera_panels_dict = {}    # cam_id -> QFrame panel
        # Paused holatini camera_state.json dan yuklash
        self.camera_paused: set = (
            config_manager.get_paused_cameras(crossing_id)
            if config_manager else set()
        )
        self.camera_labels = {}
        self.camera_status_labels = {}
        self.camera_detection_labels = {}  # Detection info labels
        self.camera_polytime_labels = {}  # Polygon time labels
        self.camera_types = {}  # cam_id -> "main"/"additional"
        self._active_main_cam_id = None  # Hozirgi aktiv asosiy kamera (pause bo'lganda o'zgaradi)
        self.stats_db = stats_db
        self.is_custom_model = is_custom_model
        self._destroyed = False

        # Shared car detector (dashboard dan keladi - bitta TensorRT engine)
        self.car_detector = car_detector

        if not self.crossing_data:
            raise ValueError(f"Crossing {crossing_id} not found")

        self._setup_ui()
        self._load_startup_counts()
        QTimer.singleShot(300, self._start_all_cameras)
        LM.language_changed.connect(self._retranslate)

    def _load_startup_counts(self):
        """DB dan bugungi sanashni yuklash"""
        self._light_offset = 0
        self._heavy_offset = 0
        if not self.stats_db:
            return
        try:
            # Asosiy kamera nomini topish
            cameras = self.crossing_data.get("cameras", [])
            main_cam_name = ""
            for cam in cameras:
                if cam.get("type", "additional") == "main":
                    main_cam_name = cam.get("name", "")
                    break
            if not main_cam_name and cameras:
                main_cam_name = cameras[0].get("name", "")

            if main_cam_name:
                light, heavy = self.stats_db.get_camera_today(
                    self.crossing_id, main_cam_name)
            else:
                light, heavy = self.stats_db.get_today_total(self.crossing_id)

            self._light_offset = light
            self._heavy_offset = heavy
            if light > 0 or heavy > 0:
                self._update_statistics_panel(light, heavy)
        except Exception:
            pass

    def _get_camera_grid_cols(self):
        """Calculate camera grid columns based on screen and camera count"""
        cameras = self.crossing_data.get("cameras", [])
        cam_count = len(cameras)

        screen = QApplication.primaryScreen()
        if screen:
            screen_w = screen.availableGeometry().width()
        else:
            screen_w = 1920

        if cam_count <= 1:
            return 1
        elif cam_count == 2:
            return 2
        elif cam_count <= 4:
            if screen_w >= 1400:
                return min(cam_count, 4)
            elif screen_w >= 1000:
                return min(cam_count, 3)
            else:
                return 2
        else:
            if screen_w >= 1600:
                return 4
            elif screen_w >= 1200:
                return 3
            else:
                return 2

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 8, 15, 8)
        layout.setSpacing(0)

        # ===== Header =====
        header = QFrame()
        header.setFixedHeight(50)
        header.setStyleSheet(f"""
            QFrame {{
                background: {C('bg_secondary')};
                border: 1px solid {C('bg_input')};
                border-radius: 10px;
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(10)

        # Back
        self._back_btn = QPushButton(t("crossing.back"))
        self._back_btn.clicked.connect(self.back_clicked.emit)
        self._back_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C('bg_input')}; color: {C('text_primary')}; border: none;
                border-radius: 6px; padding: 6px 14px; font-size: 12px;
            }}
            QPushButton:hover {{ background: {C('bg_hover')}; }}
        """)
        header_layout.addWidget(self._back_btn)

        # Separator
        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setFixedHeight(24)
        sep.setStyleSheet(f"background: {C('bg_input')};")
        header_layout.addWidget(sep)

        # Title
        title = QLabel(self.crossing_data.get("name", "Pereezd"))
        title.setStyleSheet(f"color: {C('text_primary')}; font-size: 16px; font-weight: bold; background: transparent;")
        header_layout.addWidget(title)

        # Location badge
        loc = self.crossing_data.get('location', '')
        if loc:
            loc_lbl = QLabel(loc)
            loc_lbl.setStyleSheet(f"""
                color: {C('text_secondary')}; font-size: 11px; background: {C('bg_primary')};
                border: 1px solid {C('bg_input')}; border-radius: 4px; padding: 2px 8px;
            """)
            header_layout.addWidget(loc_lbl)

        header_layout.addStretch()

        # Action buttons
        btn_css = """
            QPushButton {{
                background: {bg}; color: {fg}; border: none;
                border-radius: 6px; padding: 6px 16px; font-size: 11px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {hover}; }}
        """

        self._add_cam_btn = QPushButton(t("crossing.add_camera"))
        self._add_cam_btn.setStyleSheet(btn_css.format(bg=C('bg_input'), fg=C('accent_brand'), hover=C('bg_hover')))
        self._add_cam_btn.clicked.connect(lambda: self.add_camera_clicked.emit(self.crossing_id))
        header_layout.addWidget(self._add_cam_btn)

        self._settings_btn = QPushButton(t("crossing.settings_btn"))
        self._settings_btn.setStyleSheet(btn_css.format(bg=C('bg_input'), fg=C('accent_green'), hover=C('bg_hover')))
        self._settings_btn.clicked.connect(lambda: self.edit_crossing_clicked.emit(self.crossing_id))
        header_layout.addWidget(self._settings_btn)

        self._delete_btn = QPushButton(t("crossing.delete"))
        self._delete_btn.setStyleSheet(btn_css.format(bg=C('bg_input'), fg=C('accent_red'), hover=C('accent_red')))
        self._delete_btn.clicked.connect(lambda: self.delete_crossing_clicked.emit(self.crossing_id))
        header_layout.addWidget(self._delete_btn)

        layout.addWidget(header)
        layout.addSpacing(8)

        # ===== Content =====
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Cameras section
        cameras_widget = self._create_cameras_section()
        content_layout.addWidget(cameras_widget)

        # Info panels row
        info_row = QHBoxLayout()
        info_row.setSpacing(10)

        stats = self._create_statistics_panel()
        info_row.addWidget(stats, 3)

        plc = self._create_plc_panel()
        info_row.addWidget(plc, 1)

        content_layout.addLayout(info_row)

        # Soatlik grafik (So'nggi Hodisalar o'rniga)
        self._hourly_chart = self._create_hourly_chart()
        content_layout.addWidget(self._hourly_chart)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Time timer
        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self._update_time)
        self.time_timer.start(1000)

        # Grafik yangilash timer (60 sekundda bir)
        self._chart_timer = QTimer(self)
        self._chart_timer.timeout.connect(self._refresh_chart)
        self._chart_timer.start(60000)
        # Dastlabki yuklash
        QTimer.singleShot(500, self._refresh_chart)

    def _create_cameras_section(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        cameras = self.crossing_data.get("cameras", [])

        if not cameras:
            empty = QLabel(t("crossing.no_cameras"))
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(f"""
                color: {C('text_muted')}; font-size: 14px; padding: 60px;
                background: {C('bg_primary')}; border: 2px dashed {C('bg_input')}; border-radius: 12px;
            """)
            layout.addWidget(empty)
            return container

        cols = self._get_camera_grid_cols()

        cameras_grid = QGridLayout()
        cameras_grid.setSpacing(10)

        # Faqat yoqilgan kameralarni ko'rsatish
        active_cameras = [c for c in cameras if c.get("enabled", True)]
        cam_count = len(active_cameras)
        for i, cam in enumerate(active_cameras):
            row = i // cols
            col = i % cols
            panel = self._create_camera_panel(cam, i)
            if cam_count == 1:
                panel.setMaximumHeight(500)
            cameras_grid.addWidget(panel, row, col)

        layout.addLayout(cameras_grid)
        return container

    def _create_camera_panel(self, cam_data: dict, index: int):
        cam_id = cam_data.get("id", index)
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame#camPanel {{
                background: {C('bg_primary')};
                border: 2px solid {C('bg_input')};
                border-radius: 12px;
            }}
        """)
        panel.setObjectName("camPanel")
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.camera_panels_dict[cam_id] = panel

        p_layout = QVBoxLayout(panel)
        p_layout.setContentsMargins(10, 8, 10, 8)
        p_layout.setSpacing(6)

        # Header row
        hdr = QHBoxLayout()
        hdr.setSpacing(8)

        # Camera name
        name = QLabel(cam_data.get("name", f"Kamera {index + 1}"))
        name.setStyleSheet(f"color: {C('text_primary')}; font-size: 13px; font-weight: bold; background: transparent;")
        hdr.addWidget(name)

        # Type badge
        cam_type = cam_data.get("type", "additional")
        is_main = cam_type == "main"
        badge_color = C('accent_brand') if is_main else C('accent_green')
        badge_text = t("crossing.type.main") if is_main else t("crossing.type.additional")
        badge = QLabel(badge_text)
        badge.setStyleSheet(f"""
            color: {badge_color}; font-size: 9px; font-weight: bold;
            background: {badge_color}20; border: 1px solid {badge_color}80;
            border-radius: 4px; padding: 2px 8px;
        """)
        hdr.addWidget(badge)

        # Status indicator
        cam_id = cam_data.get("id", index)
        status_dot = QLabel("*")
        status_dot.setStyleSheet(f"color: {C('accent_orange')}; font-size: 12px; background: transparent;")
        self.camera_status_labels[cam_id] = status_dot
        hdr.addWidget(status_dot)

        hdr.addStretch()

        # Pause/Resume tugmasi (JSON ga tegmaydi, faqat in-memory)
        pause_color = C('accent_blue')
        pause_btn = QPushButton(f"⏸ {t('cam.pause')}")
        pause_btn.setFixedHeight(22)
        pause_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {pause_color};
                border: 1px solid {pause_color}; border-radius: 4px;
                padding: 2px 8px; font-size: 10px;
            }}
            QPushButton:hover {{ background: {pause_color}20; }}
        """)
        pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        pause_btn.clicked.connect(lambda _, cid=cam_id: self._pause_resume_camera(cid))
        hdr.addWidget(pause_btn)
        self.camera_toggle_btns[cam_id] = pause_btn

        # Settings gear
        gear = QPushButton(t("crossing.camera_settings"))
        gear.setFixedHeight(22)
        gear.setStyleSheet(f"""
            QPushButton {{
                background: {C('bg_input')}; color: {C('text_muted')}; border: none;
                border-radius: 4px; padding: 2px 8px; font-size: 10px;
            }}
            QPushButton:hover {{ background: {C('bg_hover')}; color: {C('text_primary')}; }}
        """)
        gear.clicked.connect(lambda _, cid=cam_id: self._open_camera_settings(cid))
        hdr.addWidget(gear)

        p_layout.addLayout(hdr)

        # Video label - responsive, no fixed size
        video = QLabel()
        video.setMinimumSize(200, 120)
        video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video.setScaledContents(False)
        video.setStyleSheet(f"""
            background: {C('bg_camera')};
            border: 1px solid {C('bg_input')};
            border-radius: 8px;
        """)
        # Set aspect ratio hint
        video.setMinimumHeight(150)
        self._set_placeholder(video, t("cam.status.connecting"), 480, 270)
        self.camera_labels[cam_id] = video
        p_layout.addWidget(video)

        # Bottom row
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        time_lbl = QLabel(time.strftime("%H:%M:%S"))
        time_lbl.setObjectName(f"time_label_{cam_id}")
        time_lbl.setStyleSheet(f"color: {C('text_secondary')}; font-size: 10px; background: transparent;")
        bottom.addWidget(time_lbl)

        # Detection info label
        det_lbl = QLabel(t("cam.detection", light=0, heavy=0, total=0, fps=0.0))
        det_lbl.setStyleSheet(f"color: {C('text_secondary')}; font-size: 10px; background: transparent;")
        self.camera_detection_labels[cam_id] = det_lbl
        bottom.addWidget(det_lbl)

        bottom.addStretch()

        # Polygon vaqt label
        poly_time_lbl = QLabel(t("cam.polygon.empty"))
        poly_time_lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 10px; background: transparent;")
        self.camera_polytime_labels[cam_id] = poly_time_lbl
        bottom.addWidget(poly_time_lbl)

        p_layout.addLayout(bottom)
        return panel

    def _set_placeholder(self, label, text, w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (17, 17, 27)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6 if w > 400 else 0.45
        sz = cv2.getTextSize(text, font, scale, 1)[0]
        x, y = (w - sz[0]) // 2, (h + sz[1]) // 2
        cv2.putText(img, text, (x, y), font, scale, (100, 100, 140), 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._show_frame(label, rgb)

    def _show_frame(self, label, qimg, worker=None):
        """Display ready QImage - minimal GUI thread work"""
        if self._destroyed or label is None:
            return
        try:
            if isinstance(qimg, QImage):
                pixmap = QPixmap.fromImage(qimg)
                scaled = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.FastTransformation)
                label.setPixmap(scaled)
            else:
                # Fallback for numpy array (placeholder)
                h, w = qimg.shape[:2]
                img = QImage(qimg.data, w, h, w * 3, QImage.Format.Format_RGB888)
                label.setPixmap(QPixmap.fromImage(img))
        except Exception:
            pass

    def _start_all_cameras(self):
        if self._destroyed:
            return

        cameras = self.crossing_data.get("cameras", [])
        for cam in cameras:
            if not cam.get("enabled", False):
                continue
            src = cam.get("source", "")
            if not src:
                continue

            cam_id = cam.get("id", 0)
            cam_name = cam.get("name", f"Cam-{cam_id}")
            cam_type = cam.get("type", "additional")
            self.camera_types[cam_id] = cam_type
            if cam_type == "main" and self._active_main_cam_id is None:
                self._active_main_cam_id = cam_id

            # Paused kamerani o'tkazib yuborish + tugmasini to'g'ri holatda ko'rsatish
            if cam_id in self.camera_paused:
                label = self.camera_labels.get(cam_id)
                if label:
                    self._set_placeholder(label, t("cam.paused"), 480, 270)
                btn = self.camera_toggle_btns.get(cam_id)
                clr = C('accent_orange')
                if btn:
                    btn.setText(f"▶ {t('cam.resume')}")
                    btn.setStyleSheet(f"""QPushButton {{
                        background: transparent; color: {clr};
                        border: 1px solid {clr}; border-radius: 4px;
                        padding: 2px 8px; font-size: 10px;
                    }} QPushButton:hover {{ background: {clr}20; }}""")
                if cam_type == "main":
                    self._failover_main()
                continue

            label = self.camera_labels.get(cam_id)
            if not label:
                continue

            # Polygon file yo'lini aniqlash
            poly_file = cam.get("polygon_file", "")
            if poly_file and not os.path.isabs(poly_file):
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                poly_file = os.path.join(project_root, poly_file)

            # Settings dan thresholdlar
            settings = self.config_manager.get_settings() if self.config_manager else {}
            warn_t = settings.get("warning_threshold", 10.0)
            viol_t = settings.get("violation_threshold", 15.0)

            # Create worker with car detector + polygon
            worker = DetailCameraWorker(
                src,
                cam_name,
                car_detector=self.car_detector,
                detection_enabled=cam.get("detection_enabled", True),
                polygon_file=poly_file if poly_file and os.path.isfile(poly_file) else None,
                warning_threshold=warn_t,
                violation_threshold=viol_t,
                is_custom_model=self.is_custom_model,
            )
            worker.frame_ready.connect(
                lambda lbl=label, w=worker: self._on_frame(lbl, w)
            )
            worker.status_changed.connect(
                lambda s, cid=cam_id: self._on_camera_status(cid, s)
            )
            worker.stats_updated.connect(
                lambda light, heavy, in_poly, max_t, fps, cid=cam_id, cname=cam_name:
                    self._on_stats_update(cid, cname, light, heavy, in_poly, max_t, fps)
            )
            worker.start()
            self.camera_workers.append(worker)
            self.camera_workers_dict[cam_id] = worker

    def _on_frame(self, label, worker):
        if not self._destroyed:
            qimg = worker.take_frame()
            if qimg is None:
                return
            try:
                self._show_frame(label, qimg)
            except RuntimeError:
                self._destroyed = True

    def _on_camera_status(self, cam_id, status):
        if self._destroyed:
            return
        try:
            dot = self.camera_status_labels.get(cam_id)
            label = self.camera_labels.get(cam_id)
            if dot:
                if status == "online":
                    dot.setStyleSheet(f"color: {C('accent_green')}; font-size: 12px; background: transparent;")
                elif status == "reconnecting":
                    dot.setStyleSheet(f"color: {C('accent_yellow')}; font-size: 12px; background: transparent;")
                    if label:
                        self._set_placeholder(label, t("cam.status.reconnecting"), 480, 270)
                elif status == "error":
                    dot.setStyleSheet(f"color: {C('accent_red')}; font-size: 12px; background: transparent;")
                    if label:
                        self._set_placeholder(label, t("cam.status.failed"), 480, 270)
        except RuntimeError:
            self._destroyed = True

    def _on_stats_update(self, cam_id, cam_name, light_count, heavy_count,
                         in_poly_count, max_time, fps):
        """Handle tracking stats from DetailCameraWorker"""
        if self._destroyed:
            return
        try:
            det_label = self.camera_detection_labels.get(cam_id)
            if det_label:
                total = light_count + heavy_count
                det_label.setText(t("cam.detection", light=light_count, heavy=heavy_count, total=total, fps=fps))
                # setStyleSheet faqat holat o'zgarganda (qimmat operatsiya)
                _prev_in_poly = getattr(self, f'_prev_in_poly_{cam_id}', -1)
                if (in_poly_count > 0) != (_prev_in_poly > 0):
                    setattr(self, f'_prev_in_poly_{cam_id}', in_poly_count)
                    if in_poly_count > 0:
                        det_label.setStyleSheet(f"color: {C('accent_green')}; font-size: 10px; font-weight: bold; background: transparent;")
                    else:
                        det_label.setStyleSheet(f"color: {C('text_secondary')}; font-size: 10px; background: transparent;")
            # Polygon vaqtni yangilash
            poly_lbl = self.camera_polytime_labels.get(cam_id)
            if poly_lbl:
                if max_time > 0:
                    poly_lbl.setText(t("cam.polygon.time", time=max_time))
                    _prev_poly_active = getattr(self, f'_prev_poly_active_{cam_id}', False)
                    if not _prev_poly_active:
                        setattr(self, f'_prev_poly_active_{cam_id}', True)
                        poly_lbl.setStyleSheet(f"color: {C('accent_red')}; font-size: 10px; font-weight: bold; background: transparent;")
                else:
                    poly_lbl.setText(t("cam.polygon.empty"))
                    _prev_poly_active = getattr(self, f'_prev_poly_active_{cam_id}', True)
                    if _prev_poly_active:
                        setattr(self, f'_prev_poly_active_{cam_id}', False)
                        poly_lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 10px; background: transparent;")
            # Active main kamera bo'lsa → statistika panelni yangilash
            if cam_id == self._active_main_cam_id:
                display_light = self._light_offset + light_count
                display_heavy = self._heavy_offset + heavy_count
                self._update_statistics_panel(display_light, display_heavy)
            # DB ga dashboard worker yozadi (detail yozsa _last_counts buziladi)
        except RuntimeError:
            self._destroyed = True

    def _update_statistics_panel(self, light: int, heavy: int):
        """Statistika panel qiymatlarini yangilash"""
        try:
            if hasattr(self, '_stat_light_label'):
                self._stat_light_label.setText(str(light))
            if hasattr(self, '_stat_heavy_label'):
                self._stat_heavy_label.setText(str(heavy))
            if hasattr(self, '_stat_total_label'):
                self._stat_total_label.setText(str(light + heavy))
        except RuntimeError:
            pass

    def _update_time(self):
        if self._destroyed:
            return
        try:
            current = time.strftime("%H:%M:%S")
            cameras = self.crossing_data.get("cameras", [])
            for cam in cameras:
                cid = cam.get("id", 0)
                lbl = self.findChild(QLabel, f"time_label_{cid}")
                if lbl:
                    lbl.setText(current)
        except RuntimeError:
            self._destroyed = True

    def _pause_resume_camera(self, camera_id):
        """Kamerani vaqtinchalik to'xtatish/davom ettirish. JSON ga tegmaydi."""
        if not camera_id:
            return
        btn = self.camera_toggle_btns.get(camera_id)
        is_paused = camera_id in self.camera_paused

        if is_paused:
            # --- RESUME ---
            self.camera_paused.discard(camera_id)
            self.config_manager.set_camera_paused(self.crossing_id, camera_id, False)
            clr = C('accent_blue')
            if btn:
                btn.setText(f"⏸ {t('cam.pause')}")
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent; color: {clr};
                        border: 1px solid {clr}; border-radius: 4px;
                        padding: 2px 8px; font-size: 10px;
                    }}
                    QPushButton:hover {{ background: {clr}20; }}
                """)
            # Agar asosiy kamera qayta yoqilsa — uni active main ga qaytarish
            if self.camera_types.get(camera_id) == "main":
                self._active_main_cam_id = camera_id
            self.crossing_data = self.config_manager.get_crossing(self.crossing_id)
            cameras = self.crossing_data.get("cameras", [])
            cam = next((c for c in cameras if c.get("id") == camera_id), None)
            if cam:
                self._start_single_camera(cam)
        else:
            # --- PAUSE ---
            self.camera_paused.add(camera_id)
            self.config_manager.set_camera_paused(self.crossing_id, camera_id, True)
            clr = C('accent_orange')
            if btn:
                btn.setText(f"▶ {t('cam.resume')}")
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: transparent; color: {clr};
                        border: 1px solid {clr}; border-radius: 4px;
                        padding: 2px 8px; font-size: 10px;
                    }}
                    QPushButton:hover {{ background: {clr}20; }}
                """)
            worker = self.camera_workers_dict.pop(camera_id, None)
            if worker:
                try:
                    self.camera_workers.remove(worker)
                except ValueError:
                    pass
                worker.stop()
            label = self.camera_labels.get(camera_id)
            if label:
                self._set_placeholder(label, t("cam.paused"), 480, 270)

            # Asosiy to'xtatilsa → birinchi aktiv qo'shimchaga failover
            if self.camera_types.get(camera_id) == "main":
                self._failover_main()

    def _failover_main(self):
        """Asosiy kamera to'xtatilganda, birinchi aktiv qo'shimchani active main qilib qo'yish."""
        cameras = self.crossing_data.get("cameras", [])
        for cam in cameras:
            cid = cam.get("id")
            if (cid not in self.camera_paused
                    and cam.get("enabled", True)
                    and cam.get("type", "additional") == "additional"):
                self._active_main_cam_id = cid
                return
        # Hech qanday qo'shimcha yo'q — active main None
        self._active_main_cam_id = None

    def _start_single_camera(self, cam: dict):
        """Bitta kamerani ishga tushirish (toggle yoki qayta ulanish uchun)"""
        cam_id = cam.get("id", 0)
        src = cam.get("source", "")
        if not src:
            return
        label = self.camera_labels.get(cam_id)
        if not label:
            return
        cam_name = cam.get("name", f"Cam-{cam_id}")
        cam_type = cam.get("type", "additional")
        self.camera_types[cam_id] = cam_type

        poly_file = cam.get("polygon_file", "")
        if poly_file and not os.path.isabs(poly_file):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            poly_file = os.path.join(project_root, poly_file)

        settings = self.config_manager.get_settings() if self.config_manager else {}
        warn_t = settings.get("warning_threshold", 10.0)
        viol_t = settings.get("violation_threshold", 15.0)

        worker = DetailCameraWorker(
            src, cam_name,
            car_detector=self.car_detector,
            detection_enabled=cam.get("detection_enabled", True),
            polygon_file=poly_file if poly_file and os.path.isfile(poly_file) else None,
            warning_threshold=warn_t,
            violation_threshold=viol_t,
            is_custom_model=self.is_custom_model,
        )
        worker.frame_ready.connect(lambda lbl=label, w=worker: self._on_frame(lbl, w))
        worker.status_changed.connect(lambda s, cid=cam_id: self._on_camera_status(cid, s))
        worker.stats_updated.connect(
            lambda light, heavy, in_poly, max_t, fps, cid=cam_id, cname=cam_name:
                self._on_stats_update(cid, cname, light, heavy, in_poly, max_t, fps)
        )
        worker.start()
        self.camera_workers.append(worker)
        self.camera_workers_dict[cam_id] = worker

    def _open_camera_settings(self, camera_id):
        from gui.ui.dialogs import AddCameraDialog
        dialog = AddCameraDialog(self.config_manager, self.crossing_id, camera_id,
                                  stats_db=self.stats_db)
        if dialog.exec():
            self.refresh()

    def _create_statistics_panel(self):
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame#statsPanel {{
                background: {C('bg_primary')};
                border: 2px solid {C('bg_input')};
                border-radius: 12px;
            }}
        """)
        panel.setObjectName("statsPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        self._stats_title = QLabel(t("stats.panel"))
        self._stats_title.setStyleSheet(f"color: {C('text_primary')}; font-size: 14px; font-weight: bold; background: transparent;")
        layout.addWidget(self._stats_title)

        # Divider
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {C('bg_input')};")
        layout.addWidget(div)

        cameras_count = len(self.crossing_data.get("cameras", []))
        active = sum(1 for c in self.crossing_data.get("cameras", []) if c.get("enabled"))

        # Kameralar (statik)
        row_cam = QHBoxLayout()
        self._stat_cameras_lbl = QLabel(t("stats.cameras"))
        self._stat_cameras_lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
        row_cam.addWidget(self._stat_cameras_lbl)
        row_cam.addStretch()
        v_cam = QLabel(f"{active}/{cameras_count}")
        v_cam.setStyleSheet(f"color: {C('accent_brand')}; font-size: 14px; font-weight: bold; background: transparent;")
        row_cam.addWidget(v_cam)
        layout.addLayout(row_cam)

        # Yengil transport (dinamik)
        row_light = QHBoxLayout()
        self._stat_light_lbl = QLabel(t("stats.light"))
        self._stat_light_lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
        row_light.addWidget(self._stat_light_lbl)
        row_light.addStretch()
        self._stat_light_label = QLabel("0")
        self._stat_light_label.setStyleSheet(f"color: {C('accent_blue')}; font-size: 14px; font-weight: bold; background: transparent;")
        row_light.addWidget(self._stat_light_label)
        layout.addLayout(row_light)

        # Og'ir transport (dinamik)
        row_heavy = QHBoxLayout()
        self._stat_heavy_lbl = QLabel(t("stats.heavy"))
        self._stat_heavy_lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
        row_heavy.addWidget(self._stat_heavy_lbl)
        row_heavy.addStretch()
        self._stat_heavy_label = QLabel("0")
        self._stat_heavy_label.setStyleSheet(f"color: {C('accent_orange')}; font-size: 14px; font-weight: bold; background: transparent;")
        row_heavy.addWidget(self._stat_heavy_label)
        layout.addLayout(row_heavy)

        # Jami transport (dinamik)
        row_total = QHBoxLayout()
        self._stat_total_lbl = QLabel(t("stats.total"))
        self._stat_total_lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
        row_total.addWidget(self._stat_total_lbl)
        row_total.addStretch()
        self._stat_total_label = QLabel("0")
        self._stat_total_label.setStyleSheet(f"color: {C('accent_green')}; font-size: 14px; font-weight: bold; background: transparent;")
        row_total.addWidget(self._stat_total_label)
        layout.addLayout(row_total)

        layout.addStretch()
        return panel

    def _create_plc_panel(self):
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame#plcPanel {{
                background: {C('bg_primary')};
                border: 2px solid {C('bg_input')};
                border-radius: 12px;
            }}
        """)
        panel.setObjectName("plcPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        title = QLabel(t("plc.title"))
        title.setStyleSheet(f"color: {C('text_primary')}; font-size: 14px; font-weight: bold; background: transparent;")
        layout.addWidget(title)

        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {C('bg_input')};")
        layout.addWidget(div)

        plc = self.crossing_data.get("plc", {})

        if plc.get("enabled", False):
            row1 = QHBoxLayout()
            s = QLabel(t("plc.status_label"))
            s.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
            row1.addWidget(s)
            row1.addStretch()
            sv = QLabel(t("plc.connected"))
            sv.setStyleSheet(f"color: {C('accent_green')}; font-size: 12px; font-weight: bold; background: transparent;")
            row1.addWidget(sv)
            layout.addLayout(row1)

            row2 = QHBoxLayout()
            ip_n = QLabel(t("plc.ip"))
            ip_n.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
            row2.addWidget(ip_n)
            row2.addStretch()
            ip_v = QLabel(plc.get("ip", "N/A"))
            ip_v.setStyleSheet(f"color: {C('text_primary')}; font-size: 12px; background: transparent;")
            row2.addWidget(ip_v)
            layout.addLayout(row2)

            row3 = QHBoxLayout()
            p_n = QLabel(t("plc.port"))
            p_n.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
            row3.addWidget(p_n)
            row3.addStretch()
            p_v = QLabel(str(plc.get("port", 102)))
            p_v.setStyleSheet(f"color: {C('text_primary')}; font-size: 12px; background: transparent;")
            row3.addWidget(p_v)
            layout.addLayout(row3)
        else:
            row = QHBoxLayout()
            s = QLabel(t("plc.status_label"))
            s.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
            row.addWidget(s)
            row.addStretch()
            sv = QLabel(t("plc.disabled"))
            sv.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; font-weight: bold; background: transparent;")
            row.addWidget(sv)
            layout.addLayout(row)

        layout.addStretch()
        return panel

    def _create_hourly_chart(self):
        """Soatlik transport grafigi (HourlyChartPanel)"""
        chart_panel = HourlyChartPanel()
        return chart_panel

    def _refresh_chart(self):
        """Grafik ma'lumotlarini DB dan yangilash"""
        if self._destroyed or not self.stats_db:
            return
        try:
            data = self.stats_db.get_hourly_data(self.crossing_id)
            self._hourly_chart.set_data(data)
        except Exception:
            pass

    def cleanup(self):
        if self._destroyed:
            return
        self._destroyed = True
        try:
            if hasattr(self, 'time_timer') and self.time_timer is not None:
                self.time_timer.stop()
        except RuntimeError:
            pass
        try:
            if hasattr(self, '_chart_timer') and self._chart_timer is not None:
                self._chart_timer.stop()
        except RuntimeError:
            pass
        # Avval signallarni uzib, keyin to'xtatish (crash prevention)
        for w in self.camera_workers:
            try:
                w.frame_ready.disconnect()
            except (TypeError, RuntimeError):
                pass
            try:
                w.status_changed.disconnect()
            except (TypeError, RuntimeError):
                pass
            try:
                w.stats_updated.disconnect()
            except (TypeError, RuntimeError):
                pass
        for w in self.camera_workers:
            try:
                w.stop()
            except (RuntimeError, Exception):
                pass
        self.camera_workers.clear()
        self.camera_workers_dict.clear()
        self.camera_toggle_btns.clear()
        self.camera_panels_dict.clear()
        self.camera_detection_labels.clear()
        self.camera_polytime_labels.clear()
        self.camera_types.clear()

    def _retranslate(self):
        """Til o'zgarganida UI textlarini yangilash."""
        try:
            if hasattr(self, '_back_btn'):
                self._back_btn.setText(t("crossing.back"))
            if hasattr(self, '_add_cam_btn'):
                self._add_cam_btn.setText(t("crossing.add_camera"))
            if hasattr(self, '_settings_btn'):
                self._settings_btn.setText(t("crossing.settings_btn"))
            if hasattr(self, '_delete_btn'):
                self._delete_btn.setText(t("crossing.delete"))
            if hasattr(self, '_stats_title'):
                self._stats_title.setText(t("stats.panel"))
            if hasattr(self, '_stat_cameras_lbl'):
                self._stat_cameras_lbl.setText(t("stats.cameras"))
            if hasattr(self, '_stat_light_lbl'):
                self._stat_light_lbl.setText(t("stats.light"))
            if hasattr(self, '_stat_heavy_lbl'):
                self._stat_heavy_lbl.setText(t("stats.heavy"))
            if hasattr(self, '_stat_total_lbl'):
                self._stat_total_lbl.setText(t("stats.total"))
            # Kamera labellarini yangilash
            for cam_id, lbl in self.camera_detection_labels.items():
                lbl.setText(t("cam.detection", light=0, heavy=0, total=0, fps=0.0))
            for cam_id, lbl in self.camera_polytime_labels.items():
                lbl.setText(t("cam.polygon.empty"))
        except (RuntimeError, Exception):
            pass

    def refresh(self):
        self.cleanup()
        # Barcha workerlar to'liq to'xtaganligini kutish
        import time as _time
        _time.sleep(0.1)
        self._destroyed = False
        self.crossing_data = self.config_manager.get_crossing(self.crossing_id)
        self.camera_labels.clear()
        self.camera_status_labels.clear()
        self.camera_detection_labels.clear()
        self.camera_polytime_labels.clear()
        self.camera_types.clear()
        self.camera_workers = []
        self.camera_workers_dict.clear()
        self.camera_toggle_btns.clear()
        self.camera_panels_dict.clear()
        self.camera_paused.clear()

        # Remove all child widgets
        old_layout = self.layout()
        if old_layout:
            while old_layout.count():
                item = old_layout.takeAt(0)
                w = item.widget()
                if w:
                    w.setParent(None)
                    w.deleteLater()
            # Transfer old layout to temp widget so it gets deleted
            QWidget().setLayout(old_layout)

        self._setup_ui()
        QTimer.singleShot(300, self._start_all_cameras)

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

    def deleteLater(self):
        self.cleanup()
        try:
            super().deleteLater()
        except RuntimeError:
            pass
