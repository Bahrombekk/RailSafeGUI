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
from gui.utils.polygon_tracker import PolygonTracker
from gui.widgets.hourly_chart import HourlyChartPanel

# RTSP ultra-low-latency (FFmpeg fallback uchun)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp|stimeout;2000000|'
    'fflags;nobuffer+discardcorrupt|flags;low_delay|'
    'analyzeduration;100000|probesize;100000|'
    'max_delay;0|reorder_queue_size;0'
)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'


def _open_camera(source: str, camera_name: str = "") -> tuple:
    """RTSP kamerani ochish: GStreamer NVDEC → GStreamer CPU → FFmpeg fallback.
    Returns (cap, backend_name) or (None, None)."""
    is_rtsp = source.lower().startswith("rtsp://")

    if is_rtsp:
        # 1) GStreamer NVDEC (GPU H.265 decode)
        gst_nvdec = (
            f"rtspsrc location={source} latency=0 protocols=tcp ! "
            f"rtph265depay ! h265parse ! nvh265dec ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst_nvdec, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[{camera_name}] GStreamer NVDEC (GPU H.265)")
            return cap, "gst-nvdec"

        # 2) GStreamer CPU (software H.265 decode)
        gst_cpu = (
            f"rtspsrc location={source} latency=0 protocols=tcp ! "
            f"rtph265depay ! h265parse ! avdec_h265 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst_cpu, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[{camera_name}] GStreamer CPU (H.265)")
            return cap, "gst-cpu"

    # 3) FFmpeg fallback (har doim ishlaydi)
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[{camera_name}] FFmpeg fallback")
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
                    # RTSP buffer tozalash - eski kadrlarni tashlash (~3 sek)
                    for _ in range(90):
                        cap.grab()

                    self.status_changed.emit("online")
                    retry_count = 0

                    # --- Dedicated grab thread: grab() HECH QACHON to'xtamaydi ---
                    _latest_frame = [None]
                    _frame_lock = threading.Lock()
                    _grab_error = [False]

                    def _grab_loop():
                        fails = 0
                        while _grab_running[0]:
                            ret = cap.grab()
                            if not ret:
                                fails += 1
                                if fails > 30:
                                    _grab_error[0] = True
                                    break
                                continue
                            fails = 0
                            # Faqat kerak bo'lganda decode (CPU 50% tejash)
                            with _frame_lock:
                                need_decode = _latest_frame[0] is None
                            if need_decode:
                                ret, frame = cap.retrieve()
                                if ret:
                                    with _frame_lock:
                                        _latest_frame[0] = frame

                    gt = threading.Thread(target=_grab_loop, daemon=True)
                    gt.start()

                    # --- Main loop: eng oxirgi kadrni process qilish ---
                    _fc = 0
                    _fps_t = time.perf_counter()
                    _cam_fps = 0.0
                    _poly_loaded = False
                    _tracker = None  # PolygonTracker (polygon yuklangandan keyin yaratiladi)

                    while self._is_running() and not _grab_error[0]:
                        with _frame_lock:
                            frame = _latest_frame[0]
                            _latest_frame[0] = None

                        if frame is None:
                            time.sleep(0.003)
                            continue

                        h, w = frame.shape[:2]
                        if w > self.display_width:
                            scale = self.display_width / w
                            frame = cv2.resize(frame, (self.display_width, int(h * scale)),
                                               interpolation=cv2.INTER_AREA)
                            h, w = frame.shape[:2]

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
                                if detections:
                                    draw_on = det_frame if det_frame is not None else frame
                                    # Tracking + counting
                                    in_poly_bboxes = None
                                    if _tracker is not None:
                                        _tracker.process_detections(detections)
                                        in_poly_count = _tracker.get_inside_count()
                                        max_time = _tracker.get_max_time()
                                        in_poly_bboxes = _tracker.get_in_polygon_bboxes()
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
        self.camera_labels = {}
        self.camera_status_labels = {}
        self.camera_detection_labels = {}  # Detection info labels
        self.camera_polytime_labels = {}  # Polygon time labels
        self.camera_types = {}  # cam_id -> "main"/"additional"
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
        back_btn = QPushButton("< Orqaga")
        back_btn.clicked.connect(self.back_clicked.emit)
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C('bg_input')}; color: {C('text_primary')}; border: none;
                border-radius: 6px; padding: 6px 14px; font-size: 12px;
            }}
            QPushButton:hover {{ background: {C('bg_hover')}; }}
        """)
        header_layout.addWidget(back_btn)

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

        add_cam_btn = QPushButton("+ Kamera")
        add_cam_btn.setStyleSheet(btn_css.format(bg=C('bg_input'), fg=C('accent_brand'), hover=C('bg_hover')))
        add_cam_btn.clicked.connect(lambda: self.add_camera_clicked.emit(self.crossing_id))
        header_layout.addWidget(add_cam_btn)

        settings_btn = QPushButton("Sozlamalar")
        settings_btn.setStyleSheet(btn_css.format(bg=C('bg_input'), fg=C('accent_green'), hover=C('bg_hover')))
        settings_btn.clicked.connect(lambda: self.edit_crossing_clicked.emit(self.crossing_id))
        header_layout.addWidget(settings_btn)

        delete_btn = QPushButton("O'chirish")
        delete_btn.setStyleSheet(btn_css.format(bg=C('bg_input'), fg=C('accent_red'), hover=C('accent_red')))
        delete_btn.clicked.connect(lambda: self.delete_crossing_clicked.emit(self.crossing_id))
        header_layout.addWidget(delete_btn)

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
            empty = QLabel("Kameralar yo'q. '+ Kamera' tugmasini bosing.")
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

        cam_count = len(cameras)
        for i, cam in enumerate(cameras):
            row = i // cols
            col = i % cols
            panel = self._create_camera_panel(cam, i)
            if cam_count == 1:
                panel.setMaximumHeight(500)
            cameras_grid.addWidget(panel, row, col)

        layout.addLayout(cameras_grid)
        return container

    def _create_camera_panel(self, cam_data: dict, index: int):
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
        badge_text = "Asosiy" if is_main else "Qo'shimcha"
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

        # Settings gear
        gear = QPushButton("Settings")
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
        self._set_placeholder(video, "Ulanmoqda...", 480, 270)
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
        det_lbl = QLabel("Yengil: 0 | Og'ir: 0 | Jami: 0 | FPS: 0.0")
        det_lbl.setStyleSheet(f"color: {C('text_secondary')}; font-size: 10px; background: transparent;")
        self.camera_detection_labels[cam_id] = det_lbl
        bottom.addWidget(det_lbl)

        bottom.addStretch()

        # Polygon vaqt label
        poly_time_lbl = QLabel("Polygon: bo'sh")
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
                                       Qt.TransformationMode.SmoothTransformation)
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
                        self._set_placeholder(label, "Qayta ulanmoqda...", 480, 270)
                elif status == "error":
                    dot.setStyleSheet(f"color: {C('accent_red')}; font-size: 12px; background: transparent;")
                    if label:
                        self._set_placeholder(label, "Ulanmadi", 480, 270)
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
                det_label.setText(f"Yengil: {light_count} | Og'ir: {heavy_count} | Jami: {total} | FPS: {fps:.1f}")
                if in_poly_count > 0:
                    det_label.setStyleSheet(f"color: {C('accent_green')}; font-size: 10px; font-weight: bold; background: transparent;")
                else:
                    det_label.setStyleSheet(f"color: {C('text_secondary')}; font-size: 10px; background: transparent;")
            # Polygon vaqtni yangilash
            poly_lbl = self.camera_polytime_labels.get(cam_id)
            if poly_lbl:
                if max_time > 0:
                    poly_lbl.setText(f"Polygon: {max_time:.1f}s")
                    poly_lbl.setStyleSheet(f"color: {C('accent_red')}; font-size: 10px; font-weight: bold; background: transparent;")
                else:
                    poly_lbl.setText("Polygon: bo'sh")
                    poly_lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 10px; background: transparent;")
            # Asosiy kamera bo'lsa → statistika panelni yangilash (offset + tracker)
            cam_type = self.camera_types.get(cam_id, "additional")
            if cam_type == "main":
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

        title = QLabel("Statistika")
        title.setStyleSheet(f"color: {C('text_primary')}; font-size: 14px; font-weight: bold; background: transparent;")
        layout.addWidget(title)

        # Divider
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {C('bg_input')};")
        layout.addWidget(div)

        cameras_count = len(self.crossing_data.get("cameras", []))
        active = sum(1 for c in self.crossing_data.get("cameras", []) if c.get("enabled"))

        # Kameralar (statik)
        row_cam = QHBoxLayout()
        n_cam = QLabel("Kameralar")
        n_cam.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
        row_cam.addWidget(n_cam)
        row_cam.addStretch()
        v_cam = QLabel(f"{active}/{cameras_count}")
        v_cam.setStyleSheet(f"color: {C('accent_brand')}; font-size: 14px; font-weight: bold; background: transparent;")
        row_cam.addWidget(v_cam)
        layout.addLayout(row_cam)

        # Yengil transport (dinamik)
        row_light = QHBoxLayout()
        n_light = QLabel("Yengil transport")
        n_light.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
        row_light.addWidget(n_light)
        row_light.addStretch()
        self._stat_light_label = QLabel("0")
        self._stat_light_label.setStyleSheet(f"color: {C('accent_blue')}; font-size: 14px; font-weight: bold; background: transparent;")
        row_light.addWidget(self._stat_light_label)
        layout.addLayout(row_light)

        # Og'ir transport (dinamik)
        row_heavy = QHBoxLayout()
        n_heavy = QLabel("Og'ir transport")
        n_heavy.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
        row_heavy.addWidget(n_heavy)
        row_heavy.addStretch()
        self._stat_heavy_label = QLabel("0")
        self._stat_heavy_label.setStyleSheet(f"color: {C('accent_orange')}; font-size: 14px; font-weight: bold; background: transparent;")
        row_heavy.addWidget(self._stat_heavy_label)
        layout.addLayout(row_heavy)

        # Jami transport (dinamik)
        row_total = QHBoxLayout()
        n_total = QLabel("Jami transport")
        n_total.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
        row_total.addWidget(n_total)
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

        title = QLabel("PLC Holati")
        title.setStyleSheet(f"color: {C('text_primary')}; font-size: 14px; font-weight: bold; background: transparent;")
        layout.addWidget(title)

        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {C('bg_input')};")
        layout.addWidget(div)

        plc = self.crossing_data.get("plc", {})

        if plc.get("enabled", False):
            row1 = QHBoxLayout()
            s = QLabel("Holat")
            s.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
            row1.addWidget(s)
            row1.addStretch()
            sv = QLabel("ULANGAN")
            sv.setStyleSheet(f"color: {C('accent_green')}; font-size: 12px; font-weight: bold; background: transparent;")
            row1.addWidget(sv)
            layout.addLayout(row1)

            row2 = QHBoxLayout()
            ip_n = QLabel("IP")
            ip_n.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
            row2.addWidget(ip_n)
            row2.addStretch()
            ip_v = QLabel(plc.get("ip", "N/A"))
            ip_v.setStyleSheet(f"color: {C('text_primary')}; font-size: 12px; background: transparent;")
            row2.addWidget(ip_v)
            layout.addLayout(row2)

            row3 = QHBoxLayout()
            p_n = QLabel("Port")
            p_n.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
            row3.addWidget(p_n)
            row3.addStretch()
            p_v = QLabel(str(plc.get("port", 102)))
            p_v.setStyleSheet(f"color: {C('text_primary')}; font-size: 12px; background: transparent;")
            row3.addWidget(p_v)
            layout.addLayout(row3)
        else:
            row = QHBoxLayout()
            s = QLabel("Holat")
            s.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
            row.addWidget(s)
            row.addStretch()
            sv = QLabel("O'CHIRILGAN")
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
        self.camera_detection_labels.clear()
        self.camera_polytime_labels.clear()
        self.camera_types.clear()

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
