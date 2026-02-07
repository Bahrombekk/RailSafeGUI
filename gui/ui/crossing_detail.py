"""
Crossing Detail View - Responsive cameras with grid layout, auto-reconnect
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QScrollArea, QFrame,
                              QTableWidget, QTableWidgetItem,
                              QHeaderView, QSizePolicy, QApplication,
                              QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QMutex
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np
import os
import time
import threading

from gui.utils.theme_colors import C

# Car detector import (optional - graceful degradation)
try:
    from detectors import RealtimeMultiCameraDetector
    CAR_DETECTOR_AVAILABLE = True
except ImportError:
    CAR_DETECTOR_AVAILABLE = False
    print("[CrossingDetail] RealtimeMultiCameraDetector not available")

# RTSP ultra-low-latency (NVIDIA GPU - VA-API o'rniga software decode)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp|stimeout;2000000|'
    'fflags;nobuffer+discardcorrupt|flags;low_delay|'
    'analyzeduration;100000|probesize;100000|'
    'max_delay;0|reorder_queue_size;0'
)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'


class DetailCameraWorker(QThread):
    """Worker thread - all heavy work here, GUI only does setPixmap, auto-reconnect"""
    frame_ready = pyqtSignal(QImage)
    status_changed = pyqtSignal(str)
    detection_updated = pyqtSignal(int, float)  # detection_count, fps

    def __init__(self, source: str, camera_name: str = "Camera", display_width: int = 960,
                 car_detector: 'CarDetector' = None, detection_enabled: bool = True):
        super().__init__()
        self.source = source
        self.camera_name = camera_name
        self.display_width = display_width
        self._running = True
        self._mutex = QMutex()
        self._frame_pending = False
        self._frame_mutex = QMutex()
        self._retry_delay = 3

        # Car detector - non-blocking real-time mode
        self.car_detector = car_detector
        self.detection_enabled = detection_enabled and car_detector is not None

    def set_frame_delivered(self):
        self._frame_mutex.lock()
        self._frame_pending = False
        self._frame_mutex.unlock()

    def run(self):
        retry_count = 0
        while self._is_running():
            cap = None
            gt = None
            _grab_running = [True]

            try:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
                    while self._is_running() and not _grab_error[0]:
                        self._frame_mutex.lock()
                        pending = self._frame_pending
                        self._frame_mutex.unlock()
                        if pending:
                            time.sleep(0.001)
                            continue

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

                        # Car detection - NON-BLOCKING
                        detection_count = 0
                        if self.detection_enabled and self.car_detector is not None:
                            try:
                                detections, det_frame = self.car_detector.detect_async(
                                    frame, camera_id=self.camera_name)
                                detection_count = len(detections)
                                if detections:
                                    draw_on = det_frame if det_frame is not None else frame
                                    frame = self.car_detector.draw_detections(
                                        draw_on, detections,
                                        thickness=2, font_scale=0.6)
                                    h, w = frame.shape[:2]
                                fps = self.car_detector.get_fps()
                                self.detection_updated.emit(detection_count, fps)
                            except Exception as e:
                                print(f"[{self.camera_name}] Detection error: {e}")

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        qimg = QImage(rgb.data, w, h, w * 3,
                                      QImage.Format.Format_RGB888).copy()

                        self._frame_mutex.lock()
                        self._frame_pending = True
                        self._frame_mutex.unlock()
                        self.frame_ready.emit(qimg)
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
        self._mutex.lock()
        r = self._running
        self._mutex.unlock()
        return r

    def stop(self):
        try:
            self._mutex.lock()
            self._running = False
            self._mutex.unlock()
        except Exception:
            self._running = False
        try:
            if self.isRunning():
                self.quit()
                self.wait(5000)
        except (RuntimeError, Exception):
            pass


class CrossingDetail(QWidget):
    """Detailed view - responsive cameras with grid, auto-reconnect"""

    back_clicked = pyqtSignal()
    add_camera_clicked = pyqtSignal(int)
    edit_crossing_clicked = pyqtSignal(int)
    delete_crossing_clicked = pyqtSignal(int)

    def __init__(self, config_manager, crossing_id: int, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.crossing_id = crossing_id
        self.crossing_data = config_manager.get_crossing(crossing_id)
        self.camera_workers = []
        self.camera_labels = {}
        self.camera_status_labels = {}
        self.camera_detection_labels = {}  # Detection info labels
        self._destroyed = False

        # Car detector initialization
        self.car_detector = None
        self._init_car_detector()

        if not self.crossing_data:
            raise ValueError(f"Crossing {crossing_id} not found")

        self._setup_ui()
        QTimer.singleShot(300, self._start_all_cameras)

    def _init_car_detector(self):
        """
        REAL-TIME Multi-Camera Detector - GPU 100%
        TensorRT qo'llab-quvvatlaydi - 3-5x tezroq inference
        """
        if not CAR_DETECTOR_AVAILABLE:
            print("[CrossingDetail] RealtimeMultiCameraDetector not available")
            return

        try:
            car_config = self.config_manager.get_car_detector_config()
            if not car_config.get("enabled", False):
                print("[CrossingDetail] Car detector disabled in config")
                return

            model_path = car_config.get("model_path", "")
            if not model_path or not os.path.exists(model_path):
                print(f"[CrossingDetail] Car detector model not found: {model_path}")
                return

            # REAL-TIME detector - barcha kameralar BITTA batch
            # TensorRT engine mavjud bo'lsa, avtomatik ishlatiladi
            self.car_detector = RealtimeMultiCameraDetector(
                model_path=model_path,
                confidence_threshold=car_config.get("confidence", 0.3),
                iou_threshold=car_config.get("iou_threshold", 0.45),
                imgsz=car_config.get("imgsz", 640),
                device=car_config.get("device", "cuda"),
                half=car_config.get("half", True),
                filter_classes=car_config.get("filter_classes"),
                batch_interval_ms=15.0,  # Har 15ms batch process
            )

            # Pre-load model (TensorRT auto-detected)
            if self.car_detector.load():
                stats = self.car_detector.get_stats()
                print(f"[CrossingDetail] Detector yuklandi! Mode: {stats['model_type'].upper()}")
            else:
                self.car_detector = None
                print("[CrossingDetail] Detector yuklanmadi")

        except Exception as e:
            print(f"[CrossingDetail] Car detector init error: {e}")
            import traceback
            traceback.print_exc()
            self.car_detector = None

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

        # Events
        events = self._create_events_table()
        content_layout.addWidget(events)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Time timer
        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self._update_time)
        self.time_timer.start(1000)

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
        det_lbl = QLabel("Detect: 0 | FPS: 0.0")
        det_lbl.setStyleSheet(f"color: {C('text_secondary')}; font-size: 10px; background: transparent;")
        self.camera_detection_labels[cam_id] = det_lbl
        bottom.addWidget(det_lbl)

        bottom.addStretch()

        plate_lbl = QLabel("Avtomobil: -- --- --")
        plate_lbl.setStyleSheet(f"color: {C('accent_orange')}; font-size: 10px; font-weight: bold; background: transparent;")
        bottom.addWidget(plate_lbl)

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
            if worker is not None:
                worker.set_frame_delivered()
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
            label = self.camera_labels.get(cam_id)
            if not label:
                continue

            # Create worker with car detector
            worker = DetailCameraWorker(
                src,
                cam.get("name", f"Cam-{cam_id}"),
                car_detector=self.car_detector,
                detection_enabled=cam.get("detection_enabled", True)
            )
            worker.frame_ready.connect(
                lambda f, lbl=label, w=worker: self._on_frame(lbl, f, w)
            )
            worker.status_changed.connect(
                lambda s, cid=cam_id: self._on_camera_status(cid, s)
            )
            worker.detection_updated.connect(
                lambda count, fps, cid=cam_id: self._on_detection_update(cid, count, fps)
            )
            worker.start()
            self.camera_workers.append(worker)

    def _on_frame(self, label, frame, worker=None):
        if not self._destroyed:
            try:
                self._show_frame(label, frame, worker)
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

    def _on_detection_update(self, cam_id, count, fps):
        """Handle detection updates from camera worker"""
        if self._destroyed:
            return
        try:
            det_label = self.camera_detection_labels.get(cam_id)
            if det_label:
                det_label.setText(f"Detect: {count} | FPS: {fps:.1f}")
                if count > 0:
                    det_label.setStyleSheet(f"color: {C('accent_green')}; font-size: 10px; font-weight: bold; background: transparent;")
                else:
                    det_label.setStyleSheet(f"color: {C('text_secondary')}; font-size: 10px; background: transparent;")
        except RuntimeError:
            self._destroyed = True

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
        dialog = AddCameraDialog(self.config_manager, self.crossing_id, camera_id)
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

        stats = [
            ("Kameralar", f"{active}/{cameras_count}", C('accent_brand')),
            ("Jami Transport", "0", C('accent_green')),
            ("Buzilishlar", "0", C('accent_red')),
            ("O'rtacha Vaqt", "0s", C('accent_yellow')),
        ]

        for name, value, color in stats:
            row = QHBoxLayout()
            n = QLabel(name)
            n.setStyleSheet(f"color: {C('text_muted')}; font-size: 12px; background: transparent;")
            row.addWidget(n)
            row.addStretch()
            v = QLabel(value)
            v.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold; background: transparent;")
            row.addWidget(v)
            layout.addLayout(row)

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

    def _create_events_table(self):
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame#eventsPanel {{
                background: {C('bg_primary')};
                border: 2px solid {C('bg_input')};
                border-radius: 12px;
            }}
        """)
        panel.setObjectName("eventsPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        title = QLabel("So'nggi Hodisalar")
        title.setStyleSheet(f"color: {C('text_primary')}; font-size: 14px; font-weight: bold; background: transparent;")
        layout.addWidget(title)

        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {C('bg_input')};")
        layout.addWidget(div)

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Vaqt", "Kamera", "Hodisa", "Transport"])
        table.setStyleSheet(f"""
            QTableWidget {{
                background: {C('bg_secondary')}; border: 1px solid {C('bg_input')};
                color: {C('text_primary')}; gridline-color: {C('bg_input')}; border-radius: 6px;
            }}
            QHeaderView::section {{
                background: {C('bg_primary')}; color: {C('text_secondary')};
                border: 1px solid {C('bg_input')}; padding: 6px;
                font-size: 11px; font-weight: bold;
            }}
            QTableWidget::item:alternate {{ background: {C('bg_primary')}; }}
        """)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        table.setAlternatingRowColors(True)
        table.setMinimumHeight(120)
        table.setMaximumHeight(180)
        table.setRowCount(0)
        table.verticalHeader().setVisible(False)

        layout.addWidget(table)
        return panel

    def cleanup(self):
        if self._destroyed:
            return
        self._destroyed = True
        try:
            if hasattr(self, 'time_timer') and self.time_timer is not None:
                self.time_timer.stop()
        except RuntimeError:
            pass
        for w in self.camera_workers:
            try:
                w.frame_ready.disconnect()
                w.status_changed.disconnect()
                w.detection_updated.disconnect()
            except Exception:
                pass
            try:
                w.stop()
            except (RuntimeError, Exception):
                pass
        self.camera_workers.clear()
        self.camera_detection_labels.clear()

    def refresh(self):
        self.cleanup()
        self._destroyed = False
        self.crossing_data = self.config_manager.get_crossing(self.crossing_id)
        self.camera_labels.clear()
        self.camera_status_labels.clear()
        self.camera_detection_labels.clear()
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
