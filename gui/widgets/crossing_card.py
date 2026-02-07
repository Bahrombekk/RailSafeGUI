"""
Crossing Card Widget - Shows crossing with main + additional camera layout
Layout: left = main camera (tall), right = additional camera + PLC + stats
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QGridLayout, QSizePolicy, QMenu,
                              QPushButton, QDialog, QFormLayout, QLineEdit,
                              QComboBox, QCheckBox, QFileDialog, QMessageBox,
                              QToolButton)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QMutex, QPoint
from PyQt6.QtGui import QFont, QPixmap, QImage, QAction
import cv2
import numpy as np
import os
import time
from gui.utils.theme_colors import C

# Car detector import (optional)
try:
    from detectors import BatchCarDetector
    CAR_DETECTOR_AVAILABLE = True
except ImportError:
    CAR_DETECTOR_AVAILABLE = False
    print("[CrossingCard] BatchCarDetector not available")

# RTSP low-latency + Intel VA-API hw decode (set once, not per-thread)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp|stimeout;2000000|'
    'fflags;nobuffer|flags;low_delay|'
    'analyzeduration;500000|probesize;500000|'
    'hwaccel;vaapi|hwaccel_device;/dev/dri/renderD128'
)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'


class CameraWorker(QThread):
    """Worker thread - all heavy work here, GUI thread only does setPixmap"""
    frame_ready = pyqtSignal(QImage)
    status_changed = pyqtSignal(str)
    detection_updated = pyqtSignal(int)  # detection count

    def __init__(self, source: str, camera_name: str = "Camera", display_width: int = 640,
                 car_detector=None, detection_enabled: bool = True):
        super().__init__()
        self.source = source
        self.camera_name = camera_name
        self.display_width = display_width
        self._running = True
        self._mutex = QMutex()
        self._frame_pending = False
        self._frame_mutex = QMutex()
        self.cap = None

        # Car detector - non-blocking real-time mode
        self.car_detector = car_detector
        self.detection_enabled = detection_enabled and car_detector is not None

    def set_frame_delivered(self):
        self._frame_mutex.lock()
        self._frame_pending = False
        self._frame_mutex.unlock()

    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.status_changed.emit("online")
                consecutive_fails = 0

                while self._is_running():
                    # ALWAYS grab to flush RTSP buffer (real-time)
                    ret = self.cap.grab()
                    if not ret:
                        consecutive_fails += 1
                        if consecutive_fails > 30:
                            self.status_changed.emit("error")
                            break
                        continue
                    consecutive_fails = 0

                    # Only decode when GUI is ready
                    self._frame_mutex.lock()
                    pending = self._frame_pending
                    self._frame_mutex.unlock()
                    if pending:
                        continue

                    ret, frame = self.cap.retrieve()
                    if ret and self._is_running():
                        # ALL heavy work in worker thread (parallel across cameras)
                        h, w = frame.shape[:2]
                        if w > self.display_width:
                            scale = self.display_width / w
                            frame = cv2.resize(frame, (self.display_width, int(h * scale)),
                                               interpolation=cv2.INTER_AREA)
                            h, w = frame.shape[:2]

                        # Car detection - SYNC mode (LAG YO'Q)
                        detection_count = 0
                        if self.detection_enabled and self.car_detector is not None:
                            try:
                                # Sync detect - real-time, lag yo'q
                                detections = self.car_detector.detect_sync(
                                    frame,
                                    camera_id=self.camera_name
                                )
                                detection_count = len(detections)
                                if detections:
                                    frame = self.car_detector.draw_detections(
                                        frame, detections,
                                        thickness=2,
                                        font_scale=0.5
                                    )
                                self.detection_updated.emit(detection_count)
                            except Exception as e:
                                print(f"[{self.camera_name}] Detection error: {e}")

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # QImage created in worker thread (thread-safe)
                        qimg = QImage(rgb.data, w, h, w * 3,
                                      QImage.Format.Format_RGB888).copy()

                        self._frame_mutex.lock()
                        self._frame_pending = True
                        self._frame_mutex.unlock()
                        self.frame_ready.emit(qimg)
            else:
                self.status_changed.emit("error")

        except Exception as e:
            print(f"[{self.camera_name}] Error: {e}")
            self.status_changed.emit("error")
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None

    def _is_running(self):
        self._mutex.lock()
        r = self._running
        self._mutex.unlock()
        return r

    def stop(self):
        # Signal thread to stop - grab() will return and loop exits
        try:
            self._mutex.lock()
            self._running = False
            self._mutex.unlock()
        except Exception:
            self._running = False

        # Wait for thread to finish naturally (cap.release happens in run())
        try:
            if self.isRunning():
                self.quit()
                self.wait(5000)
        except (RuntimeError, Exception):
            pass


class CameraSettingsDialog(QDialog):
    """Camera settings dialog with enable/disable and delete"""

    camera_deleted = pyqtSignal()  # emitted when a camera is deleted

    def __init__(self, crossing_data: dict, config_manager=None, parent=None):
        super().__init__(parent)
        self.crossing_data = crossing_data
        self.crossing_id = crossing_data.get("id", 0)
        self.config_manager = config_manager
        self._changed = False
        self.setWindowTitle(f"Kamera Sozlamalari - {crossing_data.get('name', 'Pereezd')}")
        self.setMinimumWidth(520)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel(f"📷 {self.crossing_data.get('name', 'Pereezd')} - Kameralar")
        title.setStyleSheet(f"color: {C('accent_brand')}; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        cameras = self.crossing_data.get("cameras", [])

        if not cameras:
            no_cam = QLabel("Kameralar topilmadi")
            no_cam.setStyleSheet(f"color: {C('text_muted')}; font-size: 13px; padding: 20px;")
            no_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(no_cam)
        else:
            for cam in cameras:
                cam_frame = QFrame()
                cam_frame.setStyleSheet(f"""
                    QFrame {{
                        background-color: {C('bg_panel')};
                        border: 1px solid {C('bg_panel_border')};
                        border-radius: 6px;
                    }}
                """)
                cam_inner = QVBoxLayout(cam_frame)
                cam_inner.setContentsMargins(10, 8, 10, 8)
                cam_inner.setSpacing(6)

                header = QHBoxLayout()
                header.setSpacing(6)
                cam_type = cam.get("type", "additional")
                type_color = C('accent_blue') if cam_type == "main" else C('accent_orange')
                type_text = "Asosiy" if cam_type == "main" else "Qo'shimcha"

                name_lbl = QLabel(cam.get("name", "Kamera"))
                name_lbl.setStyleSheet(f"color: {C('text_primary')}; font-size: 13px; font-weight: bold;")
                header.addWidget(name_lbl)

                tc = type_color.lstrip('#')
                r_v, g_v, b_v = int(tc[:2], 16), int(tc[2:4], 16), int(tc[4:6], 16)
                type_badge = QLabel(type_text)
                type_badge.setStyleSheet(f"""
                    color: {type_color}; font-size: 10px; font-weight: bold;
                    background-color: rgba({r_v}, {g_v}, {b_v}, 0.15);
                    border: 1px solid rgba({r_v}, {g_v}, {b_v}, 0.3);
                    border-radius: 3px; padding: 2px 6px;
                """)
                header.addWidget(type_badge)
                header.addStretch()

                enabled = cam.get("enabled", True)
                status_dot = QLabel("●")
                status_dot.setStyleSheet(f"color: {C('status_online') if enabled else C('status_error')}; font-size: 10px;")
                header.addWidget(status_dot)

                enabled_lbl = QLabel("Yoqilgan" if enabled else "O'chirilgan")
                enabled_lbl.setStyleSheet(f"color: {C('status_online') if enabled else C('status_error')}; font-size: 10px;")
                header.addWidget(enabled_lbl)

                cam_inner.addLayout(header)

                source = cam.get("source", "")
                if source:
                    display_source = source
                    if "@" in source:
                        proto_end = source.find("://")
                        at_pos = source.find("@")
                        if proto_end >= 0 and at_pos > proto_end:
                            display_source = source[:proto_end + 3] + "***@" + source[at_pos + 1:]
                    src_lbl = QLabel(f"Manba: {display_source}")
                    src_lbl.setStyleSheet(f"color: {C('text_muted')}; font-size: 10px;")
                    src_lbl.setWordWrap(True)
                    cam_inner.addWidget(src_lbl)

                # Action buttons row
                actions = QHBoxLayout()
                actions.setSpacing(6)
                actions.addStretch()

                cam_id = cam.get("id")

                # Toggle enable/disable
                toggle_text = "O'chirish" if enabled else "Yoqish"
                toggle_color = C('accent_yellow') if enabled else C('accent_green')
                toggle_btn = QPushButton(toggle_text)
                toggle_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: transparent; color: {toggle_color};
                        border: 1px solid {toggle_color}; border-radius: 3px;
                        padding: 3px 10px; font-size: 10px;
                    }}
                    QPushButton:hover {{ background-color: rgba(255,255,255,0.05); }}
                """)
                toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                toggle_btn.clicked.connect(
                    lambda checked, cid=cam_id, en=enabled: self._toggle_camera(cid, en))
                actions.addWidget(toggle_btn)

                # Delete button
                del_btn = QPushButton("O'chirish 🗑")
                del_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: transparent; color: {C('accent_red')};
                        border: 1px solid {C('accent_red')}; border-radius: 3px;
                        padding: 3px 10px; font-size: 10px;
                    }}
                    QPushButton:hover {{ background-color: rgba(243, 139, 168, 0.1); }}
                """)
                del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                del_btn.clicked.connect(
                    lambda checked, cid=cam_id, cname=cam.get("name", ""): self._delete_camera(cid, cname))
                actions.addWidget(del_btn)

                cam_inner.addLayout(actions)
                layout.addWidget(cam_frame)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Yopish")
        close_btn.setMinimumWidth(100)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {C('bg_input')}; color: {C('text_primary')};
                border: 1px solid {C('bg_hover')}; border-radius: 4px;
                padding: 6px 16px; font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {C('bg_hover')}; }}
        """)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _toggle_camera(self, camera_id, currently_enabled):
        if not self.config_manager or not camera_id:
            return
        self.config_manager.update_camera(
            self.crossing_id, camera_id, {"enabled": not currently_enabled})
        self._changed = True
        QMessageBox.information(self, "Tayyor",
            f"Kamera {'o\u2018chirildi' if currently_enabled else 'yoqildi'}. Yangilash uchun sahifani qayta yuklang.")

    def _delete_camera(self, camera_id, camera_name):
        if not self.config_manager or not camera_id:
            return
        reply = QMessageBox.question(
            self, "Tasdiqlash",
            f"'{camera_name}' kamerasini o'chirmoqchimisiz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.config_manager.delete_camera(self.crossing_id, camera_id)
            self._changed = True
            self.camera_deleted.emit()
            self.accept()


class CrossingCard(QWidget):
    """
    Crossing card with two layout modes:
    Wide (1 crossing):
      ┌─────────────────┬──────────────┐
      │  Kamera 1       │  Kamera 2    │
      │  (katta,baland) ├──────────────┤
      │                 │  PLC + Stats │
      └─────────────────┴──────────────┘
    Compact (2+ crossings):
      ┌────────────────────────────────┐
      │       Kamera 1 (keng)          │
      ├────────────────┬───────────────┤
      │  Kamera 2      │  PLC + Stats  │
      └────────────────┴───────────────┘
    """

    clicked = pyqtSignal(int)

    def __init__(self, crossing_data: dict, config_manager=None, compact=False,
                 car_detector=None, parent=None):
        super().__init__(parent)
        self.crossing_data = crossing_data
        self.crossing_id = crossing_data.get("id", 0)
        self.config_manager = config_manager
        self.compact = compact
        self.camera_workers = []
        self.main_camera_label = None
        self.additional_camera_label = None
        self._is_destroyed = False

        # Car detector - HAR BIR CROSSING O'ZINING DETECTORIGA EGA
        self.car_detector = None
        self._init_own_detector()

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._setup_ui()
        QTimer.singleShot(100, self._start_cameras)

    def _init_own_detector(self):
        """Har bir crossing uchun alohida detector - parallel ishlash"""
        if not CAR_DETECTOR_AVAILABLE or not self.config_manager:
            return

        try:
            car_config = self.config_manager.get_car_detector_config()
            if not car_config.get("enabled", False):
                return

            model_path = car_config.get("model_path", "")
            if not model_path or not os.path.exists(model_path):
                return

            # Har bir crossing uchun alohida detector
            self.car_detector = BatchCarDetector(
                model_path=model_path,
                confidence_threshold=car_config.get("confidence", 0.5),
                iou_threshold=car_config.get("iou_threshold", 0.45),
                imgsz=car_config.get("imgsz", 640),
                device=car_config.get("device", "cuda"),
                half=car_config.get("half", True),
                batch_size=2,  # Kichik batch - tezroq response
                filter_classes=car_config.get("filter_classes"),
            )

            if self.car_detector.load():
                print(f"[CrossingCard {self.crossing_id}] Detector yuklandi")
            else:
                self.car_detector = None

        except Exception as e:
            print(f"[CrossingCard {self.crossing_id}] Detector xato: {e}")
            self.car_detector = None

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)

        # Card frame
        self.frame = QFrame()
        self.frame.setObjectName("cardFrame")
        self.frame.setStyleSheet(f"""
            QFrame#cardFrame {{
                background-color: {C('bg_card')};
                border: 2px solid {C('bg_card_border')};
                border-radius: 8px;
            }}
            QFrame#cardFrame:hover {{
                border-color: {C('accent_blue')};
            }}
        """)
        self.frame.setCursor(Qt.CursorShape.PointingHandCursor)

        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)

        # ── HEADER ──
        header_frame = QFrame()
        header_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {C('bg_card_header')};
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                border-bottom: 1px solid {C('bg_card_border')};
            }}
        """)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 6, 8, 6)
        header_layout.setSpacing(8)

        name_label = QLabel(self.crossing_data.get("name", "Pereezd"))
        name_label.setStyleSheet(
            f"color: {C('text_primary')}; font-size: 14px; font-weight: bold; background: transparent; border: none;")
        header_layout.addWidget(name_label)
        header_layout.addStretch()

        self.menu_btn = QToolButton()
        self.menu_btn.setText("⋮")
        self.menu_btn.setStyleSheet(f"""
            QToolButton {{
                color: {C('text_muted')}; font-size: 18px; font-weight: bold;
                background: transparent; border: none;
                border-radius: 4px; padding: 2px 6px;
            }}
            QToolButton:hover {{ background-color: {C('bg_panel_border')}; color: {C('text_primary')}; }}
        """)
        self.menu_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.menu_btn.clicked.connect(self._show_menu)
        header_layout.addWidget(self.menu_btn)

        frame_layout.addWidget(header_frame)

        # Find cameras
        cameras = self.crossing_data.get("cameras", [])
        self.main_camera_data = None
        self.additional_camera_data = None
        for cam in cameras:
            if cam.get("type") == "main" and self.main_camera_data is None:
                self.main_camera_data = cam
            elif cam.get("type") == "additional" and self.additional_camera_data is None:
                self.additional_camera_data = cam
            elif self.main_camera_data is None:
                self.main_camera_data = cam
            elif self.additional_camera_data is None:
                self.additional_camera_data = cam

        main_name = self.main_camera_data.get("name", "Kamera 1") if self.main_camera_data else "Kamera 1"
        add_name = self.additional_camera_data.get("name", "Kamera 2") if self.additional_camera_data else "Kamera 2"

        # ── Create shared components ──

        # Main camera label
        self.main_camera_label = QLabel()
        self.main_camera_label.setMinimumSize(200, 150)
        self.main_camera_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_camera_label.setStyleSheet(f"""
            background-color: {C('bg_camera')}; border: 1px solid {C('border_light')};
            border-top-left-radius: 4px; border-top-right-radius: 4px;
            border-bottom: none;
        """)
        self._set_placeholder(self.main_camera_label, "Yuklanmoqda...")

        # Main camera bottom bar
        main_bottom = QFrame()
        main_bottom.setStyleSheet(f"""
            QFrame {{
                background-color: {C('bg_camera_bar')};
                border: 1px solid {C('border_light')};
                border-top: none;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }}
        """)
        main_bottom_layout = QHBoxLayout(main_bottom)
        main_bottom_layout.setContentsMargins(8, 3, 8, 3)
        main_bottom_layout.setSpacing(0)

        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet(
            f"color: {C('status_online')}; font-size: 8px; background: transparent; border: none;")
        main_bottom_layout.addWidget(self.status_indicator)

        main_name_lbl = QLabel(f"  {main_name}")
        main_name_lbl.setStyleSheet(
            f"color: {C('text_secondary')}; font-size: 10px; background: transparent; border: none;")
        main_bottom_layout.addWidget(main_name_lbl)

        main_bottom_layout.addStretch()

        self.time_label = QLabel("00:00:00")
        self.time_label.setStyleSheet(
            f"color: {C('text_primary')}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        main_bottom_layout.addWidget(self.time_label)

        main_bottom_layout.addStretch()

        # Additional camera label
        self.additional_camera_label = QLabel()
        self.additional_camera_label.setMinimumSize(160, 60)
        self.additional_camera_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.additional_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.additional_camera_label.setStyleSheet(f"""
            background-color: {C('bg_camera')}; border: 1px solid {C('border_light')};
            border-top-left-radius: 4px; border-top-right-radius: 4px;
            border-bottom: none;
        """)
        self._set_placeholder(self.additional_camera_label, add_name)

        # Additional camera bottom bar
        add_bottom = QFrame()
        add_bottom.setStyleSheet(f"""
            QFrame {{
                background-color: {C('bg_camera_bar')};
                border: 1px solid {C('border_light')}; border-top: none;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }}
        """)
        add_bottom_layout = QHBoxLayout(add_bottom)
        add_bottom_layout.setContentsMargins(8, 3, 8, 3)
        add_name_lbl = QLabel(add_name)
        add_name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        add_name_lbl.setStyleSheet(
            f"color: {C('text_secondary')}; font-size: 10px; background: transparent; border: none;")
        add_bottom_layout.addWidget(add_name_lbl)

        # PLC panel
        plc_frame = self._create_plc_panel()

        # Stats panel
        stats_frame = self._create_stats_panel()

        # Time timer
        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self._update_time)
        self.time_timer.start(1000)

        # ── COMPOSE LAYOUT ──
        if self.compact:
            # Compact: top=main camera, bottom row=(camera2 | PLC+stats)
            content_widget = QWidget()
            content_widget.setStyleSheet("background: transparent;")
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(8, 6, 8, 6)
            content_layout.setSpacing(4)

            # Main camera (full width, top)
            content_layout.addWidget(self.main_camera_label, stretch=3)

            # Bottom row: camera2 (left) | PLC+stats (right)
            bottom_row = QWidget()
            bottom_row.setStyleSheet("background: transparent;")
            bottom_row_layout = QHBoxLayout(bottom_row)
            bottom_row_layout.setContentsMargins(0, 0, 0, 0)
            bottom_row_layout.setSpacing(4)

            # Camera 2 + name bar
            cam2_widget = QWidget()
            cam2_widget.setStyleSheet("background: transparent;")
            cam2_layout = QVBoxLayout(cam2_widget)
            cam2_layout.setContentsMargins(0, 0, 0, 0)
            cam2_layout.setSpacing(0)
            cam2_layout.addWidget(self.additional_camera_label, stretch=1)
            cam2_layout.addWidget(add_bottom)
            bottom_row_layout.addWidget(cam2_widget, stretch=1)

            # PLC + Stats stacked
            right_panel = QWidget()
            right_panel.setStyleSheet("background: transparent;")
            right_panel_layout = QVBoxLayout(right_panel)
            right_panel_layout.setContentsMargins(0, 0, 0, 0)
            right_panel_layout.setSpacing(4)
            right_panel_layout.addWidget(plc_frame, stretch=1)
            right_panel_layout.addWidget(stats_frame, stretch=2)
            bottom_row_layout.addWidget(right_panel, stretch=1)

            content_layout.addWidget(bottom_row, stretch=2)

            frame_layout.addWidget(content_widget, stretch=1)

            # Bottom bar at very bottom
            frame_layout.addWidget(main_bottom)
        else:
            # Wide: left=main camera+bar, right=camera2+PLC+stats
            content_widget = QWidget()
            content_widget.setStyleSheet("background: transparent;")
            content_layout = QHBoxLayout(content_widget)
            content_layout.setContentsMargins(8, 6, 8, 6)
            content_layout.setSpacing(6)

            # LEFT: Main camera + bottom bar
            left_widget = QWidget()
            left_widget.setStyleSheet("background: transparent;")
            left_layout = QVBoxLayout(left_widget)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(0)
            left_layout.addWidget(self.main_camera_label, stretch=1)
            left_layout.addWidget(main_bottom)
            content_layout.addWidget(left_widget, stretch=4)

            # RIGHT: camera2 + PLC + stats
            right_widget = QWidget()
            right_widget.setStyleSheet("background: transparent;")
            right_layout = QVBoxLayout(right_widget)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.setSpacing(4)
            right_layout.addWidget(self.additional_camera_label, stretch=5)
            right_layout.addWidget(add_bottom)
            right_layout.addWidget(plc_frame, stretch=1)
            right_layout.addWidget(stats_frame, stretch=2)
            content_layout.addWidget(right_widget, stretch=2)

            frame_layout.addWidget(content_widget, stretch=1)

        main_layout.addWidget(self.frame)
        self.frame.mousePressEvent = self._on_mouse_press
        self.frame.mouseDoubleClickEvent = self._on_double_click
        self.frame.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.frame.customContextMenuRequested.connect(self._show_context_menu)

    def _create_plc_panel(self):
        plc_data = self.crossing_data.get("plc", {})
        plc_enabled = plc_data.get("enabled", False)
        plc_color = C('status_online') if plc_enabled else C('text_muted')
        plc_status_text = "ONLINE" if plc_enabled else "O'CHIRILGAN"

        plc_frame = QFrame()
        plc_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {C('bg_panel')}, stop:1 {C('bg_panel_dark')});
                border: 1px solid {C('bg_panel_border')};
                border-radius: 5px;
            }}
        """)
        plc_inner = QVBoxLayout(plc_frame)
        plc_inner.setContentsMargins(8, 5, 8, 5)
        plc_inner.setSpacing(2)

        plc_title = QLabel("PLC HOLATI")
        plc_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plc_title.setStyleSheet(
            f"color: {C('text_dim')}; font-size: 9px; font-weight: bold; letter-spacing: 1px;"
            " background: transparent; border: none;")
        plc_inner.addWidget(plc_title)

        plc_row = QHBoxLayout()
        plc_row.setSpacing(5)
        plc_row.addStretch()

        plc_icon = QLabel("🔌")
        plc_icon.setStyleSheet("font-size: 12px; background: transparent; border: none;")
        plc_row.addWidget(plc_icon)

        self.plc_indicator = QLabel("●")
        self.plc_indicator.setStyleSheet(
            f"color: {plc_color}; font-size: 10px; background: transparent; border: none;")
        plc_row.addWidget(self.plc_indicator)

        self.plc_status_label = QLabel(plc_status_text)
        self.plc_status_label.setStyleSheet(
            f"color: {plc_color}; font-size: 12px; font-weight: bold; background: transparent; border: none;")
        plc_row.addWidget(self.plc_status_label)

        plc_row.addStretch()
        plc_inner.addLayout(plc_row)

        self.plc_time_label = QLabel(time.strftime("%H:%M:%S"))
        self.plc_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plc_time_label.setStyleSheet(
            f"color: {C('text_dim')}; font-size: 9px; background: transparent; border: none;")
        plc_inner.addWidget(self.plc_time_label)

        return plc_frame

    @staticmethod
    def _format_count(n):
        """Format number: 1000 → 1,000 | 1000000 → 1.0M"""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 10_000:
            return f"{n / 1_000:.1f}K"
        return f"{n:,}".replace(",", " ")

    @staticmethod
    def _count_font_size(text):
        """Dynamic font size based on text length"""
        length = len(text)
        if length <= 2:
            return 18
        if length <= 4:
            return 15
        if length <= 6:
            return 13
        return 11

    def _update_stat_label(self, label, value, color):
        """Update stat label with dynamic font size"""
        text = self._format_count(value)
        size = self._count_font_size(text)
        label.setText(text)
        label.setStyleSheet(
            f"font-size: {size}px; font-weight: bold; color: {color};"
            " background: transparent; border: none;")

    def _create_stats_panel(self):
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {C('bg_panel')}, stop:1 {C('bg_panel_dark')});
                border: 1px solid {C('bg_panel_border')};
                border-radius: 5px;
            }}
        """)
        stats_inner = QVBoxLayout(stats_frame)
        stats_inner.setContentsMargins(6, 4, 6, 4)
        stats_inner.setSpacing(3)

        stats_title = QLabel("STATISTIKA")
        stats_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_title.setStyleSheet(
            f"color: {C('text_dim')}; font-size: 9px; font-weight: bold; letter-spacing: 1px;"
            " background: transparent; border: none;")
        stats_inner.addWidget(stats_title)

        counts_row = QHBoxLayout()
        counts_row.setSpacing(4)

        # Car badge
        car_badge = QFrame()
        car_badge.setStyleSheet(f"""
            QFrame {{
                background-color: {C('bg_panel_dark')};
                border: 1px solid {C('bg_panel_border')};
                border-radius: 4px;
            }}
        """)
        car_lay = QHBoxLayout(car_badge)
        car_lay.setContentsMargins(5, 3, 5, 3)
        car_lay.setSpacing(3)
        car_ic = QLabel("🚗")
        car_ic.setStyleSheet("font-size: 16px; background: transparent; border: none;")
        car_ic.setFixedWidth(20)
        car_lay.addWidget(car_ic)
        self.car_count = QLabel("0")
        self.car_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.car_count.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._update_stat_label(self.car_count, 0, C('accent_blue'))
        car_lay.addWidget(self.car_count)
        counts_row.addWidget(car_badge, stretch=1)

        # Truck badge
        truck_badge = QFrame()
        truck_badge.setStyleSheet(f"""
            QFrame {{
                background-color: {C('bg_panel_dark')};
                border: 1px solid {C('bg_panel_border')};
                border-radius: 4px;
            }}
        """)
        truck_lay = QHBoxLayout(truck_badge)
        truck_lay.setContentsMargins(5, 3, 5, 3)
        truck_lay.setSpacing(3)
        truck_ic = QLabel("🚚")
        truck_ic.setStyleSheet("font-size: 16px; background: transparent; border: none;")
        truck_ic.setFixedWidth(20)
        truck_lay.addWidget(truck_ic)
        self.truck_count = QLabel("0")
        self.truck_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.truck_count.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._update_stat_label(self.truck_count, 0, C('accent_orange'))
        truck_lay.addWidget(self.truck_count)
        counts_row.addWidget(truck_badge, stretch=1)

        stats_inner.addLayout(counts_row)

        # Total
        self.total_label = QLabel("Jami: 0")
        self.total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.total_label.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {C('accent_green')};"
            f" background-color: {C('bg_panel_dark')};"
            f" border: 1px solid {C('bg_panel_border')};"
            " border-radius: 4px; padding: 2px 0px;")
        stats_inner.addWidget(self.total_label)

        return stats_frame

    def update_stats(self, car_count: int, truck_count: int):
        """Update statistics with dynamic formatting"""
        self._update_stat_label(self.car_count, car_count, C('accent_blue'))
        self._update_stat_label(self.truck_count, truck_count, C('accent_orange'))
        total = car_count + truck_count
        total_text = self._format_count(total)
        total_size = self._count_font_size(total_text)
        self.total_label.setText(f"Jami: {total_text}")
        self.total_label.setStyleSheet(
            f"font-size: {total_size}px; font-weight: bold; color: {C('accent_green')};"
            f" background-color: {C('bg_panel_dark')};"
            f" border: 1px solid {C('bg_panel_border')};"
            " border-radius: 4px; padding: 2px 0px;")

    def _show_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {C('menu_bg')}; border: 1px solid {C('menu_border')};
                border-radius: 6px; padding: 4px;
            }}
            QMenu::item {{
                color: {C('text_primary')}; padding: 6px 20px; border-radius: 4px;
            }}
            QMenu::item:selected {{ background-color: {C('menu_hover')}; }}
        """)

        cam_settings = menu.addAction("📷 Kamera Sozlamalari")
        cam_settings.triggered.connect(self._open_camera_settings)

        info_action = menu.addAction("ℹ️ Pereezd Ma'lumotlari")
        info_action.triggered.connect(lambda: self.clicked.emit(self.crossing_id))

        btn_pos = self.menu_btn.mapToGlobal(QPoint(0, self.menu_btn.height()))
        menu.exec(btn_pos)

    def _open_camera_settings(self):
        dialog = CameraSettingsDialog(self.crossing_data, self.config_manager, self)
        dialog.exec()

    def _set_placeholder(self, label: QLabel, text: str):
        try:
            if self._is_destroyed or label is None:
                return
            w = min(max(label.width(), label.minimumWidth(), 180), 1920)
            h = min(max(label.height(), label.minimumHeight(), 120), 1080)
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            placeholder[:] = (13, 13, 26)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(placeholder, text, (text_x, text_y), font, 0.5, (100, 100, 150), 1)

            rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(qimg))
        except (RuntimeError, Exception) as e:
            print(f"[Placeholder] Error: {e}")

    def _display_frame(self, label: QLabel, qimg: QImage, worker=None):
        """Display ready QImage - minimal GUI thread work"""
        if qimg is None or self._is_destroyed or label is None:
            return
        try:
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
            label.setPixmap(scaled)
            if worker is not None:
                worker.set_frame_delivered()
        except (RuntimeError, Exception):
            pass

    def _update_time(self):
        if self._is_destroyed:
            return
        try:
            current_time = time.strftime("%H:%M:%S")
            if hasattr(self, 'time_label') and self.time_label is not None:
                self.time_label.setText(current_time)
            if hasattr(self, 'plc_time_label') and self.plc_time_label is not None:
                self.plc_time_label.setText(current_time)
        except RuntimeError:
            self._is_destroyed = True

    def _start_cameras(self):
        if self._is_destroyed:
            return

        try:
            cameras = self.crossing_data.get("cameras", [])

            for i, camera in enumerate(cameras[:2]):
                if not camera.get("enabled", False):
                    continue

                source = camera.get("source", "")
                if not source:
                    continue

                cam_type = camera.get("type", "main")
                cam_name = camera.get("name", f"Camera {i+1}")

                # Create worker with car detector
                worker = CameraWorker(
                    source, cam_name,
                    car_detector=self.car_detector,
                    detection_enabled=camera.get("detection_enabled", True)
                )

                if i == 0 or cam_type == "main":
                    worker.frame_ready.connect(lambda f, w=worker: self._on_main_frame(f, w))
                    worker.status_changed.connect(self._on_main_status)
                    worker.detection_updated.connect(self._on_detection_update)
                else:
                    worker.frame_ready.connect(lambda f, w=worker: self._on_additional_frame(f, w))
                    worker.status_changed.connect(self._on_additional_status)

                worker.start()
                self.camera_workers.append(worker)
        except (RuntimeError, Exception) as e:
            print(f"[StartCameras] Error: {e}")

    def _on_main_frame(self, frame, worker=None):
        if self._is_destroyed or self.main_camera_label is None:
            return
        try:
            self._display_frame(self.main_camera_label, frame, worker)
        except RuntimeError:
            self._is_destroyed = True

    def _on_additional_frame(self, frame, worker=None):
        if self._is_destroyed or self.additional_camera_label is None:
            return
        try:
            self._display_frame(self.additional_camera_label, frame, worker)
        except RuntimeError:
            self._is_destroyed = True

    def _on_main_status(self, status):
        if self._is_destroyed:
            return
        try:
            if status == "online":
                self.status_indicator.setStyleSheet(
                    f"color: {C('status_online')}; font-size: 10px; background: transparent; border: none;")
            elif status == "error":
                self.status_indicator.setStyleSheet(
                    f"color: {C('status_error')}; font-size: 10px; background: transparent; border: none;")
                if self.main_camera_label:
                    self._set_placeholder(self.main_camera_label, "Ulanmadi")
        except RuntimeError:
            self._is_destroyed = True

    def _on_additional_status(self, status):
        if self._is_destroyed:
            return
        try:
            if status == "error" and self.additional_camera_label:
                self._set_placeholder(self.additional_camera_label, "Ulanmadi")
        except RuntimeError:
            self._is_destroyed = True

    def _on_detection_update(self, count):
        """Handle detection count updates"""
        if self._is_destroyed:
            return
        try:
            # Update car count stat
            self._update_stat_label(self.car_count, count, C('accent_blue'))
        except RuntimeError:
            self._is_destroyed = True

    def _on_mouse_press(self, event):
        """Single click - do nothing, wait for double click"""
        pass

    def _on_double_click(self, event):
        """Double click - open crossing detail"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.crossing_id)

    def _show_context_menu(self, pos):
        """Right click context menu"""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {C('menu_bg')}; border: 1px solid {C('menu_border')};
                border-radius: 6px; padding: 4px;
            }}
            QMenu::item {{
                color: {C('text_primary')}; padding: 6px 20px; border-radius: 4px;
            }}
            QMenu::item:selected {{ background-color: {C('menu_hover')}; }}
        """)

        open_action = menu.addAction("📂 Ochish")
        open_action.triggered.connect(lambda: self.clicked.emit(self.crossing_id))

        cam_settings = menu.addAction("📷 Kamera Sozlamalari")
        cam_settings.triggered.connect(self._open_camera_settings)

        menu.addSeparator()

        info_action = menu.addAction("ℹ️ Ma'lumotlar")
        info_action.triggered.connect(lambda: self.clicked.emit(self.crossing_id))

        menu.exec(self.frame.mapToGlobal(pos))

    def stop_cameras(self):
        for worker in self.camera_workers:
            try:
                worker.stop()
            except (RuntimeError, Exception) as e:
                print(f"[StopCamera] Error: {e}")
        self.camera_workers.clear()

    def cleanup(self):
        self._is_destroyed = True
        try:
            if hasattr(self, 'time_timer') and self.time_timer is not None:
                self.time_timer.stop()
        except RuntimeError:
            pass
        self.stop_cameras()
        # Detector to'xtatish
        if self.car_detector is not None:
            try:
                self.car_detector.stop()
            except Exception:
                pass
            self.car_detector = None

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

    def deleteLater(self):
        self.cleanup()
        try:
            super().deleteLater()
        except RuntimeError:
            pass
