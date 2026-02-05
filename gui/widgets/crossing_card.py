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


class CameraWorker(QThread):
    """Worker thread for camera streaming - proper thread management"""
    frame_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, source: str, camera_name: str = "Camera"):
        super().__init__()
        self.source = source
        self.camera_name = camera_name
        self._running = True
        self._mutex = QMutex()
        self.cap = None

    def run(self):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;3000000'
        print(f"[{self.camera_name}] Connecting...")

        try:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 25)
                self.status_changed.emit("online")
                print(f"[{self.camera_name}] Connected!")

                while self._running:
                    self._mutex.lock()
                    is_running = self._running
                    self._mutex.unlock()

                    if not is_running:
                        break

                    ret, frame = self.cap.read()
                    if ret and self._running:
                        self.frame_ready.emit(frame.copy())
                    elif not ret:
                        print(f"[{self.camera_name}] Frame read failed")
                        self.status_changed.emit("error")
                        break

                    self.msleep(33)
            else:
                print(f"[{self.camera_name}] Failed to connect")
                self.status_changed.emit("error")

        except Exception as e:
            print(f"[{self.camera_name}] Error: {e}")
            self.status_changed.emit("error")
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None
            print(f"[{self.camera_name}] Thread finished")

    def stop(self):
        try:
            self._mutex.lock()
            self._running = False
            self._mutex.unlock()
        except Exception:
            self._running = False

        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except Exception:
            self.cap = None

        try:
            if self.isRunning():
                self.quit()
                if not self.wait(3000):
                    print(f"[{self.camera_name}] Force terminating thread")
                    self.terminate()
                    self.wait(1000)
        except (RuntimeError, Exception) as e:
            print(f"[{self.camera_name}] Stop error: {e}")


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
        title.setStyleSheet("color: #89b4fa; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        cameras = self.crossing_data.get("cameras", [])

        if not cameras:
            no_cam = QLabel("Kameralar topilmadi")
            no_cam.setStyleSheet("color: #6c7086; font-size: 13px; padding: 20px;")
            no_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(no_cam)
        else:
            for cam in cameras:
                cam_frame = QFrame()
                cam_frame.setStyleSheet("""
                    QFrame {
                        background-color: #1e1e3a;
                        border: 1px solid #2d2d50;
                        border-radius: 6px;
                    }
                """)
                cam_inner = QVBoxLayout(cam_frame)
                cam_inner.setContentsMargins(10, 8, 10, 8)
                cam_inner.setSpacing(6)

                header = QHBoxLayout()
                header.setSpacing(6)
                cam_type = cam.get("type", "additional")
                type_color = "#4a9eff" if cam_type == "main" else "#fab387"
                type_text = "Asosiy" if cam_type == "main" else "Qo'shimcha"

                name_lbl = QLabel(cam.get("name", "Kamera"))
                name_lbl.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: bold;")
                header.addWidget(name_lbl)

                r_v = 74 if cam_type == "main" else 250
                g_v = 158 if cam_type == "main" else 179
                b_v = 255 if cam_type == "main" else 135
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
                status_dot.setStyleSheet(f"color: {'#4ade80' if enabled else '#ef4444'}; font-size: 10px;")
                header.addWidget(status_dot)

                enabled_lbl = QLabel("Yoqilgan" if enabled else "O'chirilgan")
                enabled_lbl.setStyleSheet(f"color: {'#4ade80' if enabled else '#ef4444'}; font-size: 10px;")
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
                    src_lbl.setStyleSheet("color: #6c7086; font-size: 10px;")
                    src_lbl.setWordWrap(True)
                    cam_inner.addWidget(src_lbl)

                # Action buttons row
                actions = QHBoxLayout()
                actions.setSpacing(6)
                actions.addStretch()

                cam_id = cam.get("id")

                # Toggle enable/disable
                toggle_text = "O'chirish" if enabled else "Yoqish"
                toggle_color = "#f9e2af" if enabled else "#a6e3a1"
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
                del_btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent; color: #f38ba8;
                        border: 1px solid #f38ba8; border-radius: 3px;
                        padding: 3px 10px; font-size: 10px;
                    }
                    QPushButton:hover { background-color: rgba(243, 139, 168, 0.1); }
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
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 4px;
                padding: 6px 16px; font-size: 12px;
            }
            QPushButton:hover { background-color: #45475a; }
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

    def __init__(self, crossing_data: dict, config_manager=None, compact=False, parent=None):
        super().__init__(parent)
        self.crossing_data = crossing_data
        self.crossing_id = crossing_data.get("id", 0)
        self.config_manager = config_manager
        self.compact = compact
        self.camera_workers = []
        self.main_camera_label = None
        self.additional_camera_label = None
        self._is_destroyed = False

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._setup_ui()
        QTimer.singleShot(100, self._start_cameras)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)

        # Card frame
        self.frame = QFrame()
        self.frame.setObjectName("cardFrame")
        self.frame.setStyleSheet("""
            QFrame#cardFrame {
                background-color: #1a1a2e;
                border: 2px solid #2d2d44;
                border-radius: 8px;
            }
            QFrame#cardFrame:hover {
                border-color: #4a9eff;
            }
        """)
        self.frame.setCursor(Qt.CursorShape.PointingHandCursor)

        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)

        # ── HEADER ──
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #16162a;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                border-bottom: 1px solid #2d2d44;
            }
        """)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 6, 8, 6)
        header_layout.setSpacing(8)

        name_label = QLabel(self.crossing_data.get("name", "Pereezd"))
        name_label.setStyleSheet(
            "color: #ffffff; font-size: 14px; font-weight: bold; background: transparent; border: none;")
        header_layout.addWidget(name_label)
        header_layout.addStretch()

        self.menu_btn = QToolButton()
        self.menu_btn.setText("⋮")
        self.menu_btn.setStyleSheet("""
            QToolButton {
                color: #888; font-size: 18px; font-weight: bold;
                background: transparent; border: none;
                border-radius: 4px; padding: 2px 6px;
            }
            QToolButton:hover { background-color: #2d2d50; color: #cdd6f4; }
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
        self.main_camera_label.setStyleSheet("""
            background-color: #0d0d1a; border: 1px solid #333;
            border-top-left-radius: 4px; border-top-right-radius: 4px;
            border-bottom: none;
        """)
        self._set_placeholder(self.main_camera_label, "Yuklanmoqda...")

        # Main camera bottom bar
        main_bottom = QFrame()
        main_bottom.setStyleSheet("""
            QFrame {
                background-color: #13132a;
                border: 1px solid #333;
                border-top: none;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }
        """)
        main_bottom_layout = QHBoxLayout(main_bottom)
        main_bottom_layout.setContentsMargins(8, 3, 8, 3)
        main_bottom_layout.setSpacing(0)

        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet(
            "color: #4ade80; font-size: 8px; background: transparent; border: none;")
        main_bottom_layout.addWidget(self.status_indicator)

        main_name_lbl = QLabel(f"  {main_name}")
        main_name_lbl.setStyleSheet(
            "color: #a6adc8; font-size: 10px; background: transparent; border: none;")
        main_bottom_layout.addWidget(main_name_lbl)

        main_bottom_layout.addStretch()

        self.time_label = QLabel("00:00:00")
        self.time_label.setStyleSheet(
            "color: #ffffff; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        main_bottom_layout.addWidget(self.time_label)

        main_bottom_layout.addStretch()

        # Additional camera label
        self.additional_camera_label = QLabel()
        self.additional_camera_label.setMinimumSize(160, 60)
        self.additional_camera_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.additional_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.additional_camera_label.setStyleSheet("""
            background-color: #0d0d1a; border: 1px solid #333;
            border-top-left-radius: 4px; border-top-right-radius: 4px;
            border-bottom: none;
        """)
        self._set_placeholder(self.additional_camera_label, add_name)

        # Additional camera bottom bar
        add_bottom = QFrame()
        add_bottom.setStyleSheet("""
            QFrame {
                background-color: #13132a;
                border: 1px solid #333; border-top: none;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }
        """)
        add_bottom_layout = QHBoxLayout(add_bottom)
        add_bottom_layout.setContentsMargins(8, 3, 8, 3)
        add_name_lbl = QLabel(add_name)
        add_name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        add_name_lbl.setStyleSheet(
            "color: #a6adc8; font-size: 10px; background: transparent; border: none;")
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
        self.frame.mousePressEvent = self._on_click

    def _create_plc_panel(self):
        plc_data = self.crossing_data.get("plc", {})
        plc_enabled = plc_data.get("enabled", False)
        plc_color = "#4ade80" if plc_enabled else "#6c7086"
        plc_status_text = "ONLINE" if plc_enabled else "O'CHIRILGAN"

        plc_frame = QFrame()
        plc_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e3a, stop:1 #181830);
                border: 1px solid #2d2d50;
                border-radius: 5px;
            }
        """)
        plc_inner = QVBoxLayout(plc_frame)
        plc_inner.setContentsMargins(8, 5, 8, 5)
        plc_inner.setSpacing(2)

        plc_title = QLabel("PLC HOLATI")
        plc_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plc_title.setStyleSheet(
            "color: #585b70; font-size: 9px; font-weight: bold; letter-spacing: 1px;"
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
            "color: #585b70; font-size: 9px; background: transparent; border: none;")
        plc_inner.addWidget(self.plc_time_label)

        return plc_frame

    def _create_stats_panel(self):
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e3a, stop:1 #181830);
                border: 1px solid #2d2d50;
                border-radius: 5px;
            }
        """)
        stats_inner = QVBoxLayout(stats_frame)
        stats_inner.setContentsMargins(8, 5, 8, 5)
        stats_inner.setSpacing(3)

        stats_title = QLabel("STATISTIKA")
        stats_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stats_title.setStyleSheet(
            "color: #585b70; font-size: 9px; font-weight: bold; letter-spacing: 1px;"
            " background: transparent; border: none;")
        stats_inner.addWidget(stats_title)

        counts_row = QHBoxLayout()
        counts_row.setSpacing(4)

        car_badge = QFrame()
        car_badge.setStyleSheet("""
            QFrame {
                background-color: rgba(74, 158, 255, 0.08);
                border: 1px solid rgba(74, 158, 255, 0.2);
                border-radius: 4px;
            }
        """)
        car_lay = QHBoxLayout(car_badge)
        car_lay.setContentsMargins(6, 2, 6, 2)
        car_lay.setSpacing(3)
        car_ic = QLabel("🚗")
        car_ic.setStyleSheet("font-size: 35px; background: transparent; border: none;")
        car_lay.addWidget(car_ic)
        self.car_count = QLabel("16")
        self.car_count.setStyleSheet(
            "font-size: 25px; font-weight: bold; color: #4a9eff; background: transparent; border: none;")
        car_lay.addWidget(self.car_count)
        counts_row.addWidget(car_badge)

        truck_badge = QFrame()
        truck_badge.setStyleSheet("""
            QFrame {
                background-color: rgba(250, 179, 135, 0.08);
                border: 1px solid rgba(250, 179, 135, 0.2);
                border-radius: 4px;
            }
        """)
        truck_lay = QHBoxLayout(truck_badge)
        truck_lay.setContentsMargins(6, 2, 6, 2)
        truck_lay.setSpacing(3)
        truck_ic = QLabel("🚚")
        truck_ic.setStyleSheet("font-size: 35px; background: transparent; border: none;")
        truck_lay.addWidget(truck_ic)
        self.truck_count = QLabel("24")
        self.truck_count.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #fab387; background: transparent; border: none;")
        truck_lay.addWidget(self.truck_count)
        counts_row.addWidget(truck_badge)

        stats_inner.addLayout(counts_row)

        self.total_label = QLabel("Jami: 36")
        self.total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.total_label.setStyleSheet("""
            font-size: 20px; font-weight: bold; color: #a6e3a1;
            background-color: rgba(166, 227, 161, 0.06);
            border: 1px solid rgba(166, 227, 161, 0.15);
            border-radius: 4px; padding: 2px 0px;
        """)
        stats_inner.addWidget(self.total_label)

        return stats_frame

    def _show_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e1e3a; border: 1px solid #2d2d50;
                border-radius: 6px; padding: 4px;
            }
            QMenu::item {
                color: #cdd6f4; padding: 6px 20px; border-radius: 4px;
            }
            QMenu::item:selected { background-color: #313244; }
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

    def _display_frame(self, label: QLabel, frame):
        if frame is None or self._is_destroyed or label is None:
            return
        try:
            fh, fw = frame.shape[:2]
            if fh <= 0 or fw <= 0:
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, fw, fh, fw * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            lw, lh = label.width(), label.height()
            if lw <= 0 or lh <= 0:
                return
            scaled = pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(scaled)
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

                worker = CameraWorker(source, cam_name)

                if i == 0 or cam_type == "main":
                    worker.frame_ready.connect(self._on_main_frame)
                    worker.status_changed.connect(self._on_main_status)
                else:
                    worker.frame_ready.connect(self._on_additional_frame)
                    worker.status_changed.connect(self._on_additional_status)

                worker.start()
                self.camera_workers.append(worker)
        except (RuntimeError, Exception) as e:
            print(f"[StartCameras] Error: {e}")

    def _on_main_frame(self, frame):
        if self._is_destroyed or self.main_camera_label is None:
            return
        try:
            self._display_frame(self.main_camera_label, frame)
        except RuntimeError:
            self._is_destroyed = True

    def _on_additional_frame(self, frame):
        if self._is_destroyed or self.additional_camera_label is None:
            return
        try:
            self._display_frame(self.additional_camera_label, frame)
        except RuntimeError:
            self._is_destroyed = True

    def _on_main_status(self, status):
        if self._is_destroyed:
            return
        try:
            if status == "online":
                self.status_indicator.setStyleSheet(
                    "color: #4ade80; font-size: 10px; background: transparent; border: none;")
            elif status == "error":
                self.status_indicator.setStyleSheet(
                    "color: #ef4444; font-size: 10px; background: transparent; border: none;")
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

    def _on_click(self, event):
        self.clicked.emit(self.crossing_id)

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

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

    def deleteLater(self):
        self.cleanup()
        try:
            super().deleteLater()
        except RuntimeError:
            pass
