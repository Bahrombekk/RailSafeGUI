"""
Crossing Card Widget - Shows crossing with main + additional camera layout
Similar to the reference design with main camera and side panel
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QGridLayout, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QMutex, QWaitCondition
from PyQt6.QtGui import QFont, QPixmap, QImage
import cv2
import numpy as np
import os
import time


class CameraWorker(QThread):
    """Worker thread for camera streaming - proper thread management"""
    frame_ready = pyqtSignal(object)  # frame
    status_changed = pyqtSignal(str)  # status

    def __init__(self, source: str, camera_name: str = "Camera"):
        super().__init__()
        self.source = source
        self.camera_name = camera_name
        self._running = True
        self._mutex = QMutex()
        self.cap = None

    def run(self):
        """Main thread loop"""
        # Set RTSP timeout environment
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;3000000'

        print(f"[{self.camera_name}] Connecting...")

        try:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
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

                    self.msleep(33)  # ~30 FPS
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
        """Safely stop the thread"""
        self._mutex.lock()
        self._running = False
        self._mutex.unlock()

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.isRunning():
            self.quit()
            if not self.wait(3000):  # Wait up to 3 seconds
                print(f"[{self.camera_name}] Force terminating thread")
                self.terminate()
                self.wait(1000)


class CrossingCard(QWidget):
    """
    Crossing card with main camera (large) + additional camera (small) layout
    Design matches the reference image
    """

    clicked = pyqtSignal(int)  # crossing_id

    def __init__(self, crossing_data: dict, parent=None):
        super().__init__(parent)
        self.crossing_data = crossing_data
        self.crossing_id = crossing_data.get("id", 0)
        self.camera_workers = []
        self.main_camera_label = None
        self.additional_camera_label = None
        self._is_destroyed = False

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._setup_ui()
        # Start cameras immediately
        QTimer.singleShot(100, self._start_cameras)

    def _setup_ui(self):
        """Setup the user interface - matching reference design"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)

        # Main card frame
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
        frame_layout.setContentsMargins(8, 8, 8, 8)
        frame_layout.setSpacing(5)

        # Header with name and menu
        header_layout = QHBoxLayout()
        header_layout.setSpacing(5)

        name_label = QLabel(self.crossing_data.get("name", "Pereezd"))
        name_label.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold;")
        header_layout.addWidget(name_label)

        header_layout.addStretch()

        # Menu dots
        menu_label = QLabel("⋮")
        menu_label.setStyleSheet("color: #666; font-size: 16px;")
        header_layout.addWidget(menu_label)

        frame_layout.addLayout(header_layout)

        # Cameras section - main + additional side by side
        cameras_layout = QHBoxLayout()
        cameras_layout.setSpacing(5)

        cameras = self.crossing_data.get("cameras", [])
        main_camera = None
        additional_camera = None

        # Find main and additional cameras
        for cam in cameras:
            if cam.get("type") == "main" and main_camera is None:
                main_camera = cam
            elif cam.get("type") == "additional" and additional_camera is None:
                additional_camera = cam
            elif main_camera is None:
                main_camera = cam
            elif additional_camera is None:
                additional_camera = cam

        # Main camera (left side - larger)
        main_widget = self._create_main_camera_widget(main_camera)
        cameras_layout.addWidget(main_widget, stretch=3)

        # Additional camera panel (right side - smaller)
        additional_widget = self._create_additional_panel(additional_camera)
        cameras_layout.addWidget(additional_widget, stretch=2)

        frame_layout.addLayout(cameras_layout)

        # Bottom status bar
        status_layout = QHBoxLayout()
        status_layout.setSpacing(10)

        # Status indicator
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: #4ade80; font-size: 12px;")
        status_layout.addWidget(self.status_indicator)

        # Camera label
        cam_label = QLabel("Camera 1")
        cam_label.setStyleSheet("color: #888; font-size: 10px;")
        status_layout.addWidget(cam_label)

        status_layout.addStretch()

        frame_layout.addLayout(status_layout)

        main_layout.addWidget(self.frame)

        # Make clickable
        self.frame.mousePressEvent = self._on_click

    def _create_main_camera_widget(self, camera_data):
        """Create main camera widget (large)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        # Video frame - responsive, not fixed
        self.main_camera_label = QLabel()
        self.main_camera_label.setMinimumSize(180, 120)
        self.main_camera_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_camera_label.setStyleSheet("""
            background-color: #0d0d1a;
            border: 1px solid #333;
            border-radius: 4px;
        """)

        # Set placeholder
        self._set_placeholder(self.main_camera_label, "Yuklanmoqda...")

        layout.addWidget(self.main_camera_label)

        # Time display
        self.time_label = QLabel("00:00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: bold;")
        layout.addWidget(self.time_label)

        # Update time
        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self._update_time)
        self.time_timer.start(1000)

        # Store camera data for later
        if camera_data:
            widget.camera_data = camera_data

        return widget

    def _create_additional_panel(self, camera_data):
        """Create additional camera panel (right side)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Additional camera video - responsive
        self.additional_camera_label = QLabel()
        self.additional_camera_label.setMinimumSize(120, 80)
        self.additional_camera_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.additional_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.additional_camera_label.setStyleSheet("""
            background-color: #0d0d1a;
            border: 1px solid #333;
            border-radius: 4px;
        """)

        self._set_placeholder(self.additional_camera_label, "Kamera 2")

        layout.addWidget(self.additional_camera_label)

        # Info panel
        info_frame = QFrame()
        info_frame.setObjectName("infoPanel")
        info_frame.setStyleSheet("""
            QFrame#infoPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e3a, stop:1 #16162e);
                border: 1px solid #2d2d50;
                border-radius: 6px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 6, 8, 6)
        info_layout.setSpacing(4)

        # Vehicle counts row
        top_layout = QHBoxLayout()
        top_layout.setSpacing(6)

        # Car count card
        car_frame = QFrame()
        car_frame.setStyleSheet("""
            background-color: rgba(74, 158, 255, 0.1);
            border: 1px solid rgba(74, 158, 255, 0.2);
            border-radius: 4px;
        """)
        car_inner = QVBoxLayout(car_frame)
        car_inner.setContentsMargins(6, 4, 6, 4)
        car_inner.setSpacing(1)
        car_icon = QLabel("🚗")
        car_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        car_icon.setStyleSheet("font-size: 14px; border: none; background: transparent;")
        self.car_count = QLabel("12")
        self.car_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.car_count.setStyleSheet("font-size: 15px; font-weight: bold; color: #4a9eff; border: none; background: transparent;")
        car_inner.addWidget(car_icon)
        car_inner.addWidget(self.car_count)

        # Truck count card
        truck_frame = QFrame()
        truck_frame.setStyleSheet("""
            background-color: rgba(250, 179, 135, 0.1);
            border: 1px solid rgba(250, 179, 135, 0.2);
            border-radius: 4px;
        """)
        truck_inner = QVBoxLayout(truck_frame)
        truck_inner.setContentsMargins(6, 4, 6, 4)
        truck_inner.setSpacing(1)
        truck_icon = QLabel("🚚")
        truck_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        truck_icon.setStyleSheet("font-size: 14px; border: none; background: transparent;")
        self.truck_count = QLabel("24")
        self.truck_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.truck_count.setStyleSheet("font-size: 15px; font-weight: bold; color: #fab387; border: none; background: transparent;")
        truck_inner.addWidget(truck_icon)
        truck_inner.addWidget(self.truck_count)

        top_layout.addWidget(car_frame)
        top_layout.addWidget(truck_frame)

        # Total
        self.total_label = QLabel("Jami: 36")
        self.total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.total_label.setStyleSheet("""
            font-size: 11px; font-weight: bold; color: #a6e3a1;
            background-color: rgba(166, 227, 161, 0.08);
            border: 1px solid rgba(166, 227, 161, 0.15);
            border-radius: 4px;
            padding: 3px 0px;
        """)

        info_layout.addLayout(top_layout)
        info_layout.addWidget(self.total_label)


        layout.addWidget(info_frame)

        layout.addStretch()

        if camera_data:
            widget.camera_data = camera_data

        return widget

    def _set_placeholder(self, label: QLabel, text: str):
        """Set placeholder with text"""
        w = max(label.width(), label.minimumWidth(), 180)
        h = max(label.height(), label.minimumHeight(), 120)
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

    def _display_frame(self, label: QLabel, frame):
        """Display frame centered with KeepAspectRatio - no stretching"""
        if frame is None or self._is_destroyed or label is None:
            return
        try:
            fh, fw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, fw, fh, fw * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(scaled)
        except Exception:
            pass

    def _update_time(self):
        """Update time display"""
        if self._is_destroyed:
            return
        current_time = time.strftime("%H:%M:%S")
        if hasattr(self, 'time_label'):
            self.time_label.setText(current_time)

    def _start_cameras(self):
        """Start camera workers"""
        if self._is_destroyed:
            return

        cameras = self.crossing_data.get("cameras", [])

        for i, camera in enumerate(cameras[:2]):  # Max 2 cameras
            if not camera.get("enabled", False):
                continue

            source = camera.get("source", "")
            if not source:
                continue

            cam_type = camera.get("type", "main")
            cam_name = camera.get("name", f"Camera {i+1}")

            worker = CameraWorker(source, cam_name)

            # Connect signals based on camera type
            if i == 0 or cam_type == "main":
                worker.frame_ready.connect(self._on_main_frame)
                worker.status_changed.connect(self._on_main_status)
            else:
                worker.frame_ready.connect(self._on_additional_frame)
                worker.status_changed.connect(self._on_additional_status)

            worker.start()
            self.camera_workers.append(worker)

    def _on_main_frame(self, frame):
        """Handle main camera frame"""
        if self._is_destroyed or self.main_camera_label is None:
            return
        self._display_frame(self.main_camera_label, frame)

    def _on_additional_frame(self, frame):
        """Handle additional camera frame"""
        if self._is_destroyed or self.additional_camera_label is None:
            return
        self._display_frame(self.additional_camera_label, frame)

    def _on_main_status(self, status):
        """Handle main camera status"""
        if self._is_destroyed:
            return
        if status == "online":
            self.status_indicator.setStyleSheet("color: #4ade80; font-size: 12px;")
        elif status == "error":
            self.status_indicator.setStyleSheet("color: #ef4444; font-size: 12px;")
            if self.main_camera_label:
                self._set_placeholder(self.main_camera_label, "Ulanmadi")

    def _on_additional_status(self, status):
        """Handle additional camera status"""
        if self._is_destroyed:
            return
        if status == "error" and self.additional_camera_label:
            self._set_placeholder(self.additional_camera_label, "Ulanmadi")

    def _on_click(self, event):
        """Handle click"""
        self.clicked.emit(self.crossing_id)

    def stop_cameras(self):
        """Stop all camera workers safely"""
        for worker in self.camera_workers:
            worker.stop()
        self.camera_workers.clear()

    def cleanup(self):
        """Full cleanup"""
        self._is_destroyed = True

        # Stop timers
        if hasattr(self, 'time_timer'):
            self.time_timer.stop()

        # Stop cameras
        self.stop_cameras()

    def closeEvent(self, event):
        """Handle close"""
        self.cleanup()
        super().closeEvent(event)

    def deleteLater(self):
        """Override deleteLater to ensure cleanup"""
        self.cleanup()
        super().deleteLater()
