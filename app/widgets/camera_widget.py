"""
Camera Widget - Displays live camera feed with status and statistics.

LEGACY / ISHLATILMAYDI (unused):
    Bu widget dasturning hech bir joyida import qilinmaydi. Dashboard endi
    CrossingCard (app/widgets/crossing_card.py) dan foydalanadi, u kadrlarni
    alohida QThread (CameraWorker) da o'qiydi.

    Bu yerdagi CameraWidget esa RTSP'ni GUI thread'ida QTimer callback ichida
    `cap.read()` bilan o'qiydi va `open(timeout=10)` ni sinxron chaqiradi —
    ya'ni GUI'ni bloklaydi. Shu sabab u ishlatilmaydi. Yangi kod uchun
    CrossingCard/CameraWorker ishlatilsin. Fayl tarixiy sabablarga ko'ra
    saqlanmoqda; qayta yozilmagan.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont
import cv2
import numpy as np
from datetime import datetime
from app.core.camera import OptimizedCamera


class CameraWidget(QWidget):
    """Widget to display camera feed with status information"""

    clicked = pyqtSignal(int, int)  # crossing_id, camera_id

    def __init__(self, crossing_id: int, camera_data: dict, size: str = "medium", parent=None):
        super().__init__(parent)
        self.crossing_id = crossing_id
        self.camera_data = camera_data
        self.size = size  # "small", "medium", "large"
        self.is_running = False
        self.cap = None
        self.detection_count = 0
        self.violation_count = 0

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main frame (card style)
        self.frame = QFrame()
        self.frame.setObjectName("cardFrame")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setCursor(Qt.CursorShape.PointingHandCursor)
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(10, 10, 10, 10)
        frame_layout.setSpacing(10)

        # Header with camera name and status
        header_layout = QHBoxLayout()

        # Camera name
        self.name_label = QLabel(self.camera_data.get("name", "Camera"))
        self.name_label.setObjectName("subtitleLabel")
        font = QFont()
        font.setPointSize(12 if self.size == "large" else 10)
        font.setBold(True)
        self.name_label.setFont(font)
        header_layout.addWidget(self.name_label)

        header_layout.addStretch()

        # Status indicator
        self.status_label = QLabel("● OFFLINE")
        self.status_label.setObjectName("statusOffline")
        font = QFont()
        font.setPointSize(10 if self.size == "large" else 8)
        font.setBold(True)
        self.status_label.setFont(font)
        header_layout.addWidget(self.status_label)

        frame_layout.addLayout(header_layout)

        # Video display
        self.video_label = QLabel()
        self.video_label.setObjectName("cameraLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setScaledContents(False)

        # Set size based on configuration
        if self.size == "small":
            self.video_label.setFixedSize(320, 180)
        elif self.size == "medium":
            self.video_label.setFixedSize(480, 270)
        else:  # large
            self.video_label.setFixedSize(640, 360)

        # Default placeholder
        self._set_placeholder()

        frame_layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Statistics panel
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)

        # FPS
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setObjectName("statusLabel")
        stats_layout.addWidget(self.fps_label)

        # Detection count
        self.detection_label = QLabel("🚗 Detect: 0")
        stats_layout.addWidget(self.detection_label)

        # Violation count
        self.violation_label = QLabel("⚠️ Buzilish: 0")
        self.violation_label.setObjectName("statusWarning")
        stats_layout.addWidget(self.violation_label)

        stats_layout.addStretch()

        # Timestamp
        self.time_label = QLabel()
        self.time_label.setObjectName("statusLabel")
        stats_layout.addWidget(self.time_label)

        frame_layout.addLayout(stats_layout)

        layout.addWidget(self.frame)

        # Make the widget clickable
        self.frame.mousePressEvent = self._on_click

    def _setup_timer(self):
        """Setup timer for video updates"""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.update_time_timer = QTimer(self)
        self.update_time_timer.timeout.connect(self._update_time)
        self.update_time_timer.start(1000)  # Update time every second

    def _set_placeholder(self):
        """Set placeholder image when camera is offline"""
        width = self.video_label.width()
        height = self.video_label.height()

        # Create a dark placeholder with text
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        placeholder[:] = (24, 24, 37)  # Dark background

        # Add text
        text = "KAMERA O'CHIQ"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        cv2.putText(placeholder, text, (text_x, text_y), font, 1, (108, 112, 134), 2)

        # Convert to QPixmap
        self._display_frame(placeholder)

    def _display_frame(self, frame):
        """Display a frame on the video label"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale to fit while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def _update_frame(self):
        """Update video frame"""
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            # Resize frame for better performance
            target_width = self.video_label.width()
            target_height = self.video_label.height()
            frame = cv2.resize(frame, (target_width, target_height))

            self._display_frame(frame)
            self._update_status("online")
        else:
            # Frame read failed - try to reconnect
            print(f"[WARNING] Frame read failed for {self.camera_data['name']}, reconnecting...")
            self._update_status("error")
            # Note: For production, implement auto-reconnect logic here

    def _update_time(self):
        """Update timestamp display"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(current_time)

    def _update_status(self, status: str):
        """Update status indicator"""
        if status == "online":
            self.status_label.setText("● ONLINE")
            self.status_label.setObjectName("statusOK")
        elif status == "error":
            self.status_label.setText("● XATOLIK")
            self.status_label.setObjectName("statusError")
        elif status == "connecting":
            self.status_label.setText("● ULANMOQDA...")
            self.status_label.setObjectName("statusWarning")
        elif status == "warning":
            self.status_label.setText("● OGOHLANTIRISH")
            self.status_label.setObjectName("statusWarning")
        else:
            self.status_label.setText("● OFFLINE")
            self.status_label.setObjectName("statusOffline")

        # Reapply stylesheet
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _on_click(self, event):
        """Handle widget click"""
        self.clicked.emit(self.crossing_id, self.camera_data["id"])

    def start(self):
        """Start the camera feed"""
        source = self.camera_data.get("source", "")
        if not source:
            return

        self._update_status("connecting")  # Show connecting status
        print(f"[INFO] Starting camera: {self.camera_data['name']}")

        try:
            # Use OptimizedCamera with timeout
            optimized_cam = OptimizedCamera(source, self.camera_data['name'])

            # Try to open with 10 second timeout
            if optimized_cam.open(timeout=10.0):
                self.cap = optimized_cam.cap  # Get the underlying cv2.VideoCapture
                self.is_running = True
                self.timer.start(40)  # ~25 FPS (more stable for RTSP)
                self._update_status("online")
                print(f"[OK] Camera {self.camera_data['name']} started successfully")
            else:
                self._update_status("error")
                print(f"[ERROR] Failed to connect to camera: {self.camera_data['name']}")
                print(f"[INFO] Check camera URL: {source}")
        except Exception as e:
            print(f"[ERROR] Starting camera {self.camera_data['name']}: {e}")
            self._update_status("error")

    def stop(self):
        """Stop the camera feed"""
        self.is_running = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self._set_placeholder()
        self._update_status("offline")

    def update_stats(self, detections: int, violations: int, fps: float):
        """Update statistics display"""
        self.detection_count = detections
        self.violation_count = violations

        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.detection_label.setText(f"🚗 Detect: {detections}")
        self.violation_label.setText(f"⚠️ Buzilish: {violations}")

        # Update status based on violations
        if violations > 0:
            self._update_status("warning")
        elif self.is_running:
            self._update_status("online")

    def closeEvent(self, event):
        """Clean up on close"""
        self.stop()
        self.update_time_timer.stop()
        super().closeEvent(event)
