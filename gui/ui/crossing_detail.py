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

# RTSP low-latency + Intel VA-API hw decode (set once, not per-thread)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp|stimeout;5000000|'
    'fflags;nobuffer|flags;low_delay|'
    'analyzeduration;500000|probesize;500000|'
    'hwaccel;vaapi|hwaccel_device;/dev/dri/renderD128'
)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'


class DetailCameraWorker(QThread):
    """Worker thread - all heavy work here, GUI only does setPixmap, auto-reconnect"""
    frame_ready = pyqtSignal(QImage)
    status_changed = pyqtSignal(str)

    def __init__(self, source: str, camera_name: str = "Camera", display_width: int = 960):
        super().__init__()
        self.source = source
        self.camera_name = camera_name
        self.display_width = display_width
        self._running = True
        self._mutex = QMutex()
        self._frame_pending = False
        self._frame_mutex = QMutex()
        self._retry_delay = 3

    def set_frame_delivered(self):
        self._frame_mutex.lock()
        self._frame_pending = False
        self._frame_mutex.unlock()

    def run(self):
        retry_count = 0
        while self._is_running():
            cap = None
            try:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.status_changed.emit("online")
                    retry_count = 0
                    consecutive_fails = 0

                    while self._is_running():
                        # ALWAYS grab to flush RTSP buffer (real-time)
                        ret = cap.grab()
                        if not ret:
                            consecutive_fails += 1
                            if consecutive_fails > 30:
                                break
                            continue
                        consecutive_fails = 0

                        # Only decode when GUI is ready
                        self._frame_mutex.lock()
                        pending = self._frame_pending
                        self._frame_mutex.unlock()
                        if pending:
                            continue

                        ret, frame = cap.retrieve()
                        if ret and self._is_running():
                            # ALL heavy work in worker thread
                            h, w = frame.shape[:2]
                            if w > self.display_width:
                                scale = self.display_width / w
                                frame = cv2.resize(frame, (self.display_width, int(h * scale)),
                                                   interpolation=cv2.INTER_AREA)
                                h, w = frame.shape[:2]
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
        self._destroyed = False

        if not self.crossing_data:
            raise ValueError(f"Crossing {crossing_id} not found")

        self._setup_ui()
        QTimer.singleShot(300, self._start_all_cameras)

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
        header.setStyleSheet("""
            QFrame {
                background: #181825;
                border: 1px solid #313244;
                border-radius: 10px;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(10)

        # Back
        back_btn = QPushButton("< Orqaga")
        back_btn.clicked.connect(self.back_clicked.emit)
        back_btn.setStyleSheet("""
            QPushButton {
                background: #313244; color: #cdd6f4; border: none;
                border-radius: 6px; padding: 6px 14px; font-size: 12px;
            }
            QPushButton:hover { background: #45475a; }
        """)
        header_layout.addWidget(back_btn)

        # Separator
        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setFixedHeight(24)
        sep.setStyleSheet("background: #313244;")
        header_layout.addWidget(sep)

        # Title
        title = QLabel(self.crossing_data.get("name", "Pereezd"))
        title.setStyleSheet("color: #cdd6f4; font-size: 16px; font-weight: bold; background: transparent;")
        header_layout.addWidget(title)

        # Location badge
        loc = self.crossing_data.get('location', '')
        if loc:
            loc_lbl = QLabel(loc)
            loc_lbl.setStyleSheet("""
                color: #a6adc8; font-size: 11px; background: #1e1e2e;
                border: 1px solid #313244; border-radius: 4px; padding: 2px 8px;
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
        add_cam_btn.setStyleSheet(btn_css.format(bg="#313244", fg="#89b4fa", hover="#45475a"))
        add_cam_btn.clicked.connect(lambda: self.add_camera_clicked.emit(self.crossing_id))
        header_layout.addWidget(add_cam_btn)

        settings_btn = QPushButton("Sozlamalar")
        settings_btn.setStyleSheet(btn_css.format(bg="#313244", fg="#a6e3a1", hover="#45475a"))
        settings_btn.clicked.connect(lambda: self.edit_crossing_clicked.emit(self.crossing_id))
        header_layout.addWidget(settings_btn)

        delete_btn = QPushButton("O'chirish")
        delete_btn.setStyleSheet(btn_css.format(bg="#313244", fg="#f38ba8", hover="#f38ba8"))
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
            empty.setStyleSheet("""
                color: #6c7086; font-size: 14px; padding: 60px;
                background: #1e1e2e; border: 2px dashed #313244; border-radius: 12px;
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
        panel.setStyleSheet("""
            QFrame#camPanel {
                background: #1e1e2e;
                border: 2px solid #313244;
                border-radius: 12px;
            }
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
        name.setStyleSheet("color: #cdd6f4; font-size: 13px; font-weight: bold; background: transparent;")
        hdr.addWidget(name)

        # Type badge
        cam_type = cam_data.get("type", "additional")
        is_main = cam_type == "main"
        badge_color = "#89b4fa" if is_main else "#a6e3a1"
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
        status_dot.setStyleSheet("color: #fab387; font-size: 12px; background: transparent;")
        self.camera_status_labels[cam_id] = status_dot
        hdr.addWidget(status_dot)

        hdr.addStretch()

        # Settings gear
        gear = QPushButton("Settings")
        gear.setFixedHeight(22)
        gear.setStyleSheet("""
            QPushButton {
                background: #313244; color: #6c7086; border: none;
                border-radius: 4px; padding: 2px 8px; font-size: 10px;
            }
            QPushButton:hover { background: #45475a; color: #cdd6f4; }
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
        video.setStyleSheet("""
            background: #11111b;
            border: 1px solid #313244;
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
        time_lbl.setStyleSheet("color: #a6adc8; font-size: 10px; background: transparent;")
        bottom.addWidget(time_lbl)

        bottom.addStretch()

        plate_lbl = QLabel("Avtomobil: -- --- --")
        plate_lbl.setStyleSheet("color: #fab387; font-size: 10px; font-weight: bold; background: transparent;")
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

            worker = DetailCameraWorker(src, cam.get("name", f"Cam-{cam_id}"))
            worker.frame_ready.connect(
                lambda f, lbl=label, w=worker: self._on_frame(lbl, f, w)
            )
            worker.status_changed.connect(
                lambda s, cid=cam_id: self._on_camera_status(cid, s)
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
                    dot.setStyleSheet("color: #a6e3a1; font-size: 12px; background: transparent;")
                elif status == "reconnecting":
                    dot.setStyleSheet("color: #f9e2af; font-size: 12px; background: transparent;")
                    if label:
                        self._set_placeholder(label, "Qayta ulanmoqda...", 480, 270)
                elif status == "error":
                    dot.setStyleSheet("color: #f38ba8; font-size: 12px; background: transparent;")
                    if label:
                        self._set_placeholder(label, "Ulanmadi", 480, 270)
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
        panel.setStyleSheet("""
            QFrame#statsPanel {
                background: #1e1e2e;
                border: 2px solid #313244;
                border-radius: 12px;
            }
        """)
        panel.setObjectName("statsPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        title = QLabel("Statistika")
        title.setStyleSheet("color: #cdd6f4; font-size: 14px; font-weight: bold; background: transparent;")
        layout.addWidget(title)

        # Divider
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet("background: #313244;")
        layout.addWidget(div)

        cameras_count = len(self.crossing_data.get("cameras", []))
        active = sum(1 for c in self.crossing_data.get("cameras", []) if c.get("enabled"))

        stats = [
            ("Kameralar", f"{active}/{cameras_count}", "#89b4fa"),
            ("Jami Transport", "0", "#a6e3a1"),
            ("Buzilishlar", "0", "#f38ba8"),
            ("O'rtacha Vaqt", "0s", "#f9e2af"),
        ]

        for name, value, color in stats:
            row = QHBoxLayout()
            n = QLabel(name)
            n.setStyleSheet("color: #6c7086; font-size: 12px; background: transparent;")
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
        panel.setStyleSheet("""
            QFrame#plcPanel {
                background: #1e1e2e;
                border: 2px solid #313244;
                border-radius: 12px;
            }
        """)
        panel.setObjectName("plcPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        title = QLabel("PLC Holati")
        title.setStyleSheet("color: #cdd6f4; font-size: 14px; font-weight: bold; background: transparent;")
        layout.addWidget(title)

        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet("background: #313244;")
        layout.addWidget(div)

        plc = self.crossing_data.get("plc", {})

        if plc.get("enabled", False):
            row1 = QHBoxLayout()
            s = QLabel("Holat")
            s.setStyleSheet("color: #6c7086; font-size: 12px; background: transparent;")
            row1.addWidget(s)
            row1.addStretch()
            sv = QLabel("ULANGAN")
            sv.setStyleSheet("color: #a6e3a1; font-size: 12px; font-weight: bold; background: transparent;")
            row1.addWidget(sv)
            layout.addLayout(row1)

            row2 = QHBoxLayout()
            ip_n = QLabel("IP")
            ip_n.setStyleSheet("color: #6c7086; font-size: 12px; background: transparent;")
            row2.addWidget(ip_n)
            row2.addStretch()
            ip_v = QLabel(plc.get("ip", "N/A"))
            ip_v.setStyleSheet("color: #cdd6f4; font-size: 12px; background: transparent;")
            row2.addWidget(ip_v)
            layout.addLayout(row2)

            row3 = QHBoxLayout()
            p_n = QLabel("Port")
            p_n.setStyleSheet("color: #6c7086; font-size: 12px; background: transparent;")
            row3.addWidget(p_n)
            row3.addStretch()
            p_v = QLabel(str(plc.get("port", 102)))
            p_v.setStyleSheet("color: #cdd6f4; font-size: 12px; background: transparent;")
            row3.addWidget(p_v)
            layout.addLayout(row3)
        else:
            row = QHBoxLayout()
            s = QLabel("Holat")
            s.setStyleSheet("color: #6c7086; font-size: 12px; background: transparent;")
            row.addWidget(s)
            row.addStretch()
            sv = QLabel("O'CHIRILGAN")
            sv.setStyleSheet("color: #6c7086; font-size: 12px; font-weight: bold; background: transparent;")
            row.addWidget(sv)
            layout.addLayout(row)

        layout.addStretch()
        return panel

    def _create_events_table(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame#eventsPanel {
                background: #1e1e2e;
                border: 2px solid #313244;
                border-radius: 12px;
            }
        """)
        panel.setObjectName("eventsPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        title = QLabel("So'nggi Hodisalar")
        title.setStyleSheet("color: #cdd6f4; font-size: 14px; font-weight: bold; background: transparent;")
        layout.addWidget(title)

        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet("background: #313244;")
        layout.addWidget(div)

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Vaqt", "Kamera", "Hodisa", "Transport"])
        table.setStyleSheet("""
            QTableWidget {
                background: #181825; border: 1px solid #313244;
                color: #cdd6f4; gridline-color: #313244; border-radius: 6px;
            }
            QHeaderView::section {
                background: #1e1e2e; color: #a6adc8;
                border: 1px solid #313244; padding: 6px;
                font-size: 11px; font-weight: bold;
            }
            QTableWidget::item:alternate { background: #1e1e2e; }
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
            except Exception:
                pass
            try:
                w.stop()
            except (RuntimeError, Exception):
                pass
        self.camera_workers.clear()

    def refresh(self):
        self.cleanup()
        self._destroyed = False
        self.crossing_data = self.config_manager.get_crossing(self.crossing_id)
        self.camera_labels.clear()
        self.camera_status_labels.clear()
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
