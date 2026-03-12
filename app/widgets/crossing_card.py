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
import json
import threading
from datetime import date
from app.utils.theme_colors import C
from app.core.tracker import PolygonTracker
from app.utils.language import t, LM
from app.core.plc import PLCManager as _PLCManager, SNAP7_AVAILABLE as _SNAP7_OK

# Car detector import (optional)
try:
    from detectors import RealtimeMultiCameraDetector
    CAR_DETECTOR_AVAILABLE = True
except ImportError:
    CAR_DETECTOR_AVAILABLE = False
    print("[CrossingCard] RealtimeMultiCameraDetector not available")

# RTSP ultra-low-latency (FFmpeg fallback uchun)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;udp|stimeout;2000000|'
    'fflags;nobuffer+discardcorrupt|flags;low_delay|'
    'analyzeduration;0|probesize;32|'
    'max_delay;0|reorder_queue_size;0|'
    'thread_queue_size;1'
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

        # 3) GStreamer NVDEC (GPU H.264 decode)
        gst_nvdec264 = (
            f"rtspsrc location={source} latency=0 protocols=tcp ! "
            f"rtph264depay ! h264parse ! nvh264dec ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst_nvdec264, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[{camera_name}] GStreamer NVDEC (GPU H.264)")
            return cap, "gst-nvdec264"

        # 4) GStreamer CPU (software H.264 decode)
        gst_cpu264 = (
            f"rtspsrc location={source} latency=0 protocols=tcp ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst_cpu264, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[{camera_name}] GStreamer CPU (H.264)")
            return cap, "gst-cpu264"

    # 5) FFmpeg fallback (har doim ishlaydi)
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


class CameraWorker(QThread):
    """Worker thread - all heavy work here, GUI thread only does setPixmap"""
    frame_ready = pyqtSignal()  # Lightweight - payload yo'q (queue backup bo'lmaydi)
    frame_with_data = pyqtSignal(QImage)  # Shared consumers uchun (detail view)
    status_changed = pyqtSignal(str)
    stats_updated = pyqtSignal(int, int, int, float)  # light_count, heavy_count, in_poly_count, max_poly_time

    def __init__(self, source: str, camera_name: str = "Camera", display_width: int = 1280,
                 car_detector=None, detection_enabled: bool = True,
                 polygon_file: str = None, warning_threshold: float = 10.0,
                 violation_threshold: float = 15.0, is_custom_model: bool = False):
        super().__init__()
        self.source = source
        self.camera_name = camera_name
        self.display_width = display_width
        self._running = True
        self._mutex = QMutex()
        self._latest_qimg = None  # Atomic latest frame (GIL-safe)
        # Car detector - non-blocking real-time mode
        self.car_detector = car_detector
        self.detection_enabled = detection_enabled and car_detector is not None
        # Polygon
        self.polygon_file = polygon_file
        self.warning_threshold = warning_threshold
        self.violation_threshold = violation_threshold
        self.is_custom_model = is_custom_model
        self._poly_pts = None
        self._poly_mask = None

    def take_frame(self):
        """Eng oxirgi kadrni olish - eski framelar avtomatik tashlanadi"""
        qimg = self._latest_qimg
        self._latest_qimg = None
        return qimg

    def run(self):
        cap = None
        gt = None
        _pyav_container = None
        _grab_running = [True]
        _latest_frame = [None]
        _frame_lock = threading.Lock()
        _grab_error = [False]
        _want_frame = [False]

        try:
            # 1. PyAV backend (low-latency: nobuffer, direct demux)
            gt, _pyav_container = self._start_pyav(
                _grab_running, _latest_frame, _frame_lock, _grab_error)

            # 2. OpenCV fallback
            if gt is None:
                cap, backend = _open_camera(self.source, self.camera_name)
                if cap is None:
                    self.status_changed.emit("error")
                    return

                # RTSP buffer tozalash — vaqt bo'yicha (300ms), tez flush
                _flush_end = time.perf_counter() + 0.3
                while time.perf_counter() < _flush_end:
                    cap.grab()

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
                        # Faqat so'ralganda decode — har doim eng yangi packet
                        if _want_frame[0]:
                            _want_frame[0] = False
                            ret2, frm = cap.retrieve()
                            if ret2:
                                with _frame_lock:
                                    _latest_frame[0] = frm

                gt = threading.Thread(target=_grab_loop, daemon=True)
                gt.start()

            self.status_changed.emit("online")

            # --- Main loop: 30 FPS cap (PyAV va OpenCV uchun bir xil) ---
            _poly_loaded = False
            _tracker = None
            _target_interval = 1.0 / 30.0
            _last_frame_t = 0.0
            while self._is_running() and not _grab_error[0]:
                _now_t = time.perf_counter()
                if _now_t - _last_frame_t >= _target_interval:
                    _want_frame[0] = True  # OpenCV uchun; PyAV e'tibor bermaydi

                with _frame_lock:
                    frame = _latest_frame[0]
                    _latest_frame[0] = None

                if frame is None:
                    time.sleep(0.002)
                    continue
                _last_frame_t = time.perf_counter()

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
                    # Tracker yaratish (polygon mavjud bo'lsa)
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
                            frame, camera_id=self.camera_name)
                        detection_count = len(detections)
                        # Tracking (har doim — eski tracklar expire bo'lishi uchun)
                        in_poly_bboxes = None
                        if _tracker is not None:
                            _tracker.process_detections(detections)
                            in_poly_count = _tracker.get_inside_count()
                            max_time = _tracker.get_max_time()
                            if detections:
                                in_poly_bboxes = _tracker.get_in_polygon_bboxes()
                            self.stats_updated.emit(
                                _tracker.light_count,
                                _tracker.heavy_count,
                                in_poly_count,
                                max_time
                            )
                        elif detections:
                            self.stats_updated.emit(0, 0, detection_count, 0.0)
                        if detections:
                            # det_frame = aniqlashan kadr (100% mos)
                            draw_on = det_frame if det_frame is not None else frame
                            frame = self.car_detector.draw_detections(
                                draw_on, detections,
                                thickness=2, font_scale=0.5,
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
                self.frame_with_data.emit(qimg)

            if _grab_error[0]:
                self.status_changed.emit("error")

        except Exception as e:
            print(f"[{self.camera_name}] Error: {e}")
            self.status_changed.emit("error")
        finally:
            _grab_running[0] = False
            # PyAV: interrupt_callback orqali blocking demux to'xtatiladi,
            # container grab threadda yopiladi (heap corruption oldini olish)
            if _pyav_container is not None:
                try:
                    _pyav_container.interrupt_callback = lambda: True
                except Exception:
                    pass
            if gt is not None:
                gt.join(timeout=2.0)
            # cap (OpenCV fallback)
            if cap is not None:
                cap.release()

    def _start_pyav(self, _grab_running, _latest_frame, _frame_lock, _grab_error):
        """PyAV low-latency backend. Returns (grab_thread, container) or (None, None).

        PyAV ustunliklari OpenCV/FFmpeg ga nisbatan:
        - To'g'ridan-to'g'ri demux → minimal buferlanish
        - fflags=nobuffer + flags=low_delay → iVMS ga yaqin kechikish
        - UDP transport → TCP ga qaraganda 50-100ms kam
        """
        try:
            import av as _av
        except ImportError:
            print(f"[{self.camera_name}] PyAV o'rnatilmagan, OpenCV ishlatiladi")
            return None, None

        base_opts = {
            'fflags': 'nobuffer+discardcorrupt',  # discardcorrupt: UDP packet loss → tashlanadi, detection ga o'tmaydi
            'flags': 'low_delay',
            'max_delay': '0',
            'reorder_queue_size': '0',
            'analyzeduration': '0',
            'probesize': '32',
            'stimeout': '2000000',
        }
        is_rtsp = self.source.lower().startswith('rtsp://')
        # TCP: ishonchli, H.265 error concealment yo'q → buzilish yo'q
        if is_rtsp:
            variants = [
                {**base_opts, 'rtsp_transport': 'tcp'},
                {**base_opts, 'rtsp_transport': 'udp'},
            ]
        else:
            variants = [base_opts]

        for opts in variants:
            container = None
            try:
                container = _av.open(self.source, options=opts, timeout=8)
                stream = container.streams.video[0]
                # NONE: frame-threaded decode o'chiriladi → ichki bufer minimal → kam lag
                stream.thread_type = 'NONE'
                stream.codec_context.thread_count = 1
                try:
                    stream.codec_context.flags |= 0x4000  # AV_CODEC_FLAG_LOW_DELAY
                except Exception:
                    pass

                transport = opts.get('rtsp_transport', 'local')
                print(f"[{self.camera_name}] PyAV ({transport}) — low-latency mode")

                # TCP ring buffer drain: kamera eski kadrlarini tezda yuboradi,
                # ularni decode qilmasdan o'tkazib ketamiz → live pozitsiyaga tushish
                if opts.get('rtsp_transport') == 'tcp':
                    _drain_t = time.perf_counter()
                    _drained = 0
                    try:
                        for _pkt in container.demux(stream):
                            if _pkt.dts is None:
                                continue
                            _drained += 1
                            _elapsed = time.perf_counter() - _drain_t
                            # Kamera ring bufferini tez yuboradi (>>real-time);
                            # sekinlashsa (>30ms/paket) = live pozitsiyaga yetdik
                            if _drained > 1 and _elapsed / _drained > 0.030:
                                break
                            if _elapsed > 3.0:  # Eng ko'p 3 sekund kutamiz
                                break
                    except Exception:
                        pass
                    print(f"[{self.camera_name}] TCP drain: {_drained} pkt, {time.perf_counter()-_drain_t:.2f}s")

                def _grab(cont=container, st=stream):
                    try:
                        for packet in cont.demux(st):
                            if not _grab_running[0]:
                                break
                            if packet.dts is None:
                                continue
                            try:
                                for frm in packet.decode():
                                    img = frm.to_ndarray(format='bgr24')
                                    with _frame_lock:
                                        _latest_frame[0] = img
                            except Exception:
                                pass
                    except Exception as e:
                        if _grab_running[0]:
                            print(f"[{self.camera_name}] PyAV grab error: {e}")
                            _grab_error[0] = True
                    finally:
                        # Container shu threadda yopiladi (thread-safe)
                        try:
                            cont.close()
                        except Exception:
                            pass

                gt = threading.Thread(target=_grab, daemon=True,
                                      name=f"pyav-{self.camera_name}")
                gt.start()
                return gt, container

            except Exception as e:
                transport = opts.get('rtsp_transport', '')
                print(f"[{self.camera_name}] PyAV ({transport}) failed: {e}")
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        pass

        return None, None

    def _is_running(self):
        try:
            self._mutex.lock()
            r = self._running
            self._mutex.unlock()
            return r
        except Exception:
            return False

    def stop(self):
        # Signal thread to stop — NON-BLOCKING (GUI thread bloklanmaydi)
        # Resources finally block da tozalanadi (PyAV container.close, cap.release)
        try:
            self._mutex.tryLock(50)
            self._running = False
            try:
                self._mutex.unlock()
            except Exception:
                pass
        except Exception:
            self._running = False
        try:
            if self.isRunning():
                self.quit()
                # wait() CHAQIRILMAYDI — GUI freeze oldini olish uchun
                # QThread o'z-o'zicha to'xtaydi (finally block)
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
        self.setWindowTitle(t("cam_dlg.title", name=crossing_data.get('name', 'Pereezd')))
        self.setMinimumWidth(520)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel(f"📷 {self.crossing_data.get('name', 'Pereezd')} {t('cam_dlg.subtitle')}")
        title.setStyleSheet(f"color: {C('accent_brand')}; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        cameras = self.crossing_data.get("cameras", [])

        if not cameras:
            no_cam = QLabel(t("cam_dlg.no_cameras"))
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
                type_text = t("crossing.type.main") if cam_type == "main" else t("crossing.type.additional")

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

                enabled_lbl = QLabel(t("cam_dlg.enabled") if enabled else t("cam_dlg.disabled"))
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
                toggle_text = t("cam_dlg.toggle_off") if enabled else t("cam_dlg.toggle_on")
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
                del_btn = QPushButton(f"{t('cam_dlg.delete')} 🗑")
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
        close_btn = QPushButton(t("cam_dlg.close"))
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
        msg = t("cam_dlg.toggle_msg_off") if currently_enabled else t("cam_dlg.toggle_msg_on")
        QMessageBox.information(self, t("cam_dlg.toggle_title"), msg)

    def _delete_camera(self, camera_id, camera_name):
        if not self.config_manager or not camera_id:
            return
        reply = QMessageBox.question(
            self, t("confirm.title"),
            t("confirm.delete_camera", name=camera_name),
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
    Compact (4+ crossings) - horizontal:
      ┌──────────────┬──────────┬─────────┐
      │   Kamera 1   │ Kamera 2 │  PLC    │
      │   (katta)    │          ├─────────┤
      │              │          │  Stats  │
      └──────────────┴──────────┴─────────┘
    Default (1-3 crossings) - vertical:
      ┌────────────────────────────────┐
      │       Kamera 1 (keng)          │
      ├────────────────┬───────────────┤
      │  Kamera 2      │  PLC + Stats  │
      └────────────────┴───────────────┘
    """

    clicked = pyqtSignal(int)

    def __init__(self, crossing_data: dict, config_manager=None, compact=False,
                 car_detector=None, stats_db=None, is_custom_model=False, parent=None):
        super().__init__(parent)
        self.crossing_data = crossing_data
        self.crossing_id = crossing_data.get("id", 0)
        self.config_manager = config_manager
        self.compact = compact
        self.stats_db = stats_db
        self.is_custom_model = is_custom_model
        self.camera_workers = []
        self.camera_workers_by_id = {}  # cam_id -> worker (detail view uchun)
        self.main_camera_label = None
        self.additional_camera_label = None
        self._is_destroyed = False
        self._main_camera_down = False  # Asosiy kamera uzilsa True
        self._last_paused_ids: set = set()  # Oldingi paused holat (o'zgarishni kuzatish)
        self._reconnect_pending = False  # Debounce: bir marta restart

        # PLC Manager (har bir pereezd uchun alohida)
        self._plc_manager = None
        self._plc_active = False   # Hozirgi PLC holati (poyezd kelmoqdami?)
        self._plc_enabled = False  # Config da enabled?

        # Train tracking
        self._plc_first_poll = True       # Birinchi poll — faqat init, hisob yo'q
        self._train_count_today = 0       # Bugungi poyezdlar soni
        self._train_start_time = 0.0      # monotonic time, PLC active bo'lganda
        self._train_start_dt = None       # datetime — DB ga yozish uchun
        self._last_train_duration = 0.0   # Oxirgi poyezd davomiyligi (sekund)
        # Minimal haqiqiy poyezd davomiyligi (sek) — qisqa yolg'on signallarni o'tkazib yuborish
        self._min_train_duration = 60.0
        # Grace period: PLC "yo'q" desa shu qancha sek kutiladi (yolg'on yo'q signal uchun)
        self._plc_in_grace = False        # Grace period ichidamizmi?
        self._plc_grace_secs = 10         # 10 soniya kutish

        # Car detector - SHARED instance (GPU maksimal ishlatish)
        self.car_detector = car_detector
        self._owns_detector = False  # Biz yaratmadik, biz to'xtatmaymiz

        # Midnight reset uchun
        self._current_date = date.today()
        self._tracker_base_light = 0
        self._tracker_base_heavy = 0
        self._last_tracker_light = 0
        self._last_tracker_heavy = 0

        self._last_display_light = 0
        self._last_display_heavy = 0

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._setup_ui()
        self._load_startup_counts()

        # Midnight check timer (har 30 sekundda)
        self._midnight_timer = QTimer(self)
        self._midnight_timer.timeout.connect(self._check_midnight)
        self._midnight_timer.start(30000)

        # Camera state o'zgarishini kuzatish (har 3 sekundda)
        self._state_timer = QTimer(self)
        self._state_timer.timeout.connect(self._check_camera_state_change)
        self._state_timer.start(3000)

        # PLC Manager — har bir pereezd uchun alohida
        self._plc_timer = QTimer(self)
        self._plc_timer.timeout.connect(self._poll_plc_state)
        self._train_timer = QTimer(self)
        self._train_timer.setInterval(1000)
        self._train_timer.timeout.connect(self._update_train_elapsed)
        # Grace timer: PLC "yo'q" signali kelganda to'xtatmasdan kutish
        self._grace_timer = QTimer(self)
        self._grace_timer.setSingleShot(True)
        self._grace_timer.timeout.connect(self._on_plc_grace_expired)
        plc_cfg = crossing_data.get("plc", {})
        self._plc_enabled = plc_cfg.get("enabled", False)
        plc_ip = plc_cfg.get("ip", "").strip()
        if self._plc_enabled and plc_ip and _SNAP7_OK:
            plc_port = int(plc_cfg.get("port", 102))
            self._plc_manager = _PLCManager(plc_ip, plc_port)
            self._plc_manager.start()
            self._plc_timer.start(500)  # 500ms da bir poll
        elif self._plc_enabled and not _SNAP7_OK:
            print(f"[CrossingCard] PLC yoqilgan lekin snap7 o'rnatilmagan: {self.crossing_data.get('name')}")

        # DB dan bugungi poyezd sonini yuklash
        if self.stats_db and self._plc_enabled:
            try:
                self._train_count_today = self.stats_db.get_train_today_count(self.crossing_id)
            except Exception:
                pass

        LM.language_changed.connect(self._retranslate)
        QTimer.singleShot(100, self._start_cameras)

    def _check_camera_state_change(self):
        """Camera_state.json o'zgarganini tekshirish — o'zgarsa workerllarni qayta ishga tushirish."""
        try:
            if self._is_destroyed or not self.config_manager:
                return
            current = self.config_manager.get_paused_cameras(self.crossing_id)
            if current != self._last_paused_ids:
                self._last_paused_ids = current
                self._restart_camera_workers()
        except Exception:
            pass

    def _restart_camera_workers(self):
        """Barcha camera workerlarni to'xtatib, yangi holat bilan qayta ishga tushirish."""
        for w in self.camera_workers:
            try:
                w.stop()
            except Exception:
                pass
        self.camera_workers.clear()
        self.camera_workers_by_id.clear()
        self._main_camera_down = False
        QTimer.singleShot(300, self._start_cameras)

    def get_shared_workers(self) -> dict:
        """Detail view uchun: cam_id → worker lug'atini qaytaradi (workerlar to'xtatilmaydi)."""
        return dict(self.camera_workers_by_id)

    def _poll_plc_state(self):
        """PLC holatini 500ms da bir tekshirish (GUI thread da xavfsiz).

        Grace period: PLC "yo'q" desa darhol to'xtatmaydi — _plc_grace_secs soniya kutadi.
        Agar shu vaqt ichida yana "bor" kelsa, timer uzluksiz davom etadi.
        """
        try:
            if self._is_destroyed or self._plc_manager is None:
                return
            active = self._plc_manager.get_plc_active()

            # Birinchi poll — faqat holatni boshlash, hisob yo'q
            if self._plc_first_poll:
                self._plc_first_poll = False
                self._plc_active = active
                self._update_plc_ui(active)
                if active:
                    self._train_start_time = time.monotonic()
                    self._train_start_dt = None
                    self._train_timer.start()
                self._update_train_count_display()
                return

            if active:
                # PLC aktiv signal keldi
                if self._plc_in_grace:
                    # Grace period ichida yana "bor" — yolg'on "yo'q" edi, davom etamiz
                    self._plc_in_grace = False
                    self._grace_timer.stop()
                    self._plc_active = True
                    self._update_plc_ui(True)
                    # Timer to'xtatilmagan edi, davom etaveradi
                elif not self._plc_active:
                    # Yangi poyezd — timer boshlash
                    self._plc_active = True
                    self._update_plc_ui(True)
                    self._train_start_time = time.monotonic()
                    from datetime import datetime as _dt
                    self._train_start_dt = _dt.now()
                    self._train_timer.start()
            else:
                # PLC "yo'q" signali
                if self._plc_active and not self._plc_in_grace:
                    # Birinchi "yo'q" — grace period boshlash
                    self._plc_in_grace = True
                    self._plc_active = False
                    self._update_plc_ui(False)
                    self._grace_timer.start(self._plc_grace_secs * 1000)
                    # Timer va display — o'zgartirilmaydi (davom etadi)
        except Exception:
            pass  # PyQt6 abort() oldini olish

    def _on_plc_grace_expired(self):
        """Grace period tugadi — PLC haqiqatan ham o'chdi."""
        try:
            if self._is_destroyed:
                return
            self._plc_in_grace = False
            self._train_timer.stop()

            duration = 0.0
            if self._train_start_time > 0:
                duration = time.monotonic() - self._train_start_time
                self._last_train_duration = duration
                self._train_start_time = 0.0
            self._update_train_duration_display(False)

            if duration >= self._min_train_duration:
                self._train_count_today += 1
                self._update_train_count_display()
                if self.stats_db and self._train_start_dt is not None:
                    from datetime import datetime as _dt
                    try:
                        self.stats_db.record_train_event(
                            self.crossing_id,
                            self._train_start_dt,
                            _dt.now()
                        )
                    except Exception as e:
                        print(f"[CrossingCard] Train event yozish xato: {e}")
            else:
                if duration > 0:
                    print(f"[CrossingCard] Qisqa signal ({duration:.1f}s < "
                          f"{self._min_train_duration}s) — e'tiborsiz")
                self._last_train_duration = 0.0
                self._update_train_duration_display(False)

            self._train_start_dt = None
        except Exception:
            pass

    def _update_plc_ui(self, active: bool):
        """PLC holati o'zgarganda UI ni yangilash."""
        if self._is_destroyed:
            return
        try:
            if active:
                # Poyezd kelmoqda — QIZIL
                color = C('status_error')
                text = t("plc.active")
            else:
                # Normal — YASHIL
                color = C('status_online')
                text = t("plc.connected")
            self.plc_indicator.setStyleSheet(
                f"color: {color}; font-size: 10px; background: transparent; border: none;")
            self.plc_status_label.setText(text)
            self.plc_status_label.setStyleSheet(
                f"color: {color}; font-size: 12px; font-weight: bold;"
                " background: transparent; border: none;")
        except RuntimeError:
            self._is_destroyed = True

    @staticmethod
    def _fmt_duration(secs: float) -> str:
        """Sekundni MM:SS formatga aylantirish."""
        s = int(secs)
        m = s // 60
        s = s % 60
        return f"{m}:{s:02d}"

    def _update_train_count_display(self):
        """Poyezd sonini yangilash."""
        try:
            if self._is_destroyed:
                return
            self.train_count_label.setText(f"🚂  {self._train_count_today}")
        except Exception:
            pass

    def _update_train_elapsed(self):
        """Har sekundda poyezd o'tish vaqtini yangilash (timer callback)."""
        try:
            if self._is_destroyed or not self._plc_active or self._train_start_time == 0:
                self._train_timer.stop()
                return
            elapsed = time.monotonic() - self._train_start_time
            self._update_train_duration_display(True, elapsed)
        except Exception:
            pass

    def _update_train_duration_display(self, in_progress: bool, elapsed: float = 0.0):
        """Poyezd vaqt labelini yangilash."""
        try:
            if self._is_destroyed:
                return
            if in_progress and elapsed > 0:
                text = f"⏱  {self._fmt_duration(elapsed)}"
                color = C('accent_red')
            elif not in_progress and self._last_train_duration > 0:
                text = f"⏱  {self._fmt_duration(self._last_train_duration)}"
                color = C('text_muted')
            else:
                text = ""
                color = C('text_muted')
            self.train_duration_label.setText(text)
            self.train_duration_label.setStyleSheet(
                f"color: {color}; font-size: 10px;"
                " background: transparent; border: none;")
        except Exception:
            pass

    def _check_midnight(self):
        """Yarim tunda hisoblagichlarni nolga qaytarish"""
        try:
            today = date.today()
            if today != self._current_date:
                self._current_date = today
                # Tracker bazasini yangilash (hozirgi kumulyativ qiymatni ayirish uchun)
                self._tracker_base_light = self._last_tracker_light
                self._tracker_base_heavy = self._last_tracker_heavy
                # DB offset 0 (yangi kun — bazada hali hech narsa yo'q)
                self._light_offset = 0
                self._heavy_offset = 0
                # Displayni darhol yangilash
                self.update_stats(0, 0)
        except Exception:
            pass

    def _load_startup_counts(self):
        """DB dan bugungi sanashni yuklash (dastur qayta ishga tushganda davom etadi)"""
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
                self.update_stats(light, heavy)
        except Exception:
            pass

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
        self._set_placeholder(self.main_camera_label, t("cam.loading"))

        # Bottom bar: status + kamera nomi + bandlik + polygon + vaqt
        main_bottom = QFrame()
        main_bottom.setStyleSheet(f"""
            QFrame {{
                background-color: {C('bg_camera_bar')};
                border: 1px solid {C('border_light')};
                border-top: none;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
            }}
        """)
        main_bottom_layout = QHBoxLayout(main_bottom)
        main_bottom_layout.setContentsMargins(8, 3, 8, 3)
        main_bottom_layout.setSpacing(6)

        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet(
            f"color: {C('status_online')}; font-size: 8px; background: transparent; border: none;")
        main_bottom_layout.addWidget(self.status_indicator)

        main_name_lbl = QLabel(main_name)
        main_name_lbl.setStyleSheet(
            f"color: {C('text_secondary')}; font-size: 10px; background: transparent; border: none;")
        main_bottom_layout.addWidget(main_name_lbl)

        main_bottom_layout.addStretch()

        # Bandlik
        self.bandlik_label = QLabel(t("cam.bandlik", count=0))
        self.bandlik_label.setStyleSheet(
            f"color: {C('accent_green')}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        main_bottom_layout.addWidget(self.bandlik_label)

        main_bottom_layout.addStretch()

        # Polygon vaqt (soat o'rniga)
        self.poly_time_label = QLabel(t("cam.polygon.time", time=0.0))
        self.poly_time_label.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        main_bottom_layout.addWidget(self.poly_time_label)

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


        # ── COMPOSE LAYOUT ──
        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent;")

        if self.compact:
            # ── 4+ pereezd: Kamera1 (chap) | Kamera2+PLC+Stats (o'ng) ──
            content_layout = QHBoxLayout(content_widget)
            content_layout.setContentsMargins(8, 6, 8, 6)
            content_layout.setSpacing(4)

            # Asosiy kamera (chap, 65%)
            content_layout.addWidget(self.main_camera_label, stretch=65)

            # O'ng ustun: Kamera2 (tepa) + PLC + Stats (pastda)
            right_col = QWidget()
            right_col.setStyleSheet("background: transparent;")
            right_col_layout = QVBoxLayout(right_col)
            right_col_layout.setContentsMargins(0, 0, 0, 0)
            right_col_layout.setSpacing(4)

            # Qo'shimcha kamera + nomi
            cam2_widget = QWidget()
            cam2_widget.setStyleSheet("background: transparent;")
            cam2_layout = QVBoxLayout(cam2_widget)
            cam2_layout.setContentsMargins(0, 0, 0, 0)
            cam2_layout.setSpacing(0)
            cam2_layout.addWidget(self.additional_camera_label, stretch=1)
            cam2_layout.addWidget(add_bottom)
            right_col_layout.addWidget(cam2_widget, stretch=3)

            # PLC + Stats
            right_col_layout.addWidget(plc_frame, stretch=1)
            right_col_layout.addWidget(stats_frame, stretch=2)

            content_layout.addWidget(right_col, stretch=35)
        else:
            # ── 1-3 pereezd: vertikal (Kamera1 ustida, Kamera2+PLC pastda) ──
            content_layout = QVBoxLayout(content_widget)
            content_layout.setContentsMargins(8, 6, 8, 6)
            content_layout.setSpacing(4)

            # Main camera (full width, top)
            content_layout.addWidget(self.main_camera_label, stretch=3)

            # Bottom row: camera2 (left 65%) | PLC+stats (right 35%)
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
            bottom_row_layout.addWidget(cam2_widget, stretch=65)

            # PLC + Stats stacked
            right_panel = QWidget()
            right_panel.setStyleSheet("background: transparent;")
            right_panel_layout = QVBoxLayout(right_panel)
            right_panel_layout.setContentsMargins(0, 0, 0, 0)
            right_panel_layout.setSpacing(4)
            right_panel_layout.addWidget(plc_frame, stretch=1)
            right_panel_layout.addWidget(stats_frame, stretch=2)
            bottom_row_layout.addWidget(right_panel, stretch=35)

            content_layout.addWidget(bottom_row, stretch=2)

        frame_layout.addWidget(content_widget, stretch=1)

        # Bottom bar: kamera nomi + bandlik + polygon vaqt
        frame_layout.addWidget(main_bottom)

        main_layout.addWidget(self.frame)
        self.frame.mousePressEvent = self._on_mouse_press
        self.frame.mouseDoubleClickEvent = self._on_double_click
        self.frame.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.frame.customContextMenuRequested.connect(self._show_context_menu)

    def _create_plc_panel(self):
        plc_data = self.crossing_data.get("plc", {})
        plc_enabled = plc_data.get("enabled", False)
        plc_color = C('status_online') if plc_enabled else C('text_muted')
        plc_status_text = t("plc.connected") if plc_enabled else t("plc.disabled")

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

        self._plc_title_lbl = QLabel(t("plc.title").upper())
        self._plc_title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._plc_title_lbl.setStyleSheet(
            f"color: {C('text_dim')}; font-size: 9px; font-weight: bold; letter-spacing: 1px;"
            " background: transparent; border: none;")
        plc_inner.addWidget(self._plc_title_lbl)

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

        # Poyezd soni va vaqt
        train_row = QHBoxLayout()
        train_row.setSpacing(4)
        train_row.addStretch()
        self.train_count_label = QLabel(f"🚂  0")
        self.train_count_label.setStyleSheet(
            f"color: {C('text_secondary')}; font-size: 11px;"
            " background: transparent; border: none;")
        train_row.addWidget(self.train_count_label)
        train_row.addStretch()
        plc_inner.addLayout(train_row)

        self.train_duration_label = QLabel("")
        self.train_duration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.train_duration_label.setStyleSheet(
            f"color: {C('text_muted')}; font-size: 10px;"
            " background: transparent; border: none;")
        plc_inner.addWidget(self.train_duration_label)

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

        self._stats_title_lbl = QLabel(t("stats.panel").upper())
        self._stats_title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stats_title_lbl.setStyleSheet(
            f"color: {C('text_dim')}; font-size: 9px; font-weight: bold; letter-spacing: 1px;"
            " background: transparent; border: none;")
        stats_inner.addWidget(self._stats_title_lbl)

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
        self.total_label = QLabel(t("stats.total_fmt", total=0))
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
        self._last_display_light = car_count
        self._last_display_heavy = truck_count
        self._update_stat_label(self.car_count, car_count, C('accent_blue'))
        self._update_stat_label(self.truck_count, truck_count, C('accent_orange'))
        total = car_count + truck_count
        total_text = self._format_count(total)
        total_size = self._count_font_size(total_text)
        self.total_label.setText(t("stats.total_fmt", total=total_text))
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

        cam_settings = menu.addAction(f"📷 {t('menu.cam_settings')}")
        cam_settings.triggered.connect(self._open_camera_settings)

        info_action = menu.addAction(f"ℹ️ {t('menu.info')}")
        info_action.triggered.connect(lambda: self.clicked.emit(self.crossing_id))

        export_action = menu.addAction(f"💾 {t('menu.export_json')}")
        export_action.triggered.connect(self._export_json)

        btn_pos = self.menu_btn.mapToGlobal(QPoint(0, self.menu_btn.height()))
        menu.exec(btn_pos)

    def _export_json(self):
        """Pereezd ma'lumotlarini JSON faylga eksport qilish"""
        crossing = self.config_manager.get_crossing(self.crossing_id)
        if not crossing:
            return
        name = crossing.get("name", "pereezd").replace(" ", "_")
        file_path, _ = QFileDialog.getSaveFileName(
            self, t("export.json_title"), f"{name}.json",
            "JSON Files (*.json);;All Files (*)")
        if not file_path:
            return
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(crossing, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, t("export.json_title"),
                t("export.json_success", path=file_path))
        except Exception as e:
            QMessageBox.warning(self, t("error.title"), t("export.json_error", e=e))

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
        except (RuntimeError, Exception):
            pass

    def _start_cameras(self):
        if self._is_destroyed:
            return

        try:
            cameras = self.crossing_data.get("cameras", [])

            # Paused kameralar ro'yxati (camera_state.json dan)
            paused_ids = (
                self.config_manager.get_paused_cameras(self.crossing_id)
                if self.config_manager else set()
            )
            self._last_paused_ids = paused_ids

            # Asosiy to'xtatilgan/o'chirilgan bo'lsa — qo'shimcha asosiy joyni egallaydi
            main_id = self.main_camera_data.get("id") if self.main_camera_data else None
            main_enabled = self.main_camera_data.get("enabled", True) if self.main_camera_data else True
            if main_id and (main_id in paused_ids or not main_enabled):
                self._main_camera_down = True

            for i, camera in enumerate(cameras[:2]):
                if not camera.get("enabled", False):
                    continue
                if camera.get("id") in paused_ids:
                    continue

                source = camera.get("source", "")
                if not source:
                    continue

                cam_type = camera.get("type", "main")
                cam_name = camera.get("name", f"Camera {i+1}")

                # Polygon file yo'lini aniqlash
                poly_file = camera.get("polygon_file", "")
                if poly_file and not os.path.isabs(poly_file):
                    # Relative path — project root dan qidirish
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    poly_file = os.path.join(project_root, poly_file)

                # Settings dan thresholdlar
                settings = self.config_manager.get_settings() if self.config_manager else {}
                warn_t = settings.get("warning_threshold", 10.0)
                viol_t = settings.get("violation_threshold", 15.0)

                # Create worker with car detector + polygon
                worker = CameraWorker(
                    source, cam_name,
                    car_detector=self.car_detector,
                    detection_enabled=camera.get("detection_enabled", True),
                    polygon_file=poly_file if poly_file and os.path.isfile(poly_file) else None,
                    warning_threshold=warn_t,
                    violation_threshold=viol_t,
                    is_custom_model=self.is_custom_model,
                )

                # cam_type bo'yicha aniqlash, fallback: birinchi=main
                is_main = cam_type == "main" if cam_type in ("main", "additional") else (i == 0)
                if is_main:
                    worker.frame_ready.connect(lambda w=worker: self._on_main_frame(w))
                    worker.status_changed.connect(self._on_main_status)
                    worker.stats_updated.connect(self._on_stats_update)
                else:
                    worker.frame_ready.connect(lambda w=worker: self._on_additional_frame(w))
                    worker.status_changed.connect(self._on_additional_status)
                    # Asosiy to'xtatilgan bo'lsa — qo'shimcha sanashni davom ettiradi
                    if self._main_camera_down:
                        worker.stats_updated.connect(self._on_stats_update)

                worker.start()
                self.camera_workers.append(worker)
                self.camera_workers_by_id[camera.get("id")] = worker
        except (RuntimeError, Exception) as e:
            print(f"[StartCameras] Error: {e}")

    def _on_main_frame(self, worker):
        try:
            if self._is_destroyed or self.main_camera_label is None:
                return
            qimg = worker.take_frame()
            if qimg is None:
                return
            self._display_frame(self.main_camera_label, qimg)
        except Exception:
            self._is_destroyed = True

    def _on_additional_frame(self, worker):
        try:
            if self._is_destroyed or self.additional_camera_label is None:
                return
            qimg = worker.take_frame()
            if qimg is None:
                return
            self._display_frame(self.additional_camera_label, qimg)
            # Asosiy kamera uzilgan bo'lsa — qo'shimchani asosiy joyda ham ko'rsatish
            if self._main_camera_down and self.main_camera_label is not None:
                self._display_frame(self.main_camera_label, qimg)
        except Exception:
            self._is_destroyed = True

    def _schedule_reconnect(self):
        """Kamera uzilganda 5 sekunddan keyin qayta ulanish (debounced)."""
        if self._is_destroyed or self._reconnect_pending:
            return
        self._reconnect_pending = True
        QTimer.singleShot(5000, self._do_reconnect)

    def _do_reconnect(self):
        try:
            self._reconnect_pending = False
            if not self._is_destroyed:
                print(f"[{self.crossing_data.get('name','')}] Auto-reconnect...")
                self._restart_camera_workers()
        except Exception:
            pass

    def _on_main_status(self, status):
        try:
            if self._is_destroyed:
                return
            if status == "online":
                self._main_camera_down = False
                self.status_indicator.setStyleSheet(
                    f"color: {C('status_online')}; font-size: 10px; background: transparent; border: none;")
            elif status == "error":
                self._main_camera_down = True
                self.status_indicator.setStyleSheet(
                    f"color: {C('status_error')}; font-size: 10px; background: transparent; border: none;")
                if self.main_camera_label:
                    self._set_placeholder(self.main_camera_label, t("cam.status.main_down"))
                self._schedule_reconnect()
        except Exception:
            self._is_destroyed = True

    def _on_additional_status(self, status):
        try:
            if self._is_destroyed:
                return
            if status == "error":
                if self.additional_camera_label:
                    self._set_placeholder(self.additional_camera_label, t("cam.status.failed"))
                self._schedule_reconnect()
        except Exception:
            self._is_destroyed = True

    def _on_stats_update(self, light_count: int, heavy_count: int,
                         in_poly_count: int, max_time: float):
        """Handle cumulative tracking stats from CameraWorker"""
        try:
            if self._is_destroyed:
                return
            # Oxirgi tracker qiymatlarini saqlash (midnight reset uchun)
            self._last_tracker_light = light_count
            self._last_tracker_heavy = heavy_count
            # Display: DB offset + (tracker - base)
            # Midnight dan keyin base = midnight dagi qiymat, shuning uchun 0 dan boshlanadi
            display_light = self._light_offset + (light_count - self._tracker_base_light)
            display_heavy = self._heavy_offset + (heavy_count - self._tracker_base_heavy)
            self.update_stats(display_light, display_heavy)

            # Bandlik (polygon ichidagi avtomobillar soni) — faqat holat o'zgarganda style
            self.bandlik_label.setText(t("cam.bandlik", count=in_poly_count))
            _prev_in_poly = getattr(self, '_prev_in_poly_state', -1)
            if (in_poly_count > 0) != (_prev_in_poly > 0):
                self._prev_in_poly_state = in_poly_count
                if in_poly_count > 0:
                    self.bandlik_label.setStyleSheet(
                        f"color: {C('accent_red')}; font-size: 10px; font-weight: bold;"
                        " background: transparent; border: none;")
                else:
                    self.bandlik_label.setStyleSheet(
                        f"color: {C('accent_green')}; font-size: 10px; font-weight: bold;"
                        " background: transparent; border: none;")

            # Polygon vaqt — faqat qiymat o'zgarganda style
            if max_time > 0:
                self.poly_time_label.setText(t("cam.polygon.time", time=max_time))
                _new_color = C('accent_red') if max_time > 10 else C('accent_orange')
                if getattr(self, '_prev_poly_color', None) != _new_color:
                    self._prev_poly_color = _new_color
                    self.poly_time_label.setStyleSheet(
                        f"color: {_new_color}; font-size: 10px; font-weight: bold;"
                        " background: transparent; border: none;")
            else:
                self.poly_time_label.setText(t("cam.polygon.time", time=0.0))
                if getattr(self, '_prev_poly_color', None) is not None:
                    self._prev_poly_color = None
                    self.poly_time_label.setStyleSheet(
                        f"color: {C('text_muted')}; font-size: 10px; font-weight: bold;"
                        " background: transparent; border: none;")

            # PLC ga mashina holati yuborish (polygon ichida bor/yo'q)
            if self._plc_manager is not None:
                self._plc_manager.set_has_cars(in_poly_count > 0)

            # DB ga yozish (asosiy kamera uchun)
            if self.stats_db and light_count + heavy_count > 0:
                main_name = ""
                if self.main_camera_data:
                    main_name = self.main_camera_data.get("name", "Camera")
                if main_name:
                    try:
                        self.stats_db.record_count(
                            self.crossing_id, main_name, light_count, heavy_count)
                    except Exception as e:
                        print(f"[StatsDB] Record error: {e}")
        except Exception:
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

        open_action = menu.addAction(f"📂 {t('menu.open')}")
        open_action.triggered.connect(lambda: self.clicked.emit(self.crossing_id))

        cam_settings = menu.addAction(f"📷 {t('menu.cam_settings')}")
        cam_settings.triggered.connect(self._open_camera_settings)

        menu.addSeparator()

        info_action = menu.addAction(f"ℹ️ {t('menu.info')}")
        info_action.triggered.connect(lambda: self.clicked.emit(self.crossing_id))

        menu.exec(self.frame.mapToGlobal(pos))

    def _retranslate(self, _lang=None):
        """Til o'zgarganida statik label textlarni yangilash"""
        if self._is_destroyed:
            return
        try:
            if hasattr(self, '_plc_title_lbl'):
                self._plc_title_lbl.setText(t("plc.title").upper())
            if hasattr(self, 'plc_status_label'):
                if self._plc_enabled:
                    if self._plc_active:
                        self.plc_status_label.setText(t("plc.active"))
                    else:
                        self.plc_status_label.setText(t("plc.connected"))
                else:
                    self.plc_status_label.setText(t("plc.disabled"))
            if hasattr(self, '_stats_title_lbl'):
                self._stats_title_lbl.setText(t("stats.panel").upper())
            if hasattr(self, 'total_label'):
                self.update_stats(self._last_display_light, self._last_display_heavy)
        except (RuntimeError, Exception):
            pass

    def stop_cameras(self):
        # Avval signallarni uzib, keyin to'xtatish (crash prevention)
        for worker in self.camera_workers:
            try:
                worker.frame_ready.disconnect()
                worker.status_changed.disconnect()
                worker.stats_updated.disconnect()
            except (TypeError, RuntimeError):
                pass
        for worker in self.camera_workers:
            try:
                worker.stop()
            except (RuntimeError, Exception) as e:
                print(f"[StopCamera] Error: {e}")
        self.camera_workers.clear()

    def cleanup(self):
        if self._is_destroyed:
            return
        self._is_destroyed = True
        try:
            if hasattr(self, '_midnight_timer'):
                self._midnight_timer.stop()
            if hasattr(self, '_state_timer'):
                self._state_timer.stop()
            if hasattr(self, '_plc_timer'):
                self._plc_timer.stop()
            if hasattr(self, '_train_timer'):
                self._train_timer.stop()
            if hasattr(self, '_grace_timer'):
                self._grace_timer.stop()
        except RuntimeError:
            pass
        # PLC threadini to'xtatish
        if self._plc_manager is not None:
            try:
                self._plc_manager.stop()
            except Exception:
                pass
            self._plc_manager = None
        self.stop_cameras()
        # Shared detector - to'xtatmaymiz
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
