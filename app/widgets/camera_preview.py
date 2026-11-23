"""Prepare camera previews on the worker thread and display the latest one in Qt."""

from threading import Lock

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class CameraPreview(QLabel):
    _frame_available = pyqtSignal()
    FRAME_INTERVAL = 1.0 / 15.0

    def __init__(self, *, responsive_height=False):
        super().__init__()
        self._source = None
        self._lock = Lock()
        self._viewport = None
        self._latest = None
        self._deadline = None
        self._aspect_ratio = 16 / 9 if responsive_height else None
        self._frame_available.connect(self._display_latest, Qt.ConnectionType.QueuedConnection)
        # A previously displayed pixmap must not set a minimum size for the tile.
        policy = QSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred if responsive_height else QSizePolicy.Policy.Ignored,
        )
        policy.setHeightForWidth(responsive_height)
        self.setSizePolicy(policy)

    def heightForWidth(self, width):
        if self._aspect_ratio is None or width < 0:
            return super().heightForWidth(width)
        margins = self.contentsMargins()
        video_width = max(0, width - margins.left() - margins.right())
        return max(self.minimumHeight(),
                   round(video_width / self._aspect_ratio) + margins.top() + margins.bottom())

    def set_source(self, source):
        """Bind this tile to a camera worker from the GUI thread."""
        with self._lock:
            previous = self._source
            if previous is source:
                return
            self._source = source
            self._latest = None
            self._deadline = None
        if previous is not None:
            previous.preview_targets = tuple(
                target for target in previous.preview_targets if target is not self)
        if source is not None:
            source.preview_targets += (self,)

    def submit(self, source, frame, now):
        """Resize on the camera thread and queue at most one pending preview."""
        with self._lock:
            viewport = self._viewport
            if self._source is not source or viewport is None:
                return
            if self._deadline is not None and now < self._deadline:
                return
            # Carry the deadline forward through normal frame jitter. Reset after a stall.
            if self._deadline is None or now - self._deadline >= self.FRAME_INTERVAL:
                self._deadline = now
            self._deadline += self.FRAME_INTERVAL

        width, height, dpr = viewport
        frame_h, frame_w = frame.shape[:2]
        scale = min(width / frame_w, height / frame_h)
        size = (max(1, round(frame_w * scale)), max(1, round(frame_h * scale)))
        if size != (frame_w, frame_h):
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            frame = cv2.resize(frame, size, interpolation=interpolation)
        frame = np.ascontiguousarray(frame)
        image = QImage(frame.data, *size, frame.strides[0], QImage.Format.Format_BGR888).copy()
        image.setDevicePixelRatio(dpr)

        with self._lock:
            if self._source is not source or self._viewport != viewport:
                return
            notify = self._latest is None
            self._latest = (image, frame_w / frame_h)
            if notify:
                self._frame_available.emit()

    @pyqtSlot()
    def _display_latest(self):
        with self._lock:
            preview, self._latest = self._latest, None
        if preview is None:
            return
        image, aspect_ratio = preview
        if self._aspect_ratio is not None and self._aspect_ratio != aspect_ratio:
            self._aspect_ratio = aspect_ratio
            self.updateGeometry()
        self.setPixmap(QPixmap.fromImage(image))

    def _update_viewport(self):
        rect = self.contentsRect()
        dpr = self.devicePixelRatioF()
        viewport = None
        if self.isVisible() and rect.width() > 0 and rect.height() > 0:
            viewport = (round(rect.width() * dpr), round(rect.height() * dpr), dpr)
        with self._lock:
            if self._viewport != viewport:
                self._viewport = viewport
                self._latest = None
                self._deadline = None

    def event(self, event):
        result = super().event(event)
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.Hide,
                            QEvent.Type.DevicePixelRatioChange):
            self._update_viewport()
        return result
