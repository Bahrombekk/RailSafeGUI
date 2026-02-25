"""
Dashboard View - GPU MAKSIMAL ISHLATISH
TensorRT native inference (208 FPS) yoki PyTorch/ONNX fallback
"""

import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea,
                              QGridLayout, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from gui.widgets.crossing_card import CrossingCard
from gui.utils.theme_colors import C
from gui.utils.stats_db import StatsDB
from gui.utils.language import t, LM

try:
    from detectors import RealtimeMultiCameraDetector
    CAR_DETECTOR_AVAILABLE = True
except ImportError:
    CAR_DETECTOR_AVAILABLE = False


class Dashboard(QWidget):
    """Dashboard - GPU maksimal ishlatish, bitta shared detector"""

    crossing_selected = pyqtSignal(int)
    add_crossing_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.crossing_cards = []
        self._last_col_count = 0

        # BITTA detector - barcha kameralar uchun (GPU maksimal)
        self.car_detector = None
        self.is_custom_model = False

        # Statistika bazasi (barcha pereezdlar uchun bitta)
        self.stats_db = StatsDB()

        self._empty_label = None
        self._setup_ui()
        LM.language_changed.connect(self._retranslate)
        # Detektor va kameralar KEYINROQ ishga tushadi (engine tayyor bo'lgandan keyin)

    def start_detection(self):
        """Engine eksport tugagandan keyin chaqiriladi — detektor yuklash + kameralar boshlash"""
        self._init_shared_detector()
        self._load_crossings()

    def _init_shared_detector(self):
        """
        RealtimeMultiCameraDetector - TensorRT native yoki Ultralytics fallback
        """
        if not CAR_DETECTOR_AVAILABLE:
            return

        try:
            car_config = self.config_manager.get_car_detector_config()
            if not car_config.get("enabled", False):
                return

            model_path = car_config.get("model_path", "")
            if not model_path or not os.path.exists(model_path):
                print(f"[Dashboard] Model not found: {model_path}")
                return

            self.car_detector = RealtimeMultiCameraDetector(
                model_path=model_path,
                confidence_threshold=car_config.get("confidence", 0.3),
                iou_threshold=car_config.get("iou_threshold", 0.45),
                imgsz=car_config.get("imgsz", 640),
                device=car_config.get("device", "cuda"),
                half=car_config.get("half", True),
                filter_classes=car_config.get("filter_classes"),
                batch_interval_ms=15.0,
            )

            self.is_custom_model = car_config.get("is_custom_model", False)

            if self.car_detector.load():
                stats = self.car_detector.get_stats()
                model_label = "MAXSUS" if self.is_custom_model else stats['model_type'].upper()
                print(f"[Dashboard] Detector yuklandi! Mode: {model_label}")

                print(f"[Dashboard] Model type: {stats['model_type'].upper()}")
            else:
                self.car_detector = None

        except Exception as e:
            print(f"[Dashboard] Detector error: {e}")
            import traceback
            traceback.print_exc()
            self.car_detector = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(8)
        self.grid.setContentsMargins(0, 0, 0, 0)

        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)

    def _get_column_count(self):
        """Calculate columns - max 3 on normal, 4 only on 37+ inch"""
        w = self.scroll.viewport().width()
        if w < 700:
            return 1
        elif w < 1100:
            return 2
        elif w < 2200:
            return 3
        else:
            return 4

    def _apply_stretch(self, cols, rows, fill_screen=False):
        for c in range(cols):
            self.grid.setColumnStretch(c, 1)
        if fill_screen:
            for r in range(rows):
                self.grid.setRowStretch(r, 1)

    def _clear_stretch(self):
        for c in range(max(self.grid.columnCount(), 5)):
            self.grid.setColumnStretch(c, 0)
        for r in range(max(self.grid.rowCount(), 10)):
            self.grid.setRowStretch(r, 0)

    def _load_crossings(self):
        self._clear_crossings()
        self._clear_stretch()
        crossings = self.config_manager.get_crossings()

        if not crossings:
            self._empty_label = QLabel(t("dashboard.empty"))
            self._empty_label.setStyleSheet(f"color: {C('text_muted')}; font-size: 14px; padding: 40px;")
            self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.grid.addWidget(self._empty_label, 0, 0, 1, 3)
            return

        count = len(crossings)

        if count == 1:
            cols = 1
        elif count == 2:
            cols = 2
        elif count == 3:
            cols = 3
        elif count == 4:
            cols = 2
        else:
            cols = self._get_column_count()

        self._last_col_count = cols
        rows = (count + cols - 1) // cols
        fill_screen = True  # Har doim ekranga moslash

        for idx, crossing in enumerate(crossings):
            row = idx // cols
            col = idx % cols
            card = CrossingCard(
                crossing,
                config_manager=self.config_manager,
                compact=(count >= 4),
                car_detector=self.car_detector,
                stats_db=self.stats_db,
                is_custom_model=self.is_custom_model,
            )
            card.clicked.connect(self.crossing_selected.emit)
            card.setMaximumHeight(16777215)
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.grid.addWidget(card, row, col)
            self.crossing_cards.append(card)

        self._apply_stretch(cols, rows, fill_screen=True)

    def _relayout_cards(self):
        cols = self._get_column_count()
        if cols == self._last_col_count or not self.crossing_cards:
            return
        self._last_col_count = cols
        self._clear_stretch()

        for card in self.crossing_cards:
            self.grid.removeWidget(card)

        count = len(self.crossing_cards)

        if count == 1:
            cols = 1
        elif count == 2:
            cols = 2
        elif count == 3:
            cols = 3
        elif count == 4:
            cols = 2

        rows = (count + cols - 1) // cols

        for idx, card in enumerate(self.crossing_cards):
            row = idx // cols
            col = idx % cols
            card.setMaximumHeight(16777215)
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.grid.addWidget(card, row, col)

        self._apply_stretch(cols, rows, fill_screen=True)

    def _clear_crossings(self):
        for card in self.crossing_cards:
            try:
                if hasattr(card, 'cleanup'):
                    card.cleanup()
                card.setParent(None)
                card.deleteLater()
            except (RuntimeError, Exception):
                pass
        self.crossing_cards.clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_cards()

    def stop_all_cameras(self):
        """Kamera workerlarni to'xtatish (detail view uchun GPU bo'shatish)"""
        for card in self.crossing_cards:
            try:
                if hasattr(card, 'cleanup'):
                    card.cleanup()
            except (RuntimeError, Exception):
                pass

    def _retranslate(self, _lang=None):
        """Til o'zgarganida bo'sh holat labelini yangilash"""
        if self._empty_label is not None:
            try:
                self._empty_label.setText(t("dashboard.empty"))
            except RuntimeError:
                pass

    def refresh(self):
        self._empty_label = None
        self._load_crossings()
