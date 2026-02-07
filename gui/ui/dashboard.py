"""
Dashboard View - Responsive view, cards fill viewport proportionally
HAR BIR CROSSING O'ZINING DETECTORINI YARATADI
"""

import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea,
                              QGridLayout, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from gui.widgets.crossing_card import CrossingCard
from gui.utils.theme_colors import C


class Dashboard(QWidget):
    """Dashboard - max 3 columns, proportional cards"""

    crossing_selected = pyqtSignal(int)
    add_crossing_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.crossing_cards = []
        self._last_col_count = 0

        self._setup_ui()
        self._load_crossings()

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
            msg = QLabel("Pereezdlar yo'q. Toolbar dan '+ Pereezd Qo'shish' bosing.")
            msg.setStyleSheet(f"color: {C('text_muted')}; font-size: 14px; padding: 40px;")
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.grid.addWidget(msg, 0, 0, 1, 3)
            return

        count = len(crossings)

        # 1 ta = to'liq ekran, 2-4 ta = teng bo'lib to'ldiradi, 5+ = scroll grid
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
        fill_screen = (count <= 4)

        for idx, crossing in enumerate(crossings):
            row = idx // cols
            col = idx % cols
            # Har bir card o'zining detectorini yaratadi
            card = CrossingCard(
                crossing,
                config_manager=self.config_manager,
                compact=(count in (2, 3) or count >= 5)
            )
            card.clicked.connect(self.crossing_selected.emit)
            if fill_screen:
                card.setMaximumHeight(16777215)
                card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self.grid.addWidget(card, row, col)
            else:
                card.setMaximumHeight(420)
                self.grid.addWidget(card, row, col, Qt.AlignmentFlag.AlignTop)
            self.crossing_cards.append(card)

        self._apply_stretch(cols, rows, fill_screen=fill_screen)

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
        fill_screen = (count <= 4)

        for idx, card in enumerate(self.crossing_cards):
            row = idx // cols
            col = idx % cols
            if fill_screen:
                card.setMaximumHeight(16777215)
                card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                self.grid.addWidget(card, row, col)
            else:
                card.setMaximumHeight(420)
                card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                self.grid.addWidget(card, row, col, Qt.AlignmentFlag.AlignTop)

        self._apply_stretch(cols, rows, fill_screen=fill_screen)

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

    def refresh(self):
        self._load_crossings()
