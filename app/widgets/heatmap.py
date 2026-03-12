"""
HeatmapChart — 7 kun x 24 soat issiqlik xaritasi.
QPainter bilan chizilgan, tema ranglariga mos.

Usage:
    heatmap = HeatmapChart()
    data = stats_db.get_heatmap_data(crossing_id)
    heatmap.set_data(data)
"""

from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import (QPainter, QColor, QFont, QPen, QBrush,
                          QPainterPath, QLinearGradient)
from typing import List, Dict

from app.utils.theme_colors import C


class HeatmapChart(QWidget):
    """7 kun x 24 soat issiqlik xaritasi (heatmap)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = []  # [{"day": "Du", "hours": [0]*24}, ...]
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, data: List[Dict]):
        """data: [{"day": "Du", "hours": [h0, h1, ..., h23]}, ...] (7 ta)"""
        self._data = data
        self.update()

    def _get_color(self, value: int, max_val: int) -> QColor:
        """Qiymatga qarab rang: qora → yashil → sariq → qizil"""
        if max_val == 0 or value == 0:
            return QColor(C('bg_panel_dark'))

        ratio = min(value / max_val, 1.0)

        if ratio < 0.33:
            # Qora → Yashil
            t = ratio / 0.33
            r = int(20 * (1 - t) + 40 * t)
            g = int(30 * (1 - t) + 160 * t)
            b = int(20 * (1 - t) + 60 * t)
        elif ratio < 0.66:
            # Yashil → Sariq
            t = (ratio - 0.33) / 0.33
            r = int(40 * (1 - t) + 200 * t)
            g = int(160 * (1 - t) + 180 * t)
            b = int(60 * (1 - t) + 40 * t)
        else:
            # Sariq → Qizil
            t = (ratio - 0.66) / 0.34
            r = int(200 * (1 - t) + 220 * t)
            g = int(180 * (1 - t) + 60 * t)
            b = int(40 * (1 - t) + 50 * t)

        return QColor(r, g, b)

    def paintEvent(self, event):
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        left_m = 32   # Kun nomlari uchun
        top_m = 22    # Soat raqamlari uchun
        right_m = 8
        bottom_m = 6

        cw = w - left_m - right_m
        ch = h - top_m - bottom_m

        if cw <= 0 or ch <= 0:
            painter.end()
            return

        rows = len(self._data)  # 7
        cols = 24

        cell_w = cw / cols
        cell_h = ch / rows
        gap = 1.5
        radius = min(3, cell_w / 4, cell_h / 4)

        # Maksimal qiymatni topish
        max_val = 0
        for d in self._data:
            for v in d.get("hours", []):
                if v > max_val:
                    max_val = v
        if max_val == 0:
            max_val = 1

        text_color = QColor(C('text_muted'))
        font = QFont()
        font.setPixelSize(9)
        painter.setFont(font)

        # Soat raqamlari (tepada)
        painter.setPen(QPen(text_color))
        for c in range(cols):
            if c % 3 == 0:  # Har 3 soatda label
                x = left_m + c * cell_w
                painter.drawText(
                    QRectF(x, 0, cell_w * 3, top_m - 2),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
                    f"{c:02d}"
                )

        # Qatorlar (kunlar)
        for r, d in enumerate(self._data):
            y = top_m + r * cell_h
            day_label = d.get("day", "")
            hours = d.get("hours", [0] * 24)

            # Kun nomi (chapda)
            painter.setPen(QPen(text_color))
            font.setPixelSize(9)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(
                QRectF(0, y, left_m - 4, cell_h),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                day_label
            )
            font.setBold(False)
            painter.setFont(font)

            # Kataklar
            for c in range(cols):
                x = left_m + c * cell_w
                value = hours[c] if c < len(hours) else 0

                color = self._get_color(value, max_val)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))

                path = QPainterPath()
                path.addRoundedRect(
                    QRectF(x + gap / 2, y + gap / 2,
                           cell_w - gap, cell_h - gap),
                    radius, radius
                )
                painter.drawPath(path)

                # Katta qiymatlarda son yozish
                if value > 0 and cell_w >= 18 and cell_h >= 16:
                    painter.setPen(QPen(QColor(255, 255, 255, 200)))
                    font.setPixelSize(max(7, int(min(cell_w, cell_h) * 0.45)))
                    painter.setFont(font)
                    painter.drawText(
                        QRectF(x + gap / 2, y + gap / 2,
                               cell_w - gap, cell_h - gap),
                        Qt.AlignmentFlag.AlignCenter,
                        str(value)
                    )
                    font.setPixelSize(9)
                    painter.setFont(font)

        painter.end()
