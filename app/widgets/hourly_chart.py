"""
HourlyChart - 24 soatlik transport statistika bar chart (QPainter).
Tashqi kutubxona kerak emas.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QSizePolicy
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QFontMetrics, QBrush
from datetime import datetime
from typing import List, Dict

from app.utils.theme_colors import C
from app.utils.numfmt import fmt_compact, axis_label_width, fmt_full
from app.widgets.chart_hover import (HoverTipMixin, tip_lines, tip_header,
                                     tip_traffic)
from app.utils.language import t
from app.utils.language import t, LM


class HourlyBarChart(HoverTipMixin, QWidget):
    """24 soatlik stacked bar chart - yengil (ko'k) + og'ir (to'q sariq)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_hover()
        self._data: List[Dict] = [{"hour": h, "light": 0, "heavy": 0} for h in range(24)]
        self._current_hour = datetime.now().hour
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _hover_info(self, pos):
        """Kursor ostidagi SOAT uchun aniq qiymatlar."""
        i = self._index_from_bars(pos, 24)
        if i < 0 or i >= len(self._data):
            return -1, None
        d = self._data[i]
        return i, tip_lines(tip_header(f"{i:02d}:00"),
                            tip_traffic(d.get("light", 0), d.get("heavy", 0)))

    def set_data(self, data: List[Dict]):
        """Ma'lumotni yangilash: [{"hour": 0, "light": 5, "heavy": 2}, ...]"""
        self._data = data
        self._current_hour = datetime.now().hour
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Margins
        left_m = 30
        right_m = 10
        top_m = 10
        bottom_m = 25

        chart_w = w - left_m - right_m
        chart_h = h - top_m - bottom_m

        if chart_w <= 0 or chart_h <= 0:
            painter.end()
            return

        # Max qiymat
        max_val = max(
            (d["light"] + d["heavy"]) for d in self._data
        )
        if max_val == 0:
            max_val = 1

        # Chap chegarani yorliq kengligiga moslash — ma'lumot o'sib son
        # uzaysa ham kesilmasin (izoh: app/utils/numfmt.py)
        _f8 = QFont()
        _f8.setPixelSize(8)
        left_m = axis_label_width(
            (int(max_val * i / 3) for i in range(1, 4)), QFontMetrics(_f8),
            minimum=30)
        chart_w = w - left_m - right_m
        if chart_w <= 0:
            painter.end()
            return

        bar_w = chart_w / 24
        self._save_geom(left_m=left_m, top_m=top_m, cw=chart_w, ch=chart_h, n=24)
        gap = max(1, bar_w * 0.15)

        # Ranglar
        light_color = QColor(C('accent_blue'))
        heavy_color = QColor(C('accent_orange'))
        current_highlight = QColor(C('accent_green'))
        current_highlight.setAlpha(40)
        bg_color = QColor(C('bg_panel_dark') if C('bg_panel_dark') != '#000000' else '#1a1a2e')
        grid_color = QColor(C('bg_input') if C('bg_input') != '#000000' else '#2a2a3e')
        text_color = QColor(C('text_muted') if C('text_muted') != '#000000' else '#666680')

        # Background
        painter.fillRect(QRectF(left_m, top_m, chart_w, chart_h), QBrush(bg_color))

        # Grid chiziqlari (3 ta)
        painter.setPen(QPen(grid_color, 1))
        font = QFont()
        font.setPixelSize(8)
        painter.setFont(font)
        for i in range(1, 4):
            y = top_m + chart_h - (chart_h * i / 3)
            painter.drawLine(int(left_m), int(y), int(left_m + chart_w), int(y))
            val = int(max_val * i / 3)
            painter.setPen(QPen(text_color, 1))
            painter.drawText(QRectF(0, y - 6, left_m - 4, 12),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             fmt_compact(val))
            painter.setPen(QPen(grid_color, 1))

        # Barlar
        for i, d in enumerate(self._data):
            x = left_m + i * bar_w
            light = d["light"]
            heavy = d["heavy"]
            total = light + heavy

            # Hozirgi soat highlight
            if i == self._current_hour:
                painter.fillRect(
                    QRectF(x, top_m, bar_w, chart_h),
                    QBrush(current_highlight)
                )

            if total > 0:
                total_h = (total / max_val) * chart_h
                light_h = (light / max_val) * chart_h
                heavy_h = (heavy / max_val) * chart_h

                bar_x = x + gap / 2
                actual_bar_w = bar_w - gap

                # Heavy (pastda)
                if heavy > 0:
                    y_heavy = top_m + chart_h - total_h
                    painter.fillRect(
                        QRectF(bar_x, y_heavy, actual_bar_w, heavy_h),
                        QBrush(heavy_color)
                    )

                # Light (tepada)
                if light > 0:
                    y_light = top_m + chart_h - light_h
                    painter.fillRect(
                        QRectF(bar_x, y_light, actual_bar_w, light_h),
                        QBrush(light_color)
                    )

            # Soat labellari (har 3 soatda)
            if i % 3 == 0:
                painter.setPen(QPen(text_color, 1))
                painter.drawText(
                    QRectF(x, top_m + chart_h + 2, bar_w, bottom_m - 2),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    f"{i:02d}"
                )

        # Bottom line
        painter.setPen(QPen(grid_color, 1))
        painter.drawLine(int(left_m), int(top_m + chart_h),
                         int(left_m + chart_w), int(top_m + chart_h))

        painter.end()


class TrainHourlyBarChart(HoverTipMixin, QWidget):
    """24 soatlik poyezd soni bar chart (teal rangli, bitta ustun)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_hover()
        self._data: List[int] = [0] * 24   # har soat uchun poyezd soni
        self._current_hour = datetime.now().hour
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _hover_info(self, pos):
        """Kursor ostidagi SOAT uchun poyezd soni."""
        i = self._index_from_bars(pos, 24)
        if i < 0 or i >= len(self._data):
            return -1, None
        return i, tip_lines(tip_header(f"{i:02d}:00"),
                            t("tip.trains_count",
                              count=fmt_full(self._data[i])))

    def set_data(self, data: List[int]):
        """Ma'lumotni yangilash: 24 elementli list [count_hour0, ..., count_hour23]"""
        self._data = data if len(data) == 24 else [0] * 24
        self._current_hour = datetime.now().hour
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        left_m, right_m, top_m, bottom_m = 30, 10, 10, 25
        chart_w = w - left_m - right_m
        chart_h = h - top_m - bottom_m

        if chart_w <= 0 or chart_h <= 0:
            painter.end()
            return

        max_val = max(self._data) if self._data else 0
        if max_val == 0:
            max_val = 1

        # Chegarani yorliq kengligiga moslash (uzun son kesilmasin)
        _f8 = QFont()
        _f8.setPixelSize(8)
        left_m = axis_label_width(
            (int(max_val * i / 3) for i in range(1, 4)), QFontMetrics(_f8),
            minimum=30)
        chart_w = w - left_m - right_m
        if chart_w <= 0:
            painter.end()
            return

        bar_w = chart_w / 24
        self._save_geom(left_m=left_m, top_m=top_m, cw=chart_w, ch=chart_h, n=24)
        gap = max(1, bar_w * 0.15)

        teal_color = QColor(C('accent_teal'))
        current_hl = QColor(C('accent_teal'))
        current_hl.setAlpha(30)
        bg_color = QColor(C('bg_panel_dark'))
        grid_color = QColor(C('bg_input'))
        text_color = QColor(C('text_muted'))

        painter.fillRect(QRectF(left_m, top_m, chart_w, chart_h), QBrush(bg_color))

        painter.setPen(QPen(grid_color, 1))
        font = QFont()
        font.setPixelSize(8)
        painter.setFont(font)
        for i in range(1, 4):
            y = top_m + chart_h - (chart_h * i / 3)
            painter.drawLine(int(left_m), int(y), int(left_m + chart_w), int(y))
            val = int(max_val * i / 3)
            painter.setPen(QPen(text_color, 1))
            painter.drawText(QRectF(0, y - 6, left_m - 4, 12),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             fmt_compact(val))
            painter.setPen(QPen(grid_color, 1))

        for i, count in enumerate(self._data):
            x = left_m + i * bar_w
            if i == self._current_hour:
                painter.fillRect(QRectF(x, top_m, bar_w, chart_h), QBrush(current_hl))
            if count > 0:
                bar_h = (count / max_val) * chart_h
                bar_x = x + gap / 2
                actual_w = bar_w - gap
                y0 = top_m + chart_h - bar_h
                painter.fillRect(QRectF(bar_x, y0, actual_w, bar_h), QBrush(teal_color))
            if i % 3 == 0:
                painter.setPen(QPen(text_color, 1))
                painter.drawText(
                    QRectF(x, top_m + chart_h + 2, bar_w, bottom_m - 2),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    f"{i:02d}"
                )

        painter.setPen(QPen(grid_color, 1))
        painter.drawLine(int(left_m), int(top_m + chart_h),
                         int(left_m + chart_w), int(top_m + chart_h))
        painter.end()


class HourlyChartPanel(QWidget):
    """So'nggi Hodisalar paneli - title + legend + chart"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame#chartPanel {{
                background: {C('bg_primary')};
                border: 2px solid {C('bg_input')};
                border-radius: 12px;
            }}
        """)
        panel.setObjectName("chartPanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Title
        self._title = QLabel(t("chart.hourly_title"))
        self._title.setStyleSheet(f"color: {C('text_primary')}; font-size: 14px; font-weight: bold; background: transparent;")
        layout.addWidget(self._title)

        # Divider
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet(f"background: {C('bg_input')};")
        layout.addWidget(div)

        # Legend
        from PyQt6.QtWidgets import QHBoxLayout
        legend_layout = QHBoxLayout()
        legend_layout.setSpacing(16)

        self._light_dot = QLabel(f"● {t('stats.light')}")
        self._light_dot.setStyleSheet(f"color: {C('accent_blue')}; font-size: 10px; background: transparent;")
        legend_layout.addWidget(self._light_dot)

        self._heavy_dot = QLabel(f"● {t('stats.heavy')}")
        self._heavy_dot.setStyleSheet(f"color: {C('accent_orange')}; font-size: 10px; background: transparent;")
        legend_layout.addWidget(self._heavy_dot)

        legend_layout.addStretch()

        self._total_light = 0
        self._total_heavy = 0
        self._total_label = QLabel(t("chart.hourly_total").format(total=0, light=0, heavy=0))
        self._total_label.setStyleSheet(f"color: {C('accent_green')}; font-size: 11px; font-weight: bold; background: transparent;")
        legend_layout.addWidget(self._total_label)

        layout.addLayout(legend_layout)

        LM.language_changed.connect(self._retranslate)

        # Chart
        self.chart = HourlyBarChart()
        self.chart.setMinimumHeight(120)
        self.chart.setMaximumHeight(180)
        layout.addWidget(self.chart)

        # Outer layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)

    def set_data(self, data: List[Dict]):
        self.chart.set_data(data)
        self._total_light = sum(d["light"] for d in data)
        self._total_heavy = sum(d["heavy"] for d in data)
        self._update_total_label()

    def _update_total_label(self):
        self._total_label.setText(t("chart.hourly_total").format(
            total=self._total_light + self._total_heavy,
            light=self._total_light, heavy=self._total_heavy))

    def _retranslate(self, _lang=None):
        try:
            self._title.setText(t("chart.hourly_title"))
            self._light_dot.setText(f"● {t('stats.light')}")
            self._heavy_dot.setText(f"● {t('stats.heavy')}")
            self._update_total_label()
        except RuntimeError:
            pass  # widget o'chirilgan
