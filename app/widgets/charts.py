"""
Charts - Chiroyli QPainter diagrammalar (tashqi kutubxona kerak emas).
DonutChart, LineChart, BarChart — analitika sahifasi uchun.
"""

import math
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (QPainter, QColor, QPen, QFont, QBrush,
                          QLinearGradient, QPainterPath, QConicalGradient,
                          QFontMetrics)
from typing import List, Dict

from app.utils.theme_colors import C
from app.utils.numfmt import fmt_compact, fmt_fit, axis_label_width, fits, fmt_full
from app.widgets.chart_hover import (HoverTipMixin, tip_lines, tip_header,
                                     tip_traffic)
from app.utils.language import t


# ═══════════════════════════════════════════════════════════════
#  DONUT CHART — transport turlari taqsimoti (yengil / og'ir)
# ═══════════════════════════════════════════════════════════════

class DonutChart(HoverTipMixin, QWidget):
    """Animated donut chart with center text"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_hover()
        self._segments = []  # [{"value": 10, "color": "#4a9eff", "label": "Yengil"}, ...]
        self._center_text = "0"
        self._center_sub = "Jami"
        self.setMinimumSize(140, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, segments: List[Dict], center_text: str = "", center_sub: str = ""):
        self._segments = segments
        self._center_text = center_text
        self._center_sub = center_sub
        self.update()

    def _hover_info(self, pos):
        """Halqaning kursor ostidagi bo'lagi: qiymat va ulush."""
        g = self._geom
        if not g or not self._segments:
            return -1, None
        dx = pos.x() - g["cx"]
        dy = pos.y() - g["cy"]
        r = math.hypot(dx, dy)
        if not (g["inner_r"] <= r <= g["outer_r"]):
            return -1, None
        total = sum(seg.get("value", 0) for seg in self._segments) or 1
        # Chizish 90 gradusdan (tepadan) SOAT YO'NALISHIDA boradi
        ang = (90 - math.degrees(math.atan2(-dy, dx))) % 360
        acc = 0.0
        for i, seg in enumerate(self._segments):
            span = (seg.get("value", 0) / total) * 360.0
            if span <= 0:
                continue
            if acc <= ang < acc + span:
                val = seg.get("value", 0)
                return i, tip_lines(
                    tip_header(seg.get("label", "")),
                    fmt_full(val),
                    t("tip.share", pct=round(val / total * 100)))
            acc += span
        return -1, None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        size = min(w, h) - 8
        if size < 40:
            painter.end()
            return

        cx, cy = w / 2, h / 2
        outer_r = size / 2
        inner_r = outer_r * 0.62
        ring_w = outer_r - inner_r
        self._save_geom(cx=cx, cy=cy, inner_r=inner_r, outer_r=outer_r)

        rect = QRectF(cx - outer_r, cy - outer_r, size, size)

        total = sum(s["value"] for s in self._segments)

        if total == 0:
            # Bo'sh donut — kulrang
            painter.setPen(QPen(QColor(C('bg_input')), ring_w))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(
                cx - (outer_r - ring_w / 2),
                cy - (outer_r - ring_w / 2),
                size - ring_w, size - ring_w
            ))
        else:
            start = 90.0  # 12 soat pozitsiyasidan boshlash
            for seg in self._segments:
                if seg["value"] <= 0:
                    continue
                span = (seg["value"] / total) * 360.0

                color = QColor(seg["color"])
                pen = QPen(color, ring_w)
                pen.setCapStyle(Qt.PenCapStyle.FlatCap)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                arc_rect = QRectF(
                    cx - (outer_r - ring_w / 2),
                    cy - (outer_r - ring_w / 2),
                    size - ring_w, size - ring_w
                )
                painter.drawArc(arc_rect, int(start * 16), int(-span * 16))
                start -= span

        # Markaziy matn
        painter.setPen(Qt.PenStyle.NoPen)

        # Katta raqam
        font = QFont()
        font.setPixelSize(int(inner_r * 0.55))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(C('text_primary')))
        painter.drawText(
            QRectF(cx - inner_r, cy - inner_r * 0.4, inner_r * 2, inner_r * 0.6),
            Qt.AlignmentFlag.AlignCenter, self._center_text
        )

        # Kichik label
        font.setPixelSize(int(inner_r * 0.25))
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(C('text_muted')))
        painter.drawText(
            QRectF(cx - inner_r, cy + inner_r * 0.1, inner_r * 2, inner_r * 0.4),
            Qt.AlignmentFlag.AlignCenter, self._center_sub
        )

        painter.end()


# ═══════════════════════════════════════════════════════════════
#  LINE CHART — trend grafik (30 kun yoki 12 oy)
# ═══════════════════════════════════════════════════════════════

class LineChart(HoverTipMixin, QWidget):
    """Gradient to'ldirilgan line chart (nuqta ustida tooltip)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_hover()
        self._data = []  # [{"label": "01", "light": 5, "heavy": 2}, ...]
        self._label_key = "label"
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, data: List[Dict], label_key: str = "label"):
        self._data = data
        self._label_key = label_key
        self.update()

    def _hover_info(self, pos):
        """Kursorga eng yaqin nuqta uchun aniq qiymatlar."""
        i = self._index_from_points(pos, len(self._data))
        if i < 0:
            return -1, None
        d = self._data[i]
        return i, tip_lines(tip_header(d.get(self._label_key, "")),
                            tip_traffic(d.get("light", 0), d.get("heavy", 0)))

    def paintEvent(self, event):
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        n = len(self._data)
        max_val = max((d.get("light", 0) + d.get("heavy", 0)) for d in self._data)
        if max_val == 0:
            max_val = 1

        # Chap chegara — yorliqlarning HAQIQIY kengligidan. Qat'iy 35 px
        # bo'lganda ma'lumot o'sishi bilan uzun son kesilib qolardi
        # ("1511023" → "|511023").
        _f = QFont()
        _f.setPixelSize(9)
        left_m = axis_label_width(
            (int(max_val * i / 3) for i in range(1, 4)), QFontMetrics(_f))
        right_m = 10
        top_m = 10
        bottom_m = 28

        cw = w - left_m - right_m
        ch = h - top_m - bottom_m

        if cw <= 0 or ch <= 0:
            painter.end()
            return

        text_color = QColor(C('text_muted'))
        grid_color = QColor(C('bg_input'))
        light_color = QColor(C('accent_blue'))
        heavy_color = QColor(C('accent_orange'))

        # Background
        bg = QColor(C('bg_panel_dark'))
        painter.fillRect(QRectF(left_m, top_m, cw, ch), QBrush(bg))

        # Grid (3 ta gorizontal chiziq)
        painter.setPen(QPen(grid_color, 1))
        font = QFont()
        font.setPixelSize(9)
        painter.setFont(font)
        for i in range(1, 4):
            y = top_m + ch - (ch * i / 3)
            painter.drawLine(int(left_m), int(y), int(left_m + cw), int(y))
            val = int(max_val * i / 3)
            painter.setPen(QPen(text_color))
            painter.drawText(QRectF(0, y - 6, left_m - 4, 12),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             fmt_compact(val))
            painter.setPen(QPen(grid_color, 1))

        # Bottom line
        painter.drawLine(int(left_m), int(top_m + ch), int(left_m + cw), int(top_m + ch))

        step = cw / max(n - 1, 1) if n > 1 else cw
        # Kursor ostidagi nuqtada vertikal yo'naltiruvchi chiziq
        _hi = getattr(self, "_hover_idx", -1)
        if 0 <= _hi < n:
            _hx = left_m + _hi * step
            _hp = QPen(QColor(C('text_muted')), 1, Qt.PenStyle.DashLine)
            painter.setPen(_hp)
            painter.drawLine(int(_hx), int(top_m), int(_hx), int(top_m + ch))
        self._save_geom(left_m=left_m, top_m=top_m, cw=cw, ch=ch, n=n)

        def get_points(key):
            pts = []
            for i, d in enumerate(self._data):
                x = left_m + i * step
                v = d.get(key, 0)
                y = top_m + ch - (v / max_val) * ch
                pts.append(QPointF(x, y))
            return pts

        def draw_filled_line(points, color):
            if len(points) < 2:
                return
            # Gradient fill
            path = QPainterPath()
            path.moveTo(QPointF(points[0].x(), top_m + ch))
            for p in points:
                path.lineTo(p)
            path.lineTo(QPointF(points[-1].x(), top_m + ch))
            path.closeSubpath()

            grad = QLinearGradient(0, top_m, 0, top_m + ch)
            fill_color = QColor(color)
            fill_color.setAlpha(60)
            grad.setColorAt(0, fill_color)
            fill_color.setAlpha(5)
            grad.setColorAt(1, fill_color)
            painter.fillPath(path, QBrush(grad))

            # Line
            pen = QPen(QColor(color), 2)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])

            # Dots
            painter.setBrush(QBrush(QColor(color)))
            painter.setPen(Qt.PenStyle.NoPen)
            for p in points:
                painter.drawEllipse(p, 3, 3)

        # Heavy (pastda, orange)
        heavy_pts = get_points("heavy")
        draw_filled_line(heavy_pts, C('accent_orange'))

        # Light (ustida, blue)
        light_pts = get_points("light")
        draw_filled_line(light_pts, C('accent_blue'))

        # Labels (har n-tasi)
        painter.setPen(QPen(text_color))
        font.setPixelSize(8)
        painter.setFont(font)
        label_skip = max(1, n // 10)
        for i, d in enumerate(self._data):
            if i % label_skip == 0 or i == n - 1:
                x = left_m + i * step
                lbl = str(d.get(self._label_key, ""))
                painter.drawText(
                    QRectF(x - 15, top_m + ch + 3, 30, bottom_m - 3),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    lbl
                )

        painter.end()


# ═══════════════════════════════════════════════════════════════
#  BAR CHART — haftalik / oylik stacked bar chart
# ═══════════════════════════════════════════════════════════════

class BarChart(HoverTipMixin, QWidget):
    """Rounded stacked bar chart with labels"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_hover()
        self._data = []  # [{"label": "Du", "light": 5, "heavy": 2}, ...]
        self._label_key = "label"
        self._show_values = True
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _hover_info(self, pos):
        """Kursor ostidagi ustun uchun aniq qiymatlar."""
        i = self._index_from_bars(pos, len(self._data))
        if i < 0:
            return -1, None
        d = self._data[i]
        return i, tip_lines(tip_header(d.get(self._label_key, "")),
                            tip_traffic(d.get("light", 0), d.get("heavy", 0)))

    def set_data(self, data: List[Dict], label_key: str = "label", show_values: bool = True):
        self._data = data
        self._label_key = label_key
        self._show_values = show_values
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        n = len(self._data)
        max_val = max((d.get("light", 0) + d.get("heavy", 0)) for d in self._data)
        if max_val == 0:
            max_val = 1

        # Chap chegara yorliq kengligidan hisoblanadi (uzun son kesilmasin)
        _f = QFont()
        _f.setPixelSize(9)
        left_m = axis_label_width(
            (int(max_val * i / 3) for i in range(1, 4)), QFontMetrics(_f))
        right_m = 10
        top_m = 18
        bottom_m = 28

        cw = w - left_m - right_m
        ch = h - top_m - bottom_m

        if cw <= 0 or ch <= 0:
            painter.end()
            return

        text_color = QColor(C('text_muted'))
        grid_color = QColor(C('bg_input'))
        light_color = QColor(C('accent_blue'))
        heavy_color = QColor(C('accent_orange'))
        bg_color = QColor(C('bg_panel_dark'))

        # Background
        painter.fillRect(QRectF(left_m, top_m, cw, ch), QBrush(bg_color))

        # Grid
        painter.setPen(QPen(grid_color, 1))
        font = QFont()
        font.setPixelSize(9)
        painter.setFont(font)
        for i in range(1, 4):
            y = top_m + ch - (ch * i / 3)
            painter.drawLine(int(left_m), int(y), int(left_m + cw), int(y))
            val = int(max_val * i / 3)
            painter.setPen(QPen(text_color))
            painter.drawText(QRectF(0, y - 6, left_m - 4, 12),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             fmt_compact(val))
            painter.setPen(QPen(grid_color, 1))

        # Bottom line
        painter.drawLine(int(left_m), int(top_m + ch), int(left_m + cw), int(top_m + ch))

        bar_w = cw / n
        self._save_geom(left_m=left_m, top_m=top_m, cw=cw, ch=ch, n=n)
        gap = max(2, bar_w * 0.2)
        radius = min(4, (bar_w - gap) / 4)

        for i, d in enumerate(self._data):
            x = left_m + i * bar_w
            light = d.get("light", 0)
            heavy = d.get("heavy", 0)
            total = light + heavy

            # Kursor ostidagi ustunni ajratib ko'rsatish (tooltip qaysi
            # ustunga tegishli ekani aniq bo'lsin)
            if i == getattr(self, "_hover_idx", -1):
                hl = QColor(C('bg_hover'))
                hl.setAlpha(90)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(hl))
                painter.drawRect(QRectF(x, top_m, bar_w, ch))

            if total > 0:
                total_h = (total / max_val) * ch
                light_h = (light / max_val) * ch
                heavy_h = (heavy / max_val) * ch

                bar_x = x + gap / 2
                actual_w = bar_w - gap

                # Heavy (pastda) — to'g'ri burchakli
                if heavy > 0:
                    y_start = top_m + ch - total_h
                    painter.setBrush(QBrush(heavy_color))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(QRectF(bar_x, y_start, actual_w, heavy_h))

                # Light (tepada) — yuqori burchaklari yumaloq
                if light > 0:
                    y_light = top_m + ch - light_h
                    painter.setBrush(QBrush(light_color))
                    painter.setPen(Qt.PenStyle.NoPen)
                    path = QPainterPath()
                    path.addRoundedRect(QRectF(bar_x, y_light, actual_w, light_h), radius, radius)
                    painter.drawPath(path)

                # Qiymat (bar ustida) — ustun kengligiga SIG'ADIGAN ko'rinishda
                # ("1 511 023" sig'masa "1.5M"), aks holda son kesilardi
                if self._show_values and total > 0 and n <= 12:
                    font.setPixelSize(8)
                    painter.setFont(font)
                    painter.setPen(QPen(text_color))
                    y_top = top_m + ch - total_h
                    _fm = QFontMetrics(font)
                    _txt = fmt_fit(total, actual_w - 2, _fm)
                    if fits(_txt, actual_w - 2, _fm):
                        painter.drawText(
                            QRectF(bar_x, y_top - 14, actual_w, 12),
                            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                            _txt
                        )

            # Label
            painter.setPen(QPen(text_color))
            font.setPixelSize(9)
            painter.setFont(font)
            lbl = str(d.get(self._label_key, ""))
            painter.drawText(
                QRectF(x, top_m + ch + 3, bar_w, bottom_m - 3),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                lbl
            )

        painter.end()


# ═══════════════════════════════════════════════════════════════
#  MINI SPARKLINE — kichik inline grafik
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  TRAIN BAR CHART — poyezd soni bar chart (bitta qiymat)
# ═══════════════════════════════════════════════════════════════

class TrainBarChart(HoverTipMixin, QWidget):
    """Poyezd soni uchun bar chart (bitta "count" qiymat)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_hover()
        self._data = []  # [{"label": "Du", "count": 3, "avg": 65.0}, ...]
        self._label_key = "label"
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _hover_info(self, pos):
        """Kursor ostidagi kun uchun poyezd soni va o'rtacha davomiylik."""
        i = self._index_from_bars(pos, len(self._data))
        if i < 0:
            return -1, None
        d = self._data[i]
        avg = d.get("avg", 0) or 0
        return i, tip_lines(
            tip_header(d.get(self._label_key, "")),
            t("tip.trains_count", count=fmt_full(d.get("count", 0))),
            t("tip.trains_avg", sec=int(avg)) if avg else None)

    def set_data(self, data: List[Dict], label_key: str = "label"):
        self._data = data
        self._label_key = label_key
        self.update()

    def paintEvent(self, event):
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        n = len(self._data)
        max_val = max(d.get("count", 0) for d in self._data)
        if max_val == 0:
            max_val = 1

        # Chap chegara yorliq kengligidan (uzun son kesilmasin)
        _f = QFont()
        _f.setPixelSize(9)
        left_m = axis_label_width(
            (int(max_val * i / 3) for i in range(1, 4)), QFontMetrics(_f))
        right_m = 10
        top_m = 18
        bottom_m = 28

        cw = w - left_m - right_m
        ch = h - top_m - bottom_m

        if cw <= 0 or ch <= 0:
            painter.end()
            return

        text_color = QColor(C('text_muted'))
        grid_color = QColor(C('bg_input'))
        bar_color = QColor(C('accent_teal'))
        bg_color = QColor(C('bg_panel_dark'))

        # Background
        painter.fillRect(QRectF(left_m, top_m, cw, ch), QBrush(bg_color))

        # Grid
        painter.setPen(QPen(grid_color, 1))
        font = QFont()
        font.setPixelSize(9)
        painter.setFont(font)
        for i in range(1, 4):
            y = top_m + ch - (ch * i / 3)
            painter.drawLine(int(left_m), int(y), int(left_m + cw), int(y))
            val = int(max_val * i / 3)
            painter.setPen(QPen(text_color))
            painter.drawText(QRectF(0, y - 6, left_m - 4, 12),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             fmt_compact(val))
            painter.setPen(QPen(grid_color, 1))

        painter.drawLine(int(left_m), int(top_m + ch), int(left_m + cw), int(top_m + ch))

        bar_w = cw / n
        self._save_geom(left_m=left_m, top_m=top_m, cw=cw, ch=ch, n=n)
        gap = max(2, bar_w * 0.25)
        radius = min(4, (bar_w - gap) / 4)

        for i, d in enumerate(self._data):
            x = left_m + i * bar_w
            count = d.get("count", 0)

            if i == getattr(self, "_hover_idx", -1):
                hl = QColor(C('bg_hover'))
                hl.setAlpha(90)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(hl))
                painter.drawRect(QRectF(x, top_m, bar_w, ch))

            if count > 0:
                bar_h = (count / max_val) * ch
                bar_x = x + gap / 2
                actual_w = bar_w - gap
                y_start = top_m + ch - bar_h

                # Gradient effect
                grad = QLinearGradient(bar_x, y_start, bar_x, top_m + ch)
                c1 = QColor(bar_color)
                c1.setAlpha(220)
                grad.setColorAt(0, c1)
                c2 = QColor(bar_color)
                c2.setAlpha(140)
                grad.setColorAt(1, c2)

                painter.setBrush(QBrush(grad))
                painter.setPen(Qt.PenStyle.NoPen)
                path = QPainterPath()
                path.addRoundedRect(QRectF(bar_x, y_start, actual_w, bar_h), radius, radius)
                painter.drawPath(path)

                # Value on top
                if n <= 12:
                    font.setPixelSize(9)
                    font.setBold(True)
                    painter.setFont(font)
                    painter.setPen(QPen(bar_color))
                    _fm = QFontMetrics(font)
                    _txt = fmt_fit(count, actual_w - 2, _fm)
                    if fits(_txt, actual_w - 2, _fm):
                        painter.drawText(
                            QRectF(bar_x, y_start - 16, actual_w, 14),
                            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                            _txt
                        )
                    font.setBold(False)

            # Label
            painter.setPen(QPen(text_color))
            font.setPixelSize(9)
            painter.setFont(font)
            lbl = str(d.get(self._label_key, ""))
            painter.drawText(
                QRectF(x, top_m + ch + 3, bar_w, bottom_m - 3),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                lbl
            )

        painter.end()


class SparkLine(QWidget):
    """Kichik inline trend chiziq"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._values = []
        self._color = C('accent_blue')
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, values: List[int], color: str = None):
        self._values = values
        if color:
            self._color = color
        self.update()

    def paintEvent(self, event):
        if len(self._values) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        pad = 2

        max_v = max(self._values) or 1
        n = len(self._values)
        step = (w - pad * 2) / (n - 1)

        color = QColor(self._color)

        # Gradient fill
        path = QPainterPath()
        points = []
        for i, v in enumerate(self._values):
            x = pad + i * step
            y = pad + (h - pad * 2) * (1 - v / max_v)
            points.append(QPointF(x, y))

        path.moveTo(QPointF(points[0].x(), h - pad))
        for p in points:
            path.lineTo(p)
        path.lineTo(QPointF(points[-1].x(), h - pad))
        path.closeSubpath()

        grad = QLinearGradient(0, 0, 0, h)
        fc = QColor(color)
        fc.setAlpha(40)
        grad.setColorAt(0, fc)
        fc.setAlpha(5)
        grad.setColorAt(1, fc)
        painter.fillPath(path, QBrush(grad))

        # Line
        pen = QPen(color, 1.5)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

        painter.end()
