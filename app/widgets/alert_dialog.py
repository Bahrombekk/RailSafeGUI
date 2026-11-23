"""
AlertDialog — ilova uslubidagi ogohlantirish / xabar oynasi.

Standart QMessageBox tizim ramkasi va oddiy tugmalar bilan chiqadi, ilova
mavzusiga (dark / military / light) mos kelmaydi. Bu oyna:
  * ramkasiz, yumaloq burchakli karta + soya;
  * chap tomonda turi bo'yicha rangli ikonka nishoni (warning/success/info);
  * sarlavha, ixtiyoriy "chip" (masalan pereezd nomi), asosiy matn va
    rangli chiziqli izoh bloki;
  * asosiy (to'ldirilgan) va ikkilamchi (kontur) tugmalar;
  * sarlavha qismidan sudrab ko'chirish, ✕ / Esc — rad etish.

Bloklamaydi: open() bilan ochiladi, natija QDialog.finished(int) orqali
(Accepted — asosiy tugma, Rejected — ikkilamchi / ✕ / Esc).
"""

from PyQt6.QtWidgets import (QDialog, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QWidget, QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QPainterPath, QBrush

from app.utils.theme_colors import C, contrast_on

_KIND_COLOR = {
    "warning": "status_warning",
    "success": "status_online",
    "error": "status_error",
    "info": "accent_brand",
}


def _alpha(hex_color: str, a: int) -> str:
    c = QColor(hex_color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {a})"


class _KindIcon(QWidget):
    """Doira ichida chizilgan belgi: ogohlantirish uchburchagi / galochka / i."""

    def __init__(self, kind: str, color: str, size: int = 44, parent=None):
        super().__init__(parent)
        self._kind = kind
        self._color = QColor(color)
        self.setFixedSize(size, size)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = float(self.width())
        bg = QColor(self._color)
        bg.setAlpha(38)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawEllipse(QRectF(0, 0, s, s))
        ring = QColor(self._color)
        ring.setAlpha(90)
        p.setPen(QPen(ring, 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(0.6, 0.6, s - 1.2, s - 1.2))

        c = s / 2
        pen = QPen(self._color, s * 0.075, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        if self._kind == "warning":
            path = QPainterPath()
            path.moveTo(c, s * 0.27)
            path.lineTo(s * 0.73, s * 0.70)
            path.lineTo(s * 0.27, s * 0.70)
            path.closeSubpath()
            p.drawPath(path)
            p.drawLine(QPointF(c, s * 0.42), QPointF(c, s * 0.54))
            p.setBrush(QBrush(self._color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(c, s * 0.62), s * 0.035, s * 0.035)
        elif self._kind == "success":
            path = QPainterPath()
            path.moveTo(s * 0.32, s * 0.51)
            path.lineTo(s * 0.45, s * 0.63)
            path.lineTo(s * 0.69, s * 0.38)
            p.drawPath(path)
        else:   # info / error
            p.drawLine(QPointF(c, s * 0.46), QPointF(c, s * 0.68))
            p.setBrush(QBrush(self._color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(c, s * 0.33), s * 0.04, s * 0.04)
        p.end()


class AlertDialog(QDialog):
    def __init__(self, parent=None, *, kind: str = "warning", title: str = "",
                 chip: str = "", text: str = "", note: str = "",
                 primary: str = "OK", secondary: str = None, width: int = 500):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowTitle(title)
        self._drag = None

        accent = C(_KIND_COLOR.get(kind, "accent_brand"))
        shadow_m = 18   # soya uchun bo'sh joy

        outer = QVBoxLayout(self)
        outer.setContentsMargins(shadow_m, shadow_m, shadow_m, shadow_m)

        card = QFrame(self)
        card.setObjectName("alertCard")
        card.setFixedWidth(width)
        card.setStyleSheet(f"""
            QFrame#alertCard {{
                background: {C('bg_card')};
                border: 1px solid {C('border_card')};
                border-top: 3px solid {accent};
                border-radius: 12px;
            }}
            QLabel {{ background: transparent; border: none; }}
        """)
        sh = QGraphicsDropShadowEffect(card)
        sh.setBlurRadius(36)
        sh.setOffset(0, 8)
        sh.setColor(QColor(0, 0, 0, 150))
        card.setGraphicsEffect(sh)
        outer.addWidget(card)

        root = QVBoxLayout(card)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        # ── Sarlavha qatori: ikonka + (sarlavha, chip) + ✕
        head = QHBoxLayout()
        head.setSpacing(14)
        head.addWidget(_KindIcon(kind, accent), 0, Qt.AlignmentFlag.AlignTop)

        titles = QVBoxLayout()
        titles.setSpacing(6)
        t_lbl = QLabel(title)
        t_lbl.setWordWrap(True)
        t_lbl.setStyleSheet(
            f"color: {C('text_primary')}; font-size: 16px; font-weight: 700;")
        titles.addWidget(t_lbl)
        if chip:
            chip_row = QHBoxLayout()
            chip_row.setContentsMargins(0, 0, 0, 0)
            chip_lbl = QLabel(chip)
            chip_lbl.setStyleSheet(f"""
                color: {accent};
                background: {_alpha(accent, 30)};
                border: 1px solid {_alpha(accent, 80)};
                border-radius: 9px;
                padding: 2px 10px;
                font-size: 11px; font-weight: 600;
            """)
            chip_row.addWidget(chip_lbl)
            chip_row.addStretch()
            titles.addLayout(chip_row)
        head.addLayout(titles, 1)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; color: {C('text_muted')};
                           border: none; border-radius: 6px; font-size: 13px; }}
            QPushButton:hover {{ background: {C('bg_hover')}; color: {C('text_primary')}; }}
        """)
        close_btn.clicked.connect(self.reject)
        head.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(head)

        # ── Asosiy matn
        if text:
            body = QLabel(text)
            body.setWordWrap(True)
            body.setStyleSheet(
                f"color: {C('text_secondary')}; font-size: 13px; line-height: 140%;")
            body.setContentsMargins(58, 0, 0, 0)
            root.addWidget(body)

        # ── Izoh bloki (rangli chap chiziq)
        if note:
            note_box = QFrame()
            note_box.setObjectName("alertNote")
            note_box.setStyleSheet(f"""
                QFrame#alertNote {{
                    background: {_alpha(accent, 22)};
                    border: none;
                    border-left: 3px solid {accent};
                    border-radius: 6px;
                }}
            """)
            nl = QVBoxLayout(note_box)
            nl.setContentsMargins(12, 9, 12, 9)
            n_lbl = QLabel(note)
            n_lbl.setWordWrap(True)
            n_lbl.setStyleSheet(f"color: {C('text_secondary')}; font-size: 12px;")
            nl.addWidget(n_lbl)
            wrap = QHBoxLayout()
            wrap.setContentsMargins(58, 0, 0, 0)
            wrap.addWidget(note_box)
            root.addLayout(wrap)

        # ── Tugmalar
        btns = QHBoxLayout()
        btns.setSpacing(10)
        btns.addStretch()
        if secondary:
            sec = QPushButton(secondary)
            sec.setCursor(Qt.CursorShape.PointingHandCursor)
            sec.setMinimumHeight(36)
            sec.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; color: {C('text_secondary')};
                    border: 1px solid {C('border_card')}; border-radius: 8px;
                    padding: 0 18px; font-size: 13px; font-weight: 600;
                }}
                QPushButton:hover {{ background: {C('bg_hover')}; color: {C('text_primary')}; }}
            """)
            sec.clicked.connect(self.reject)
            btns.addWidget(sec)
        prim = QPushButton(primary)
        prim.setCursor(Qt.CursorShape.PointingHandCursor)
        prim.setMinimumHeight(36)
        prim.setDefault(True)
        on = contrast_on(accent).name()
        hover = QColor(accent).lighter(112).name()
        prim.setStyleSheet(f"""
            QPushButton {{
                background: {accent}; color: {on};
                border: none; border-radius: 8px;
                padding: 0 20px; font-size: 13px; font-weight: 700;
            }}
            QPushButton:hover {{ background: {hover}; }}
        """)
        prim.clicked.connect(self.accept)
        btns.addWidget(prim)
        root.addSpacing(2)
        root.addLayout(btns)

        self.adjustSize()

    # Ota oyna markaziga joylash
    def showEvent(self, e):
        super().showEvent(e)
        par = self.parentWidget()
        if par is not None:
            g = par.window().frameGeometry()
            self.move(g.center() - QPoint(self.width() // 2, self.height() // 2))

    # Sudrab ko'chirish (ramka yo'q)
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, e):
        self._drag = None
