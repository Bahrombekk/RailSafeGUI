"""
Diagrammalar uchun umumiy "sichqoncha izohi" (tooltip) mexanizmi.

Heatmapdagi kabi: kursor ustun/nuqta ustiga borganda o'sha kun (yoki soat)
uchun ANIQ qiymatlar chiqadi. Bu grafikdagi mayda yozuvlarga bo'lgan
ehtiyojni kamaytiradi — ekranda faqat rang va shakl qoladi, aniq sonlar
esa so'ralganda ko'rsatiladi.

Ishlatish:
    class BarChart(HoverTipMixin, QWidget):
        def __init__(...):
            super().__init__(parent)
            self._init_hover()

        def _hover_info(self, pos):
            # (indeks, HTML matn) yoki (-1, None)
            ...

        def paintEvent(self, e):
            ...
            # oxirida geometriyani saqlaymiz — hit-test chizish bilan
            # BIR XIL hisobdan foydalanishi uchun
            self._save_geom(left_m=..., top_m=..., cw=..., ch=..., n=...)
"""

from PyQt6.QtWidgets import QToolTip
from app.utils.numfmt import fmt_full
from app.utils.language import t


def tip_lines(*lines) -> str:
    """Tooltip matnini yig'ish (bo'sh qatorlar tashlanadi)."""
    return "<br>".join(str(x) for x in lines if x)


def tip_header(text) -> str:
    return f"<b>{text}</b>"


def tip_traffic(light: int, heavy: int) -> str:
    """Yengil / og'ir / jami qatorlari — barcha diagrammalarda bir xil."""
    return tip_lines(
        t("stats.light_fmt", light=fmt_full(light)),
        t("stats.heavy_fmt", heavy=fmt_full(heavy)),
        t("stats.total_fmt", total=fmt_full(light + heavy)),
    )


class HoverTipMixin:
    """Diagramma ustida kursor harakatini kuzatib tooltip ko'rsatadi.

    Bola klass `_hover_info(pos) -> (index, text)` ni amalga oshiradi.
    `self._hover_idx` chizishda ishlatilishi mumkin (ustunni ajratib ko'rsatish).
    """

    def _init_hover(self):
        self._hover_idx = -1
        self._geom = None
        self.setMouseTracking(True)

    def _save_geom(self, **kw):
        """paintEvent oxirida chaqiriladi — hit-test uchun geometriya."""
        self._geom = kw

    def _hover_info(self, pos):        # bola klass qayta yozadi
        return -1, None

    def mouseMoveEvent(self, event):
        try:
            idx, text = self._hover_info(event.position())
        except Exception:
            idx, text = -1, None
        if idx != getattr(self, "_hover_idx", -1):
            self._hover_idx = idx
            self.update()
        if text:
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        else:
            QToolTip.hideText()

    def leaveEvent(self, event):
        if getattr(self, "_hover_idx", -1) != -1:
            self._hover_idx = -1
            self.update()
        QToolTip.hideText()

    # ─── Yordamchi hit-testlar ────────────────────────────────

    def _index_from_bars(self, pos, count):
        """Teng kenglikdagi ustunlar uchun: kursor qaysi ustunda."""
        g = self._geom
        if not g or count <= 0:
            return -1
        x = pos.x() - g["left_m"]
        y = pos.y() - g["top_m"]
        if x < 0 or x > g["cw"] or y < 0 or y > g["ch"]:
            return -1
        i = int(x // (g["cw"] / count))
        return i if 0 <= i < count else -1

    def _index_from_points(self, pos, count):
        """Chiziqli grafik uchun: kursorga eng YAQIN nuqta."""
        g = self._geom
        if not g or count <= 0:
            return -1
        x = pos.x() - g["left_m"]
        y = pos.y() - g["top_m"]
        if x < -10 or x > g["cw"] + 10 or y < 0 or y > g["ch"]:
            return -1
        if count == 1:
            return 0
        step = g["cw"] / (count - 1)
        i = int(round(x / step))
        return min(max(i, 0), count - 1)
