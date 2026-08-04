"""
HeatmapChart — 7 kun x 24 soat issiqlik xaritasi.
QPainter bilan chizilgan, tema ranglariga mos.

Usage:
    heatmap = HeatmapChart()
    data = stats_db.get_heatmap_data(crossing_id)
    heatmap.set_data(data)
"""

from PyQt6.QtWidgets import QWidget, QSizePolicy, QToolTip
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import (QPainter, QColor, QFont, QFontMetrics, QPen,
                          QBrush, QPainterPath, QLinearGradient)
from typing import List, Dict

from app.utils.theme_colors import C, TM, get_theme, contrast_on
from app.utils.language import t
from app.utils.numfmt import fmt_fit, fmt_full, fits


# ОДМ 218.2.020-2012 xizmat darajalari: yuklanish koeffitsienti
# z = soatlik oqim / amaliy sig'im (P). Har daraja uchun (z yuqori chegara,
# daraja belgisi, zaxira rang RGB).
# DIQQAT: ranglar MAVZUGA qarab o'zgaradi — `level_rgb()` ishlatilsin.
# Bu ro'yxatdagi RGB faqat zaxira (dark mavzu qiymatlari).
ODM_LEVELS = [
    (0.20, "A", (31, 107, 70)),
    (0.45, "B", (63, 157, 92)),
    (0.70, "C", (201, 162, 39)),
    (0.90, "D", (224, 123, 57)),
    (1.00, "E", (210, 75, 62)),
    (float("inf"), "F", (142, 32, 54)),
]

# Har mavzu uchun A→F shkalasi. Yorqinlik ketma-ket o'sib boradi, shu sabab
# rangni ajratib olmaydigan (rang ko'rlik) foydalanuvchi ham darajalarni
# to'yinganlik/yorqinlik farqi bilan ajratadi.
LEVEL_COLORS = {
    # Qoraygan ko'k fonda (bg_card #1a1a2e) — to'yingan, lekin ko'zni
    # qamashtirmaydigan qiymatlar
    "dark": (
        (31, 107, 70), (63, 157, 92), (201, 162, 39),
        (224, 123, 57), (210, 75, 62), (142, 32, 54),
    ),
    # Harbiy (zaytun) mavzu — palitra shu oilada qoladi, aks holda xarita
    # interfeysdan ajralib, "yopishtirilgan" ko'rinardi
    "military": (
        (74, 106, 52), (109, 148, 64), (184, 160, 72),
        (200, 133, 60), (184, 72, 47), (125, 31, 36),
    ),
    # Oq fon — ranglar to'qroq bo'lishi kerak, aks holda katak fon bilan
    # qo'shilib ketadi
    "light": (
        (34, 139, 87), (86, 179, 105), (222, 170, 30),
        (233, 128, 34), (214, 69, 55), (150, 28, 48),
    ),
}


def level_rgb(index: int) -> tuple:
    """Daraja rangi (0=A..5=F) — JORIY MAVZUGA moslashadi."""
    pal = LEVEL_COLORS.get(get_theme(), LEVEL_COLORS["dark"])
    return pal[max(0, min(index, len(pal) - 1))]


def level_qcolor(index: int) -> QColor:
    return QColor(*level_rgb(index))


def contrast_text(bg: QColor) -> QColor:
    """Fon rangiga qarab o'qiladigan matn rangi. Sariq/yashil kataklarda oq
    matn zaif ko'rinadi — ular uchun to'q matn olinadi.
    (Umumiy yordamchi: app/utils/theme_colors.contrast_on)"""
    return contrast_on(bg)


def odm_level_index(z: float) -> int:
    """z bo'yicha ОДМ darajasining indeksi (0 = A ... 5 = F)."""
    for i, (upper, _key, _rgb) in enumerate(ODM_LEVELS):
        if z <= upper:
            return i
    return len(ODM_LEVELS) - 1


def odm_level(z: float):
    """z bo'yicha ОДМ darajasi. Returns (belgi, (r, g, b)) — rang mavzuga mos."""
    i = odm_level_index(z)
    return ODM_LEVELS[i][1], level_rgb(i)


# Turish vaqti bo'yicha daraja: ogohlantirish chegarasiga (W, sozlamalarda
# "warning_threshold") nisbatan koeffitsientlar. W = 10 s bo'lsa:
#   A < 4s, B < 7s, C < 10s, D < 15s, E < 30s, F >= 30s
#
# NEGA turish vaqti: probkani oqim (soatda o'tgan mashina) ham, zona band
# vaqti ham ko'rsatmaydi — probka boshlanganda oqim KAMAYADI, band vaqt esa
# gavjum-lekin-oqadigan holatda ham to'yingan bo'ladi. Bitta mashinaning
# zonada turgan vaqti esa erkin oqimda bir necha sekund, probkada bir necha
# barobar ko'p — ya'ni ikki holatni aniq ajratadi.
DWELL_LEVEL_FACTORS = (0.4, 0.7, 1.0, 1.5, 3.0)


def dwell_level_index(dwell_seconds: float, warn_seconds: float) -> int:
    """O'rtacha turish vaqti bo'yicha daraja indeksi. Chegara noma'lum
    (warn_seconds <= 0) bo'lsa -1 (baholanmaydi)."""
    if warn_seconds <= 0 or dwell_seconds <= 0:
        return -1
    ratio = dwell_seconds / warn_seconds
    for i, factor in enumerate(DWELL_LEVEL_FACTORS):
        if ratio < factor:
            return i
    return len(ODM_LEVELS) - 1


class HeatmapChart(QWidget):
    """7 kun x 24 soat issiqlik xaritasi (heatmap).

    Ikki rang rejimi:
      - capacity berilmagan: eski nisbiy shkala (haftalik maksimumga nisbatan)
      - set_capacity(P) chaqirilgan: ОДМ z-shkala (mutlaq, A-F darajalar)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = []  # [{"day": "Du", "hours": [0]*24}, ...]
        self._capacity = 0.0  # amaliy sig'im P (0 = nisbiy rejim)
        self._scale_max = 0.0  # nisbiy rejimda qo'lda berilgan yuqori chegara
        # Asosiy qiymatlar turish vaqti (sekund) bo'lsa — shu chegara bilan
        # baholanadi (0 = asosiy qiymat turish vaqti emas)
        self._dwell_warn = 0.0
        # Ikkinchi signal: soatlik o'rtacha turish vaqti {sana: {soat: sekund}}
        self._dwell_map = {}
        self._dwell_map_warn = 0.0
        # Zonadan foydalanish (band daq/soat) — faqat tooltip uchun
        self._util_map = {}
        self._hover = None          # (row, col) — takroriy tooltipni oldini oladi
        self.setMouseTracking(True)  # tooltip uchun harakatni kuzatish
        self.setMinimumHeight(160)
        # Mavzu almashtirilsa ranglar darhol yangilanadi (dark/military/light
        # uchun alohida A-F palitrasi bor — izoh: LEVEL_COLORS)
        try:
            TM.theme_changed.connect(self._on_theme_changed)
        except Exception:
            pass
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, data: List[Dict]):
        """data: [{"day": "Du", "hours": [h0, h1, ..., h23]}, ...] (7 ta)"""
        self._data = data
        self.update()

    def set_capacity(self, capacity: float):
        """Amaliy sig'im P ni o'rnatish — ОДМ z-shkala rejimiga o'tkazadi.
        0 yoki manfiy qiymat nisbiy rejimga qaytaradi."""
        self._capacity = max(0.0, float(capacity or 0))
        self.update()

    def set_scale_max(self, scale_max: float):
        """Nisbiy rejim uchun MUTLAQ yuqori chegara (masalan soat = 60 daqiqa).
        Aks holda gradient haftalik maksimumga nisbatan bo'lib, kam bandlikli
        hafta ham 'to'la qizil' ko'rinardi."""
        self._scale_max = max(0.0, float(scale_max or 0))
        self.update()

    def set_dwell_scale(self, warn_seconds: float):
        """Asosiy qiymatlar — soatlik O'RTACHA TURISH VAQTI (sekund). Ranglar
        `dwell_level_index` bo'yicha A-F. warn_seconds = sozlamalardagi
        ogohlantirish chegarasi."""
        self._dwell_warn = max(0.0, float(warn_seconds or 0))
        self.update()

    def set_congestion(self, data: List[Dict], warn_seconds: float):
        """IKKINCHI signal: soatlik o'rtacha turish vaqti. Katak darajasi
        oqim va turish vaqtining YOMONI bo'yicha olinadi.

        NEGA KERAK: asosiy qiymat — soatda o'tgan mashina soni (oqim). Probka
        boshlanganda oqim KAMAYADI, to'liq to'xtashda 0 ga tushadi — ya'ni
        faqat oqim bo'yicha baholansa eng og'ir zator "A — erkin oqim" (yashil)
        bo'lib ko'rinadi. Turish vaqti esa aynan teskari harakat qiladi.

        Args:
            data: get_dwell_heatmap() natijasi (hours = o'rtacha sekund)
            warn_seconds: ogohlantirish chegarasi (sozlamalardan)
        """
        self._dwell_map = {}
        for row in (data or []):
            ds = row.get("date")
            if not ds:
                continue
            self._dwell_map[ds] = {
                h: v for h, v in enumerate(row.get("hours", [])) if v}
        self._dwell_map_warn = max(0.0, float(warn_seconds or 0))
        self.update()

    def set_utilization(self, data: List[Dict]):
        """Zonadan foydalanish (band daqiqa/soat) — FAQAT tooltip uchun.
        Rangga ta'sir qilmaydi: bu probka o'lchovi emas (gavjum lekin erkin
        oqadigan soatlarda ham zona deyarli doim band bo'ladi)."""
        self._util_map = {}
        for row in (data or []):
            ds = row.get("date")
            if not ds:
                continue
            self._util_map[ds] = {
                h: v for h, v in enumerate(row.get("hours", [])) if v}
        self.update()

    def _dwell_at(self, date_str: str, hour: int) -> float:
        """Katak uchun o'rtacha turish vaqti (ma'lumot yo'q bo'lsa 0)."""
        if not self._dwell_map or not date_str:
            return 0.0
        return float(self._dwell_map.get(date_str, {}).get(hour, 0))

    def _cell_at(self, pos):
        """Sichqoncha koordinatasi ostidagi katak: (row, col) yoki None."""
        lay = self._layout()
        if lay is None:
            return None
        cell_w, cell_h, rows, cols = lay
        x = pos.x() - self._LEFT_M
        y = pos.y() - self._TOP_M
        if x < 0 or y < 0:
            return None
        col = int(x // cell_w)
        row = int(y // cell_h)
        if 0 <= row < rows and 0 <= col < cols:
            return row, col
        return None

    def _on_theme_changed(self, *_a):
        """Mavzu almashdi — palitra boshqa, qayta chizamiz."""
        try:
            self.update()
        except Exception:
            pass

    def mouseMoveEvent(self, event):
        """Katak ustida turganda uchala qiymatni ko'rsatish: o'tgan mashina,
        o'rtacha turish vaqti, zona band vaqti va daraja. Uchta alohida
        xarita o'rniga bitta xarita + tooltip."""
        cell = self._cell_at(event.position())
        if cell is None:
            QToolTip.hideText()
            if self._hover is not None:
                self._hover = None
                self.update()
            return
        if cell == self._hover:
            return
        self._hover = cell
        self.update()   # tanlangan katak ramkasi
        row, col = cell
        d = self._data[row]
        hours = d.get("hours", [])
        value = hours[col] if col < len(hours) else 0
        ds = d.get("date", "")
        dwell = self._dwell_at(ds, col)
        util = self._util_map.get(ds, {}).get(col, 0) if self._util_map else 0

        # Sana oddiy ko'rinishda: "Se, 04.08.2026 08:00"
        ds_h = ds
        try:
            y, m, dd = ds.split("-")
            ds_h = f"{dd}.{m}.{y}"
        except Exception:
            pass
        lines = [f"<b>{d.get('day', '')}, {ds_h} {col:02d}:00</b>"]
        if self._dwell_warn > 0:
            # Bu xaritada asosiy qiymat — turish vaqti
            lines.append(t("heatmap.tip_dwell", sec=int(value)))
        else:
            lines.append(t("heatmap.tip_flow", count=int(value)))
            if dwell > 0:
                lines.append(t("heatmap.tip_dwell", sec=int(dwell)))
        if util:
            lines.append(t("heatmap.tip_util", min=int(util)))

        lvl = self._cell_level(value, dwell)
        if lvl >= 0:
            # Harf emas, SODDA nom ko'rsatiladi ("To'xtab-to'xtab (E)") —
            # legendadagi bilan bir xil matn, bitta manba.
            lines.append(t("heatmap.tip_level",
                           level=t(f"legend.los_{ODM_LEVELS[lvl][1].lower()}")))
            if self._is_congestion_driven(value, dwell):
                lines.append(t("heatmap.tip_from_dwell"))

        QToolTip.showText(event.globalPosition().toPoint(),
                          "<br>".join(lines), self)

    def leaveEvent(self, event):
        if self._hover is not None:
            self._hover = None
            self.update()
        QToolTip.hideText()

    def _is_congestion_driven(self, value: int, dwell: float) -> bool:
        """Katak rangi oqimdan emas, turish vaqtidan kelib chiqdimi — shunday
        kataklarda kichik nuqta chiziladi ("son kichik, rang qizil" = zator)."""
        d_idx = dwell_level_index(dwell, self._dwell_map_warn)
        if d_idx < 0:
            return False
        if value == 0:
            return d_idx > 0
        if self._capacity > 0:
            return d_idx > odm_level_index(value / self._capacity)
        return d_idx >= 3

    def _cell_level(self, value: int, dwell: float = 0.0) -> int:
        """Katakning A-F darajasi (0..5) yoki -1 — daraja qo'llanmaydi
        (nisbiy rejim yoki ma'lumot yo'q). Rang ham, tooltip ham shu bittadan
        foydalanadi, shuning uchun ular hech qachon bir-biriga qarama-qarshi
        bo'lmaydi."""
        # Asosiy qiymatlarning o'zi turish vaqti bo'lsa
        if self._dwell_warn > 0:
            return dwell_level_index(value, self._dwell_warn)

        d_idx = dwell_level_index(dwell, self._dwell_map_warn)

        if value == 0:
            # Mashina o'tmagan. Lekin turish vaqti bor bo'lsa — bu bo'sh emas,
            # TO'XTAB qolgan holat (zatorning eng og'ir ko'rinishi).
            return d_idx if d_idx > 0 else -1

        if self._capacity > 0:
            # Oqim va turish vaqtining YOMONI
            return max(odm_level_index(value / self._capacity), d_idx)

        # Nisbiy rejim: turish vaqti aniq zator ko'rsatsa (D va yomonroq) —
        # oqim shkalasi mutlaq bo'lmasa ham shu daraja ustun.
        return d_idx if d_idx >= 3 else -1

    def _get_color(self, value: int, max_val: int,
                   dwell: float = 0.0) -> QColor:
        """Qiymatga qarab rang.
          - A-F daraja aniqlansa (izoh: _cell_level) — shu darajaning rangi
          - aks holda nisbiy gradient: qora → yashil → sariq → qizil
        """
        lvl = self._cell_level(value, dwell)
        if lvl >= 0:
            return level_qcolor(lvl)

        if value == 0:
            return QColor(C('bg_panel_dark'))

        if self._scale_max > 0:
            max_val = self._scale_max
        if max_val == 0:
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

    # Katakcha o'lchamlari — chizish va sichqoncha ostidagi katakni topish
    # uchun BIR XIL hisob ishlatiladi (aks holda tooltip boshqa katakni
    # ko'rsatib qo'yardi).
    # Chap chegara "Se 04" yorlig'iga yetadi
    _LEFT_M, _TOP_M, _RIGHT_M, _BOTTOM_M = 42, 22, 8, 6

    def _layout(self):
        """(cell_w, cell_h, rows, cols) yoki None (joy yetmasa)."""
        rows = len(self._data)
        if rows == 0:
            return None
        cols = 24
        cw = self.width() - self._LEFT_M - self._RIGHT_M
        ch = self.height() - self._TOP_M - self._BOTTOM_M
        if cw <= 0 or ch <= 0:
            return None
        return cw / cols, ch / rows, rows, cols

    def paintEvent(self, event):
        if not self._data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        left_m, top_m = self._LEFT_M, self._TOP_M

        lay = self._layout()
        if lay is None:
            painter.end()
            return
        cell_w, cell_h, rows, cols = lay
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
        axis_color = QColor(C('text_secondary'))
        font = QFont()
        font.setPixelSize(9)
        painter.setFont(font)

        # Soat o'qi: har 3 soatda raqam, ustida ozgina chizgi. 00/06/12/18
        # yorqinroq — kun qismlarini tez topish uchun.
        step = 3 if cell_w >= 26 else (4 if cell_w >= 18 else 6)
        for c in range(cols):
            if c % step:
                continue
            x = left_m + c * cell_w
            major = c % 6 == 0
            painter.setPen(QPen(axis_color if major else text_color))
            f2 = QFont(font)
            f2.setPixelSize(9)
            f2.setBold(major)
            painter.setFont(f2)
            painter.drawText(
                QRectF(x, 0, cell_w * step, top_m - 4),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
                f"{c:02d}")
            painter.setPen(QPen(QColor(text_color.red(), text_color.green(),
                                       text_color.blue(), 90), 1))
            painter.drawLine(int(x + gap / 2), int(top_m - 3),
                             int(x + gap / 2), int(top_m - 1))
        painter.setFont(font)

        # Qatorlar (kunlar)
        for r, d in enumerate(self._data):
            y = top_m + r * cell_h
            day_label = d.get("day", "")
            hours = d.get("hours", [0] * 24)
            date_str = d.get("date", "")

            # Chapda: kun nomi + sana raqami ("Se 04") — qaysi kun ekani
            # aniq bo'lsin. Joy tor bo'lsa faqat kun nomi.
            day_num = ""
            try:
                day_num = date_str.split("-")[2]
            except Exception:
                pass
            label = f"{day_label} {day_num}".strip() if cell_h >= 15 else day_label
            painter.setPen(QPen(axis_color))
            f2 = QFont(font)
            f2.setPixelSize(9)
            f2.setBold(True)
            painter.setFont(f2)
            painter.drawText(
                QRectF(0, y, left_m - 5, cell_h),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label)
            painter.setFont(font)

            # Kataklar
            for c in range(cols):
                x = left_m + c * cell_w
                value = hours[c] if c < len(hours) else 0
                dwell = self._dwell_at(date_str, c)

                color = self._get_color(value, max_val, dwell)
                rect = QRectF(x + gap / 2, y + gap / 2,
                              cell_w - gap, cell_h - gap)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(color))
                path = QPainterPath()
                path.addRoundedRect(rect, radius, radius)
                painter.drawPath(path)

                # Rang turish vaqtidan kelgan bo'lsa — kichik nuqta.
                # "Son kichik, lekin rang qizil" xato emas: mashina kam
                # o'tgan, chunki zonada uzoq turgan (zator).
                if cell_w >= 12 and self._is_congestion_driven(value, dwell):
                    marker = contrast_text(color)
                    painter.setBrush(QBrush(marker))
                    dot = max(2.0, min(3.5, cell_w * 0.09))
                    painter.drawEllipse(
                        QRectF(rect.right() - dot - 2, rect.top() + 2, dot, dot))

                # Son: fon rangiga qarab to'q yoki oq matn (sariq katakda oq
                # matn o'qilmaydi). Joy tor bo'lsa qisqartiriladi: 2.4k
                if value > 0 and cell_w >= 18 and cell_h >= 15:
                    f3 = QFont(font)
                    f3.setPixelSize(max(7, int(min(cell_w * 0.34, cell_h * 0.45))))
                    fm3 = QFontMetrics(f3)
                    avail = cell_w - gap - 4
                    # Matn KATAKKA o'lchab moslanadi: sig'sa "2 431",
                    # sig'masa "2.4K" (izoh: app/utils/numfmt.py)
                    txt = fmt_fit(value, avail, fm3)
                    # Eng qisqa shakl ham sig'masa — SONNI CHIZMAYMIZ. Yarim
                    # kesilgan son ("15110…") noto'g'ri o'qiladi; rang darajani
                    # ko'rsatadi, aniq qiymat tooltipda bor.
                    if fits(txt, avail, fm3):
                        painter.setPen(QPen(contrast_text(color)))
                        painter.setFont(f3)
                        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, txt)
                    painter.setFont(font)

                # Sichqoncha ostidagi katak — nozik ramka (qaysi katak
                # tooltipda ko'rsatilayotgani aniq bo'lsin)
                if self._hover == (r, c):
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(contrast_text(color), 1.4))
                    painter.drawRoundedRect(rect.adjusted(0.7, 0.7, -0.7, -0.7),
                                            radius, radius)

        painter.end()
