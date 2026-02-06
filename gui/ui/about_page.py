"""
About Page - System information and documentation
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QScrollArea, QSizePolicy, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
from pathlib import Path

from gui.utils.theme_colors import C


class AboutPage(QWidget):
    """About page with system information"""

    back_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_section = "umumiy"
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left sidebar
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar)

        # Main content area
        content_area = self._create_content_area()
        main_layout.addWidget(content_area, 1)

    def _create_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background-color: {C('bg_secondary')};
                border-right: 1px solid {C('border_light')};
            }}
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 20, 12, 20)
        layout.setSpacing(4)

        # Title
        title = QLabel("Tizim haqida")
        title.setStyleSheet(f"""
            color: {C('text_primary')};
            font-size: 16px;
            font-weight: bold;
            padding: 8px 12px;
        """)
        layout.addWidget(title)

        layout.addSpacing(10)

        # Navigation items
        nav_items = [
            ("umumiy", "◇", "Umumiy"),
            ("arxitektura", "⚙", "Arxitektura"),
            ("ishlash", "▷", "Ishlash"),
            ("xavfsizlik", "⊕", "Xavfsizlik"),
            ("analitika", "∿", "Analitika"),
            ("bashorat", "⚡", "Bashorat"),
            ("versiya", "▣", "Versiya"),
        ]

        self.nav_buttons = {}
        for key, icon, text in nav_items:
            btn = QPushButton(f"  {icon}  {text}")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self._nav_btn_style(key == self._current_section))
            btn.clicked.connect(lambda checked, k=key: self._select_section(k))
            self.nav_buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch()

        # Version info at bottom
        version_label = QLabel("Versiya 1.0.5 • 2024  OEM Design")
        version_label.setStyleSheet(f"""
            color: {C('text_muted')};
            font-size: 10px;
            padding: 12px;
        """)
        layout.addWidget(version_label)

        return sidebar

    def _nav_btn_style(self, active=False):
        if active:
            return f"""
                QPushButton {{
                    background-color: {C('bg_hover')};
                    color: {C('accent_brand')};
                    border: none;
                    border-radius: 6px;
                    padding: 10px 12px;
                    text-align: left;
                    font-size: 13px;
                    font-weight: bold;
                }}
            """
        return f"""
            QPushButton {{
                background-color: transparent;
                color: {C('text_secondary')};
                border: none;
                border-radius: 6px;
                padding: 10px 12px;
                text-align: left;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {C('bg_hover')};
                color: {C('text_primary')};
            }}
        """

    def _select_section(self, section_key):
        self._current_section = section_key
        for key, btn in self.nav_buttons.items():
            btn.setStyleSheet(self._nav_btn_style(key == section_key))

        # Scroll to section
        if hasattr(self, 'scroll_area') and section_key in self.sections:
            widget = self.sections[section_key]
            self.scroll_area.ensureWidgetVisible(widget)

    def _create_content_area(self):
        container = QWidget()
        container.setStyleSheet(f"background-color: {C('bg_primary')};")

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {C('bg_primary')};
            }}
            QScrollBar:vertical {{
                background: {C('bg_secondary')};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {C('text_muted')};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)

        # Content widget
        content_widget = QWidget()
        content_widget.setStyleSheet(f"background-color: {C('bg_primary')};")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(40, 30, 40, 40)
        content_layout.setSpacing(30)

        self.sections = {}

        # Page title
        page_title = QLabel("Tizim haqida")
        page_title.setStyleSheet(f"""
            color: {C('text_primary')};
            font-size: 28px;
            font-weight: bold;
        """)
        content_layout.addWidget(page_title)

        # Umumiy section
        umumiy = self._create_section("umumiy", "Umumiy maqsad", """
Ushbu tizim avtomobil yo'llari temir yo'l kesishuvlaridagi xavfsizlikni real vaqt rejimida nazorat qilish uchun ishlab chiqilgan bo'lib, temir yo'l orqali transport harakatini xavf-xatarlar yuzaga kelishidan oldin aniqlash va ogohlantirish imkonini beradi.

<b>Asosiy vazifalar:</b>
• Pereezdlardagi transport harakatini real vaqtda kuzatish
• Sun'iy intellekt yordamida transport vositalarini aniqlash
• Xavfli vaziyatlarni oldindan bashorat qilish
• Operatorlarni darhol ogohlantirish
• Statistik ma'lumotlarni to'plash va tahlil qilish
        """)
        content_layout.addWidget(umumiy)
        self.sections["umumiy"] = umumiy

        # Arxitektura section
        arxitektura = self._create_section("arxitektura", "Tizim arxitekturasi", """
Tizim markaziy server asosidagi arxitekturaga ega va quyidagi komponentlardan iborat:

<b>Asosiy komponentlar:</b>
• <b>Kameralar va sensorlar</b> - Pereezdlarda o'rnatilgan IP kameralar
• <b>Markaziy server</b> - Ma'lumotlarni qayta ishlash va AI modellari
• <b>Operator paneli</b> - Real vaqtda monitoring va boshqaruv
• <b>PLC kontrollerlar</b> - Shlagbaum va signalizatsiya boshqaruvi

<b>Texnologiyalar:</b>
• Python + PyQt6 (Grafik interfeys)
• OpenCV (Video qayta ishlash)
• YOLO v8 (Ob'ektlarni aniqlash)
• RTSP/TCP (Video uzatish protokoli)
        """)
        content_layout.addWidget(arxitektura)
        self.sections["arxitektura"] = arxitektura

        # Ishlash section
        ishlash = self._create_section("ishlash", "Qanday ishlaydi", """
<b>1. Video oqimini olish</b>
Kameralar RTSP protokoli orqali real vaqtda video oqimini serverga uzatadi.

<b>2. AI tahlili</b>
YOLO neyron tarmog'i har bir kadrda transport vositalarini aniqlaydi va klassifikatsiya qiladi.

<b>3. Xavfni baholash</b>
Tizim transport vositasining tezligi, yo'nalishi va pereezdga yaqinligini hisoblab, xavf darajasini aniqlaydi.

<b>4. Ogohlantirish</b>
Xavfli vaziyat aniqlanganda operator darhol ogohlantiriladi va zarur bo'lsa avtomatik choralar ko'riladi.
        """)
        content_layout.addWidget(ishlash)
        self.sections["ishlash"] = ishlash

        # Xavfsizlik section
        xavfsizlik = self._create_section("xavfsizlik", "Xavfsizlik darajasi", """
Tizim xavfsizlikni ta'minlash uchun quyidagi mexanizmlardan foydalanadi:

<b>Ma'lumotlar xavfsizligi:</b>
• Barcha video oqimlari shifrlangan holda uzatiladi
• Foydalanuvchi autentifikatsiyasi va ruxsatlar tizimi
• Muntazam zaxira nusxalar yaratish

<b>Tizim barqarorligi:</b>
• Kameralar bilan aloqa uzilganda avtomatik qayta ulanish
• Serverda xatolik yuz berganda zaxira rejim
• 24/7 uzluksiz ishlash qobiliyati
        """)
        content_layout.addWidget(xavfsizlik)
        self.sections["xavfsizlik"] = xavfsizlik

        # Analitika section
        analitika = self._create_section("analitika", "Analitika", """
Tizim statistik ma'lumotlarni to'playdi va tahlil qiladi:

<b>Real vaqt statistikasi:</b>
• Pereezddan o'tgan transport vositalari soni
• Transport turlari bo'yicha taqsimot (yengil/og'ir)
• O'rtacha o'tish vaqti

<b>Tarixiy tahlil:</b>
• Kunlik, haftalik, oylik hisobotlar
• Eng band vaqtlar tahlili
• Xavfli vaziyatlar statistikasi
        """)
        content_layout.addWidget(analitika)
        self.sections["analitika"] = analitika

        # Bashorat section
        bashorat = self._create_section("bashorat", "Bashorat", """
Sun'iy intellekt va quyidagi asoslangan algoritmlar yordamida tizim temir yo'l kesishuvlarida kelajak xavflarini bashorat qiladi va operatorni oldindan ogohlantiradi.

<b>Bashorat qilish usullari:</b>
• Transport vositasi trayektoriyasini tahlil qilish
• Tezlik va masofani hisoblash
• Tarixiy ma'lumotlar asosida tendentsiyalarni aniqlash
• Ob-havo sharoitlarini hisobga olish
        """)
        content_layout.addWidget(bashorat)
        self.sections["bashorat"] = bashorat

        # Versiya section
        versiya = self._create_section("versiya", "Versiya va muallif", f"""
<b>Versiya:</b> 1.0.5

<b>Ishlab chiquvchi:</b> OEM Design, Uzbekistan

<b>Texnologiyalar:</b>
• Python 3.12
• PyQt6
• OpenCV
• NumPy

<b>Aloqa:</b>
Savollar va takliflar uchun ishlab chiquvchiga murojaat qiling.

© 2024 OEM Design, Uzbekistan
Barcha huquqlar himoyalangan.
        """)
        content_layout.addWidget(versiya)
        self.sections["versiya"] = versiya

        content_layout.addStretch()

        self.scroll_area.setWidget(content_widget)
        layout.addWidget(self.scroll_area)

        return container

    def _create_section(self, key, title, content):
        section = QFrame()
        section.setStyleSheet(f"""
            QFrame {{
                background-color: {C('bg_card')};
                border: 1px solid {C('border_light')};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(section)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(12)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            color: {C('accent_brand')};
            font-size: 18px;
            font-weight: bold;
            border: none;
            background: transparent;
        """)
        layout.addWidget(title_label)

        # Content
        content_label = QLabel(content.strip())
        content_label.setWordWrap(True)
        content_label.setTextFormat(Qt.TextFormat.RichText)
        content_label.setStyleSheet(f"""
            color: {C('text_secondary')};
            font-size: 13px;
            line-height: 1.6;
            border: none;
            background: transparent;
        """)
        layout.addWidget(content_label)

        return section
