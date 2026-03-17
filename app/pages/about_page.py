"""
About Page - To'liq foydalanuvchi qo'llanmasi
"""

import os

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QScrollArea, QPushButton, QTextBrowser,
                              QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal

from app.utils.theme_colors import C
from app.utils.language import t, LM



_NAV_ITEMS = [
    ("umumiy",      "◇", "about.nav.general",      "about.section.general"),
    ("arxitektura", "⚙", "about.nav.architecture",  "about.section.architecture"),
    ("ishlash",     "▷", "about.nav.howworks",      "about.section.howworks"),
    ("xavfsizlik",  "⊕", "about.nav.security",      "about.section.security"),
    ("analitika",   "∿", "about.nav.analytics",     "about.section.analytics"),
    ("bashorat",    "⚡", "about.nav.forecast",      "about.section.forecast"),
    ("versiya",     "▣", "about.nav.version",       "about.section.version"),
]


# ── HTML helpers ─────────────────────────────────────────────────────────────
def _step_html(n, text, accent, text_color):
    return (
        f'<table width="100%" border="0" cellpadding="0" cellspacing="4">'
        f'<tr>'
        f'<td width="32" bgcolor="{accent}" align="center" style="border-radius:6px; padding:5px 0; font-size:12px;">'
        f'<font color="#0a0a14"><b>{n}</b></font></td>'
        f'<td style="padding-left:14px; vertical-align:middle;">'
        f'<font color="{text_color}">{text}</font></td>'
        f'</tr></table>'
    )


def _warn_html(text, bg, border_col, text_color):
    return (
        f'<table width="100%" border="0" cellpadding="10" cellspacing="0" '
        f'bgcolor="{bg}" style="margin:6px 0; border-left:3px solid {border_col}; border-radius:4px;">'
        f'<tr><td><font color="{text_color}">{text}</font></td></tr></table>'
    )


def _info_html(text, bg, text_color):
    return (
        f'<table width="100%" border="0" cellpadding="10" cellspacing="0" '
        f'bgcolor="{bg}" style="margin:6px 0; border-radius:6px;">'
        f'<tr><td><font color="{text_color}">{text}</font></td></tr></table>'
    )


def _field_row(name, desc, required, accent, text, muted):
    req = f' <font color="{accent}">*majburiy</font>' if required else \
          f' <font color="{muted}">(ixtiyoriy)</font>'
    return (
        f'<table width="100%" border="0" cellpadding="5" cellspacing="0">'
        f'<tr>'
        f'<td width="140"><font color="{accent}"><b>{name}</b></font>{req}</td>'
        f'<td><font color="{text}">— {desc}</font></td>'
        f'</tr></table>'
    )


def _tip_html(text, accent, bg):
    return (
        f'<table width="100%" border="0" cellpadding="8" cellspacing="0" bgcolor="{bg}"'
        f' style="margin:6px 0; border-radius:4px; border-left:3px solid {accent};">'
        f'<tr><td><font color="{accent}">💡 </font>'
        f'<font color="{accent}">{text}</font></td></tr></table>'
    )



class AboutPage(QWidget):
    """About page — to'liq foydalanuvchi qo'llanmasi"""

    back_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_section = "umumiy"
        self._section_title_lbls = {}
        self._browsers: dict[str, QTextBrowser] = {}
        self._setup_ui()
        LM.language_changed.connect(self._retranslate)

    # ── HTML content ─────────────────────────────────────────────────────────
    def _build_html(self, key: str) -> str:
        acc      = C('accent_brand')
        txt      = C('text_secondary')
        muted    = C('text_muted')
        bg       = C('bg_card')
        bg_in    = C('bg_input')
        ok       = C('status_online')
        err      = C('status_error')
        warn     = C('status_warning')
        warn_bg  = "#1e1700"
        warn_brd = "#f59e0b"
        warn_txt = "#fbbf24"
        info_bg  = "#001830"
        info_txt = "#60a5fa"
        tip_bg   = "#001a10"
        tip_col  = "#4ade80"

        parts = [f'<body style="background-color:{bg}; font-family:Segoe UI,Arial; '
                 f'font-size:16px; color:{txt}; margin:0; padding:0;">']

        if key == "umumiy":
            parts += [
                f'<p><font color="{acc}"><b>RailSafe AI nima?</b></font></p>',
                f'<p><font color="{txt}"><b>RailSafe AI</b> — temir yo\'l kesishmalarini (pereezdlarni) '
                f'kameralar va sun\'iy intellekt yordamida <b>avtomatik nazorat qiluvchi</b> dastur. '
                f'Dastur pereezddan o\'tayotgan avtomobillarni sanaydi, poyezd kelayotganini aniqlaydi '
                f'va barcha ma\'lumotlarni statistika shaklida saqlaydi.</font></p>',

                f'<p><font color="{txt}">Oddiy qilib aytganda: dastur kamera orqali pereezdni kuzatadi, '
                f'har bir mashina va poyezdni sanab boradi, xavfli vaziyat bo\'lsa signal beradi.</font></p>',

                f'<p><font color="{acc}"><b>Dastur ishga tushganda nima ko\'rasiz?</b></font></p>',
                f'<p><font color="{txt}">Dasturni ochdingiz — ekranda <b>Boshqaruv paneli</b> ko\'rinadi. '
                f'Bu panel — sizning asosiy ish joyingiz. Har bir pereezd alohida karta shaklida '
                f'ko\'rsatiladi. Kartada pereezdning jonli video tasviri va statistikasi bo\'ladi.</font></p>',

                _info_html(
                    f'<b>Har bir pereezd kartasida nima ko\'rinadi:</b><br><br>'
                    f'📺 <b>Jonli video</b> — kamera orqali pereezdning hozirgi ko\'rinishi<br>'
                    f'🚗 <b>Yengil transport soni</b> — bugun o\'tgan avtomobillar, mikroavtobuslar<br>'
                    f'🚛 <b>Og\'ir transport soni</b> — bugun o\'tgan yuk mashinalari, avtobuslар<br>'
                    f'🚆 <b>Poyezd hisobi</b> — bugun nechta poyezd o\'tgani<br>'
                    f'🔌 <b>PLC holati</b> — hozir poyezd kelayotgani yoki yo\'qligi',
                    info_bg, info_txt
                ),

                f'<p><font color="{acc}"><b>Yuqori paneldagi tugmalar:</b></font></p>',
                f'<table width="100%" border="0" cellpadding="6" cellspacing="0">',
                *[f'<tr><td width="180"><font color="{acc}"><b>{btn}</b></font></td>'
                  f'<td><font color="{txt}">{desc}</font></td></tr>'
                  for btn, desc in [
                      ("+ Pereezd Qo'shish", "Yangi pereezd qo'shadi — birinchi marta ishlatganda bosing"),
                      ("⟳ Yangilash", "Kameralar uzilgan bo'lsa qayta ulanishga majbur qiladi"),
                      ("⚙ Sozlamalar", "Til, rang mavzusi va AI model sozlamalarini ochadi"),
                      ("∿ Tahlil", "Grafik va statistika sahifasiga o'tadi"),
                      ("Tizim haqida", "Ushbu qo'llanma sahifasini ochadi"),
                  ]],
                f'</table>',

                _tip_html(
                    'Pereezd kartasini <b>bir marta bosing</b> — kameralar kattaroq ekranda ochiladi. '
                    'Kameralar ro\'yxatini ko\'rish va boshqarish uchun <b>"Sozlamalar"</b> tugmasini bosing.',
                    tip_col, tip_bg
                ),

                f'<p><font color="{acc}"><b>Dastur qanday ishlaydi — qisqacha:</b></font></p>',
                _info_html(
                    f'1️⃣ Dastur ishga tushadi → AI model yuklanadi (10–30 sekund)<br>'
                    f'2️⃣ Kameralar RTSP orqali ulanadi → video oqimi boshlanadi<br>'
                    f'3️⃣ AI har bir kadrni tahlil qiladi → transport aniqlanadi va sanaladi<br>'
                    f'4️⃣ PLC dan signal kelsa → "POYEZD KELMOQDA" holati ko\'rsatiladi<br>'
                    f'5️⃣ Barcha ma\'lumotlar avtomatik ma\'lumotlar bazasiga saqlanadi<br>'
                    f'6️⃣ Tahlil sahifasida grafiklar va hisobotlar ko\'rish mumkin',
                    info_bg, info_txt
                ),

                _warn_html(
                    '⚠ Dastur <b>birinchi marta</b> ishga tushganda AI model yuklanadi — '
                    'bu <b>10–30 sekund</b> vaqt olishi mumkin. Shu payt kamera signallari '
                    'ko\'rinmaydi — bu normal holat, kuting.',
                    warn_bg, warn_brd, warn_txt
                ),
            ]

        elif key == "arxitektura":
            parts += [
                f'<p><font color="{acc}"><b>Pereezd nima va nima uchun qo\'shiladi?</b></font></p>',
                f'<p><font color="{txt}"><b>Pereezd</b> — bu temir yo\'l bilan avtomobil yo\'li '
                f'kesishgan joy. Dasturda har bir pereezd alohida ro\'yxatga olinadi. '
                f'Ro\'yxatga olingandan so\'ng unga kameralar va PLC qurilmasi biriktiriladi.</font></p>',

                f'<p><font color="{txt}">Masalan: sizda 3 ta pereezd bo\'lsa — uchatasini ham '
                f'alohida qo\'shasiz. Har biri mustaqil ishlaydi, o\'z statistikasiga ega bo\'ladi.</font></p>',

                f'<p><font color="{acc}"><b>Yangi pereezd qo\'shish — bosqichma-bosqich:</b></font></p>',

                _step_html(1,
                    'Ekranning yuqori qismidagi asboblar panelida <b>"+ Pereezd Qo\'shish"</b> '
                    'tugmasini toping va bosing. Yangi oyna ochiladi.', acc, txt),
                _step_html(2,
                    '<b>"Asosiy Ma\'lumotlar"</b> bo\'limida (tab) quyidagi '
                    'maydonlarni to\'ldiring:', acc, txt),

                f'<table width="100%" border="0" cellpadding="5" cellspacing="2">',
                _field_row("Nomi",
                    "Pereezdning qisqa nomi. Masalan: <i>Pereezd 8</i> yoki <i>Shimoliy pereezd</i>",
                    True, acc, txt, muted),
                _field_row("Manzil",
                    "Pereezd qayerda joylashgani. Masalan: <i>Toshkent sh., Chilonzor tumani</i>",
                    True, acc, txt, muted),
                _field_row("Tavsif",
                    "Qo\'shimcha izoh — masalan: <i>Asosiy yo\'l, 24/7 nazorat</i>. To\'ldirish shart emas.",
                    False, acc, txt, muted),
                f'</table>',

                _step_html(3,
                    'Agar PLC qurilmasi hali tayyor bo\'lmasa — hozircha PLC bo\'limini '
                    'o\'tkazib yuboring. <b>"Saqlash"</b> tugmasini bosing. '
                    'PLC ni keyinroq tahrirlash orqali qo\'shish mumkin.', acc, txt),
                _step_html(4,
                    'Pereezd boshqaruv paneliga qo\'shildi! Endi uning '
                    '<b>kartasini bosing</b> — ichiga kiring va kamera qo\'shing.', acc, txt),

                f'<p><font color="{acc}"><b>Mavjud pereezdni o\'zgartirish:</b></font></p>',
                _info_html(
                    f'<b>Tahrirlash:</b> Pereezd kartasini <b>o\'ng tugma bilan bosing</b> '
                    f'→ "Pereezd Ma\'lumotlari" → "Tahrirlash" — nomi, manzili yoki PLC '
                    f'sozlamalarini o\'zgartiring.<br><br>'
                    f'<b>Nusxa olish (JSON eksport):</b> Kartani o\'ng bosing → "JSON Eksport" — '
                    f'pereezdning barcha sozlamalari fayl sifatida saqlanadi. '
                    f'Bu <b>zaxira nusxa</b> sifatida ishlatiladi.<br><br>'
                    f'<b>Tiklash (JSON import):</b> Yangi pereezd qo\'shish oynasida '
                    f'"📥 JSON dan yuklash" tugmasini bosib, avval saqlangan faylni tanlang — '
                    f'barcha sozlamalar avtomatik to\'ldiriladi.',
                    info_bg, info_txt
                ),

                _warn_html(
                    '⚠ <b>Pereezdni o\'chirish:</b> Pereezd ichiga kirib, '
                    'ekranning yuqori qismidagi <b>"O\'chirish"</b> tugmasini bosing. '
                    'Diqqat — pereezd o\'chirilsa, unga biriktirilgan <b>barcha kameralar '
                    'va statistika ham butunlay o\'chib ketadi</b>. Bu amalni qaytarib '
                    'bo\'lmaydi!',
                    warn_bg, warn_brd, warn_txt
                ),
            ]

        elif key == "ishlash":
            parts += [
                f'<p><font color="{acc}"><b>Kamera nima va nima uchun qo\'shiladi?</b></font></p>',
                f'<p><font color="{txt}"><b>Kamera</b> — pereezdga o\'rnatilgan IP kamera. '
                f'Dastur kamera orqali video oladi, sun\'iy intellekt yordamida transport '
                f'vositalarini aniqlaydi va sanaydi. Bir pereezdga bir nechta kamera '
                f'qo\'shish mumkin.</font></p>',

                f'<p><font color="{txt}">Kameralar ikki turga bo\'linadi: '
                f'<b>Asosiy</b> (katta ekranda ko\'rinadi, bir pereezdda faqat 1 ta bo\'ladi) '
                f'va <b>Qo\'shimcha</b> (kichik oynada, xohlagancha qo\'shish mumkin).</font></p>',

                f'<p><font color="{acc}"><b>Kamera qo\'shish — bosqichma-bosqich:</b></font></p>',

                _step_html(1,
                    'Boshqaruv panelida kerakli <b>pereezd kartasini bosib ichiga kiring</b>. '
                    'Pereezd sahifasi ochiladi.', acc, txt),
                _step_html(2,
                    'Ekranning yuqori qismida <b>"+ Kamera"</b> tugmasini toping va bosing. '
                    'Kamera qo\'shish oynasi ochiladi.', acc, txt),
                _step_html(3, 'Oynada quyidagi maydonlarni to\'ldiring:', acc, txt),

                f'<table width="100%" border="0" cellpadding="5" cellspacing="2">',
                _field_row("Nomi",
                    "Kameraga nom bering. Masalan: <i>Shimoliy yo\'nalish</i> yoki <i>Kamera 1</i>",
                    True, acc, txt, muted),
                _field_row("Turi",
                    '<b>Asosiy</b> — katta ekranda ko\'rsatiladi (har bir pereezdda 1 ta). '
                    '<b>Qo\'shimcha</b> — yonida kichik oynada ko\'rsatiladi.',
                    True, acc, txt, muted),
                _field_row("Manba",
                    "Kameraning tarmoqdagi manzili (RTSP) yoki lokal video fayl yo\'li",
                    True, acc, txt, muted),
                _field_row("Polygon fayli",
                    "Kamera tasvirida hisoblash zonasini belgilash uchun fayl. "
                    "Mavjud bo\'lmasa bo\'sh qoldiring.",
                    False, acc, txt, muted),
                f'</table>',

                f'<p><font color="{acc}"><b>RTSP manzil qanday ko\'rinadi?</b></font></p>',
                _info_html(
                    f'RTSP — bu IP kameraning tarmoqdagi video manzili. '
                    f'Kamera sotib olganingizda uning qo\'llanmasida yozilgan bo\'ladi.<br><br>'
                    f'<b>Umumiy ko\'rinish:</b><br>'
                    f'<font color="{acc}">rtsp://login:parol@IP_manzil:port/kanal</font><br><br>'
                    f'<b>Misol:</b> '
                    f'<font color="{acc}">rtsp://admin:12345@192.168.1.50:554/stream1</font><br><br>'
                    f'<b>Hikvision kameralar uchun:</b><br>'
                    f'<font color="{acc}">rtsp://admin:parol@IP/Streaming/Channels/101</font><br><br>'
                    f'<b>Dahua kameralar uchun:</b><br>'
                    f'<font color="{acc}">rtsp://admin:parol@IP/cam/realmonitor?channel=1</font><br><br>'
                    f'💡 Manzilni tekshirish uchun <b>VLC Media Player</b> dasturini oching: '
                    f'Media → "Open Network Stream" → manzilni kiriting.',
                    info_bg, info_txt
                ),

                _step_html(4,
                    '<b>"Kamerani yoqish"</b> va <b>"Aniqlov yoqish"</b> '
                    'katagichlarining ikkalasi ham belgilangan bo\'lsin. '
                    'So\'ng <b>"Saqlash"</b> tugmasini bosing.', acc, txt),
                _step_html(5,
                    'Kamera <b>3–10 sekund ichida</b> ulanishni boshlaydi. '
                    'Ulanish muvaffaqiyatli bo\'lsa, ekranda video ko\'rinadi. '
                    'Agar "Ulanmadi" xatosi chiqsa — RTSP manzilni tekshiring.', acc, txt),

                f'<p><font color="{acc}"><b>Polygon nima va nima uchun kerak?</b></font></p>',
                _info_html(
                    f'<b>Polygon</b> — kamera tasvirida siz belgilagan maxsus hudud (zona). '
                    f'Dastur transport vositalarini faqat <b>shu hudud ichida</b> sanaydi.<br><br>'
                    f'Masalan: kamera keng maydonni ko\'rsa ham, siz faqat yo\'lning '
                    f'pereezddan o\'tadigan qismini belgilaysiz — shu joydan o\'tgan '
                    f'mashinalar sanaladi.<br><br>'
                    f'Polygon bo\'lmasa ham kamera ishlaydi, lekin zona vaqti hisoblanmaydi.<br><br>'
                    f'<b>Polygon yaratish:</b> Alohida <b>polygon_editor</b> dasturi bilan '
                    f'yaratiladi va JSON fayl sifatida saqlanadi. So\'ngra bu faylni '
                    f'"Polygon fayli" maydoniga yuklaysiz.',
                    info_bg, info_txt
                ),

                _warn_html(
                    '⚠ Bir pereezdda <b>faqat 1 ta asosiy kamera</b> bo\'lishi mumkin. '
                    'Ikkinchi asosiy kamera qo\'shsangiz, avvalgisi avtomatik qo\'shimchaga '
                    'o\'tib qoladi.<br><br>'
                    '⚠ Kamerani <b>to\'xtatish (⏸ tugmasi)</b> ma\'lumotlarni o\'chirmaydi — '
                    'faqat shu kamera video oqimini vaqtincha to\'xtatadi. '
                    'Statistika saqlanib qoladi.',
                    warn_bg, warn_brd, warn_txt
                ),
            ]

        elif key == "xavfsizlik":
            parts += [
                f'<p><font color="{acc}"><b>PLC nima?</b></font></p>',
                f'<p><font color="{txt}"><b>PLC (Dasturlanadigan Mantiqiy Kontroller)</b> — '
                f'bu poyezd kelayotganini sezib, dasturga signal beruvchi elektron qurilma. '
                f'Temir yo\'l relslari yoniga o\'rnatiladi. Poyezd yaqinlashganda PLC '
                f'signal yuboradi — dastur "POYEZD KELMOQDA" deb belgilaydi.</font></p>',

                f'<p><font color="{txt}">Eng keng tarqalgan model: <b>Siemens S7-1200</b>. '
                f'PLC va kompyuter bir xil lokal tarmoqda (kabel yoki Wi-Fi orqali) '
                f'ulangan bo\'lishi kerak.</font></p>',

                f'<p><font color="{acc}"><b>PLC ulash — bosqichma-bosqich:</b></font></p>',

                _step_html(1,
                    'Pereezd qo\'shish yoki tahrirlash oynasini oching. '
                    'Yuqoridan <b>"PLC Sozlamalari"</b> tabini (bo\'limini) bosing.',
                    acc, txt),
                _step_html(2,
                    '<b>"PLC ni yoqish"</b> katagichini toping va belgilang. '
                    'Shu zahoti quyidagi maydonlar faol bo\'ladi.',
                    acc, txt),
                _step_html(3, 'Quyidagi ma\'lumotlarni to\'ldiring:', acc, txt),

                f'<table width="100%" border="0" cellpadding="5" cellspacing="2">',
                _field_row("IP Manzil",
                    "PLC qurilmasining tarmoqdagi IP manzili. "
                    "Masalan: <i>192.168.1.100</i>. "
                    "Bu manzilni PLC sozlamalaridan yoki IT mutaxassisdan bilib oling.",
                    True, acc, txt, muted),
                _field_row("Port",
                    "Siemens S7-1200 va S7-1500 uchun standart port: <b>102</b>. "
                    "Modbus TCP protokoli uchun: <b>502</b>. "
                    "Noaniq bo\'lsa 102 qoldiring.",
                    True, acc, txt, muted),
                _field_row("PLC Turi",
                    "Qurilmangizning modelini tanlang: Siemens S7-1200, S7-1500, "
                    "Modbus TCP yoki Boshqa. Noto\'g\'ri tanlansa ham urinib ko\'ring.",
                    False, acc, txt, muted),
                f'</table>',

                _step_html(4,
                    '<b>"Ulanishni Tekshirish"</b> tugmasini bosing. '
                    'Dastur PLC ga ulanishga harakat qiladi:', acc, txt),

                _info_html(
                    f'<font color="{ok}">✅ <b>Ulandi</b></font>'
                    f'<font color="{txt}"> — PLC topildi va javob berdi. '
                    f'"Saqlash" tugmasini bosishingiz mumkin.</font><br><br>'
                    f'<font color="{err}">❌ <b>Ulanmadi</b></font>'
                    f'<font color="{txt}"> — Muammo bor. Quyidagilarni tekshiring:</font><br>'
                    f'&nbsp;&nbsp;• IP manzilni to\'g\'ri kiritdingizmi?<br>'
                    f'&nbsp;&nbsp;• PLC qurilmasi yoqilganmi va tarmoqqa ulanganmi?<br>'
                    f'&nbsp;&nbsp;• Kompyuter va PLC bir xil tarmoqdami? '
                    f'(masalan ikkalasi ham 192.168.1.xxx da bo\'lishi kerak)<br>'
                    f'&nbsp;&nbsp;• Kompyuterda xavfsizlik devori (Firewall) '
                    f'<b>102-portni</b> bloklamayanmi?<br>'
                    f'&nbsp;&nbsp;• PLC dasturlash muhitida (TIA Portal) ulanish '
                    f'ruxsat etilganmi?',
                    info_bg, info_txt
                ),

                _step_html(5,
                    'Hamma narsa to\'g\'ri bo\'lsa — <b>"Saqlash"</b> ni bosing. '
                    'PLC endi pereezdga biriktirildi.', acc, txt),

                f'<p><font color="{acc}"><b>Boshqaruv panelida PLC holati qanday ko\'rinadi:</b></font></p>',
                f'<table width="100%" border="0" cellpadding="6" cellspacing="2">',
                *[f'<tr><td width="28">{ic}</td>'
                  f'<td width="210"><font color="{col}"><b>{lbl}</b></font></td>'
                  f'<td><font color="{txt}">{desc}</font></td></tr>'
                  for ic, col, lbl, desc in [
                      ("🟢", ok,    "ULANGAN",          "PLC ishlayapti, hozircha poyezd yo'q"),
                      ("🔴", err,   "POYEZD KELMOQDA",  "PLC signal berdi — poyezd kesishmaga yaqinlashmoqda"),
                      ("⚫", muted, "ALOQA YO'Q",        "PLC ga ulanib bo'lmadi — tarmoqni tekshiring"),
                      ("🔵", acc,   "O'CHIRILGAN",       "Sozlamalarda PLC yoqilmagan"),
                  ]],
                f'</table>',

                _warn_html(
                    '⚠ PLC va kompyuter <b>bir xil lokal tarmoqda</b> bo\'lishi shart. '
                    'Agar ular turli tarmoqlarda bo\'lsa — IT mutaxassisga murojaat qiling, '
                    'chunki <b>102-port ochiq</b> bo\'lishi kerak.<br><br>'
                    '⚠ PLC ga ulanish 1–3 sekund vaqt oladi. Agar doimiy "ALOQA YO\'Q" '
                    'ko\'rsatsa — IP manzil yoki port noto\'g\'ri ehtimoli yuqori.',
                    warn_bg, warn_brd, warn_txt
                ),
            ]

        elif key == "analitika":
            parts += [
                f'<p><font color="{acc}"><b>Sozlamalar qayerda?</b></font></p>',
                f'<p><font color="{txt}">Yuqori paneldagi <b>"⚙ Sozlamalar"</b> tugmasini bosing. '
                f'Sozlamalar oynasi ochiladi. O\'zgartirishlarni kiritib <b>"Saqlash"</b> '
                f'ni bossangiz — dasturni qayta yoqmasdan darhol qo\'llaniladi.</font></p>',

                f'<p><font color="{acc}"><b>🌐 Interfeys bo\'limi — til va rang:</b></font></p>',
                _info_html(
                    f'<b>Til (Language)</b> — dasturning barcha yozuvlari qaysi tilda bo\'lishini tanlaysiz:<br>'
                    f'&nbsp;&nbsp;• <b>O\'zbekcha (uz)</b> — barcha yozuvlar o\'zbek tilida<br>'
                    f'&nbsp;&nbsp;• <b>Русский (ru)</b> — rus tilida<br>'
                    f'&nbsp;&nbsp;• <b>English (en)</b> — ingliz tilida<br><br>'
                    f'Tilni o\'zgartirish tugmasini bossangiz — interfeys darhol yangilanadi.<br><br>'
                    f'<b>Mavzu (Theme)</b> — ekran rang sxemasini tanlaysiz:<br>'
                    f'&nbsp;&nbsp;• <b>Dark (Qorong\'i)</b> — qora fon, ko\'z charchamaydi, '
                    f'tungi kuzatish uchun qulay<br>'
                    f'&nbsp;&nbsp;• <b>Military (Harbiy)</b> — yashil-qora, nazorat punktlari uchun<br>'
                    f'&nbsp;&nbsp;• <b>Light (Yorug\')</b> — oq fon, kun yorug\'ida yaxshi ko\'rinadi',
                    info_bg, info_txt
                ),

                f'<p><font color="{acc}"><b>⏱ Monitoring bo\'limi — ogohlantirish vaqtlari:</b></font></p>',
                f'<p><font color="{txt}">Bu bo\'limda transport qancha vaqt pereezdda turganda '
                f'ogohlantirish berilishini sozlaysiz. Bu pereezd to\'silganda xabar berish '
                f'uchun kerak.</font></p>',
                _info_html(
                    f'<b>Ogohlantirish vaqti</b> (sekund):<br>'
                    f'Mashina yoki boshqa transport polygon (hisoblash zonasi) ichida '
                    f'bu vaqtdan <b>ko\'proq turganda</b> — ekranda '
                    f'<font color="{warn}"><b>sariq rang</b></font> bilan belgilanadi. '
                    f'Bu "diqqat, uzoq turibdi" belgisi.<br>'
                    f'Standart: <b>10 sekund</b><br><br>'
                    f'<b>Buzilish vaqti</b> (sekund):<br>'
                    f'Transport yanada ko\'proq turganda — '
                    f'<font color="{err}"><b>qizil rang</b></font> bilan xavf belgisi beriladi. '
                    f'Bu "pereezd to\'silgan" holati.<br>'
                    f'Standart: <b>15 sekund</b><br><br>'
                    f'Ikkala qiymatni ham o\'zingizning xohishingizga ko\'ra o\'zgartiring.',
                    info_bg, info_txt
                ),

                f'<p><font color="{acc}"><b>🤖 AI Model bo\'limi — qaysi model ishlatiladi:</b></font></p>',
                f'<p><font color="{txt}">Dastur transport vositalarini aniqlash uchun '
                f'sun\'iy intellekt (AI) modelidan foydalanadi. Ikki xil model mavjud:</font></p>',
                _info_html(
                    f'<b>Standart model (YOLOv8 COCO)</b><br>'
                    f'80 xil ob\'ektni taniydi (mashina, odam, velosiped va hokazo). '
                    f'Tezkor yuklash, alohida GPU talab qilmaydi. '
                    f'Oddiy testlash uchun qulay.<br><br>'
                    f'<b>Maxsus model (pereezd_v12)</b><br>'
                    f'Faqat pereezdlar uchun maxsus o\'qitilgan. Faqat 2 sinfni taniydi: '
                    f'<b>yengil transport</b> (avtomobil, mikroavtobus) va '
                    f'<b>og\'ir transport</b> (yuk mashinasi, avtobus). '
                    f'Aniqlik: <b>~95%</b>. NVIDIA GPU bo\'lsa juda tez ishlaydi.<br><br>'
                    f'⚠ Maxsus model <b>birinchi marta</b> tanlanganida NVIDIA GPU orqali '
                    f'TensorRT formatiga o\'tkaziladi. Bu jarayon <b>5–20 daqiqa</b> '
                    f'davom etadi. Shu payt dasturni yopmang!',
                    info_bg, info_txt
                ),

                _tip_html(
                    'Agar kompyuteringizda NVIDIA GPU bo\'lsa — maxsus modelni tanlang. '
                    'Aniqroq va tezroq ishlaydi. GPU bo\'lmasa — standart modeldan foydalaning.',
                    tip_col, tip_bg
                ),

                _warn_html(
                    '⚠ Model o\'zgartirilgandan so\'ng dastur kameralarni qayta ishga tushiradi. '
                    'Bir necha sekund kamera signali ko\'rinmasligi mumkin — bu normal holat.',
                    warn_bg, warn_brd, warn_txt
                ),
            ]

        elif key == "bashorat":
            parts += [
                f'<p><font color="{acc}"><b>Tahlil sahifasi nima?</b></font></p>',
                f'<p><font color="{txt}">Tahlil sahifasida barcha yig\'ilgan statistik '
                f'ma\'lumotlar grafiklar ko\'rinishida ko\'rsatiladi. Bu sahifada '
                f'qaysi pereezdda qancha transport o\'tgani, qaysi soat va kunlarda '
                f'eng ko\'p harakatlanish bo\'lgani, poyezdlar qachon o\'tgani — '
                f'barchasini ko\'rish mumkin.</font></p>',

                f'<p><font color="{txt}"><b>Kirish:</b> Yuqori paneldagi '
                f'<b>"∿ Tahlil"</b> tugmasini bosing. Chap tomondagi ro\'yxatdan '
                f'pereezdni tanlang.</font></p>',

                f'<p><font color="{acc}"><b>📊 Grafik turlari va ularning ma\'nosi:</b></font></p>',
                _info_html(
                    f'<b>Bugungi taqsimot</b> (doira grafik)<br>'
                    f'Bugun o\'tgan transport nechta yengil, nechta og\'ir ekanini '
                    f'foiz va sonda ko\'rsatadi. Bir qarashda umumiy manzarani beradi.<br><br>'

                    f'<b>Haftalik statistika</b> (ustunli grafik)<br>'
                    f'So\'nggi 7 kunlik transport soni. Qaysi kunlarda ko\'p, '
                    f'qaysi kunlarda kam harakatlanganini ko\'rasiz. '
                    f'Dam olish kunlari va ish kunlari farqini ko\'rish mumkin.<br><br>'

                    f'<b>Oylik trend</b> (chiziqli grafik)<br>'
                    f'So\'nggi 30 kunlik o\'zgarish. Transport soni oshyaptimi yoki '
                    f'kamayayaptimi — shu chiziqdan ko\'rinadi.<br><br>'

                    f'<b>Soatlik taqsimot</b> (bugun)<br>'
                    f'Bugun qaysi soatda qancha transport o\'tgani. '
                    f'Eng band soat va bo\'sh soatlarni aniqlashga yordam beradi.<br><br>'

                    f'<b>Issiqlik xaritasi (Heat Map)</b><br>'
                    f'7 kun × 24 soat kesimida bandlik. Qaysi kun va qaysi soatda '
                    f'eng ko\'p transport bo\'lgani rangli jadvalda ko\'rsatiladi. '
                    f'Quyuq rang = ko\'p, och rang = kam.<br><br>'

                    f'<b>Poyezd harakati</b><br>'
                    f'Poyezdlar qachon o\'tgani, o\'tish vaqtlari (sekund), '
                    f'so\'nggi 7 va 30 kundagi poyezd soni.',
                    info_bg, info_txt
                ),

                f'<p><font color="{acc}"><b>📄 Word formatda hisobot olish:</b></font></p>',
                f'<p><font color="{txt}">Hisobot — tanlangan davr uchun barcha statistikani '
                f'o\'z ichiga olgan rasmiy hujjat. Microsoft Word (.docx) formatida '
                f'saqlanadi va chop etish uchun tayyor bo\'ladi.</font></p>',

                _step_html(1,
                    'Tahlil sahifasining yuqori qismida <b>"📄 Hisobot yuklash"</b> '
                    'tugmasini bosing.', acc, txt),
                _step_html(2,
                    'Qaysi davr uchun hisobot olishni tanlang:<br>'
                    '&nbsp;&nbsp;• <b>Bugun</b> — faqat bugungi ma\'lumotlar<br>'
                    '&nbsp;&nbsp;• <b>7 kun</b> — so\'nggi bir hafta<br>'
                    '&nbsp;&nbsp;• <b>30 kun</b> — so\'nggi bir oy<br>'
                    '&nbsp;&nbsp;• <b>1 yil</b> — so\'nggi 12 oy<br>'
                    '&nbsp;&nbsp;• Yoki "Dan / Gacha" maydonlariga o\'z sanangizni '
                    'qo\'lda kiriting.',
                    acc, txt),
                _step_html(3,
                    '<b>"Yuklash (.docx)"</b> tugmasini bosing. Fayl saqlash joyi '
                    'so\'raladi — xohlagan joyni tanlang va saqlang.', acc, txt),
                _step_html(4,
                    'Saqlangan faylni <b>Microsoft Word</b> yoki <b>LibreOffice Writer</b> '
                    'dasturida oching. Chop etishga (print) tayyor.', acc, txt),

                _info_html(
                    f'<b>Hisobotda quyidagilar bo\'ladi:</b><br>'
                    f'&nbsp;&nbsp;✅ Tanlangan davr uchun umumiy transport soni<br>'
                    f'&nbsp;&nbsp;✅ Har bir pereezd va kamera bo\'yicha alohida jadvallar<br>'
                    f'&nbsp;&nbsp;✅ Yengil va og\'ir transport alohida ko\'rsatilgan<br>'
                    f'&nbsp;&nbsp;✅ Poyezd o\'tishlari vaqti va sanasi ro\'yxati<br>'
                    f'&nbsp;&nbsp;✅ Eng band kunlar va soatlar tahlili<br>'
                    f'&nbsp;&nbsp;✅ Tashkilot nomi va sana avtomatik qo\'yiladi',
                    info_bg, info_txt
                ),

                _tip_html(
                    'Oylik rasmiy hisobot uchun har oyning oxirida '
                    '"30 kun" ni tanlab, hisobotni yuklab oling. '
                    'Pereezdni o\'ng bosib "JSON Eksport" orqali zaxira nusxa ham oling.',
                    tip_col, tip_bg
                ),
            ]

        elif key == "versiya":
            parts += [
                f'<p><font color="{acc}" style="font-size:16px;"><b>Dastur haqida:</b></font></p>',
                _info_html(
                    f'<font style="font-size:16px;">'
                    f'<b>Dastur nomi:</b> RailSafe AI<br>'
                    f'<b>Versiya:</b> 1.0.5<br>'
                    f'<b>Chiqarilgan yil:</b> 2026<br>'
                    f'<b>Tegishli tashkilot:</b> DAS-UTY LLC<br>'
                    f'<b>Muallif:</b> Muhammadiyev Bahrombek<br>'
                    f'<b>Telefon:</b> +998 94 021 62 27 &nbsp;|&nbsp; +998 91 518 02 62<br>'
                    f'<b>Mos keluvchi tizimlar:</b> Windows 10 va Windows 11 (64-bit)'
                    f'</font>',
                    info_bg, info_txt
                ),

                f'<p><font color="{acc}" style="font-size:16px;"><b>💻 Dastur ishlashi uchun minimal talablar:</b></font></p>',
                _info_html(
                    f'<font style="font-size:16px;">'
                    f'<b>Operatsion tizim:</b> Windows 10 yoki Windows 11 (64-bit)<br>'
                    f'<b>Protsessor (CPU):</b> Intel Core i5 8-avlod va yuqori, '
                    f'yoki AMD Ryzen 5<br>'
                    f'<b>Grafik karta (GPU):</b> NVIDIA RTX 3035 va yuqori — '
                    f'AI tezlashuvi (TensorRT) uchun. GPU bo\'lmasa ham ishlaydi, '
                    f'lekin sekinroq.<br>'
                    f'<b>Operativ xotira (RAM):</b> Kamida 8 GB, 16 GB tavsiya etiladi<br>'
                    f'<b>Tarmoq:</b> Kameralar va PLC qurilmasi bilan bir lokal tarmoqda '
                    f'bo\'lishi shart<br>'
                    f'<b>Kamera protokoli:</b> RTSP (H.264 yoki H.265 siqish)'
                    f'</font>',
                    info_bg, info_txt
                ),

                f'<br>',
                _info_html(
                    f'<font style="font-size:16px;">'
                    f'© 2026 DAS-UTY LLC, O\'zbekiston. Barcha huquqlar himoyalangan.<br>'
                    f'<font color="{muted}">Muallif: Muhammadiyev Bahrombek<br>'
                    f'Tel: +998 94 021 62 27 &nbsp;|&nbsp; +998 91 518 02 62</font>'
                    f'</font>',
                    info_bg, info_txt
                ),
            ]

        parts.append('</body>')
        return ''.join(parts)

    # ── UI setup ─────────────────────────────────────────────────────────────
    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self._create_sidebar())
        main_layout.addWidget(self._create_content_area(), 1)

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
        layout.setContentsMargins(10, 20, 10, 20)
        layout.setSpacing(3)

        self._sidebar_title_lbl = QLabel(t("about.title"))
        self._sidebar_title_lbl.setStyleSheet(f"""
            color: {C('text_primary')}; font-size: 15px; font-weight: bold;
            padding: 8px 12px;
        """)
        layout.addWidget(self._sidebar_title_lbl)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"background:{C('border_light')}; border:none; max-height:1px; margin:4px 8px;")
        layout.addWidget(div)
        layout.addSpacing(4)

        self.nav_buttons = {}
        self._nav_icon_keys = {}
        for key, icon, nav_t_key, _ in _NAV_ITEMS:
            btn = QPushButton(f"  {icon}  {t(nav_t_key)}")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(self._nav_btn_style(key == self._current_section))
            btn.clicked.connect(lambda checked, k=key: self._select_section(k))
            self.nav_buttons[key] = btn
            self._nav_icon_keys[key] = (icon, nav_t_key)
            layout.addWidget(btn)

        layout.addStretch()

        self._version_lbl = QLabel(t("about.version_badge"))
        self._version_lbl.setWordWrap(True)
        self._version_lbl.setStyleSheet(f"color:{C('text_muted')}; font-size:9px; padding:8px 12px;")
        layout.addWidget(self._version_lbl)
        return sidebar

    def _nav_btn_style(self, active=False):
        if active:
            return f"""
                QPushButton {{
                    background-color: {C('accent_brand')}22;
                    color: {C('accent_brand')};
                    border: 1px solid {C('accent_brand')}44;
                    border-radius: 6px; padding: 9px 12px;
                    text-align: left; font-size: 12px; font-weight: bold;
                }}"""
        return f"""
            QPushButton {{
                background-color: transparent; color: {C('text_secondary')};
                border: none; border-radius: 6px; padding: 9px 12px;
                text-align: left; font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {C('bg_hover')}; color: {C('text_primary')};
            }}"""

    def _create_content_area(self):
        container = QWidget()
        container.setStyleSheet(f"background-color: {C('bg_primary')};")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(f"""
            QScrollArea {{ border: none; background-color: {C('bg_primary')}; }}
            QScrollBar:vertical {{
                background: {C('bg_secondary')}; width: 6px; border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {C('text_muted')}; border-radius: 3px; min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        content_widget = QWidget()
        content_widget.setStyleSheet(f"background-color: {C('bg_primary')};")
        self._content_layout = QVBoxLayout(content_widget)
        self._content_layout.setContentsMargins(20, 16, 20, 20)
        self._content_layout.setSpacing(0)

        self.sections: dict[str, QFrame] = {}
        self._section_title_lbls: dict[str, tuple] = {}
        self._browsers: dict[str, QTextBrowser] = {}

        for key, _icon, _nav_t_key, section_t_key in _NAV_ITEMS:
            section, title_lbl, browser = self._create_section(key, t(section_t_key))
            self._section_title_lbls[key] = (title_lbl, section_t_key)
            self._browsers[key] = browser
            self._content_layout.addWidget(section)
            self.sections[key] = section
            section.setVisible(key == self._current_section)

        self._content_layout.addStretch()
        self.scroll_area.setWidget(content_widget)
        layout.addWidget(self.scroll_area)
        return container


    def _create_section(self, key: str, title: str):
        section = QFrame()
        section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        section.setStyleSheet(f"""
            QFrame {{
                background-color: {C('bg_card')};
                border: 1px solid {C('border_card')};
                border-radius: 10px;
            }}
        """)
        outer = QVBoxLayout(section)
        outer.setContentsMargins(20, 14, 20, 16)
        outer.setSpacing(10)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            color: {C('accent_brand')}; font-size: 16px; font-weight: bold;
            border: none; background: transparent;
        """)
        outer.addWidget(title_label)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"background:{C('border_light')}; border:none; max-height:1px;")
        outer.addWidget(divider)

        # Text browser (faqat matn, rasm yo'q)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        browser.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {C('bg_card')};
                border: none;
                color: {C('text_secondary')};
                font-size: 16px;
                padding: 0px;
            }}
            QScrollBar:vertical {{ width: 0px; }}
            QScrollBar:horizontal {{ height: 0px; }}
        """)
        browser.setHtml(self._build_html(key))
        outer.addWidget(browser)

        # Browser balandligini hisoblash
        browser.document().setTextWidth(900)
        browser.document().adjustSize()
        h = int(browser.document().size().height()) + 8
        browser.document().setTextWidth(-1)
        browser.setFixedHeight(h)

        return section, title_label, browser

    def _select_section(self, section_key: str):
        self._current_section = section_key
        for key, btn in self.nav_buttons.items():
            btn.setStyleSheet(self._nav_btn_style(key == section_key))
        for key, section in self.sections.items():
            section.setVisible(key == section_key)
        if hasattr(self, 'scroll_area'):
            self.scroll_area.verticalScrollBar().setValue(0)

    def _retranslate(self, _lang=None):
        if hasattr(self, '_sidebar_title_lbl'):
            self._sidebar_title_lbl.setText(t("about.title"))
        for key, btn in self.nav_buttons.items():
            icon, nav_t_key = self._nav_icon_keys[key]
            btn.setText(f"  {icon}  {t(nav_t_key)}")
        if hasattr(self, '_version_lbl'):
            self._version_lbl.setText(t("about.version_badge"))
        for key, (lbl, section_t_key) in self._section_title_lbls.items():
            if lbl is not None:
                lbl.setText(t(section_t_key))
