"""
ReportGenerator — Word (.docx) formatida rasmiy hisobot yaratish.
Belgilangan sana oralig'ida pereezd monitoring statistikasini eksport qiladi.
"""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from datetime import datetime, date, timedelta


def generate_report(config_manager, stats_db, date_from: str, date_to: str,
                    file_path: str) -> bool:
    """Word hisobot yaratish.

    Args:
        config_manager: CrossingManager instance
        stats_db: StatsDB instance
        date_from: "2026-02-01" format
        date_to: "2026-02-28" format
        file_path: Saqlash yo'li (.docx)

    Returns:
        True if successful
    """
    try:
        doc = Document()

        # Sahifa sozlamalari
        section = doc.sections[0]
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2)

        crossings = config_manager.get_crossings()

        _add_title_page(doc, date_from, date_to, crossings)
        _add_summary(doc, stats_db, crossings, date_from, date_to)

        for crossing in crossings:
            _add_crossing_section(doc, stats_db, crossing, date_from, date_to)

        _add_footer(doc, date_from, date_to)

        doc.save(file_path)
        return True

    except Exception as e:
        print(f"[ReportGenerator] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def _add_title_page(doc, date_from, date_to, crossings):
    """Sarlavha sahifasi"""
    # Bo'sh joy
    for _ in range(4):
        doc.add_paragraph("")

    # Asosiy sarlavha
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("TEMIR YO'L PEREEZD\nMONITORING HISOBOTI")
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(0x1a, 0x56, 0x76)

    doc.add_paragraph("")

    # Sana oralig'i
    period = doc.add_paragraph()
    period.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = period.add_run(f"Hisobot davri: {_format_date(date_from)} — {_format_date(date_to)}")
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    doc.add_paragraph("")

    # Qisqacha ma'lumot
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    total_cams = sum(len(cr.get("cameras", [])) for cr in crossings)
    run = info.add_run(f"Pereezdlar: {len(crossings)}  |  Kameralar: {total_cams}")
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # Yaratilgan vaqt
    doc.add_paragraph("")
    gen = doc.add_paragraph()
    gen.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = gen.add_run(f"Yaratilgan: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)

    doc.add_page_break()


def _add_summary(doc, stats_db, crossings, date_from, date_to):
    """Umumiy statistika bo'limi"""
    _section_title(doc, "1. UMUMIY STATISTIKA")

    grand_light = 0
    grand_heavy = 0
    for cr in crossings:
        l, h = stats_db.get_date_range_total(cr["id"], date_from, date_to)
        grand_light += l
        grand_heavy += h
    grand_total = grand_light + grand_heavy

    # Umumiy jadval
    table = doc.add_table(rows=4, cols=2)
    table.style = "Light Shading Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    data = [
        ("Ko'rsatkich", "Qiymat"),
        ("Jami transport", str(grand_total)),
        ("Yengil transport", str(grand_light)),
        ("Og'ir transport", str(grand_heavy)),
    ]
    for i, (label, value) in enumerate(data):
        row = table.rows[i]
        row.cells[0].text = label
        row.cells[1].text = value
        if i == 0:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.bold = True

    _set_table_font(table)
    doc.add_paragraph("")

    # Pereezdlar bo'yicha qisqacha
    _section_subtitle(doc, "Pereezdlar bo'yicha")
    table2 = doc.add_table(rows=1 + len(crossings), cols=5)
    table2.style = "Light Shading Accent 1"
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers = ["Pereezd", "Yengil", "Og'ir", "Jami", "Kameralar"]
    for i, h in enumerate(headers):
        table2.rows[0].cells[i].text = h
        for p in table2.rows[0].cells[i].paragraphs:
            for r in p.runs:
                r.bold = True

    for idx, cr in enumerate(crossings):
        l, h = stats_db.get_date_range_total(cr["id"], date_from, date_to)
        row = table2.rows[idx + 1]
        row.cells[0].text = cr.get("name", f"Pereezd #{cr['id']}")
        row.cells[1].text = str(l)
        row.cells[2].text = str(h)
        row.cells[3].text = str(l + h)
        row.cells[4].text = str(len(cr.get("cameras", [])))

    _set_table_font(table2)
    doc.add_paragraph("")


def _add_crossing_section(doc, stats_db, crossing, date_from, date_to):
    """Har bir pereezd uchun batafsil bo'lim"""
    cid = crossing["id"]
    name = crossing.get("name", f"Pereezd #{cid}")
    location = crossing.get("location", "")
    cameras = crossing.get("cameras", [])

    doc.add_page_break()
    _section_title(doc, f"PEREEZD: {name}")

    if location:
        p = doc.add_paragraph()
        run = p.add_run(f"Joylashuv: {location}")
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # Jami
    light, heavy = stats_db.get_date_range_total(cid, date_from, date_to)
    total = light + heavy

    p = doc.add_paragraph()
    run = p.add_run(f"Jami transport: {total}  (yengil: {light}, og'ir: {heavy})")
    run.bold = True
    run.font.size = Pt(11)

    doc.add_paragraph("")

    # Kameralar statistikasi
    if cameras:
        _section_subtitle(doc, "Kameralar")
        cam_table = doc.add_table(rows=1 + len(cameras), cols=5)
        cam_table.style = "Light Shading Accent 1"
        cam_table.alignment = WD_TABLE_ALIGNMENT.CENTER

        cam_headers = ["Kamera", "Turi", "Yengil", "Og'ir", "Jami"]
        for i, h in enumerate(cam_headers):
            cam_table.rows[0].cells[i].text = h
            for p in cam_table.rows[0].cells[i].paragraphs:
                for r in p.runs:
                    r.bold = True

        for idx, cam in enumerate(cameras):
            cn = cam.get("name", "?")
            ct = "Asosiy" if cam.get("type") == "main" else "Qo'shimcha"
            cl, ch = stats_db.get_date_range_camera(cid, cn, date_from, date_to)
            row = cam_table.rows[idx + 1]
            row.cells[0].text = cn
            row.cells[1].text = ct
            row.cells[2].text = str(cl)
            row.cells[3].text = str(ch)
            row.cells[4].text = str(cl + ch)

        _set_table_font(cam_table)
        doc.add_paragraph("")

    # Kunlik statistika jadvali
    daily_data = stats_db.get_date_range_daily(cid, date_from, date_to)
    if daily_data:
        _section_subtitle(doc, "Kunlik statistika")
        dt = doc.add_table(rows=1 + len(daily_data), cols=4)
        dt.style = "Light Shading Accent 1"
        dt.alignment = WD_TABLE_ALIGNMENT.CENTER

        for i, h in enumerate(["Sana", "Yengil", "Og'ir", "Jami"]):
            dt.rows[0].cells[i].text = h
            for p in dt.rows[0].cells[i].paragraphs:
                for r in p.runs:
                    r.bold = True

        for idx, d in enumerate(daily_data):
            row = dt.rows[idx + 1]
            row.cells[0].text = _format_date(d["date"])
            row.cells[1].text = str(d["light"])
            row.cells[2].text = str(d["heavy"])
            row.cells[3].text = str(d["light"] + d["heavy"])

        _set_table_font(dt)


def _add_footer(doc, date_from, date_to):
    """Xulosa bo'limi"""
    doc.add_paragraph("")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("— Hisobot tugadi —")
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
    run.italic = True

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = p2.add_run("RailSafe Monitoring System")
    run2.font.size = Pt(9)
    run2.font.color.rgb = RGBColor(0xaa, 0xaa, 0xaa)


# ─── Helpers ──────────────────────────────────────────────

def _section_title(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0x1a, 0x56, 0x76)
    p.space_after = Pt(6)


def _section_subtitle(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    p.space_after = Pt(4)


def _format_date(date_str):
    """2026-02-01 → 01.02.2026"""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return d.strftime("%d.%m.%Y")
    except (ValueError, TypeError):
        return date_str


def _set_table_font(table, size=9):
    """Jadval shriftini kichikroq qilish"""
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(size)
