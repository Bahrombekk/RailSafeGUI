"""
RailSafe AI — tanishtiruv roliki montaji (o'zbekcha, ovozsiz).

demo_record.py yozgan xom materialdan 1920x1080 / 30 fps rolik yig'adi:
kirish kartasi → jonli ekran lavhalari (matnli izohlar bilan) → yakuniy karta.

    .venv/Scripts/python.exe tools/make_promo.py

Natija: build/video/RailSafeAI_promo_uz.mp4
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

VID = ROOT / "build" / "video"
TMP = VID / "parts"
TMP.mkdir(parents=True, exist_ok=True)
OUT = VID / "RailSafeAI_promo_uz.mp4"

W, H, FPS = 1920, 1080, 30

# Dastur "military" mavzusidan olingan ranglar
BG = (28, 33, 24)
BG_DEEP = (16, 20, 13)
CARD = (34, 43, 28)
TEXT = (210, 216, 190)
MUTED = (160, 168, 120)
ACCENT = (139, 166, 90)
ACCENT2 = (122, 174, 106)

F_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
F_REG = "C:/Windows/Fonts/segoeui.ttf"
F_LIGHT = "C:/Windows/Fonts/segoeuisl.ttf"
LOGO = ROOT / "installer_assets" / "railsafe_icon_src.png"


def font(path, size):
    return ImageFont.truetype(path, size)


def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        probe = (cur + " " + w_).strip()
        if draw.textlength(probe, font=fnt) <= max_w:
            cur = probe
        else:
            if cur:
                lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def draw_logo(img, size, xy):
    if not LOGO.exists():
        return
    logo = Image.open(LOGO).convert("RGBA").resize((size, size), Image.LANCZOS)
    img.paste(logo, xy, logo)


# ─── Kartalar (to'liq ekran) ──────────────────────────────────────────

def card_intro(path):
    img = Image.new("RGB", (W, H), BG_DEEP)
    d = ImageDraw.Draw(img)
    # yumshoq fon yoritmasi
    for i in range(H):
        k = 1 - abs(i - H * 0.45) / (H * 0.9)
        c = int(6 * max(0, k))
        d.line([(0, i), (W, i)], fill=(BG_DEEP[0] + c, BG_DEEP[1] + c, BG_DEEP[2] + c))

    draw_logo(img, 210, (W // 2 - 105, 250))
    f1, f2 = font(F_BOLD, 104), font(F_LIGHT, 40)
    t1 = "RailSafe AI"
    d.text((W // 2 - d.textlength(t1, font=f1) / 2, 500), t1, font=f1, fill=TEXT)
    d.rectangle([W // 2 - 90, 640, W // 2 + 90, 644], fill=ACCENT)
    t2 = "Temir yo'l kesishmalarini sun'iy intellekt bilan nazorat qilish"
    d.text((W // 2 - d.textlength(t2, font=f2) / 2, 690), t2, font=f2, fill=MUTED)
    img.save(path)


def card_text(path, title, body_lines, kicker=None):
    img = Image.new("RGB", (W, H), BG_DEEP)
    d = ImageDraw.Draw(img)
    x = 190
    y = 400
    if kicker:
        fk = font(F_BOLD, 28)
        d.text((x, y - 70), kicker.upper(), font=fk, fill=ACCENT)
    d.rectangle([x - 40, y, x - 32, y + 150], fill=ACCENT)
    ft = font(F_BOLD, 72)
    for line in wrap(d, title, ft, W - x - 190):
        d.text((x, y), line, font=ft, fill=TEXT)
        y += 88
    y += 30
    fb = font(F_LIGHT, 40)
    for line in body_lines:
        for sub in wrap(d, line, fb, W - x - 220):
            d.text((x, y), sub, font=fb, fill=MUTED)
            y += 56
        y += 14
    draw_logo(img, 86, (W - 190, H - 190))
    img.save(path)


def card_outro(path, stats):
    img = Image.new("RGB", (W, H), BG_DEEP)
    d = ImageDraw.Draw(img)
    draw_logo(img, 150, (W // 2 - 75, 190))
    f1 = font(F_BOLD, 76)
    t1 = "RailSafe AI"
    d.text((W // 2 - d.textlength(t1, font=f1) / 2, 370), t1, font=f1, fill=TEXT)

    # Raqamlar qatori
    fn, fl = font(F_BOLD, 62), font(F_REG, 28)
    n = len(stats)
    colw = 1500 // n
    x0 = (W - 1500) // 2
    for i, (val, label) in enumerate(stats):
        cx = x0 + colw * i + colw // 2
        d.text((cx - d.textlength(val, font=fn) / 2, 540), val, font=fn, fill=ACCENT)
        d.text((cx - d.textlength(label, font=fl) / 2, 625), label, font=fl, fill=MUTED)
        if i:
            d.line([(x0 + colw * i - 10, 555), (x0 + colw * i - 10, 655)],
                   fill=(60, 72, 48), width=2)

    d.rectangle([W // 2 - 70, 730, W // 2 + 70, 733], fill=ACCENT)
    f3 = font(F_LIGHT, 34)
    for i, line in enumerate(
            ["Yagona ekranda jonli nazorat, statistika va hisobot",
             "Demo va joriy etish bo'yicha murojaat qiling"]):
        d.text((W // 2 - d.textlength(line, font=f3) / 2, 780 + i * 56),
               line, font=f3, fill=MUTED if i else TEXT)
    img.save(path)


# ─── Lavha ustidagi izoh (shaffof qatlam) ─────────────────────────────

def caption(path, kicker, title, subtitle):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # pastdan yuqoriga qorayish
    top = 700
    for i in range(top, H):
        a = int(235 * ((i - top) / (H - top)) ** 0.75)
        d.line([(0, i), (W, i)], fill=(BG_DEEP[0], BG_DEEP[1], BG_DEEP[2], a))
    x, y = 120, 836
    d.rectangle([x - 34, y + 6, x - 26, y + 150], fill=ACCENT + (255,))
    if kicker:
        d.text((x, y - 44), kicker.upper(), font=font(F_BOLD, 26),
               fill=ACCENT + (255,))
    d.text((x, y), title, font=font(F_BOLD, 58), fill=TEXT + (255,))
    d.text((x, y + 78), subtitle, font=font(F_LIGHT, 36), fill=MUTED + (255,))
    draw_logo(img, 64, (W - 150, H - 150))
    img.save(path)


# ─── ffmpeg yordamchilari ─────────────────────────────────────────────

def run(args):
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print(" ".join(str(a) for a in args))
        print(r.stderr[-2500:])
        raise SystemExit("ffmpeg xatosi")


ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
       "-pix_fmt", "yuv420p", "-r", str(FPS), "-an"]


def seg_card(png, dur, out):
    run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-t", str(dur), "-i", png,
         "-vf", f"scale={W}:{H},fade=t=in:st=0:d=0.5,"
                f"fade=t=out:st={dur - 0.5:.2f}:d=0.5,format=yuv420p",
         *ENC, out])


def seg_clip(src, ss, dur, cap_png, out, crop=None, zoom=None):
    """Xom yozuvdan lavha + izoh qatlami."""
    vf = []
    if crop:
        cw, ch, cx, cy = crop
        vf.append(f"crop={cw}:{ch}:{cx}:{cy},scale={W}:{H}")
    if zoom:
        # sekin yaqinlashish (Ken Burns)
        vf.append(f"scale={int(W * 1.12)}:-1,"
                  f"zoompan=z='min(zoom+0.00035,1.12)':d={int(dur * FPS)}:"
                  f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS}")
    vf.append(f"fps={FPS}")
    chain = ",".join(vf) if vf else f"fps={FPS}"
    # Izoh PNG'i ham OQIM bo'lishi shart (-loop 1): aks holda uning yagona
    # kadri t=0 da qoladi va fade-in uni butunlay shaffof qilib qo'yadi.
    run(["ffmpeg", "-v", "error", "-y",
         "-ss", str(ss), "-t", str(dur), "-i", str(src),
         "-loop", "1", "-framerate", str(FPS), "-t", str(dur), "-i", cap_png,
         "-filter_complex",
         f"[0:v]{chain},format=rgba[v];"
         f"[1:v]format=rgba,fade=t=in:st=0.3:d=0.6:alpha=1[c];"
         f"[v][c]overlay=0:0,format=yuv420p,"
         f"fade=t=in:st=0:d=0.4,fade=t=out:st={dur - 0.5:.2f}:d=0.5[o]",
         "-map", "[o]", *ENC, out])


def db_stats():
    """Yakuniy kartadagi raqamlar — haqiqiy bazadan."""
    try:
        from app.core.database import StatsDB
        from app.core.config import ConfigManager
        db = StatsDB(str(ROOT / "app" / "data" / "stats.db"))
        cfg = ConfigManager()
        crossings = cfg.get_crossings()
        total, trains = 0, 0
        for c in crossings:
            cat = None
            try:
                from app.reports.category import compute_category
                from datetime import date
                cat = compute_category(db, c["id"], "2000-01-01",
                                       date.today().isoformat())
            except Exception:
                pass
            if cat:
                total += cat["total_light"] + cat["total_heavy"]
                trains += int(cat["avg_trains"] * cat["days"])
        cams = sum(1 for c in crossings for cam in c.get("cameras", [])
                   if cam.get("enabled", True))
        def m(n):
            return f"{n / 1_000_000:.1f} mln" if n >= 1_000_000 else f"{n:,}".replace(",", " ")
        return [(m(total), "qayd etilgan transport"),
                (m(trains), "poyezd o'tishi"),
                (str(cams), "kamera / real vaqtda")]
    except Exception as e:
        print("stats xatosi:", e)
        return [("24/7", "uzluksiz nazorat"), ("I-IV", "toifa hisoboti"),
                ("API", "tashqi tizimga uzatish")]


def main():
    scenes = json.loads((VID / "scenes.json").read_text(encoding="utf-8"))
    raw = VID / "raw_app.mp4"
    dash = VID / "raw_dash.mp4"
    dash_src = dash if dash.exists() else raw
    dash_ss = float(os.environ.get("DASH_SS", "45"))

    parts = []

    # 1. Kirish
    p = TMP / "c_intro.png"; card_intro(p)
    o = TMP / "01.mp4"; seg_card(str(p), 4.5, str(o)); parts.append(o)

    # 2. Muammo
    p = TMP / "c_problem.png"
    card_text(p, "Pereezd — eng xavfli nuqta",
              ["Poyezd yaqinlashganda zonada qolgan transport,",
               "qancha mashina o'tgani, qachon tirbandlik bo'lgani —",
               "bularning hammasi odatda hech qayerda qayd etilmaydi."],
              kicker="Muammo")
    o = TMP / "02.mp4"; seg_card(str(p), 6.0, str(o)); parts.append(o)

    # 3. Jonli panel
    p = TMP / "cap_dash.png"
    caption(p, "Boshqaruv paneli", "Barcha pereezdlar — bitta ekranda",
            "Har bir kamera jonli, uzluksiz 24/7 kuzatuv")
    o = TMP / "03.mp4"
    seg_clip(dash_src, dash_ss, 13, str(p), str(o))
    parts.append(o)

    # 4. Aniqlash
    p = TMP / "cap_detect.png"
    caption(p, "Aniqlash", "Har bir transport aniqlanadi va kuzatiladi",
            "YOLO + TensorRT, belgilangan zona (polygon) bo'yicha hisob")
    o = TMP / "04.mp4"
    seg_clip(raw, 40.5, 13, str(p), str(o))
    parts.append(o)

    # 5. PLC
    p = TMP / "cap_plc.png"
    caption(p, "PLC bog'lanishi", "Poyezd signali — to'g'ridan-to'g'ri avtomatikadan",
            "Siemens S7 orqali: shlagbaum yopilganda zona nazoratga olinadi")
    o = TMP / "05.mp4"
    seg_clip(raw, 46.0, 8, str(p), str(o))
    parts.append(o)

    # 6. Tahlil
    p = TMP / "cap_stats.png"
    caption(p, "Tahlil", "Soatlik, haftalik va oylik statistika",
            "Kesishma toifasi (I–IV) va zator xaritasi avtomatik hisoblanadi")
    o = TMP / "06.mp4"
    seg_clip(raw, 62.0, 14, str(p), str(o))
    parts.append(o)

    # 7. Hisobot / integratsiya
    p = TMP / "cap_report.png"
    caption(p, "Hisobot va integratsiya", "Word hisobot va tashqi tizimga uzatish",
            "Statistika HTTP API orqali markaziy tizimga avtomatik yuboriladi")
    o = TMP / "07.mp4"
    seg_clip(raw, 87.5, 8, str(p), str(o))
    parts.append(o)

    # 8. Yakun
    p = TMP / "c_outro.png"; card_outro(p, db_stats())
    o = TMP / "08.mp4"; seg_card(str(p), 6.0, str(o)); parts.append(o)

    lst = TMP / "list.txt"
    lst.write_text("".join(f"file '{x.as_posix()}'\n" for x in parts),
                   encoding="utf-8")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", str(lst), "-c", "copy", str(OUT)])

    size = OUT.stat().st_size / 1_048_576
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of",
                          "default=noprint_wrappers=1:nokey=1", str(OUT)],
                         capture_output=True, text=True).stdout.strip()
    print(f"Tayyor: {OUT}  ({float(dur):.1f} s, {size:.1f} MB)")


if __name__ == "__main__":
    main()
