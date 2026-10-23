"""
RailSafe AI — sotuv (taqdimot) roliki: animatsiyali, o'zbekcha, ovozsiz.

Animatsion kartalar kadrma-kadr PIL bilan chiziladi va to'g'ridan-to'g'ri
ffmpeg'ga uzatiladi; jonli lavhalar demo_record.py yozgan materialdan
olinadi (izoh paneli chapdan sirg'alib chiqadi + sekin yaqinlashish).

    .venv/Scripts/python.exe tools/make_sales_video.py

Natija: build/video/RailSafeAI_sotuv_uz.mp4
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

VID = ROOT / "build" / "video"
TMP = VID / "parts_sales"
TMP.mkdir(parents=True, exist_ok=True)
OUT = VID / "RailSafeAI_sotuv_uz.mp4"

W, H, FPS = 1920, 1080, 30

BG = (16, 20, 13)
PANEL = (28, 34, 22)
TEXT = (214, 220, 196)
MUTED = (154, 164, 124)
DIM = (96, 106, 78)
ACCENT = (139, 166, 90)
ACCENT_SOFT = (122, 174, 106)

F_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
F_REG = "C:/Windows/Fonts/segoeui.ttf"
F_LIGHT = "C:/Windows/Fonts/segoeuisl.ttf"
LOGO_P = ROOT / "installer_assets" / "railsafe_icon_src.png"
_LOGO = Image.open(LOGO_P).convert("RGBA") if LOGO_P.exists() else None
_fonts = {}


def F(path, size):
    key = (path, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, size)
    return _fonts[key]


# ─── yordamchi: tezlanish egri chiziqlari ─────────────────────────────

def ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def seg(t, start, dur=0.6):
    """start dan boshlab dur ichida 0→1 (tashqarida 0 yoki 1)."""
    return ease_out((t - start) / dur) if t > start else 0.0


def mix(c1, c2, k):
    return tuple(int(a + (b - a) * k) for a, b in zip(c1, c2))


def fade_text(d, xy, text, fnt, color, k, dy=26, anchor=None):
    """Pastdan ko'tarilib paydo bo'luvchi matn."""
    if k <= 0.001:
        return
    x, y = xy
    d.text((x, y + dy * (1 - k)), text, font=fnt,
           fill=mix(BG, color, k), anchor=anchor)


def logo(img, size, xy, k=1.0):
    if _LOGO is None or k <= 0.01:
        return
    s = max(1, int(size * (0.86 + 0.14 * k)))
    lg = _LOGO.resize((s, s), Image.LANCZOS)
    if k < 1:
        a = lg.getchannel("A").point(lambda v: int(v * k))
        lg.putalpha(a)
    img.paste(lg, (xy[0] - (s - size) // 2, xy[1] - (s - size) // 2), lg)


def wrap(d, text, fnt, max_w):
    out, cur = [], ""
    for w_ in text.split():
        probe = (cur + " " + w_).strip()
        if d.textlength(probe, font=fnt) <= max_w:
            cur = probe
        else:
            out.append(cur)
            cur = w_
    if cur:
        out.append(cur)
    return out


def bg_frame():
    """Yengil vinyetkali fon."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for i in range(0, H, 2):
        k = 1 - abs(i - H * 0.42) / (H * 1.1)
        c = int(7 * max(0.0, k))
        d.line([(0, i), (W, i + 1)], fill=(BG[0] + c, BG[1] + c, BG[2] + c))
    return img


# ─── ffmpeg ga kadr uzatish ───────────────────────────────────────────

ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
       "-pix_fmt", "yuv420p", "-r", str(FPS), "-an"]


def render_anim(draw_fn, dur, out, fade_out=0.55):
    n = int(dur * FPS)
    vf = (f"fade=t=in:st=0:d=0.45,"
          f"fade=t=out:st={dur - fade_out:.2f}:d={fade_out}")
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-vf", vf, *ENC, str(out)],
        stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(draw_fn(i / FPS).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit(f"ffmpeg xatosi: {out}")


def run(args):
    r = subprocess.run(args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(" ".join(str(a) for a in args))
        print(r.stderr[-2000:])
        raise SystemExit("ffmpeg xatosi")


def caption_png(path, kicker, title, subtitle, note=None):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if note and os.environ.get("DEMO_BADGE", "1") != "0":
        # Grafiklardagi raqamlar namuna ekanini ochiq ko'rsatamiz
        fn = F(F_BOLD, 24)
        tw = d.textlength(note.upper(), font=fn)
        d.rounded_rectangle([W - 90 - tw, 54, W - 46, 104], 12,
                            fill=(34, 42, 26, 215), outline=(74, 92, 58, 255))
        d.text((W - 68 - tw, 66), note.upper(), font=fn, fill=MUTED + (255,))
    top = 690
    for i in range(top, H):
        a = int(238 * ((i - top) / (H - top)) ** 0.7)
        d.line([(0, i), (W, i)], fill=BG + (a,))
    x, y = 120, 832
    d.rectangle([x - 34, y + 4, x - 26, y + 152], fill=ACCENT + (255,))
    d.text((x, y - 46), kicker.upper(), font=F(F_BOLD, 26), fill=ACCENT + (255,))
    d.text((x, y), title, font=F(F_BOLD, 58), fill=TEXT + (255,))
    d.text((x, y + 78), subtitle, font=F(F_LIGHT, 36), fill=MUTED + (255,))
    img.save(path)


def seg_clip(src, ss, dur, cap, out, zoom=True):
    """Jonli lavha: sekin yaqinlashish + chapdan sirg'aluvchi izoh paneli."""
    z = (f"scale=-2:{int(H * 1.08)},crop={W}:{H},"
         f"zoompan=z='min(1+0.0007*on,1.07)':d=1:x='iw/2-(iw/zoom/2)':"
         f"y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS}") if zoom else f"fps={FPS}"
    run(["ffmpeg", "-v", "error", "-y",
         "-ss", str(ss), "-t", str(dur), "-i", str(src),
         "-loop", "1", "-framerate", str(FPS), "-t", str(dur), "-i", str(cap),
         "-filter_complex",
         f"[0:v]{z},format=rgba[v];"
         f"[v][1:v]overlay=x='-w+w*min(1,(max(0,t-0.25)/0.55))':y=0:"
         f"eval=frame,format=yuv420p,"
         f"fade=t=in:st=0:d=0.4,fade=t=out:st={dur - 0.5:.2f}:d=0.5[o]",
         "-map", "[o]", *ENC, str(out)])


# ─── Animatsion sahnalar ──────────────────────────────────────────────

def sc_intro(t):
    img = bg_frame()
    d = ImageDraw.Draw(img)
    logo(img, 200, (W // 2 - 100, 268), seg(t, 0.15, 0.7))
    fade_text(d, (W // 2, 500), "RailSafe AI", F(F_BOLD, 104), TEXT,
              seg(t, 0.55, 0.7), anchor="ma")
    k = seg(t, 1.0, 0.6)
    if k > 0:
        d.rectangle([W // 2 - 95 * k, 648, W // 2 + 95 * k, 652], fill=ACCENT)
    fade_text(d, (W // 2, 700),
              "Temir yo'l kesishmalarida sun'iy intellekt nazorati",
              F(F_LIGHT, 40), MUTED, seg(t, 1.3, 0.7), anchor="ma")
    return img


def sc_problem(t):
    img = bg_frame()
    d = ImageDraw.Draw(img)
    x = 200
    fade_text(d, (x, 236), "MUAMMO", F(F_BOLD, 28), ACCENT, seg(t, 0.2))
    ft = F(F_BOLD, 68)
    for i, line in enumerate(["Pereezdda nima bo'layotganini",
                              "hech kim aniq bilmaydi"]):
        fade_text(d, (x, 296 + i * 84), line, ft, TEXT, seg(t, 0.4 + i * 0.18))
    items = [
        "Transport hisobi qo'lda yuritiladi — raqamlar taxminiy",
        "Hodisa yoki nizoda obyektiv dalil qolmaydi",
        "Toifa va statistika hisoboti haftalab tayyorlanadi",
    ]
    fi = F(F_LIGHT, 42)
    for i, it in enumerate(items):
        k = seg(t, 1.5 + i * 0.5, 0.7)
        if k <= 0.001:
            continue
        y = 560 + i * 92
        dx = int(60 * (1 - k))
        d.rectangle([x - dx, y + 20, x + 26 - dx, y + 24],
                    fill=mix(BG, ACCENT, k))
        d.text((x + 56 - dx, y), it, font=fi, fill=mix(BG, MUTED, k))
    return img


PIPE_NODES = [("Kamera", "mavjud kameralar"),
              ("Sun'iy intellekt", "transportni taniydi"),
              ("Zona nazorati", "belgilangan hudud"),
              ("Poyezd signali", "avtomatikadan"),
              ("Hisobot", "statistika va dalil")]


def sc_pipeline(t):
    img = bg_frame()
    d = ImageDraw.Draw(img)
    fade_text(d, (W // 2, 200), "QANDAY ISHLAYDI", F(F_BOLD, 28), ACCENT,
              seg(t, 0.15), anchor="ma")
    fade_text(d, (W // 2, 256), "Kameradan hisobotgacha — avtomatik",
              F(F_BOLD, 62), TEXT, seg(t, 0.35), anchor="ma")

    n = len(PIPE_NODES)
    bw, bh, gap = 290, 150, 44
    total = n * bw + (n - 1) * gap
    x0, y = (W - total) // 2, 520
    fb, fs = F(F_BOLD, 32), F(F_REG, 24)
    for i, (name, sub) in enumerate(PIPE_NODES):
        k = seg(t, 1.0 + i * 0.3, 0.55)
        if k <= 0.001:
            continue
        bx = x0 + i * (bw + gap)
        oy = int(24 * (1 - k))
        d.rounded_rectangle([bx, y + oy, bx + bw, y + bh + oy], 14,
                            fill=mix(BG, PANEL, k),
                            outline=mix(BG, (58, 74, 46), k), width=2)
        d.text((bx + bw // 2, y + 44 + oy), name, font=fb,
               fill=mix(BG, TEXT, k), anchor="ma")
        d.text((bx + bw // 2, y + 92 + oy), sub, font=fs,
               fill=mix(BG, DIM, k), anchor="ma")
        if i < n - 1:
            ka = seg(t, 1.22 + i * 0.3, 0.4)
            if ka > 0:
                ax, ay = bx + bw + 8, y + bh // 2
                d.line([(ax, ay), (ax + int((gap - 16) * ka), ay)],
                       fill=mix(BG, ACCENT, ka), width=3)

    # oqimni ko'rsatuvchi yorug' nuqta
    if t > 2.9:
        p = ((t - 2.9) / 2.6) % 1.0
        px = x0 + p * total
        py = y + bh // 2
        for r, a in ((13, 0.18), (8, 0.45), (4, 1.0)):
            d.ellipse([px - r, py - r, px + r, py + r],
                      fill=mix(PANEL, ACCENT_SOFT, a))
    return img


# ─── "Mashina zonada qolib ketsa?" sahnasi ────────────────────────────

ROAD_Y0, ROAD_Y1 = 452, 648
RAIL_X = 960
ZONE = [(852, 436), (1082, 436), (1108, 664), (826, 664)]


def _car(d, cx, cy, w=156, h=74, color=(196, 204, 186), alert=0.0):
    body = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
    d.rounded_rectangle(body, 14, fill=mix(color, (200, 70, 60), alert * 0.55))
    d.rounded_rectangle((body[0] + 26, body[1] + 9, body[0] + 96, body[1] + 32),
                        6, fill=(52, 60, 48))
    for wx in (body[0] + 34, body[2] - 34):
        d.ellipse([wx - 15, body[3] - 12, wx + 15, body[3] + 16],
                  fill=(26, 30, 24))


def _train(d, y):
    # Yuqoridan "kadrga kirib kelayotgan" poyezd: sarlavha ustiga chiqmasligi
    # uchun tepasi 300 px da kesiladi.
    x0, x1 = RAIL_X - 54, RAIL_X + 54
    top = max(y - 300, 300)
    if y - top < 30:
        return
    d.rounded_rectangle([x0, top, x1, y], 22, fill=(58, 70, 86))
    d.rounded_rectangle([x0 + 10, y - 86, x1 - 10, y - 34], 10,
                        fill=(96, 120, 146))
    d.ellipse([RAIL_X - 16, y - 26, RAIL_X + 16, y - 6], fill=(226, 212, 150))


def sc_scenario(t):
    img = bg_frame()
    d = ImageDraw.Draw(img)
    fade_text(d, (W // 2, 150), "SSENARIY", F(F_BOLD, 28), ACCENT,
              seg(t, 0.1), anchor="ma")
    fade_text(d, (W // 2, 204), "Mashina zonada qolib ketsa?",
              F(F_BOLD, 64), TEXT, seg(t, 0.3), anchor="ma")

    k_draw = seg(t, 1.0, 0.8)
    if k_draw > 0.02:
        # yo'l
        d.rectangle([180, ROAD_Y0, 1740, ROAD_Y1], fill=mix(BG, (44, 48, 40), k_draw))
        for mx in range(220, 1740, 120):      # markaziy chiziq
            d.rectangle([mx, 546, mx + 58, 552], fill=mix(BG, (108, 112, 92), k_draw))
        # relslar
        for rx in (RAIL_X - 30, RAIL_X + 30):
            d.rectangle([rx - 5, 300, rx + 5, 820], fill=mix(BG, (120, 126, 112), k_draw))
        for sy in range(312, 820, 46):        # shpallar
            d.rectangle([RAIL_X - 56, sy, RAIL_X + 56, sy + 11],
                        fill=mix(BG, (74, 66, 54), k_draw))

    # zona (polygon) — yashildan qizilga
    alert = seg(t, 4.6, 1.0)
    k_zone = seg(t, 1.6, 0.7)
    if k_zone > 0.02:
        zc = mix(ACCENT, (206, 72, 62), alert)
        d.line(ZONE + [ZONE[0]], fill=mix(BG, zc, k_zone), width=5, joint="curve")

    # mashina: chapdan kirib, zonada to'xtaydi
    if t > 1.8:
        p = ease_out(min(1.0, (t - 1.8) / 2.0))
        _car(d, 150 + (960 - 150) * p, 550, alert=alert)

    # poyezd signali + shlagbaum + yaqinlashayotgan poyezd
    k_sig = seg(t, 3.6, 0.5)
    if k_sig > 0.02:
        bx, by = 180, 322
        d.rounded_rectangle([bx, by, bx + 420, by + 56], 12,
                            fill=mix(BG, (84, 34, 32), k_sig))
        r = 11 + 3 * math.sin(t * 9)
        d.ellipse([bx + 24 - r, by + 28 - r, bx + 24 + r, by + 28 + r],
                  fill=mix(BG, (224, 86, 76), k_sig))
        d.text((bx + 56, by + 12), "POYEZD SIGNALI KELDI", font=F(F_BOLD, 30),
               fill=mix(BG, (238, 206, 200), k_sig))
    if t > 3.9:
        a = ease_out(min(1.0, (t - 3.9) / 0.9))
        for sx, sgn in ((700, 1), (1220, -1)):     # shlagbaum tushadi
            ang = math.radians(90 - 90 * a)
            ex = sx + sgn * 150 * math.cos(ang)
            ey = 560 - 150 * math.sin(ang)
            d.line([(sx, 560), (ex, ey)], fill=(214, 84, 70), width=11)
    if t > 4.0:
        # Poyezd yuqoridan chiqib, kesishmaga yaqinlashadi (yo'l chetigacha)
        _train(d, int(330 + ease_out(min(1.0, (t - 4.0) / 2.6)) * 110))

    # aniqlash ramkasi + "suratga olindi" chaqnashi
    if alert > 0.05:
        pulse = 0.5 + 0.5 * math.sin(t * 7)
        box = (880, 500, 1042, 602)
        d.rectangle(box, outline=mix((120, 40, 36), (236, 96, 84), pulse), width=4)
        d.text((box[0], box[1] - 38), "QOIDABUZARLIK", font=F(F_BOLD, 28),
               fill=(236, 120, 104))

    # dalil kartasi — o'ngdan sirg'alib chiqadi
    k_card = seg(t, 6.0, 0.8)
    if k_card > 0.02:
        cw, ch = 560, 300
        cx = int(W - 150 - cw * k_card + (1 - k_card) * 80)
        cy = 740
        d.rounded_rectangle([cx, cy, cx + cw, cy + ch], 16,
                            fill=mix(BG, PANEL, k_card),
                            outline=mix(BG, (62, 78, 48), k_card), width=2)
        d.text((cx + 28, cy + 24), "AVTOMATIK QAYD ETILDI", font=F(F_BOLD, 24),
               fill=mix(BG, ACCENT, k_card))
        rows = [("Vaqt", "13:42:07"), ("Kamera", "Pereezd 8 / A2"),
                ("Davlat raqami", "01 A 123 BC"),
                ("Saqlandi", "to'liq kadr + raqam surati")]
        for i, (a_, b_) in enumerate(rows):
            kk = seg(t, 6.3 + i * 0.22, 0.4)
            if kk <= 0.01:
                continue
            y = cy + 78 + i * 52
            d.text((cx + 28, y), a_, font=F(F_REG, 26), fill=mix(BG, DIM, kk))
            d.text((cx + cw - 28, y), b_, font=F(F_BOLD, 26),
                   fill=mix(BG, TEXT, kk), anchor="ra")

    # pastdagi xulosa
    fade_text(d, (150, 790), "Operator ko'rmasa ham — tizim ko'radi",
              F(F_BOLD, 46), TEXT, seg(t, 8.0))
    fade_text(d, (150, 860),
              "Hodisa vaqti, joyi va video dalili bilan hisobotga tushadi",
              F(F_LIGHT, 34), MUTED, seg(t, 8.4))
    return img


BENEFITS = [
    ("Uzluksiz, obyektiv hisob", "Operator shart emas — tizim 24/7 o'zi sanaydi"),
    ("Har bir hodisa qayd etiladi", "Vaqt, kamera, transport turi va video dalil"),
    ("Toifa hisoboti bir tugmada", "I–IV toifa va Word hisobot avtomatik tayyorlanadi"),
    ("Markaziy tizimga ulanadi", "Statistika yuqori tashkilotga o'zi uzatiladi"),
]


def sc_benefits(t):
    img = bg_frame()
    d = ImageDraw.Draw(img)
    fade_text(d, (200, 210), "NIMA BERADI", F(F_BOLD, 28), ACCENT, seg(t, 0.15))
    fade_text(d, (200, 266), "Qo'lda hisobdan — avtomatik nazoratga",
              F(F_BOLD, 62), TEXT, seg(t, 0.35))
    fb, fs = F(F_BOLD, 42), F(F_LIGHT, 32)
    for i, (title, sub) in enumerate(BENEFITS):
        k = seg(t, 1.2 + i * 0.6, 0.6)
        if k <= 0.001:
            continue
        y = 430 + i * 140
        dx = int(70 * (1 - k))
        cx, cy = 210 - dx, y + 30
        d.ellipse([cx - 26, cy - 26, cx + 26, cy + 26],
                  outline=mix(BG, ACCENT, k), width=3)
        kk = seg(t, 1.35 + i * 0.6, 0.35)
        if kk > 0:  # belgi chizilishi
            pts = [(cx - 12, cy), (cx - 3, cy + 10), (cx + 13, cy - 10)]
            seg1 = min(1.0, kk * 2)
            d.line([pts[0], (pts[0][0] + (pts[1][0] - pts[0][0]) * seg1,
                             pts[0][1] + (pts[1][1] - pts[0][1]) * seg1)],
                   fill=mix(BG, ACCENT_SOFT, kk), width=5)
            if kk > 0.5:
                s2 = (kk - 0.5) * 2
                d.line([pts[1], (pts[1][0] + (pts[2][0] - pts[1][0]) * s2,
                                 pts[1][1] + (pts[2][1] - pts[1][1]) * s2)],
                       fill=mix(BG, ACCENT_SOFT, kk), width=5)
        d.text((cx + 64, y - 6), title, font=fb, fill=mix(BG, TEXT, k))
        d.text((cx + 64, y + 52), sub, font=fs, fill=mix(BG, MUTED, k))
    return img


def make_outro(stats):
    fn, fl = F(F_BOLD, 66), F(F_REG, 28)

    def sc(t):
        img = bg_frame()
        d = ImageDraw.Draw(img)
        logo(img, 140, (W // 2 - 70, 200), seg(t, 0.1, 0.6))
        fade_text(d, (W // 2, 378), "RailSafe AI", F(F_BOLD, 78), TEXT,
                  seg(t, 0.45), anchor="ma")
        n = len(stats)
        colw, x0 = 1560 // n, (W - 1560) // 2
        for i, (val, label, num) in enumerate(stats):
            k = seg(t, 1.0 + i * 0.25, 0.5)
            if k <= 0.001:
                continue
            cx = x0 + colw * i + colw // 2
            kc = min(1.0, max(0.0, (t - (1.0 + i * 0.25)) / 1.1))
            shown = val if num is None else num(ease_out(kc))
            d.text((cx, 548), shown, font=fn, fill=mix(BG, ACCENT, k),
                   anchor="ma")
            d.text((cx, 642), label, font=fl, fill=mix(BG, MUTED, k),
                   anchor="ma")
            if i:
                d.line([(x0 + colw * i - 8, 562), (x0 + colw * i - 8, 668)],
                       fill=mix(BG, (56, 68, 44), k), width=2)
        k2 = seg(t, 2.3, 0.7)
        if k2 > 0:
            d.rectangle([W // 2 - 72 * k2, 756, W // 2 + 72 * k2, 759],
                        fill=mix(BG, ACCENT, k2))
        fade_text(d, (W // 2, 806),
                  "Yagona ekranda jonli nazorat, statistika va hisobot",
                  F(F_LIGHT, 38), TEXT, seg(t, 2.5), anchor="ma")
        fade_text(d, (W // 2, 874),
                  "Demo va joriy etish bo'yicha murojaat qiling",
                  F(F_LIGHT, 30), MUTED, seg(t, 2.9), anchor="ma")
        return img
    return sc


def db_stats():
    """Yakundagi raqamlar — haqiqiy bazadan (sanab ko'rsatish uchun)."""
    try:
        from datetime import date
        from app.core.config import ConfigManager
        from app.core.database import StatsDB
        from app.reports.category import compute_category
        db = StatsDB(str(ROOT / "app" / "data" / "stats.db"))
        cfg = ConfigManager()
        crossings = cfg.get_crossings()
        total = trains = 0
        for c in crossings:
            cat = compute_category(db, c["id"], "2000-01-01",
                                   date.today().isoformat())
            total += cat["total_light"] + cat["total_heavy"]
            trains += int(cat["avg_trains"] * cat["days"])
        cams = sum(1 for c in crossings for cam in c.get("cameras", [])
                   if cam.get("enabled", True))
        return [
            (f"{total / 1e6:.1f} mln", "qayd etilgan transport",
             lambda k, v=total: f"{v / 1e6 * k:.1f} mln"),
            (f"{trains:,}".replace(",", " "), "poyezd o'tishi",
             lambda k, v=trains: f"{int(v * k):,}".replace(",", " ")),
            (str(cams), "kamera real vaqtda",
             lambda k, v=cams: str(int(round(v * k)))),
        ]
    except Exception as e:
        print("stats xatosi:", e)
        return [("24/7", "uzluksiz nazorat", None),
                ("I–IV", "toifa hisoboti", None),
                ("24/7", "uzluksiz uzatish", None)]


def main():
    raw = VID / "raw_app.mp4"
    dash = VID / "raw_dash.mp4"
    dash_src = dash if dash.exists() else raw
    dash_ss = float(os.environ.get("DASH_SS", "48"))
    parts = []

    def add(name):
        p = TMP / name
        parts.append(p)
        return p

    print("1/8 kirish");    render_anim(sc_intro, 5.5, add("01.mp4"))
    print("2/8 muammo");    render_anim(sc_problem, 9.5, add("02.mp4"))
    print("3/8 sxema");     render_anim(sc_pipeline, 8.5, add("03.mp4"))

    print("4/10 panel")
    cap = TMP / "cap_dash.png"
    caption_png(cap, "Boshqaruv paneli", "Barcha pereezdlar — bitta ekranda",
                "Har bir kamera jonli, uzluksiz 24/7 kuzatuv",
                note="namuna ma'lumotlar")
    seg_clip(dash_src, dash_ss, 12, cap, add("04.mp4"))

    print("5/10 aniqlash")
    cap = TMP / "cap_detect.png"
    caption_png(cap, "Aniqlash", "Har bir transport aniqlanadi va kuzatiladi",
                "Belgilangan hududdagi mashinalar turi bo'yicha sanab boriladi")
    seg_clip(raw, 40.5, 12, cap, add("05.mp4"))

    print("6/10 ssenariy")
    render_anim(sc_scenario, 12.0, add("06.mp4"))

    print("7/10 PLC")
    cap = TMP / "cap_plc.png"
    caption_png(cap, "Poyezd signali", "Pereezd avtomatikasi bilan bevosita bog'lanadi",
                "Shlagbaum yopilganda zona avtomatik nazoratga olinadi")
    seg_clip(raw, 46.5, 7, cap, add("07.mp4"), zoom=False)

    print("8/10 tahlil")
    cap = TMP / "cap_stats.png"
    caption_png(cap, "Tahlil", "Soatlik, haftalik va oylik statistika",
                "Kesishma toifasi (I–IV) va zator xaritasi avtomatik hisoblanadi",
                note="namuna ma'lumotlar")
    seg_clip(raw, 62.0, 13, cap, add("08.mp4"), zoom=False)

    print("9/10 foyda")
    render_anim(sc_benefits, 11.5, add("09.mp4"))
    print("10/10 yakun")
    render_anim(make_outro(db_stats()), 8.0, add("10.mp4"), fade_out=1.0)

    lst = TMP / "list.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts),
                   encoding="utf-8")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", str(lst), "-c", "copy", str(OUT)])
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(OUT)],
        capture_output=True, text=True).stdout.strip()
    print(f"\nTayyor: {OUT}  ({float(dur):.1f} s, "
          f"{OUT.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
