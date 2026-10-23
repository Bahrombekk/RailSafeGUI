"""
Rolik uchun NAMUNA (demo) statistika bazasini yaratish.

Haqiqiy `app/data/stats.db` ga TEGILMAYDI — alohida fayl yaratiladi:
    build/video/demo_stats.db

Dastur uni RAILSAFE_DB muhit o'zgaruvchisi orqali oladi:
    RAILSAFE_DB=build/video/demo_stats.db .venv/Scripts/python.exe tools/demo_record.py

Ma'lumot haqiqiyga o'xshash qilib generatsiya qilinadi: sutkalik egri chiziq
(tong/kech cho'qqilari), dam olish kunlari pasayishi, poyezd o'tishlari va
zona bandligi (probka xaritasi uchun).
"""

import math
import os
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.core.config import ConfigManager  # noqa: E402
from app.core.database import StatsDB  # noqa: E402

OUT = ROOT / "build" / "video" / "demo_stats.db"
DAYS = 75
random.seed(7)

# Kesishma "og'irligi": kunlik yengil transport taxminiy darajasi
PROFILE = {
    1: {"light": 31000, "heavy": 1900, "trains": (16, 24), "busy": 1.0},
    2: {"light": 11500, "heavy": 420, "trains": (9, 15), "busy": 0.62},
}
DEFAULT_PROFILE = {"light": 9000, "heavy": 300, "trains": (6, 12), "busy": 0.5}

# Sutkalik egri chiziq (0..23) — tong va kech cho'qqilari
HOUR_W = [0.22, 0.13, 0.09, 0.08, 0.14, 0.38, 0.72, 1.35, 1.62, 1.28, 1.08,
          1.06, 1.12, 1.10, 1.06, 1.14, 1.38, 1.66, 1.44, 1.02, 0.74, 0.56,
          0.42, 0.30]


def day_factor(d: date) -> float:
    """Dam olish kunlari kamroq + kundan kunga tabiiy tebranish."""
    base = 0.68 if d.weekday() >= 5 else 1.0
    wave = 1 + 0.07 * math.sin(d.toordinal() / 5.0)
    return base * wave * random.uniform(0.93, 1.07)


def build():
    if OUT.exists():
        OUT.unlink()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    db = StatsDB(str(OUT))
    conn = db._conn

    cfg = ConfigManager()
    crossings = []
    for c in cfg.get_crossings():
        cams = [cam.get("name", "") for cam in c.get("cameras", [])
                if cam.get("enabled", True)] or ["CAM1"]
        crossings.append((c["id"], cams))
    if not crossings:
        crossings = [(1, ["A1", "A2"]), (2, ["B1", "B2"])]

    today = date.today()
    hourly, trains, occ = [], [], []

    for cid, cams in crossings:
        p = PROFILE.get(cid, DEFAULT_PROFILE)
        for back in range(DAYS, -1, -1):
            d = today - timedelta(days=back)
            f = day_factor(d)
            # bugungi kun — hozirgi soatgacha
            last_hour = datetime.now().hour if back == 0 else 23

            for h in range(last_hour + 1):
                w = HOUR_W[h] / sum(HOUR_W)
                light_h = p["light"] * f * w * 24 / 24
                heavy_h = p["heavy"] * f * w * 24 / 24
                hour_start = datetime.combine(d, datetime.min.time()) \
                    + timedelta(hours=h)
                hs = hour_start.isoformat()

                # kameralar o'rtasida taqsimlash
                for i, cam in enumerate(cams):
                    share = (0.62 if i == 0 else 0.38 / max(1, len(cams) - 1))
                    lc = max(0, int(light_h * share * random.uniform(0.88, 1.12)))
                    hc = max(0, int(heavy_h * share * random.uniform(0.8, 1.2)))
                    hourly.append((cid, cam, hs, lc, hc, hs))

                    # Zona bandligi: oqim qancha zich bo'lsa, shuncha uzoq band
                    load = (HOUR_W[h] / max(HOUR_W)) * f * p["busy"]
                    occupied = min(3400, 3600 * load * random.uniform(0.3, 0.55))
                    veh = max(1, int((lc + hc) * 0.45))
                    # o'rtacha turish vaqti: tirbandlikda keskin oshadi
                    avg_dwell = 2.5 + 26 * (load ** 3.2) * random.uniform(0.75, 1.3)
                    occ.append((cid, cam, hs, round(occupied, 1),
                                int(veh * 0.12) + 1, hs,
                                round(avg_dwell * veh, 1), veh))

            # Poyezd o'tishlari
            n = random.randint(*p["trains"])
            if back == 0:
                n = max(1, int(n * (last_hour + 1) / 24))
            slots = sorted(random.sample(range(5, max(6, (last_hour + 1) * 60 - 10)),
                                         min(n, max(1, (last_hour + 1) * 60 - 16))))
            for m in slots:
                st = datetime.combine(d, datetime.min.time()) + timedelta(minutes=m)
                dur = random.uniform(150, 430)
                en = st + timedelta(seconds=dur)
                trains.append((cid, st.isoformat(), en.isoformat(),
                               round(dur, 2), d.isoformat()))

    conn.executemany(
        "INSERT OR REPLACE INTO hourly_stats"
        " (crossing_id, camera_name, hour_start, light_count, heavy_count,"
        "  updated_at) VALUES (?,?,?,?,?,?)", hourly)
    conn.executemany(
        "INSERT INTO train_events"
        " (crossing_id, start_time, end_time, duration_seconds, event_date)"
        " VALUES (?,?,?,?,?)", trains)
    conn.executemany(
        "INSERT OR REPLACE INTO occupancy_stats"
        " (crossing_id, camera_name, bin_start, occupied_seconds, max_inside,"
        "  updated_at, dwell_seconds, dwell_vehicles) VALUES (?,?,?,?,?,?,?,?)",
        occ)
    conn.commit()

    print(f"Baza: {OUT}")
    print(f"  hourly_stats : {len(hourly):>7}")
    print(f"  train_events : {len(trains):>7}")
    print(f"  occupancy    : {len(occ):>7}")
    for cid, _ in crossings:
        l, h = db.get_today_total(cid)
        ts = db.get_train_today_stats(cid)
        print(f"  pereezd {cid}: bugun yengil={l} og'ir={h} poyezd={ts['count']}")


def build_hybrid():
    """HAQIQIY bazadan nusxa olib, FAQAT bo'sh kunlarni to'ldiradi.

    Raqamlarning ko'pi haqiqiy bo'lib qoladi; sintetika faqat qurilma
    o'chiq turgan kunlarga va poyezd hodisalariga (PLC uzilgan bo'lgani
    uchun yozilmay qolgan) qo'shiladi. Shakl — o'sha bazadagi haqiqiy
    sutkalik taqsimotdan o'rganiladi.
    """
    import shutil
    import sqlite3

    real = ROOT / "app" / "data" / "stats.db"
    if not real.exists():
        raise SystemExit(f"haqiqiy baza topilmadi: {real}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    shutil.copy2(real, OUT)
    conn = sqlite3.connect(str(OUT))
    today = date.today()
    first = today - timedelta(days=DAYS)

    # ── haqiqiy ma'lumotdan profil o'rganish ──────────────────────────
    # Bazada o'chirilgan pereezdlarning eski yozuvlari qolgan bo'lishi
    # mumkin — faqat configdagilar bilan ishlaymiz.
    live_ids = {c["id"] for c in ConfigManager().get_crossings()}
    daily = {}
    for cid, d, tot in conn.execute(
            "SELECT crossing_id, date(hour_start), SUM(light_count+heavy_count)"
            " FROM hourly_stats GROUP BY 1,2"):
        if cid in live_ids:
            daily.setdefault(cid, {})[d] = tot
    if not daily:
        raise SystemExit("haqiqiy bazada ma'lumot yo'q — oddiy rejimni ishlating")

    profiles = {}
    for cid, days_map in daily.items():
        vals = sorted(days_map.values())
        median = vals[len(vals) // 2]
        good = [d for d, t in days_map.items() if t >= median * 0.6]
        qs = ",".join("?" * len(good))
        hours = [0.0] * 24
        for h, s in conn.execute(
                f"SELECT CAST(strftime('%H',hour_start) AS INTEGER),"
                f" SUM(light_count+heavy_count) FROM hourly_stats"
                f" WHERE crossing_id=? AND date(hour_start) IN ({qs})"
                f" GROUP BY 1", (cid, *good)):
            hours[h] = float(s or 0)
        tot_h = sum(hours) or 1.0
        hours = [x / tot_h for x in hours]
        cams = {}
        for cam, l, h_ in conn.execute(
                f"SELECT camera_name, SUM(light_count), SUM(heavy_count)"
                f" FROM hourly_stats WHERE crossing_id=?"
                f" AND date(hour_start) IN ({qs}) GROUP BY 1", (cid, *good)):
            cams[cam] = (l or 0, h_ or 0)
        tot_cam = sum(l + h_ for l, h_ in cams.values()) or 1
        profiles[cid] = {
            "median": median,
            "hours": hours,
            "cams": {c: ((l + h_) / tot_cam, h_ / max(1, l + h_))
                     for c, (l, h_) in cams.items()},
        }
        print(f"  pereezd {cid}: median {median:,} / kun, {len(good)} ta to'liq kun, "
              f"{len(cams)} kamera".replace(",", " "))

    filled_days, filled_trains = 0, 0
    for cid, prof in profiles.items():
        med = prof["median"]
        for back in range(DAYS, -1, -1):
            d = today - timedelta(days=back)
            ds = d.isoformat()
            have = daily.get(cid, {}).get(ds, 0)
            last_hour = datetime.now().hour if back == 0 else 23
            # bugun uchun kutilayotgan ulush (hali kun tugamagan)
            expect = med * (sum(prof["hours"][:last_hour + 1]) if back == 0 else 1.0)
            if have >= expect * 0.55:
                continue                      # yetarli haqiqiy ma'lumot bor
            conn.execute("DELETE FROM hourly_stats WHERE crossing_id=? AND"
                         " date(hour_start)=?", (cid, ds))
            conn.execute("DELETE FROM occupancy_stats WHERE crossing_id=? AND"
                         " date(bin_start)=?", (cid, ds))
            f = day_factor(d)
            rows_h, rows_o = [], []
            for h in range(last_hour + 1):
                w = prof["hours"][h]
                for cam, (share, heavy_ratio) in prof["cams"].items():
                    tot = med * f * w * share * random.uniform(0.9, 1.1)
                    hc = int(tot * heavy_ratio)
                    lc = max(0, int(tot) - hc)
                    hs = (datetime.combine(d, datetime.min.time())
                          + timedelta(hours=h)).isoformat()
                    rows_h.append((cid, cam, hs, lc, hc, hs))
                    load = min(1.0, w / max(prof["hours"]) * f)
                    veh = max(1, int((lc + hc) * 0.45))
                    dwell = 2.5 + 24 * (load ** 3.0) * random.uniform(0.8, 1.25)
                    rows_o.append((cid, cam, hs,
                                   round(min(3400, 3600 * load * 0.45), 1),
                                   int(veh * 0.12) + 1, hs,
                                   round(dwell * veh, 1), veh))
            conn.executemany(
                "INSERT OR REPLACE INTO hourly_stats (crossing_id, camera_name,"
                " hour_start, light_count, heavy_count, updated_at)"
                " VALUES (?,?,?,?,?,?)", rows_h)
            conn.executemany(
                "INSERT OR REPLACE INTO occupancy_stats (crossing_id,"
                " camera_name, bin_start, occupied_seconds, max_inside,"
                " updated_at, dwell_seconds, dwell_vehicles)"
                " VALUES (?,?,?,?,?,?,?,?)", rows_o)
            filled_days += 1

    # ── poyezd hodisalari: PLC uzilgani uchun deyarli yozilmagan ──────
    for cid, _ in profiles.items():
        p = PROFILE.get(cid, DEFAULT_PROFILE)
        for back in range(DAYS, -1, -1):
            d = today - timedelta(days=back)
            ds = d.isoformat()
            have = conn.execute("SELECT COUNT(*) FROM train_events WHERE"
                                " crossing_id=? AND event_date=?",
                                (cid, ds)).fetchone()[0]
            if have >= 5:
                continue
            last_hour = datetime.now().hour if back == 0 else 23
            n = random.randint(*p["trains"])
            if back == 0:
                n = max(1, int(n * (last_hour + 1) / 24))
            n = max(0, n - have)
            span = max(30, (last_hour + 1) * 60 - 12)
            for m in sorted(random.sample(range(5, span), min(n, span - 6))):
                st = datetime.combine(d, datetime.min.time()) + timedelta(minutes=m)
                dur = random.uniform(150, 430)
                conn.execute(
                    "INSERT INTO train_events (crossing_id, start_time,"
                    " end_time, duration_seconds, event_date) VALUES (?,?,?,?,?)",
                    (cid, st.isoformat(), (st + timedelta(seconds=dur)).isoformat(),
                     round(dur, 2), ds))
                filled_trains += 1
    conn.commit()

    db = StatsDB(str(OUT))
    print(f"\nAralash baza: {OUT}")
    print(f"  to'ldirilgan kunlar : {filled_days}")
    print(f"  qo'shilgan poyezdlar: {filled_trains}")
    for cid in profiles:
        l, h = db.get_today_total(cid)
        ts = db.get_train_today_stats(cid)
        print(f"  pereezd {cid}: bugun yengil={l} og'ir={h} poyezd={ts['count']}")


if __name__ == "__main__":
    if "--hybrid" in sys.argv:
        build_hybrid()
    else:
        build()
