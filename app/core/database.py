"""
StatsDB - SQLite database for hourly/daily vehicle counting statistics.
Thread-safe. Stores per-camera, per-hour counts. Auto-resets at midnight.
"""

import sqlite3
import threading
import os
import time
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("RailSafe.database")


class StatsDB:
    """SQLite baza - soatlik va kunlik transport statistikasi.

    Usage:
        db = StatsDB()
        db.record_count(crossing_id=1, camera_name="A1", light=5, heavy=2)
        light, heavy = db.get_today_total(crossing_id=1)
        hourly = db.get_hourly_data(crossing_id=1)
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "stats.db")

        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-4096")  # 4MB kesh
        self._create_tables()
        # Delta tracking: tracker har safar 0 dan boshlaydi,
        # shuning uchun oxirgi yuborilgan qiymatni saqlaymiz
        self._last_counts = {}  # (crossing_id, camera_name) -> (light, heavy)
        # Occupancy (bandlik vaqti) akkumulatorlari:
        # (crossing_id, camera_name) -> holat dict (record_occupancy ga qarang)
        self._occ_state = {}

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS hourly_stats (
                crossing_id INTEGER NOT NULL,
                camera_name TEXT NOT NULL,
                hour_start TEXT NOT NULL,
                light_count INTEGER DEFAULT 0,
                heavy_count INTEGER DEFAULT 0,
                updated_at TEXT,
                UNIQUE(crossing_id, camera_name, hour_start)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS train_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crossing_id INTEGER NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_seconds REAL,
                event_date TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS occupancy_stats (
                crossing_id INTEGER NOT NULL,
                camera_name TEXT NOT NULL,
                bin_start TEXT NOT NULL,
                occupied_seconds REAL DEFAULT 0,
                max_inside INTEGER DEFAULT 0,
                updated_at TEXT,
                dwell_seconds REAL DEFAULT 0,
                dwell_vehicles INTEGER DEFAULT 0,
                UNIQUE(crossing_id, camera_name, bin_start)
            )
        """)

        # Migratsiya: eski bazalarda turish vaqti ustunlari yo'q.
        # dwell_seconds — zonadan chiqqan mashinalarning turish vaqtlari
        # yig'indisi, dwell_vehicles — shu mashinalar soni. O'rtacha turish
        # vaqti = dwell_seconds / dwell_vehicles — probkaning HAQIQIY o'lchovi
        # (oqim ham, zona band vaqti ham gavjum-lekin-oqadigan holatni
        # probkadan ajratmaydi).
        try:
            cols = {r[1] for r in self._conn.execute(
                "PRAGMA table_info(occupancy_stats)")}
            for col, ddl in (("dwell_seconds", "REAL DEFAULT 0"),
                             ("dwell_vehicles", "INTEGER DEFAULT 0")):
                if col not in cols:
                    self._conn.execute(
                        f"ALTER TABLE occupancy_stats ADD COLUMN {col} {ddl}")
                    logger.info("occupancy_stats: '%s' ustuni qo'shildi", col)
        except Exception as e:
            logger.error("occupancy_stats migratsiya xatosi: %s", e)

        self._conn.commit()

    def _current_hour(self) -> str:
        """Joriy soatning boshi: "2026-02-10T13:00:00" """
        now = datetime.now()
        return now.replace(minute=0, second=0, microsecond=0).isoformat()

    def record_count(self, crossing_id: int, camera_name: str,
                     light: int, heavy: int):
        """Joriy soat uchun delta qo'shish.

        light/heavy - tracker kumulyativ soni (har sessiyada 0 dan boshlanadi).
        Oldingi chaqiruvdagi qiymat bilan farqni (delta) hisoblaydi va DB ga qo'shadi.
        Thread-safe: barcha operatsiyalar _lock ichida.
        """
        with self._lock:
            key = (crossing_id, camera_name)
            last_l, last_h = self._last_counts.get(key, (0, 0))
            delta_l = max(0, light - last_l)
            delta_h = max(0, heavy - last_h)

            if delta_l == 0 and delta_h == 0:
                # Delta yo'q bo'lsa ham oxirgi qiymatni yangilaymiz (INSERT talab qilinmaydi)
                self._last_counts[key] = (light, heavy)
                return

            hour = self._current_hour()
            now = datetime.now().isoformat()
            try:
                self._conn.execute("""
                    INSERT INTO hourly_stats
                        (crossing_id, camera_name, hour_start, light_count, heavy_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(crossing_id, camera_name, hour_start)
                    DO UPDATE SET
                        light_count = hourly_stats.light_count + ?,
                        heavy_count = hourly_stats.heavy_count + ?,
                        updated_at = ?
                """, (crossing_id, camera_name, hour, delta_l, delta_h, now,
                      delta_l, delta_h, now))
                self._conn.commit()
                # Faqat INSERT muvaffaqiyatli bo'lgandan keyin oxirgi qiymatni
                # yangilaymiz — aks holda xatoda delta butunlay yo'qoladi.
                self._last_counts[key] = (light, heavy)
            except Exception as e:
                logger.error("record_count error: %s", e)

    def reset_baseline(self, crossing_id: int, camera_name: str):
        """Kamera qayta ulanib tracker 0 dan boshlaganda delta bazasini nollash.
        Aks holda delta = max(0, kichik - katta) = 0 bo'lib, tracker eski
        qiymatga yetguncha barcha sanoq DB ga yozilmay yo'qoladi."""
        with self._lock:
            self._last_counts[(crossing_id, camera_name)] = (0, 0)

    # ── Occupancy (bandlik vaqti) — YHQ 18 / ОДМ F-daraja uchun ──────────
    # FHWA queue-detektor standarti: zona kamida MIN_PRESENCE soniya uzluksiz
    # band bo'lgandagina "band" hisoblanadi (o'tib ketayotgan mashina emas).
    OCC_MIN_PRESENCE = 3.0
    _OCC_FLUSH_SEC = 60.0      # DB ga yozish oralig'i
    _OCC_GAP_RESET_SEC = 5.0   # kadr oqimi uzilsa segment ishonchsiz — yopamiz

    def _current_bin(self) -> str:
        """Joriy 15 daqiqalik bin boshi: "2026-07-23T13:15:00" """
        now = datetime.now()
        return now.replace(minute=(now.minute // 15) * 15,
                           second=0, microsecond=0).isoformat()

    def record_occupancy(self, crossing_id: int, camera_name: str,
                         inside_count: int):
        """Har kadrda chaqiriladi: polygon ichida mashina bor vaqtni yig'adi.
        3s dan qisqa segmentlar sanalmaydi; 15 daqiqalik binlarda saqlanadi."""
        now_m = time.monotonic()
        with self._lock:
            key = (crossing_id, camera_name)
            st = self._occ_state.get(key)
            if st is None:
                st = {"seg_start": None, "credited": 0.0, "pending": 0.0,
                      "max_inside": 0, "last_call": now_m,
                      "last_flush": now_m, "bin": self._current_bin(),
                      "dwell_sum": 0.0, "dwell_n": 0}
                self._occ_state[key] = st
            if now_m - st["last_call"] > self._OCC_GAP_RESET_SEC:
                st["seg_start"] = None
                st["credited"] = 0.0
            st["last_call"] = now_m

            if inside_count > 0:
                if st["seg_start"] is None:
                    st["seg_start"] = now_m
                    st["credited"] = 0.0
                seg = now_m - st["seg_start"]
                if seg >= self.OCC_MIN_PRESENCE:
                    st["pending"] += seg - st["credited"]
                    st["credited"] = seg
                if inside_count > st["max_inside"]:
                    st["max_inside"] = inside_count
            else:
                st["seg_start"] = None
                st["credited"] = 0.0

            cur_bin = self._current_bin()
            if cur_bin != st["bin"] or (st["pending"] > 0 and
                                        now_m - st["last_flush"] >= self._OCC_FLUSH_SEC):
                self._flush_occupancy(key, st)
                st["bin"] = cur_bin
                st["last_flush"] = now_m

    def record_dwells(self, crossing_id: int, camera_name: str, dwells):
        """Zonadan chiqqan mashinalarning turish vaqtlarini (sekund) yozish.

        `PolygonTracker.pop_completed_dwells()` natijasi beriladi. O'rtacha
        turish vaqti (= dwell_seconds/dwell_vehicles) probkaning yagona ishonchli
        o'lchovi: oqim probkada KAMAYADI, zona band vaqti esa gavjum oqimda ham
        to'yinadi — ikkalasi ham zatorni ajratmaydi. Turish vaqti erkin oqimda
        bir necha sekund, probkada bir necha barobar ko'p bo'ladi.
        """
        if not dwells:
            return
        now_m = time.monotonic()
        with self._lock:
            key = (crossing_id, camera_name)
            st = self._occ_state.get(key)
            if st is None:
                st = {"seg_start": None, "credited": 0.0, "pending": 0.0,
                      "max_inside": 0, "last_call": now_m,
                      "last_flush": now_m, "bin": self._current_bin(),
                      "dwell_sum": 0.0, "dwell_n": 0}
                self._occ_state[key] = st
            st["dwell_sum"] = st.get("dwell_sum", 0.0) + float(sum(dwells))
            st["dwell_n"] = st.get("dwell_n", 0) + len(dwells)

            cur_bin = self._current_bin()
            if cur_bin != st["bin"] or now_m - st["last_flush"] >= self._OCC_FLUSH_SEC:
                self._flush_occupancy(key, st)
                st["bin"] = cur_bin
                st["last_flush"] = now_m

    def _flush_occupancy(self, key, st):
        """Yig'ilgan bandlik vaqtini DB ga yozish. _lock ichida chaqiriladi."""
        if (st["pending"] <= 0 and st["max_inside"] <= 0
                and st.get("dwell_n", 0) <= 0):
            return
        crossing_id, camera_name = key
        dwell_sum = float(st.get("dwell_sum", 0.0))
        dwell_n = int(st.get("dwell_n", 0))
        try:
            self._conn.execute("""
                INSERT INTO occupancy_stats
                    (crossing_id, camera_name, bin_start,
                     occupied_seconds, max_inside, updated_at,
                     dwell_seconds, dwell_vehicles)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(crossing_id, camera_name, bin_start)
                DO UPDATE SET
                    occupied_seconds = occupancy_stats.occupied_seconds + ?,
                    max_inside = MAX(occupancy_stats.max_inside, ?),
                    updated_at = ?,
                    dwell_seconds = occupancy_stats.dwell_seconds + ?,
                    dwell_vehicles = occupancy_stats.dwell_vehicles + ?
            """, (crossing_id, camera_name, st["bin"],
                  st["pending"], st["max_inside"], datetime.now().isoformat(),
                  dwell_sum, dwell_n,
                  st["pending"], st["max_inside"], datetime.now().isoformat(),
                  dwell_sum, dwell_n))
            self._conn.commit()
            st["pending"] = 0.0
            st["max_inside"] = 0
            st["dwell_sum"] = 0.0
            st["dwell_n"] = 0
        except Exception as e:
            logger.error("record_occupancy flush error: %s", e)

    def get_occupancy_heatmap(self, crossing_id: int,
                              date_to: Optional[date] = None) -> List[Dict]:
        """Oxirgi 7 kun: har soatda kesishma zonasi necha DAQIQA band bo'lgan.
        Returns: [{"date": "...", "day": "Du", "hours": [daq0..daq23]}, ...]"""
        days_uz = ["Du", "Se", "Cho", "Pa", "Ju", "Sha", "Ya"]
        if date_to is None:
            date_to = date.today()
        date_from = (date_to - timedelta(days=6)).isoformat()
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT date(bin_start) as d,
                           CAST(strftime('%H', bin_start) AS INTEGER) as h,
                           COALESCE(SUM(occupied_seconds), 0)
                    FROM occupancy_stats
                    WHERE crossing_id = ?
                      AND date(bin_start) >= ? AND date(bin_start) <= ?
                    GROUP BY d, h
                """, (crossing_id, date_from, date_to.isoformat())).fetchall()
            db_map = {}
            for d_str, h, secs in rows:
                db_map.setdefault(d_str, {})[h] = min(60, int(round(secs / 60.0)))
            data = []
            for i in range(6, -1, -1):
                d = date_to - timedelta(days=i)
                ds = d.isoformat()
                hours = [db_map.get(ds, {}).get(h, 0) for h in range(24)]
                data.append({"date": ds, "day": days_uz[d.weekday()], "hours": hours})
            return data
        except Exception as e:
            logger.error("get_occupancy_heatmap error: %s", e)
            return []

    def get_dwell_heatmap(self, crossing_id: int,
                          date_to: Optional[date] = None) -> List[Dict]:
        """Oxirgi 7 kun: har soatda bitta mashinaning zonada O'RTACHA turish
        vaqti (sekund, butun songa yaxlitlangan).

        = SUM(dwell_seconds) / SUM(dwell_vehicles). Bu probkaning haqiqiy
        o'lchovi: erkin oqimda mashina zonadan bir necha sekundda o'tadi,
        probkada esa unda uzoq turadi. Oqim (o'tgan mashina soni) probkada
        kamayadi, zona band vaqti esa gavjum oqimda ham to'yinadi — shu sabab
        ikkalasi ham zatorni ko'rsatmaydi.

        Returns: [{"date": "...", "day": "Du", "hours": [sek0..sek23]}, ...]
        """
        days_uz = ["Du", "Se", "Cho", "Pa", "Ju", "Sha", "Ya"]
        if date_to is None:
            date_to = date.today()
        date_from = (date_to - timedelta(days=6)).isoformat()
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT date(bin_start) as d,
                           CAST(strftime('%H', bin_start) AS INTEGER) as h,
                           COALESCE(SUM(dwell_seconds), 0),
                           COALESCE(SUM(dwell_vehicles), 0)
                    FROM occupancy_stats
                    WHERE crossing_id = ?
                      AND date(bin_start) >= ? AND date(bin_start) <= ?
                    GROUP BY d, h
                """, (crossing_id, date_from, date_to.isoformat())).fetchall()
            db_map = {}
            for d_str, h, secs, n in rows:
                if n > 0:
                    db_map.setdefault(d_str, {})[h] = int(round(secs / n))
            data = []
            for i in range(6, -1, -1):
                d = date_to - timedelta(days=i)
                ds = d.isoformat()
                hours = [db_map.get(ds, {}).get(h, 0) for h in range(24)]
                data.append({"date": ds, "day": days_uz[d.weekday()], "hours": hours})
            return data
        except Exception as e:
            logger.error("get_dwell_heatmap error: %s", e)
            return []

    def get_practical_capacity(self, crossing_id: int,
                               date_to: Optional[date] = None,
                               days: int = 30) -> int:
        """Amaliy sig'im P (ОДМ 218.2.020-2012 uslubida, kuzatuvdan):
        oxirgi `days` kundagi soatlik oqimlarning 95-percentili.
        Kam ma'lumotda (24 soatdan kam nol bo'lmagan soat) 0 qaytadi —
        chaqiruvchi nisbiy shkalaga qaytishi kerak."""
        if date_to is None:
            date_to = date.today()
        date_from = (date_to - timedelta(days=days - 1)).isoformat()
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT COALESCE(SUM(light_count), 0) + COALESCE(SUM(heavy_count), 0) AS total
                    FROM hourly_stats
                    WHERE crossing_id = ?
                      AND date(hour_start) >= ? AND date(hour_start) <= ?
                    GROUP BY hour_start
                """, (crossing_id, date_from, date_to.isoformat())).fetchall()
            vals = sorted(r[0] for r in rows if r[0] > 0)
            if len(vals) < 24:
                return 0
            return int(vals[int(0.95 * (len(vals) - 1))])
        except Exception as e:
            logger.error("get_practical_capacity error: %s", e)
            return 0

    def get_today_total(self, crossing_id: int,
                        target_date: Optional[str] = None) -> Tuple[int, int]:
        """Berilgan kun uchun jami (barcha kameralar).
        target_date: "2026-05-13" format (None = bugun).
        Returns: (light_total, heavy_total)"""
        if target_date is None:
            target_date = date.today().isoformat()
        try:
            with self._lock:
                row = self._conn.execute("""
                    SELECT COALESCE(SUM(light_count), 0),
                           COALESCE(SUM(heavy_count), 0)
                    FROM hourly_stats
                    WHERE crossing_id = ?
                      AND date(hour_start) = ?
                """, (crossing_id, target_date)).fetchone()
            return (row[0], row[1]) if row else (0, 0)
        except Exception as e:
            logger.error("get_today_total error: %s", e)
            return (0, 0)

    def get_camera_today(self, crossing_id: int,
                         camera_name: str,
                         target_date: Optional[str] = None) -> Tuple[int, int]:
        """Bitta kamera uchun berilgan kun jamisi.
        target_date: "2026-05-13" format (None = bugun).
        Returns: (light, heavy)"""
        if target_date is None:
            target_date = date.today().isoformat()
        try:
            with self._lock:
                row = self._conn.execute("""
                    SELECT COALESCE(SUM(light_count), 0),
                           COALESCE(SUM(heavy_count), 0)
                    FROM hourly_stats
                    WHERE crossing_id = ?
                      AND camera_name = ?
                      AND date(hour_start) = ?
                """, (crossing_id, camera_name, target_date)).fetchone()
            return (row[0], row[1]) if row else (0, 0)
        except Exception as e:
            logger.error("get_camera_today error: %s", e)
            return (0, 0)

    def get_hourly_data(self, crossing_id: int,
                        target_date: Optional[str] = None) -> List[Dict]:
        """24 soatlik ma'lumot (grafik uchun).
        Returns: [{"hour": 0, "light": 5, "heavy": 2}, ...] (24 ta element)"""
        if target_date is None:
            target_date = date.today().isoformat()

        # 24 soat uchun to'liq massiv
        data = [{"hour": h, "light": 0, "heavy": 0} for h in range(24)]
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT CAST(strftime('%H', hour_start) AS INTEGER) as hour,
                           SUM(light_count) as light,
                           SUM(heavy_count) as heavy
                    FROM hourly_stats
                    WHERE crossing_id = ?
                      AND date(hour_start) = ?
                    GROUP BY hour
                    ORDER BY hour
                """, (crossing_id, target_date)).fetchall()
            for hour, light, heavy in rows:
                if 0 <= hour < 24:
                    data[hour]["light"] = light or 0
                    data[hour]["heavy"] = heavy or 0
        except Exception as e:
            logger.error("get_hourly_data error: %s", e)
        return data

    def get_weekly_data(self, crossing_id: int,
                        date_to: Optional[date] = None) -> List[Dict]:
        """Oxirgi 7 kun (kunlik jami), hafta kuni tartibi bilan (Du→Ya).
        date_to: oxirgi kun (None = bugun).
        Returns: [{"date": "2026-02-11", "day": "Du", "light": 10, "heavy": 3}, ...]"""
        days_uz = ["Du", "Se", "Cho", "Pa", "Ju", "Sha", "Ya"]
        if date_to is None:
            date_to = date.today()
        date_from = (date_to - timedelta(days=6)).isoformat()
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT date(hour_start) as d,
                           COALESCE(SUM(light_count), 0),
                           COALESCE(SUM(heavy_count), 0)
                    FROM hourly_stats
                    WHERE crossing_id = ?
                      AND date(hour_start) >= ? AND date(hour_start) <= ?
                    GROUP BY d
                """, (crossing_id, date_from, date_to.isoformat())).fetchall()
            db_map = {r[0]: (r[1], r[2]) for r in rows}
            data = []
            # 7 kunlik oyna eng eski kundan eng yangisiga qarab quriladi —
            # sana (xronologik) tartibi shu bo'yicha saqlanadi, hafta kuni nomi bilan emas.
            for i in range(6, -1, -1):
                d = date_to - timedelta(days=i)
                ds = d.isoformat()
                light, heavy = db_map.get(ds, (0, 0))
                data.append({"date": ds, "day": days_uz[d.weekday()], "light": light, "heavy": heavy})
            return data
        except Exception as e:
            logger.error("get_weekly_data error: %s", e)
            return []

    def get_monthly_data(self, crossing_id: int) -> List[Dict]:
        """Oxirgi 30 kun (kunlik jami).
        Returns: [{"date": "2026-01-13", "light": 10, "heavy": 3}, ...]"""
        today = date.today()
        date_from = (today - timedelta(days=29)).isoformat()
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT date(hour_start) as d,
                           COALESCE(SUM(light_count), 0),
                           COALESCE(SUM(heavy_count), 0)
                    FROM hourly_stats
                    WHERE crossing_id = ?
                      AND date(hour_start) >= ? AND date(hour_start) <= ?
                    GROUP BY d
                """, (crossing_id, date_from, today.isoformat())).fetchall()
            db_map = {r[0]: (r[1], r[2]) for r in rows}
            data = []
            for i in range(29, -1, -1):
                d = today - timedelta(days=i)
                ds = d.isoformat()
                light, heavy = db_map.get(ds, (0, 0))
                data.append({"date": ds, "day": d.day, "light": light, "heavy": heavy})
            return data
        except Exception as e:
            logger.error("get_monthly_data error: %s", e)
            return []

    def get_yearly_data(self, crossing_id: int) -> List[Dict]:
        """Oxirgi 12 oy (oylik jami).
        Returns: [{"month": "2025-03", "label": "Mar", "light": 100, "heavy": 30}, ...]"""
        months_uz = ["Yan", "Fev", "Mar", "Apr", "May", "Iyn",
                     "Iyl", "Avg", "Sen", "Okt", "Noy", "Dek"]
        today = date.today()
        data = []
        try:
            with self._lock:
                for i in range(11, -1, -1):
                    # i oy oldin
                    m = today.month - i
                    y = today.year
                    while m <= 0:
                        m += 12
                        y -= 1
                    ms = f"{y}-{m:02d}"
                    row = self._conn.execute("""
                        SELECT COALESCE(SUM(light_count), 0),
                               COALESCE(SUM(heavy_count), 0)
                        FROM hourly_stats
                        WHERE crossing_id = ?
                          AND strftime('%Y-%m', hour_start) = ?
                    """, (crossing_id, ms)).fetchone()
                    data.append({
                        "month": ms,
                        "label": months_uz[m - 1],
                        "light": row[0] if row else 0,
                        "heavy": row[1] if row else 0,
                    })
            return data
        except Exception as e:
            logger.error("get_yearly_data error: %s", e)
            return []

    def get_heatmap_data(self, crossing_id: int,
                         date_to: Optional[date] = None) -> List[Dict]:
        """Oxirgi 7 kun heatmap: har kun uchun 24 soatlik ma'lumot, hafta kuni tartibi (Du→Ya).
        date_to: oxirgi kun (None = bugun).
        Returns: [{"date": "...", "day": "Du", "hours": [0]*24}, ...] (7 ta)
        """
        days_uz = ["Du", "Se", "Cho", "Pa", "Ju", "Sha", "Ya"]
        if date_to is None:
            date_to = date.today()
        date_from = (date_to - timedelta(days=6)).isoformat()
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT date(hour_start) as d,
                           CAST(strftime('%H', hour_start) AS INTEGER) as h,
                           COALESCE(SUM(light_count), 0) + COALESCE(SUM(heavy_count), 0)
                    FROM hourly_stats
                    WHERE crossing_id = ?
                      AND date(hour_start) >= ? AND date(hour_start) <= ?
                    GROUP BY d, h
                """, (crossing_id, date_from, date_to.isoformat())).fetchall()
            db_map = {}
            for d_str, h, total in rows:
                db_map.setdefault(d_str, {})[h] = total
            data = []
            # Sana (xronologik) tartibida — eng eski kundan eng yangisiga.
            for i in range(6, -1, -1):
                d = date_to - timedelta(days=i)
                ds = d.isoformat()
                hours = [db_map.get(ds, {}).get(h, 0) for h in range(24)]
                data.append({"date": ds, "day": days_uz[d.weekday()], "hours": hours})
            return data
        except Exception as e:
            logger.error("get_heatmap_data error: %s", e)
            return []

    def get_date_range_daily(self, crossing_id: int,
                             date_from: str, date_to: str) -> List[Dict]:
        """Belgilangan sana oralig'ida kunlik statistika.
        date_from, date_to: "2026-02-01" format.
        Returns: [{"date": "2026-02-01", "light": 10, "heavy": 3}, ...]"""
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT date(hour_start) as d,
                           COALESCE(SUM(light_count), 0),
                           COALESCE(SUM(heavy_count), 0)
                    FROM hourly_stats
                    WHERE crossing_id = ?
                      AND date(hour_start) >= ?
                      AND date(hour_start) <= ?
                    GROUP BY d
                    ORDER BY d
                """, (crossing_id, date_from, date_to)).fetchall()
            return [{"date": r[0], "light": r[1], "heavy": r[2]} for r in rows]
        except Exception as e:
            logger.error("get_date_range_daily error: %s", e)
            return []

    def get_date_range_total(self, crossing_id: int,
                             date_from: str, date_to: str) -> Tuple[int, int]:
        """Belgilangan sana oralig'ida jami (light, heavy)."""
        try:
            with self._lock:
                row = self._conn.execute("""
                    SELECT COALESCE(SUM(light_count), 0),
                           COALESCE(SUM(heavy_count), 0)
                    FROM hourly_stats
                    WHERE crossing_id = ?
                      AND date(hour_start) >= ?
                      AND date(hour_start) <= ?
                """, (crossing_id, date_from, date_to)).fetchone()
            return (row[0], row[1]) if row else (0, 0)
        except Exception as e:
            logger.error("get_date_range_total error: %s", e)
            return (0, 0)

    def get_date_range_camera(self, crossing_id: int, camera_name: str,
                              date_from: str, date_to: str) -> Tuple[int, int]:
        """Bitta kamera uchun sana oralig'ida jami."""
        try:
            with self._lock:
                row = self._conn.execute("""
                    SELECT COALESCE(SUM(light_count), 0),
                           COALESCE(SUM(heavy_count), 0)
                    FROM hourly_stats
                    WHERE crossing_id = ?
                      AND camera_name = ?
                      AND date(hour_start) >= ?
                      AND date(hour_start) <= ?
                """, (crossing_id, camera_name, date_from, date_to)).fetchone()
            return (row[0], row[1]) if row else (0, 0)
        except Exception as e:
            logger.error("get_date_range_camera error: %s", e)
            return (0, 0)

    def get_all_totals(self) -> Dict[int, Tuple[int, int]]:
        """Barcha pereezdlar uchun bugungi jami.
        Returns: {crossing_id: (light, heavy), ...}"""
        today = date.today().isoformat()
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT crossing_id,
                           COALESCE(SUM(light_count), 0),
                           COALESCE(SUM(heavy_count), 0)
                    FROM hourly_stats
                    WHERE date(hour_start) = ?
                    GROUP BY crossing_id
                """, (today,)).fetchall()
            return {r[0]: (r[1], r[2]) for r in rows}
        except Exception as e:
            logger.error("get_all_totals error: %s", e)
            return {}

    # ─── TRAIN EVENTS ────────────────────────────────────────

    @staticmethod
    def _merge_intervals(rows: list, gap_secs: float = 180.0) -> list:
        """Yaqin joylashgan eventlarni birlashtirish.

        rows: [(start_str, end_str, duration), ...]  — start_time bo'yicha ASC tartibda.
        gap_secs: ikki event orasidagi maksimal bo'shliq (sek). Bu dan kichik bo'lsa — birlashadi.
        Returns: [(start_dt, end_dt, duration_secs), ...]
        """
        parsed = []
        for start_str, end_str, _dur in rows:
            try:
                s = datetime.fromisoformat(start_str)
                e = datetime.fromisoformat(end_str) if end_str else None
                parsed.append([s, e])
            except Exception:
                pass
        if not parsed:
            return []

        merged = [list(parsed[0])]
        for cur_s, cur_e in parsed[1:]:
            prev_e = merged[-1][1]
            if prev_e is not None and cur_s is not None:
                gap = (cur_s - prev_e).total_seconds()
                if gap < gap_secs:
                    # Birlashtirish: yangi end ni kattaroq qilib olish
                    if cur_e is None:
                        merged[-1][1] = None
                    elif prev_e is None or cur_e > prev_e:
                        merged[-1][1] = cur_e
                    continue
            merged.append([cur_s, cur_e])

        result = []
        for s, e in merged:
            dur = (e - s).total_seconds() if e else None
            result.append((s, e, dur))
        return result

    def record_train_event(self, crossing_id: int,
                           start_dt: datetime, end_dt: datetime):
        """Tugallangan poyezd o'tishini bir vaqtda yozish (start + end birgalikda).
        Yolg'on qisqa signallardan himoya: faqat davomiylik >= MIN_DURATION bo'lganda chaqiriladi.
        """
        duration = (end_dt - start_dt).total_seconds()
        try:
            with self._lock:
                self._conn.execute("""
                    INSERT INTO train_events
                        (crossing_id, start_time, end_time, duration_seconds, event_date)
                    VALUES (?, ?, ?, ?, ?)
                """, (crossing_id, start_dt.isoformat(), end_dt.isoformat(),
                      duration, start_dt.date().isoformat()))
                self._conn.commit()
        except Exception as e:
            logger.error("record_train_event error: %s", e)

    def record_train_start(self, crossing_id: int):
        """Eski usul (to'g'ridan-to'g'ri foydalanilmaydi, moslik uchun saqlanadi)."""
        now = datetime.now()
        try:
            with self._lock:
                existing = self._conn.execute("""
                    SELECT id FROM train_events
                    WHERE crossing_id = ? AND end_time IS NULL
                    LIMIT 1
                """, (crossing_id,)).fetchone()
                if existing:
                    return
                self._conn.execute("""
                    INSERT INTO train_events (crossing_id, start_time, event_date)
                    VALUES (?, ?, ?)
                """, (crossing_id, now.isoformat(), now.date().isoformat()))
                self._conn.commit()
        except Exception as e:
            logger.error("record_train_start error: %s", e)

    def record_train_end(self, crossing_id: int):
        """Eski usul (to'g'ridan-to'g'ri foydalanilmaydi, moslik uchun saqlanadi)."""
        now = datetime.now()
        try:
            with self._lock:
                row = self._conn.execute("""
                    SELECT id, start_time FROM train_events
                    WHERE crossing_id = ? AND end_time IS NULL
                    ORDER BY id DESC LIMIT 1
                """, (crossing_id,)).fetchone()
                if row:
                    start = datetime.fromisoformat(row[1])
                    duration = (now - start).total_seconds()
                    self._conn.execute("""
                        UPDATE train_events
                        SET end_time = ?, duration_seconds = ?
                        WHERE id = ?
                    """, (now.isoformat(), duration, row[0]))
                    self._conn.commit()
        except Exception as e:
            logger.error("record_train_end error: %s", e)

    def get_train_today_stats(self, crossing_id: int,
                               target_date: Optional[str] = None) -> Dict:
        """Bugungi poyezd statistikasi (birlashtirilgan).
        target_date: "2026-05-13" format (None = bugun).
        Returns: {"count": 5, "min": 45.2, "max": 120.5, "avg": 78.3}"""
        if target_date is None:
            target_date = date.today().isoformat()
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT start_time, end_time, duration_seconds
                    FROM train_events
                    WHERE crossing_id = ? AND event_date = ?
                      AND end_time IS NOT NULL
                    ORDER BY start_time ASC
                """, (crossing_id, target_date)).fetchall()
            merged = self._merge_intervals(rows)
            durations = [d for _, _, d in merged if d is not None]
            if not durations:
                return {"count": 0, "min": 0, "max": 0, "avg": 0}
            return {
                "count": len(durations),
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
            }
        except Exception as e:
            logger.error("get_train_today_stats error: %s", e)
            return {"count": 0, "min": 0, "max": 0, "avg": 0}

    def _get_raw_events(self, crossing_id: int,
                        date_from: str, date_to: str) -> list:
        """Berilgan sana oralig'idagi xom eventlarni qaytarish (ASC tartib)."""
        rows = self._conn.execute("""
            SELECT start_time, end_time, duration_seconds
            FROM train_events
            WHERE crossing_id = ?
              AND event_date >= ? AND event_date <= ?
            ORDER BY start_time ASC
        """, (crossing_id, date_from, date_to)).fetchall()
        return rows

    def get_train_weekly(self, crossing_id: int,
                         date_to: Optional[date] = None) -> List[Dict]:
        """Oxirgi 7 kun poyezd soni (birlashtirilgan), hafta kuni tartibi (Du→Ya).
        date_to: oxirgi kun (None = bugun)."""
        days_uz = ["Du", "Se", "Cho", "Pa", "Ju", "Sha", "Ya"]
        if date_to is None:
            date_to = date.today()
        date_from = (date_to - timedelta(days=6)).isoformat()
        try:
            with self._lock:
                raw = self._get_raw_events(crossing_id, date_from, date_to.isoformat())
            # Har kun uchun alohida birlashtirish
            by_day: Dict[str, list] = {}
            for r in raw:
                d = r[0][:10]
                by_day.setdefault(d, []).append(r)
            db_map = {}
            for ds, day_rows in by_day.items():
                merged = self._merge_intervals(day_rows)
                durations = [d for _, _, d in merged if d is not None]
                avg = sum(durations) / len(durations) if durations else 0
                db_map[ds] = (len(merged), avg)
            data = []
            # Sana (xronologik) tartibida — eng eski kundan eng yangisiga.
            for i in range(6, -1, -1):
                d = date_to - timedelta(days=i)
                ds = d.isoformat()
                count, avg = db_map.get(ds, (0, 0))
                data.append({"date": ds, "day": days_uz[d.weekday()], "count": count, "avg": avg})
            return data
        except Exception as e:
            logger.error("get_train_weekly error: %s", e)
            return []

    def get_train_monthly(self, crossing_id: int) -> List[Dict]:
        """Oxirgi 30 kun poyezd soni (birlashtirilgan)."""
        today = date.today()
        date_from = (today - timedelta(days=29)).isoformat()
        try:
            with self._lock:
                raw = self._get_raw_events(crossing_id, date_from, today.isoformat())
            by_day: Dict[str, list] = {}
            for r in raw:
                d = r[0][:10]
                by_day.setdefault(d, []).append(r)
            db_map = {}
            for ds, day_rows in by_day.items():
                merged = self._merge_intervals(day_rows)
                durations = [d for _, _, d in merged if d is not None]
                avg = sum(durations) / len(durations) if durations else 0
                db_map[ds] = (len(merged), avg)
            data = []
            for i in range(29, -1, -1):
                d = today - timedelta(days=i)
                ds = d.isoformat()
                count, avg = db_map.get(ds, (0, 0))
                data.append({"date": ds, "day": d.day, "count": count, "avg": avg})
            return data
        except Exception as e:
            logger.error("get_train_monthly error: %s", e)
            return []

    def get_all_train_today(self) -> Dict[int, int]:
        """Barcha pereezdlar bugungi poyezd soni (birlashtirilgan)."""
        today = date.today().isoformat()
        try:
            with self._lock:
                cids = [r[0] for r in self._conn.execute(
                    "SELECT DISTINCT crossing_id FROM train_events WHERE event_date = ?",
                    (today,)).fetchall()]
            result = {}
            for cid in cids:
                result[cid] = self.get_train_today_stats(cid)["count"]
            return result
        except Exception as e:
            logger.error("get_all_train_today error: %s", e)
            return {}

    def get_train_today_count(self, crossing_id: int) -> int:
        """Bugungi birlashtirilgan poyezdlar soni."""
        return self.get_train_today_stats(crossing_id)["count"]

    def get_train_events_today(self, crossing_id: int,
                               target_date: Optional[str] = None) -> List[Dict]:
        """Bugungi har bir poyezd o'tishini birlashtirilib qaytarish.
        Returns: [{"start": "12:00", "end": "12:06", "duration": 360.0, "in_progress": False}, ...]
        Oxirgi event birinchi (teskari tartib).
        """
        if target_date is None:
            target_date = date.today().isoformat()
        result = []
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT start_time, end_time, duration_seconds
                    FROM train_events
                    WHERE crossing_id = ? AND event_date = ?
                    ORDER BY start_time ASC
                """, (crossing_id, target_date)).fetchall()

            # Ochiq (jarayondagi) eventni ajratish
            closed = [r for r in rows if r[1] is not None]
            open_ev = [r for r in rows if r[1] is None]

            merged = self._merge_intervals(closed)

            # Ochiq event (hozir o'tayotgan) — eng birinchi
            for start_str, _, _ in open_ev:
                try:
                    s = datetime.fromisoformat(start_str)
                    result.append({
                        "start": s.strftime("%H:%M"),
                        "end": "...",
                        "duration": 0.0,
                        "in_progress": True,
                    })
                except Exception:
                    pass

            for s_dt, e_dt, dur in reversed(merged):
                try:
                    result.append({
                        "start": s_dt.strftime("%H:%M"),
                        "end": e_dt.strftime("%H:%M") if e_dt else "...",
                        "duration": dur or 0.0,
                        "in_progress": False,
                    })
                except Exception:
                    pass
        except Exception as e:
            logger.error("get_train_events_today error: %s", e)
        return result

    def get_train_hourly_data(self, crossing_id: int,
                              target_date: Optional[str] = None) -> List[int]:
        """24 soatlik poyezd soni (grafik uchun, birlashtirilgan). Returns: [0]*24 list."""
        if target_date is None:
            target_date = date.today().isoformat()
        counts = [0] * 24
        try:
            with self._lock:
                rows = self._conn.execute("""
                    SELECT start_time, end_time, duration_seconds
                    FROM train_events
                    WHERE crossing_id = ? AND event_date = ?
                      AND end_time IS NOT NULL
                    ORDER BY start_time ASC
                """, (crossing_id, target_date)).fetchall()
            merged = self._merge_intervals(rows)
            for s_dt, _e, _d in merged:
                h = s_dt.hour
                if 0 <= h < 24:
                    counts[h] += 1
        except Exception as e:
            logger.error("get_train_hourly_data error: %s", e)
        return counts

    def get_train_range_stats(self, crossing_id: int,
                              date_from: str, date_to: str) -> Dict:
        """Sana oralig'ida birlashtirilgan poyezd statistikasi.
        Returns: {"count": N, "min": s, "max": s, "avg": s}
        """
        try:
            with self._lock:
                raw = self._get_raw_events(crossing_id, date_from, date_to)
            # Kunlik birlashtirish
            by_day: Dict[str, list] = {}
            for r in raw:
                d = r[0][:10]
                by_day.setdefault(d, []).append(r)
            all_durations = []
            for day_rows in by_day.values():
                for _, _, dur in self._merge_intervals(day_rows):
                    if dur is not None:
                        all_durations.append(dur)
            if not all_durations:
                return {"count": 0, "min": 0, "max": 0, "avg": 0}
            return {
                "count": len(all_durations),
                "min": min(all_durations),
                "max": max(all_durations),
                "avg": sum(all_durations) / len(all_durations),
            }
        except Exception as e:
            logger.error("get_train_range_stats error: %s", e)
            return {"count": 0, "min": 0, "max": 0, "avg": 0}

    def get_train_events_range(self, crossing_id: int,
                               date_from: str, date_to: str) -> List[Dict]:
        """Sana oralig'idagi har bir birlashtirilgan poyezd o'tishi.
        Returns: [{"date": "12.03.2026", "start": "09:30", "end": "09:36",
                   "duration_secs": 360.0, "duration_fmt": "6 daq 0 son"}, ...]
        """
        result = []
        try:
            with self._lock:
                raw = self._get_raw_events(crossing_id, date_from, date_to)
            by_day: Dict[str, list] = {}
            for r in raw:
                d = r[0][:10]
                by_day.setdefault(d, []).append(r)
            for ds in sorted(by_day.keys()):
                for s_dt, e_dt, dur in self._merge_intervals(by_day[ds]):
                    if e_dt is None or dur is None:
                        continue
                    m = int(dur) // 60
                    s = int(dur) % 60
                    result.append({
                        "date": s_dt.strftime("%d.%m.%Y"),
                        "start": s_dt.strftime("%H:%M"),
                        "end": e_dt.strftime("%H:%M"),
                        "duration_secs": dur,
                        "duration_fmt": f"{m} daq {s} son" if m > 0 else f"{s} son",
                    })
        except Exception as e:
            logger.error("get_train_events_range error: %s", e)
        return result

    def rename_camera(self, crossing_id: int, old_name: str, new_name: str):
        """Kamera nomi o'zgarganda barcha statslarni yangi nomga ko'chirish."""
        if old_name == new_name:
            return
        with self._lock:
            self._conn.execute("""
                UPDATE hourly_stats SET camera_name = ?
                WHERE crossing_id = ? AND camera_name = ?
            """, (new_name, crossing_id, old_name))
            self._conn.commit()
            # Delta tracking cache ni ham yangilash (lock ichida)
            old_key = (crossing_id, old_name)
            if old_key in self._last_counts:
                self._last_counts[(crossing_id, new_name)] = self._last_counts.pop(old_key)

    def close(self):
        with self._lock:
            # Yopishdan oldin yig'ilgan bandlik vaqtini yozib qo'yamiz
            for key, st in self._occ_state.items():
                try:
                    self._flush_occupancy(key, st)
                except Exception:
                    pass
            self._conn.close()
