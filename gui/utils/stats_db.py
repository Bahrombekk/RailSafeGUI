"""
StatsDB - SQLite database for hourly/daily vehicle counting statistics.
Thread-safe. Stores per-camera, per-hour counts. Auto-resets at midnight.
"""

import sqlite3
import threading
import os
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Optional


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
            self._last_counts[key] = (light, heavy)

            if delta_l == 0 and delta_h == 0:
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
            except Exception as e:
                print(f"[StatsDB] record_count error: {e}")

    def get_today_total(self, crossing_id: int) -> Tuple[int, int]:
        """Bugungi kun uchun jami (barcha kameralar, 00:00 dan hozirga).
        Returns: (light_total, heavy_total)"""
        today = date.today().isoformat()
        with self._lock:
            row = self._conn.execute("""
                SELECT COALESCE(SUM(light_count), 0),
                       COALESCE(SUM(heavy_count), 0)
                FROM hourly_stats
                WHERE crossing_id = ?
                  AND date(hour_start) = ?
            """, (crossing_id, today)).fetchone()
        return (row[0], row[1]) if row else (0, 0)

    def get_camera_today(self, crossing_id: int,
                         camera_name: str) -> Tuple[int, int]:
        """Bitta kamera uchun bugungi jami.
        Returns: (light, heavy)"""
        today = date.today().isoformat()
        with self._lock:
            row = self._conn.execute("""
                SELECT COALESCE(SUM(light_count), 0),
                       COALESCE(SUM(heavy_count), 0)
                FROM hourly_stats
                WHERE crossing_id = ?
                  AND camera_name = ?
                  AND date(hour_start) = ?
            """, (crossing_id, camera_name, today)).fetchone()
        return (row[0], row[1]) if row else (0, 0)

    def get_hourly_data(self, crossing_id: int,
                        target_date: Optional[str] = None) -> List[Dict]:
        """24 soatlik ma'lumot (grafik uchun).
        Returns: [{"hour": 0, "light": 5, "heavy": 2}, ...] (24 ta element)"""
        if target_date is None:
            target_date = date.today().isoformat()

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

        # 24 soat uchun to'liq massiv
        data = [{"hour": h, "light": 0, "heavy": 0} for h in range(24)]
        for hour, light, heavy in rows:
            if 0 <= hour < 24:
                data[hour]["light"] = light or 0
                data[hour]["heavy"] = heavy or 0
        return data

    def get_weekly_data(self, crossing_id: int) -> List[Dict]:
        """Oxirgi 7 kun (kunlik jami), hafta kuni tartibi bilan (Du→Ya).
        Returns: [{"date": "2026-02-11", "day": "Du", "light": 10, "heavy": 3}, ...]"""
        days_uz = ["Du", "Se", "Cho", "Pa", "Ju", "Sha", "Ya"]
        today = date.today()
        date_from = (today - timedelta(days=6)).isoformat()
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
        for i in range(6, -1, -1):
            d = today - timedelta(days=i)
            ds = d.isoformat()
            light, heavy = db_map.get(ds, (0, 0))
            data.append({"date": ds, "day": days_uz[d.weekday()], "light": light, "heavy": heavy})
        data.sort(key=lambda x: days_uz.index(x["day"]))
        return data

    def get_monthly_data(self, crossing_id: int) -> List[Dict]:
        """Oxirgi 30 kun (kunlik jami).
        Returns: [{"date": "2026-01-13", "light": 10, "heavy": 3}, ...]"""
        today = date.today()
        date_from = (today - timedelta(days=29)).isoformat()
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

    def get_yearly_data(self, crossing_id: int) -> List[Dict]:
        """Oxirgi 12 oy (oylik jami).
        Returns: [{"month": "2025-03", "label": "Mar", "light": 100, "heavy": 30}, ...]"""
        months_uz = ["Yan", "Fev", "Mar", "Apr", "May", "Iyn",
                     "Iyl", "Avg", "Sen", "Okt", "Noy", "Dek"]
        today = date.today()
        data = []
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

    def get_heatmap_data(self, crossing_id: int) -> List[Dict]:
        """Oxirgi 7 kun heatmap: har kun uchun 24 soatlik ma'lumot, hafta kuni tartibi (Du→Ya).
        Returns: [{"date": "...", "day": "Du", "hours": [0]*24}, ...] (7 ta)
        """
        days_uz = ["Du", "Se", "Cho", "Pa", "Ju", "Sha", "Ya"]
        today = date.today()
        date_from = (today - timedelta(days=6)).isoformat()
        with self._lock:
            rows = self._conn.execute("""
                SELECT date(hour_start) as d,
                       CAST(strftime('%H', hour_start) AS INTEGER) as h,
                       COALESCE(SUM(light_count), 0) + COALESCE(SUM(heavy_count), 0)
                FROM hourly_stats
                WHERE crossing_id = ?
                  AND date(hour_start) >= ? AND date(hour_start) <= ?
                GROUP BY d, h
            """, (crossing_id, date_from, today.isoformat())).fetchall()
        db_map = {}
        for d_str, h, total in rows:
            db_map.setdefault(d_str, {})[h] = total
        data = []
        for i in range(6, -1, -1):
            d = today - timedelta(days=i)
            ds = d.isoformat()
            hours = [db_map.get(ds, {}).get(h, 0) for h in range(24)]
            data.append({"date": ds, "day": days_uz[d.weekday()], "hours": hours})
        data.sort(key=lambda x: days_uz.index(x["day"]))
        return data

    def get_date_range_daily(self, crossing_id: int,
                             date_from: str, date_to: str) -> List[Dict]:
        """Belgilangan sana oralig'ida kunlik statistika.
        date_from, date_to: "2026-02-01" format.
        Returns: [{"date": "2026-02-01", "light": 10, "heavy": 3}, ...]"""
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

    def get_date_range_total(self, crossing_id: int,
                             date_from: str, date_to: str) -> Tuple[int, int]:
        """Belgilangan sana oralig'ida jami (light, heavy)."""
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

    def get_date_range_camera(self, crossing_id: int, camera_name: str,
                              date_from: str, date_to: str) -> Tuple[int, int]:
        """Bitta kamera uchun sana oralig'ida jami."""
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

    def get_all_totals(self) -> Dict[int, Tuple[int, int]]:
        """Barcha pereezdlar uchun bugungi jami.
        Returns: {crossing_id: (light, heavy), ...}"""
        today = date.today().isoformat()
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
        with self._lock:
            self._conn.execute("""
                INSERT INTO train_events
                    (crossing_id, start_time, end_time, duration_seconds, event_date)
                VALUES (?, ?, ?, ?, ?)
            """, (crossing_id, start_dt.isoformat(), end_dt.isoformat(),
                  duration, start_dt.date().isoformat()))
            self._conn.commit()

    def record_train_start(self, crossing_id: int):
        """Eski usul (to'g'ridan-to'g'ri foydalanilmaydi, moslik uchun saqlanadi)."""
        now = datetime.now()
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

    def record_train_end(self, crossing_id: int):
        """Eski usul (to'g'ridan-to'g'ri foydalanilmaydi, moslik uchun saqlanadi)."""
        now = datetime.now()
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

    def get_train_today_stats(self, crossing_id: int) -> Dict:
        """Bugungi poyezd statistikasi (birlashtirilgan).
        Returns: {"count": 5, "min": 45.2, "max": 120.5, "avg": 78.3}"""
        today = date.today().isoformat()
        with self._lock:
            rows = self._conn.execute("""
                SELECT start_time, end_time, duration_seconds
                FROM train_events
                WHERE crossing_id = ? AND event_date = ?
                  AND end_time IS NOT NULL
                ORDER BY start_time ASC
            """, (crossing_id, today)).fetchall()
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

    def get_train_weekly(self, crossing_id: int) -> List[Dict]:
        """Oxirgi 7 kun poyezd soni (birlashtirilgan), hafta kuni tartibi (Du→Ya)."""
        days_uz = ["Du", "Se", "Cho", "Pa", "Ju", "Sha", "Ya"]
        today = date.today()
        date_from = (today - timedelta(days=6)).isoformat()
        with self._lock:
            raw = self._get_raw_events(crossing_id, date_from, today.isoformat())
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
        for i in range(6, -1, -1):
            d = today - timedelta(days=i)
            ds = d.isoformat()
            count, avg = db_map.get(ds, (0, 0))
            data.append({"date": ds, "day": days_uz[d.weekday()], "count": count, "avg": avg})
        data.sort(key=lambda x: days_uz.index(x["day"]))
        return data

    def get_train_monthly(self, crossing_id: int) -> List[Dict]:
        """Oxirgi 30 kun poyezd soni (birlashtirilgan)."""
        today = date.today()
        date_from = (today - timedelta(days=29)).isoformat()
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

    def get_all_train_today(self) -> Dict[int, int]:
        """Barcha pereezdlar bugungi poyezd soni (birlashtirilgan)."""
        today = date.today().isoformat()
        with self._lock:
            cids = [r[0] for r in self._conn.execute(
                "SELECT DISTINCT crossing_id FROM train_events WHERE event_date = ?",
                (today,)).fetchall()]
        result = {}
        for cid in cids:
            result[cid] = self.get_train_today_stats(cid)["count"]
        return result

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

        result = []
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
        return result

    def get_train_hourly_data(self, crossing_id: int,
                              target_date: Optional[str] = None) -> List[int]:
        """24 soatlik poyezd soni (grafik uchun, birlashtirilgan). Returns: [0]*24 list."""
        if target_date is None:
            target_date = date.today().isoformat()
        with self._lock:
            rows = self._conn.execute("""
                SELECT start_time, end_time, duration_seconds
                FROM train_events
                WHERE crossing_id = ? AND event_date = ?
                  AND end_time IS NOT NULL
                ORDER BY start_time ASC
            """, (crossing_id, target_date)).fetchall()
        merged = self._merge_intervals(rows)
        counts = [0] * 24
        for s_dt, _e, _d in merged:
            h = s_dt.hour
            if 0 <= h < 24:
                counts[h] += 1
        return counts

    def get_train_range_stats(self, crossing_id: int,
                              date_from: str, date_to: str) -> Dict:
        """Sana oralig'ida birlashtirilgan poyezd statistikasi.
        Returns: {"count": N, "min": s, "max": s, "avg": s}
        """
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

    def get_train_events_range(self, crossing_id: int,
                               date_from: str, date_to: str) -> List[Dict]:
        """Sana oralig'idagi har bir birlashtirilgan poyezd o'tishi.
        Returns: [{"date": "12.03.2026", "start": "09:30", "end": "09:36",
                   "duration_secs": 360.0, "duration_fmt": "6 daq 0 son"}, ...]
        """
        with self._lock:
            raw = self._get_raw_events(crossing_id, date_from, date_to)
        by_day: Dict[str, list] = {}
        for r in raw:
            d = r[0][:10]
            by_day.setdefault(d, []).append(r)
        result = []
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
            self._conn.close()
