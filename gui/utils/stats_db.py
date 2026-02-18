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
        """
        key = (crossing_id, camera_name)
        last_l, last_h = self._last_counts.get(key, (0, 0))
        delta_l = max(0, light - last_l)
        delta_h = max(0, heavy - last_h)
        self._last_counts[key] = (light, heavy)

        if delta_l == 0 and delta_h == 0:
            return

        hour = self._current_hour()
        now = datetime.now().isoformat()
        with self._lock:
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
        """Oxirgi 7 kun (kunlik jami).
        Returns: [{"date": "2026-02-11", "day": "Du", "light": 10, "heavy": 3}, ...]"""
        days_uz = ["Du", "Se", "Cho", "Pa", "Ju", "Sha", "Ya"]
        today = date.today()
        data = []
        with self._lock:
            for i in range(6, -1, -1):
                d = today - timedelta(days=i)
                ds = d.isoformat()
                row = self._conn.execute("""
                    SELECT COALESCE(SUM(light_count), 0),
                           COALESCE(SUM(heavy_count), 0)
                    FROM hourly_stats
                    WHERE crossing_id = ? AND date(hour_start) = ?
                """, (crossing_id, ds)).fetchone()
                data.append({
                    "date": ds,
                    "day": days_uz[d.weekday()],
                    "light": row[0] if row else 0,
                    "heavy": row[1] if row else 0,
                })
        return data

    def get_monthly_data(self, crossing_id: int) -> List[Dict]:
        """Oxirgi 30 kun (kunlik jami).
        Returns: [{"date": "2026-01-13", "light": 10, "heavy": 3}, ...]"""
        today = date.today()
        data = []
        with self._lock:
            for i in range(29, -1, -1):
                d = today - timedelta(days=i)
                ds = d.isoformat()
                row = self._conn.execute("""
                    SELECT COALESCE(SUM(light_count), 0),
                           COALESCE(SUM(heavy_count), 0)
                    FROM hourly_stats
                    WHERE crossing_id = ? AND date(hour_start) = ?
                """, (crossing_id, ds)).fetchone()
                data.append({
                    "date": ds,
                    "day": d.day,
                    "light": row[0] if row else 0,
                    "heavy": row[1] if row else 0,
                })
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
        """Oxirgi 7 kun heatmap: har kun uchun 24 soatlik ma'lumot.
        Returns: [{"date": "...", "day": "Du", "hours": [0]*24}, ...] (7 ta)
        """
        days_uz = ["Du", "Se", "Cho", "Pa", "Ju", "Sha", "Ya"]
        today = date.today()
        data = []
        with self._lock:
            for i in range(6, -1, -1):
                d = today - timedelta(days=i)
                ds = d.isoformat()
                row = [0] * 24
                rows = self._conn.execute("""
                    SELECT CAST(strftime('%H', hour_start) AS INTEGER) as hour,
                           COALESCE(SUM(light_count), 0) + COALESCE(SUM(heavy_count), 0)
                    FROM hourly_stats
                    WHERE crossing_id = ? AND date(hour_start) = ?
                    GROUP BY hour
                """, (crossing_id, ds)).fetchall()
                for hour, total in rows:
                    if 0 <= hour < 24:
                        row[hour] = total or 0
                data.append({
                    "date": ds,
                    "day": days_uz[d.weekday()],
                    "hours": row
                })
        return data

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

    def record_train_start(self, crossing_id: int):
        """PLC ON bo'lganda — poyezd kelmoqda."""
        now = datetime.now()
        with self._lock:
            self._conn.execute("""
                INSERT INTO train_events (crossing_id, start_time, event_date)
                VALUES (?, ?, ?)
            """, (crossing_id, now.isoformat(), now.date().isoformat()))
            self._conn.commit()

    def record_train_end(self, crossing_id: int):
        """PLC OFF bo'lganda — poyezd o'tdi. Oxirgi ochiq eventni yopish."""
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
        """Bugungi poyezd statistikasi.
        Returns: {"count": 5, "min": 45.2, "max": 120.5, "avg": 78.3}"""
        today = date.today().isoformat()
        with self._lock:
            row = self._conn.execute("""
                SELECT COUNT(*),
                       COALESCE(MIN(duration_seconds), 0),
                       COALESCE(MAX(duration_seconds), 0),
                       COALESCE(AVG(duration_seconds), 0)
                FROM train_events
                WHERE crossing_id = ? AND event_date = ?
                  AND duration_seconds IS NOT NULL
            """, (crossing_id, today)).fetchone()
        return {
            "count": row[0] if row else 0,
            "min": row[1] if row else 0,
            "max": row[2] if row else 0,
            "avg": row[3] if row else 0,
        }

    def get_train_weekly(self, crossing_id: int) -> List[Dict]:
        """Oxirgi 7 kun poyezd soni + o'rtacha vaqt.
        Returns: [{"date": "...", "day": "Du", "count": 3, "avg": 65.0}, ...]"""
        days_uz = ["Du", "Se", "Cho", "Pa", "Ju", "Sha", "Ya"]
        today = date.today()
        data = []
        with self._lock:
            for i in range(6, -1, -1):
                d = today - timedelta(days=i)
                ds = d.isoformat()
                row = self._conn.execute("""
                    SELECT COUNT(*), COALESCE(AVG(duration_seconds), 0)
                    FROM train_events
                    WHERE crossing_id = ? AND event_date = ?
                      AND duration_seconds IS NOT NULL
                """, (crossing_id, ds)).fetchone()
                data.append({
                    "date": ds,
                    "day": days_uz[d.weekday()],
                    "count": row[0] if row else 0,
                    "avg": row[1] if row else 0,
                })
        return data

    def get_train_monthly(self, crossing_id: int) -> List[Dict]:
        """Oxirgi 30 kun poyezd soni.
        Returns: [{"date": "...", "day": 13, "count": 3, "avg": 65.0}, ...]"""
        today = date.today()
        data = []
        with self._lock:
            for i in range(29, -1, -1):
                d = today - timedelta(days=i)
                ds = d.isoformat()
                row = self._conn.execute("""
                    SELECT COUNT(*), COALESCE(AVG(duration_seconds), 0)
                    FROM train_events
                    WHERE crossing_id = ? AND event_date = ?
                      AND duration_seconds IS NOT NULL
                """, (crossing_id, ds)).fetchone()
                data.append({
                    "date": ds,
                    "day": d.day,
                    "count": row[0] if row else 0,
                    "avg": row[1] if row else 0,
                })
        return data

    def get_all_train_today(self) -> Dict[int, int]:
        """Barcha pereezdlar bugungi poyezd soni.
        Returns: {crossing_id: count, ...}"""
        today = date.today().isoformat()
        with self._lock:
            rows = self._conn.execute("""
                SELECT crossing_id, COUNT(*)
                FROM train_events
                WHERE event_date = ? AND duration_seconds IS NOT NULL
                GROUP BY crossing_id
            """, (today,)).fetchall()
        return {r[0]: r[1] for r in rows}

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
        # Delta tracking cache ni ham yangilash
        old_key = (crossing_id, old_name)
        if old_key in self._last_counts:
            self._last_counts[(crossing_id, new_name)] = self._last_counts.pop(old_key)

    def close(self):
        with self._lock:
            self._conn.close()
